# benchmarks.py

import argparse
import os
import re
import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
import sys
import traceback  # Ensure traceback is imported

# --- Configuration & Constants ---
METRICS_CONFIG = OrderedDict(
    [
        ("PSNR", ("psnr_metric.py", True)),
        ("SSIM", ("ssim_metric.py", True)),
        ("LPIPS", ("lpips_metric.py", False)),
        ("DreamSim", ("dreamsim_metric.py", False)),
        ("DINO", ("dino_metric.py", True)),
        ("CLIP", ("clip_metric.py", True)),
    ]
)
LOFTR_FILTER_RATES = [0.0, 0.25, 0.50, 0.75]
DEFAULT_NUM_IMAGES = 16
MASTER_CACHE_FILENAME = "master_results_cache.json"
PER_IMAGE_CACHE_BASE = "per_scene_cache"
LOFTR_RANKING_FILENAME = "loftr_ranking_scores.json"


# --- Helper Functions ---
def find_gt_mask_paths(results_dir_path, dataset_dirs_map):
    """Finds GT and Mask paths based on a results folder name."""
    dir_name = results_dir_path.name
    match = re.match(r"^(RealBench|Custom)-(\d+)(-results.*)?$", dir_name)
    if not match:
        return None, None
    benchmark_type, scene_number = match.group(1), match.group(2)
    dataset_base_dir = dataset_dirs_map.get(benchmark_type)
    if not dataset_base_dir or not dataset_base_dir.is_dir():
        return None, None
    # Construct path based on expected structure
    # Example: dataset_base_dir / "RealBench" / "0" / "target" / "gt.png"
    base_path = dataset_base_dir / benchmark_type / scene_number / "target"
    gt_path, mask_path = base_path / "gt.png", base_path / "mask.png"
    if not gt_path.is_file() or not mask_path.is_file():
        # Fallback for slightly different structures if needed, or just return None
        # print(f"Debug: Checked {gt_path}, {mask_path} - Not found.") # Optional debug
        return None, None
    return str(gt_path), str(mask_path)


def count_result_images(folder_path: Path) -> int:
    """Counts the number of files named like '0.png', '1.png', etc."""
    if not folder_path.is_dir():
        return 0
    count = 0
    try:
        for item in folder_path.iterdir():
            if item.is_file() and item.suffix == ".png" and item.stem.isdigit():
                count += 1
        return count
    except OSError as e:
        # Use tqdm.write if possible, otherwise print
        write_func = tqdm.write if sys.stdout.isatty() else print
        write_func(f"Warning: Could not count images in {folder_path}: {e}")
        return 0


def parse_final_score(stdout_str):
    """Parses the FINAL_SCORE line from metric script output."""
    if not isinstance(stdout_str, str):  # Basic type check
        return None
    for line in stdout_str.splitlines():
        if line.startswith("FINAL_SCORE:"):
            score_part = line.split(":", 1)[1].strip()
            if score_part == "ERROR":
                return None
            try:
                return float(score_part)
            except ValueError:
                # Use tqdm.write if possible, otherwise print
                write_func = tqdm.write if sys.stdout.isatty() else print
                write_func(f"Warning: Could not parse score from line: {line}")
                return None
    # Added warning if the line is not found at all
    # write_func = tqdm.write if sys.stdout.isatty() else print
    # write_func(f"Warning: FINAL_SCORE line not found in output.")
    return None


def load_json_cache(file_path):
    """Safely loads JSON data from a file path."""
    if not file_path or not Path(file_path).is_file():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, TypeError) as e:
        print(f"Warn: Cache load fail {file_path}: {e}")
        return None


def save_json_cache(data, file_path):
    """Safely saves data to a JSON file, handling numpy types."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure data is JSON serializable (convert numpy types if necessary)
        serializable_data = convert_numpy_types(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)
    except (OSError, TypeError) as e:
        print(f"Error saving JSON cache {file_path}: {e}")
        # Print details about the problematic data if TypeError
        if isinstance(e, TypeError):
            print(f"Problematic data snippet: {str(data)[:500]}")  # Log beginning of data


def convert_numpy_types(obj):
    """Recursively converts numpy types in a dict/list to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(
        obj,
        (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64),
    ):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        # Check for NaN/Inf which are not valid JSON
        if np.isnan(obj):
            return None  # Represent NaN as null in JSON
        elif np.isinf(obj):
            # Represent Inf as a large number or None, depending on context. None is safer.
            return None  # Represent Inf as null
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())  # Convert arrays to lists
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.void):
        return None  # Represent void/structured types as null
    return obj  # Return object itself if not a numpy type


def get_scene_key(folder_name):
    """Extracts the benchmark type and scene number (e.g., 'RealBench-0')."""
    match = re.match(r"^(RealBench|Custom)-(\d+)(-results.*)?$", folder_name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def run_metric_script_parallel(
    metric_name, script_path_str, gt_path_str, mask_path_str, results_dir_str, cache_dir_str, num_images
):
    """Wrapper to run a metric script, designed for multiprocessing pool.
    Returns (metric_name, score, folder_name)
    """
    script_name = Path(script_path_str).name  # Expecting just the name now
    results_dir = Path(results_dir_str)
    cache_dir = Path(cache_dir_str)
    folder_name = results_dir.name

    # Find the benchmark directory relative to *this* script's location
    benchmark_dir = Path(__file__).parent / "benchmark"
    absolute_script_path = benchmark_dir / script_name

    # Use tqdm.write within the worker if possible (it might not render correctly but won't break parent tqdm)
    write_func = tqdm.write if sys.stdout.isatty() else print

    # Ensure script exists
    if not absolute_script_path.is_file():
        write_func(f"MP_ERROR: Script not found at {absolute_script_path}")
        return metric_name, None, folder_name

    command = [
        sys.executable,  # Use the same python interpreter that runs this script
        str(absolute_script_path),
        "--gt_path",
        str(gt_path_str),
        "--mask_path",
        str(mask_path_str),
        "--results_dir",
        str(results_dir),
        "--cache_dir",
        str(cache_dir),
        "--num_images",
        str(num_images),
    ]
    try:
        # Set cwd to the directory containing the metric scripts for relative imports etc.
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=720,  # 12 minute timeout per metric
            cwd=benchmark_dir,  # Run script from its directory
            encoding="utf-8",  # Be explicit about encoding
            errors="replace",  # Handle potential decoding errors in output
        )
        stdout_content = process.stdout or ""
        stderr_content = process.stderr or ""

        if process.returncode != 0:
            stderr_suffix = stderr_content.strip()[-500:]
            # Use write_func for output from worker
            write_func(
                f"\nMP_ERROR: {metric_name} fail {folder_name} (RC:{process.returncode}).\n"
                # f"  Cmd: {' '.join(command)}\n" # Maybe too verbose
                f"  Dir: {benchmark_dir}\n"
                f"  Stderr: ...{stderr_suffix}\n"
            )
            return metric_name, None, folder_name

        score = parse_final_score(stdout_content)
        if score is None:
            stdout_suffix = stdout_content.strip()[-500:]
            write_func(
                f"\nMP_WARN: {metric_name} score parse fail {folder_name} (RC:0).\n" f"  Stdout: ...{stdout_suffix}\n"
            )
        return metric_name, score, folder_name

    except subprocess.TimeoutExpired:
        write_func(f"\nMP_ERROR: Timeout expired for {metric_name} on {folder_name}\n")
        return metric_name, None, folder_name
    except Exception as e:
        write_func(f"\nMP_ERROR: Exception ({type(e).__name__}) for {metric_name} on {folder_name}: {e}\n")
        # Printing traceback here might intersperse badly with tqdm, maybe log instead
        # traceback.print_exc()
        return metric_name, None, folder_name


# --- Main Orchestration Class ---
class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.base_results_dir = Path(args.results_base_dir).resolve()
        self.dataset_dirs_map = {}
        # Validate and store dataset paths
        if args.realbench_dataset_dir:
            rb_path = Path(args.realbench_dataset_dir).resolve()
            if rb_path.is_dir():
                self.dataset_dirs_map["RealBench"] = rb_path
            else:
                print(f"Warn: RealBench dataset dir not found: {rb_path}")
        if args.custom_dataset_dir:
            cu_path = Path(args.custom_dataset_dir).resolve()
            if cu_path.is_dir():
                self.dataset_dirs_map["Custom"] = cu_path
            else:
                print(f"Warn: Custom dataset dir not found: {cu_path}")
        if not self.dataset_dirs_map:
            raise ValueError("No valid dataset directories provided or found.")

        self.cache_dir = Path(args.cache_dir).resolve()
        self.output_file = Path(args.output_file).resolve() if args.output_file else None
        self.num_images = args.num_images
        # Normalize force_recalc list
        self.force_recalc_metrics = [m.lower() for m in (args.force_recalc or [])]
        # Ensure metrics_to_run is a list
        self.metrics_to_run = args.metrics if isinstance(args.metrics, list) else list(METRICS_CONFIG.keys())
        # Resolve LoFTR script path relative to this file
        if args.loftr_script_path:
            candidate_path = Path(__file__).parent / args.loftr_script_path
            if candidate_path.is_file():
                self.loftr_script_path = candidate_path.resolve()
            else:
                print(
                    f"Warn: LoFTR script '{args.loftr_script_path}' not found relative to benchmarks.py. LoFTR analysis will be skipped."
                )
                self.loftr_script_path = None
        else:
            self.loftr_script_path = None

        self.master_cache_path = self.cache_dir / MASTER_CACHE_FILENAME
        self.per_image_cache_dir = self.cache_dir / PER_IMAGE_CACHE_BASE
        self.discovered_folders = []
        self.master_results = defaultdict(dict)
        self.per_image_scores = defaultdict(lambda: defaultdict(dict))
        self.loftr_ranks = defaultdict(list)

        print("Benchmark Runner Initialized.")
        print(f"Results Base: {self.base_results_dir}")
        print(f"Datasets: {self.dataset_dirs_map}")
        print(f"Cache: {self.cache_dir}")
        print(f"Metrics: {self.metrics_to_run}")
        print(f"Force Recalc: {self.force_recalc_metrics if self.force_recalc_metrics else 'None'}")
        if self.loftr_script_path:
            print(f"LoFTR Script: {self.loftr_script_path}")
        if self.output_file:
            print(f"Report Output: {self.output_file}")

        # Determine pool size
        try:
            self.cpu_cores = os.cpu_count()
            # Be conservative: Use max(1, cores - 2) or slightly less than half for hyperthreading
            self.pool_size = max(1, (self.cpu_cores or 1) // 4) if self.cpu_cores and self.cpu_cores > 1 else 1
            # self.pool_size = max(1, (self.cpu_cores or 1) - 2) if self.cpu_cores and self.cpu_cores > 2 else 1 # Alternative
            print(f"Using multiprocessing pool size: {self.pool_size} (Detected Cores: {self.cpu_cores})")
        except NotImplementedError:
            print("Could not detect CPU cores, using pool size 1.")
            self.pool_size = 1
            self.cpu_cores = None

    def discover_folders(self):
        """Discovers result folders matching the expected pattern."""
        print(f"\nScanning results folders in {self.base_results_dir}...")
        self.discovered_folders = []
        potential, skip_map, skip_gt = 0, 0, 0
        if not self.base_results_dir.is_dir():
            print(f"Error: Base results directory not found: {self.base_results_dir}")
            return

        for item in sorted(self.base_results_dir.iterdir()):  # Sort here for predictable order
            if item.is_dir() and re.match(r"^(RealBench|Custom)-\d+(-results.*)?$", item.name):
                potential += 1
                match = re.match(r"^(RealBench|Custom)", item.name)
                if match:
                    b_type = match.group(1)
                    if b_type in self.dataset_dirs_map:
                        gt, mask = find_gt_mask_paths(item, self.dataset_dirs_map)
                        if gt and mask:
                            self.discovered_folders.append(item)
                        else:
                            skip_gt += 1
                    else:
                        skip_map += 1
        # No need to sort again if sorted during iteration
        # self.discovered_folders.sort(key=lambda p: p.name)
        print(f"Scan complete. Found {potential} potential *results* folders.")
        print(f" - Added {len(self.discovered_folders)} folders with valid dataset mapping & GT/Mask files.")
        if skip_map > 0:
            print(f" - Skipped {skip_map} folders (dataset type not specified/mapped).")
        if skip_gt > 0:
            print(f" - Skipped {skip_gt} folders (missing corresponding GT/Mask).")

    def load_master_cache(self):
        """Loads the master results cache file."""
        print(f"Loading master cache: {self.master_cache_path}")
        cached = load_json_cache(self.master_cache_path)
        self.master_results = defaultdict(dict)
        if cached and isinstance(cached, dict):
            loaded_count = 0
            for folder_name, metrics_dict in cached.items():
                if isinstance(metrics_dict, dict):
                    self.master_results[folder_name] = metrics_dict
                    loaded_count += 1
                else:
                    print(f"Warn: Invalid cache entry for '{folder_name}'. Skipping.")
            print(f"Loaded {loaded_count} folder entries from master cache.")
        else:
            print("No valid master cache found or cache is empty.")

    def save_master_cache(self):
        """Saves the current master results to the cache file."""
        print(f"Saving master cache ({len(self.master_results)} entries)...")
        save_json_cache(dict(self.master_results), self.master_cache_path)
        print(f"Master cache saved to: {self.master_cache_path}")

    def check_folder_contents(self, folder_path: Path) -> bool:
        """Checks if the folder contains the expected number of result images."""
        if not folder_path.is_dir():
            return False
        try:
            return all((folder_path / f"{i}.png").is_file() for i in range(self.num_images))
        except Exception as e:
            write_func = tqdm.write if sys.stdout.isatty() else print
            write_func(f"Error checking contents of {folder_path}: {e}")
            return False

    def run_all_metrics(self):
        """Runs metrics in parallel across all folders, showing overall progress."""
        if not self.discovered_folders:
            print("No result folders discovered to run metrics on.")
            return
        print(f"\n--- Preparing Metric Tasks ---")
        self.load_master_cache()  # Load cache before identifying tasks
        skipped_incomplete = 0
        tasks_to_run = []
        folders_processed_count = 0
        folders_with_tasks = set()
        metrics_being_run = set()
        valid_metrics_found = set(METRICS_CONFIG.keys())  # Start with all configured metrics

        # --- Stage 1: Collect all tasks that need running ---
        print("Checking folders and identifying required metric calculations...")
        benchmark_base_dir = Path(__file__).parent
        for fp in tqdm(
            self.discovered_folders, desc="Checking Folders & Cache", unit="folder", disable=not sys.stdout.isatty()
        ):
            fname = fp.name
            gt, mask = find_gt_mask_paths(fp, self.dataset_dirs_map)
            if not gt or not mask:
                tqdm.write(f"Skip {fname}: Internal error - GT/Mask path invalid.")
                continue

            if not self.check_folder_contents(fp):
                actual_images = count_result_images(fp)
                tqdm.write(f"Skip {fname}: Incomplete ({actual_images}/{self.num_images} images).")
                skipped_incomplete += 1
                if fname in self.master_results:
                    tqdm.write(f"  - Removing stale cache entry for {fname}.")
                    del self.master_results[fname]
                continue

            folders_processed_count += 1
            if fname not in self.master_results:
                self.master_results[fname] = {}

            folder_has_new_tasks = False
            current_metrics_to_run = list(self.metrics_to_run)  # Copy to allow modification
            for metric in current_metrics_to_run:
                if metric not in valid_metrics_found:  # Skip if already deemed invalid
                    continue
                if metric not in METRICS_CONFIG:
                    tqdm.write(f"Warn: Metric '{metric}' invalid. Skipping.")
                    valid_metrics_found.discard(metric)  # Mark as invalid
                    self.metrics_to_run.remove(metric)  # Remove from list for future folders
                    continue

                script_name, _ = METRICS_CONFIG[metric]
                script_path_relative = Path("benchmark") / script_name
                absolute_script_path = benchmark_base_dir / script_path_relative
                if not absolute_script_path.is_file():
                    tqdm.write(f"Warn: Script '{absolute_script_path}' missing for {metric}. Skipping.")
                    valid_metrics_found.discard(metric)
                    self.metrics_to_run.remove(metric)
                    continue

                # Pass only the script *name* to the worker
                script_path_arg = script_name

                force = "all" in self.force_recalc_metrics or metric.lower() in self.force_recalc_metrics
                is_cached_valid = (
                    metric in self.master_results.get(fname, {}) and self.master_results[fname][metric] is not None
                )

                if force or not is_cached_valid:
                    task_args = (metric, script_path_arg, gt, mask, str(fp), str(self.cache_dir), self.num_images)
                    tasks_to_run.append(task_args)
                    folder_has_new_tasks = True
                    folders_with_tasks.add(fname)
                    metrics_being_run.add(metric)
                    self.master_results[fname][metric] = None  # Pre-initialize

        # --- End of Stage 1 ---

        if not tasks_to_run:
            print("\nNo metric calculations required (all results cached or folders incomplete/skipped).")
            self.save_master_cache()
            print("\n--- Metric Execution Phase Skipped ---")
            print(f"Folders checked: {len(self.discovered_folders)}")
            if skipped_incomplete > 0:
                print(f"Folders skipped (incomplete): {skipped_incomplete}")
            print(f"Folders ready for metrics (complete): {folders_processed_count}")
            print(f"Metric tasks needing calculation: 0")
            return

        print(f"\n--- Running {len(tasks_to_run)} Metric Tasks ({self.pool_size} parallel processes) ---")
        final_metrics_being_run = sorted([m for m in metrics_being_run if m in valid_metrics_found])
        if not final_metrics_being_run:
            print("Error: Tasks were identified, but all corresponding metrics/scripts are invalid. Aborting.")
            return
        print(f"Metrics involved: {', '.join(final_metrics_being_run)}")
        print(f"Results will be stored for {len(folders_with_tasks)} folders.")

        # --- Stage 2: Run all collected tasks in parallel using apply_async ---
        pool = multiprocessing.Pool(processes=self.pool_size)
        results_futures = []
        pbar = None
        try:
            print("Submitting tasks to pool...")
            for task_args in tasks_to_run:
                future = pool.apply_async(run_metric_script_parallel, args=task_args)
                results_futures.append(future)

            print("Calculating metrics...")
            # Force tqdm display by removing disable check. Use 'leave=True' to keep completed bar.
            pbar = tqdm(total=len(results_futures), desc="Calculating Metrics", unit="task", leave=True)

            # Iterate directly over the futures, get results as they complete
            for future in results_futures:
                try:
                    # Wait for the result of the current future
                    metric_name_result, score_result, folder_name_result = future.get()

                    if folder_name_result and metric_name_result:
                        if folder_name_result not in self.master_results:
                            tqdm.write(f"Warn: Result for unexpected folder '{folder_name_result}'.")
                            self.master_results[folder_name_result] = {}
                        # Update master results only if the metric is still considered valid
                        if metric_name_result in valid_metrics_found:
                            self.master_results[folder_name_result][metric_name_result] = score_result
                        # else: Silently ignore results for metrics deemed invalid earlier
                    else:
                        # Log if worker returned unexpected None values
                        tqdm.write(
                            f"Warn: Invalid result tuple from worker: ({metric_name_result}, {score_result}, {folder_name_result})"
                        )

                except Exception as e:
                    # Error likely occurred *within* the worker (run_metric_script_parallel) and was raised by .get()
                    # Worker should have printed details via tqdm.write/print.
                    tqdm.write(
                        f"\nERROR: Task failed in worker process. See worker output above. Error during get(): {e}"
                    )
                    # Score remains None in master_results
                finally:
                    # Update progress bar regardless of success or failure
                    if pbar:
                        pbar.update(1)

            print("All metric tasks processed.")  # Printed after loop finishes

        except Exception as pool_exc:
            # Catch errors during pool setup, task submission, or potentially loop logic
            print(f"\nFATAL ERROR during parallel processing pool execution: {pool_exc}")
            traceback.print_exc()
        finally:
            if pbar:
                pbar.close()  # Close the progress bar
            print("Shutting down multiprocessing pool...")
            pool.close()
            pool.join()
            print("Pool shut down.")

        self.save_master_cache()  # Save results collected, including None for errors
        print("\n--- Metric Execution Finished ---")
        print(f"Folders checked: {len(self.discovered_folders)}")
        if skipped_incomplete > 0:
            print(f"Folders skipped (incomplete): {skipped_incomplete}")
        print(f"Folders involved in metric tasks this run: {len(folders_with_tasks)}")
        print(f"Total metric tasks submitted/executed: {len(tasks_to_run)}")

    def load_per_image_results(self, fname, metric):
        """Loads per-image scores for LoFTR analysis."""
        if fname in self.per_image_scores and metric in self.per_image_scores[fname]:
            return self.per_image_scores[fname][metric]
        masked = metric in ["PSNR", "SSIM", "LPIPS"]
        sub = metric.lower() + "_masked" if masked else metric.lower()
        cache_f = self.per_image_cache_dir / sub / f"{fname}.json"
        data = load_json_cache(cache_f)
        if data and isinstance(data, dict) and "per_image" in data and isinstance(data["per_image"], dict):
            self.per_image_scores[fname][metric] = data["per_image"]
            return data["per_image"]
        return None

    def run_loftr_ranking(self):
        """Runs LoFTR ranking script."""
        if not self.loftr_script_path:  # Check resolved path
            print("\nLoFTR ranking skipped: Script path invalid or not found.")
            return

        print("\n--- Running LoFTR Ranking ---")
        folders_for_loftr = [
            f
            for f in self.discovered_folders
            if f.name.startswith("RealBench")
            and "gen" not in f.name
            and "fp32" not in f.name
            and self.check_folder_contents(f)
        ]
        if not folders_for_loftr:
            print("No suitable RealBench folders found for LoFTR ranking.")
            return

        print(f"Found {len(folders_for_loftr)} folders for LoFTR ranking.")
        ranked, skip_ex, skip_ref, errors = 0, 0, 0, 0
        force_loftr = "all" in self.force_recalc_metrics or "loftr" in self.force_recalc_metrics

        print("Executing LoFTR ranking script (this may take time)...")
        for fp in tqdm(
            folders_for_loftr, desc="LoFTR Ranking", unit="folder", leave=False
        ):  # leave=False to clear after loop
            out_j = fp / LOFTR_RANKING_FILENAME
            if out_j.is_file() and not force_loftr:
                skip_ex += 1
                continue

            gt, _ = find_gt_mask_paths(fp, self.dataset_dirs_map)
            if not gt:
                tqdm.write(f"Warn: No GT path for {fp.name}, skipping LoFTR.")
                skip_ref += 1
                continue
            ref_d = Path(gt).parent.parent / "ref"
            if not ref_d.is_dir():
                tqdm.write(f"Warn: Ref dir '{ref_d}' missing for {fp.name}, skipping LoFTR.")
                skip_ref += 1
                continue

            if force_loftr:
                tqdm.write(f"Forcing LoFTR for {fp.name}...")

            cmd = [
                sys.executable,
                str(self.loftr_script_path),
                "--source-dir",
                str(fp),
                "--ref-dir",
                str(ref_d),
                "--rank-only",
                "--ranking-output-file",
                LOFTR_RANKING_FILENAME,
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, check=False, timeout=600, encoding="utf-8", errors="replace"
                )
                if proc.returncode != 0:
                    stderr = proc.stderr or ""
                    tqdm.write(f"\nERROR LoFTR {fp.name} (RC:{proc.returncode}). Stderr: ...{stderr.strip()[-500:]}\n")
                    errors += 1
                else:
                    if (fp / LOFTR_RANKING_FILENAME).is_file():
                        ranked += 1
                    else:
                        tqdm.write(f"\nERROR LoFTR {fp.name}: Script ran (RC:0) but output missing.\n")
                        errors += 1
            except subprocess.TimeoutExpired:
                tqdm.write(f"\nERROR LoFTR Timeout for {fp.name}\n")
                errors += 1
            except Exception as e:
                tqdm.write(f"\nERROR LoFTR Exception for {fp.name}: {e}\n")
                errors += 1

        print("--- LoFTR Ranking Finished ---")
        if ranked > 0:
            print(f"Ran/updated ranks for {ranked} folders.")
        if skip_ex > 0:
            print(f"Skipped {skip_ex} folders (ranks existed).")
        if skip_ref > 0:
            print(f"Skipped {skip_ref} folders (missing ref dir/GT).")
        if errors > 0:
            print(f"Errors during LoFTR ranking for {errors} folders.")

    def load_loftr_ranks(self):
        """Loads LoFTR ranking results."""
        if not self.loftr_script_path:  # Check if LoFTR was available
            print("Skipping LoFTR rank loading: script path was invalid.")
            self.loftr_ranks = defaultdict(list)
            return

        print("Loading LoFTR ranks...")
        self.loftr_ranks = defaultdict(list)
        folders_to_check = [
            f
            for f in self.discovered_folders
            if f.name.startswith("RealBench")
            and "gen" not in f.name
            and "fp32" not in f.name
            and self.check_folder_contents(f)
        ]
        loaded, missing = 0, 0
        if not folders_to_check:
            print("No suitable folders to load LoFTR ranks from.")
            return

        for fp in folders_to_check:
            rank_f = fp / LOFTR_RANKING_FILENAME
            data = load_json_cache(rank_f)
            if data and isinstance(data, dict) and "ranking" in data and isinstance(data["ranking"], list):
                fnames = [
                    item["filename"]
                    for item in data["ranking"]
                    if isinstance(item, dict) and "filename" in item and isinstance(item["filename"], str)
                ]
                if fnames and all(re.match(r"^\d+\.png$", fn) for fn in fnames):
                    self.loftr_ranks[fp.name] = fnames
                    loaded += 1
                else:
                    if fnames:
                        print(f"Warn: Invalid filenames in LoFTR rank file {rank_f}")
                    missing += 1  # Treat invalid format as missing
            else:
                missing += 1
        print(f"Finished loading LoFTR ranks.")
        if loaded > 0:
            print(f"Loaded ranks for {loaded} folders.")
        if missing > 0:
            print(f"Ranks missing/invalid for {missing} folders.")

    def analyze_results(self):
        """Performs comparative analysis on the collected master results."""
        if not self.master_results:
            print("Analysis skipped: No metric results available.")
            return {}

        print("\n--- Analyzing Results ---")
        analysis = {}
        valid_folders = list(self.master_results.keys())

        # Define groups
        rb_f16_ng = [f for f in valid_folders if f.startswith("RealBench") and "fp32" not in f and "gen" not in f]
        cu_f16_ng = [f for f in valid_folders if f.startswith("Custom") and "fp32" not in f and "gen" not in f]

        # Overall averages
        analysis["overall_realbench_fp16_nongen"] = self._calculate_average(rb_f16_ng)
        analysis["overall_custom_fp16_nongen"] = self._calculate_average(cu_f16_ng)

        # --- Comparisons (FP16/FP32 RealBench) ---
        f16_rb_map = {get_scene_key(f): f for f in rb_f16_ng if get_scene_key(f)}
        f32_rb_map = {
            get_scene_key(f): f
            for f in valid_folders
            if f.startswith("RealBench") and "fp32" in f and "gen" not in f and get_scene_key(f)
        }
        common_rb_fp = sorted([k for k in f16_rb_map if k in f32_rb_map])
        if common_rb_fp:
            analysis["fp16_vs_fp32_realbench"] = {
                "common_scenes": common_rb_fp,
                "fp16_avg": self._calculate_average([f16_rb_map[k] for k in common_rb_fp]),
                "fp32_avg": self._calculate_average([f32_rb_map[k] for k in common_rb_fp]),
            }
        else:
            analysis["fp16_vs_fp32_realbench"] = "N/A (No common RealBench FP16/FP32 non-gen scenes)"

        # --- Comparisons (FP16/FP32 Custom) ---
        f16_cu_map = {get_scene_key(f): f for f in cu_f16_ng if get_scene_key(f)}
        f32_cu_map = {
            get_scene_key(f): f
            for f in valid_folders
            if f.startswith("Custom") and "fp32" in f and "gen" not in f and get_scene_key(f)
        }
        common_cu_fp = sorted([k for k in f16_cu_map if k in f32_cu_map])
        if common_cu_fp:
            analysis["fp16_vs_fp32_custom"] = {
                "common_scenes": common_cu_fp,
                "fp16_avg": self._calculate_average([f16_cu_map[k] for k in common_cu_fp]),
                "fp32_avg": self._calculate_average([f32_cu_map[k] for k in common_cu_fp]),
            }
        else:
            analysis["fp16_vs_fp32_custom"] = "N/A (No common Custom FP16/FP32 non-gen scenes)"

        # --- Comparisons (Non-Gen/Gen RealBench FP16) ---
        g_rb_f16_map = {
            get_scene_key(f): f
            for f in valid_folders
            if f.startswith("RealBench") and "gen" in f and "fp32" not in f and get_scene_key(f)
        }
        common_rb_gen = sorted([k for k in f16_rb_map if k in g_rb_f16_map])
        if common_rb_gen:
            analysis["gen_vs_nongen_realbench_fp16"] = {
                "common_scenes": common_rb_gen,
                "nongen_avg": self._calculate_average([f16_rb_map[k] for k in common_rb_gen]),
                "gen_avg": self._calculate_average([g_rb_f16_map[k] for k in common_rb_gen]),
            }
        else:
            analysis["gen_vs_nongen_realbench_fp16"] = "N/A (No common RealBench FP16 gen/non-gen scenes)"

        # --- LoFTR Filtering Analysis ---
        if self.loftr_script_path:
            self.load_loftr_ranks()  # Ensure ranks are loaded
        else:
            print("LoFTR Filtering analysis skipped: LoFTR script not available.")

        loftr_results = defaultdict(lambda: defaultdict(dict))
        base_loftr = rb_f16_ng  # Analyze RealBench FP16 Non-Gen

        if not self.loftr_script_path:
            analysis["loftr_filtering"] = "Skipped (LoFTR script unavailable)"
        elif not base_loftr:
            analysis["loftr_filtering"] = "Skipped (No base RealBench results)"
        elif not self.loftr_ranks:
            analysis["loftr_filtering"] = "Skipped (No LoFTR ranks loaded)"
        else:
            print(f"Performing LoFTR filtering analysis on {len(base_loftr)} scenes...")
            missing_data_notes = set()
            for rate in LOFTR_FILTER_RATES:
                keep = max(1, int(round(self.num_images * (1.0 - rate))))
                key = f"{int(rate*100)}% Filtered (Top {keep})"
                for metric in self.metrics_to_run:
                    scores = []
                    count = 0
                    for fname in base_loftr:
                        img_sc = self.load_per_image_results(fname, metric)
                        ranks = self.loftr_ranks.get(fname)
                        if img_sc is None or ranks is None:
                            if ranks is None:
                                missing_data_notes.add(f"{fname} (rank missing)")
                            if img_sc is None:
                                missing_data_notes.add(f"{fname} ({metric} scores missing)")
                            continue

                        valid_sc = [img_sc[f] for f in ranks if f in img_sc and img_sc[f] is not None]
                        top = valid_sc[:keep]
                        if top:
                            try:
                                scores.append(np.mean([float(s) for s in top]))
                                count += 1
                            except (ValueError, TypeError):
                                missing_data_notes.add(f"{fname} ({metric} invalid score in top {keep})")

                    if scores:
                        overall = np.mean(scores)
                        loftr_results[key][metric] = overall
                        loftr_results[key][f"_count_{metric}"] = count

            if missing_data_notes:
                print(f"Note: LoFTR filtering skipped some data points:")
                for item in sorted(list(missing_data_notes))[:5]:
                    print(f"  - {item}")  # Print first few
                if len(missing_data_notes) > 5:
                    print("  - ... and possibly others")

            analysis["loftr_filtering"] = dict(loftr_results) if loftr_results else "N/A (No results after filtering)"

        print("--- Analysis Finished ---")
        return analysis

    def _calculate_average(self, folder_list):
        """Calculates average scores for a list of folders."""
        if not folder_list:
            return {"_counts": {m: 0 for m in self.metrics_to_run}}
        avg_s, counts = defaultdict(list), defaultdict(int)
        metrics_in_scope = set(self.metrics_to_run)  # Use a set for faster checks

        for fname in folder_list:
            if fname in self.master_results:
                for m, s in self.master_results[fname].items():
                    if m in metrics_in_scope and s is not None:
                        try:
                            avg_s[m].append(float(s))
                            counts[m] += 1
                        except (ValueError, TypeError):
                            pass  # Ignore non-floatable scores

        final = {m: np.mean(s) for m, s in avg_s.items() if s}
        final["_counts"] = {m: counts.get(m, 0) for m in self.metrics_to_run}
        # Convert numpy types before returning, just in case mean results in numpy type
        return convert_numpy_types(final)

    def format_results(self, analysis_results):
        """Formats the analysis results into a readable string report."""
        lines = []
        pd.set_option("display.precision", 4)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        lines.append("=" * 80)
        lines.append(" Benchmark Results Summary ".center(80, "="))
        lines.append("=" * 80)
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Base Results Dir: {self.base_results_dir}")
        lines.append(f"Folders Discovered: {len(self.discovered_folders)}")
        lines.append(f"Metrics Evaluated: {', '.join(self.metrics_to_run)}")
        lines.append("-" * 80)

        def fmt_avg(data, title):
            sub_lines = [f"\n--- {title} ---"]
            if not isinstance(data, dict) or not data:
                sub_lines.append("N/A (No data)")
                return sub_lines
            counts = data.pop("_counts", {})
            if not any(k != "_counts" for k in data):  # Check if only counts key exists
                sub_lines.append("N/A (No valid results)")
                data["_counts"] = counts
                return sub_lines
            df = pd.DataFrame([data]).T
            df.columns = ["Average Score"]
            df["Num Scenes (N)"] = df.index.map(lambda m: counts.get(m, 0))
            order = [m for m in METRICS_CONFIG.keys() if m in df.index]
            df = df.reindex(order).dropna(how="all")
            sub_lines.append(df.to_string() if not df.empty else "N/A (No valid results for specified metrics)")
            data["_counts"] = counts  # Add back counts
            return sub_lines

        lines.extend(
            fmt_avg(
                analysis_results.get("overall_realbench_fp16_nongen", {}),
                "Overall Average (RealBench FP16 Non-Generated)",
            )
        )
        lines.extend(
            fmt_avg(
                analysis_results.get("overall_custom_fp16_nongen", {}), "Overall Average (Custom FP16 Non-Generated)"
            )
        )

        def fmt_comp(data, title, labels):
            sub_lines = [f"\n\n--- {title} ---"]
            if not isinstance(data, dict):
                sub_lines.append(f"{data}")
                return sub_lines
            com = data.get("common_scenes", [])
            avg1 = data.get(f"{labels[0].lower()}_avg", {})
            avg2 = data.get(f"{labels[1].lower()}_avg", {})
            sub_lines.append(f"Based on {len(com)} common scene(s): {', '.join(com)}")
            cts1 = avg1.pop("_counts", {})
            cts2 = avg2.pop("_counts", {})
            if not any(k != "_counts" for k in avg1) and not any(k != "_counts" for k in avg2):
                sub_lines.append("N/A (No valid results for comparison)")
                avg1["_counts"], avg2["_counts"] = cts1, cts2
                return sub_lines
            df = pd.DataFrame({f"{labels[0]} Avg": avg1, f"{labels[1]} Avg": avg2})
            df[f"N {labels[0]}"] = df.index.map(lambda x: cts1.get(x, 0))
            df[f"N {labels[1]}"] = df.index.map(lambda x: cts2.get(x, 0))
            order = [m for m in METRICS_CONFIG.keys() if m in df.index]
            df = df.reindex(order).dropna(how="all", subset=[f"{labels[0]} Avg", f"{labels[1]} Avg"])
            sub_lines.append(df.to_string() if not df.empty else "N/A (No valid results for specified metrics)")
            avg1["_counts"], avg2["_counts"] = cts1, cts2  # Add back counts
            return sub_lines

        lines.extend(
            fmt_comp(
                analysis_results.get("fp16_vs_fp32_realbench"),
                "FP16 vs FP32 (Common RealBench Non-Gen)",
                ["FP16", "FP32"],
            )
        )
        lines.extend(
            fmt_comp(
                analysis_results.get("fp16_vs_fp32_custom"), "FP16 vs FP32 (Common Custom Non-Gen)", ["FP16", "FP32"]
            )
        )
        lines.extend(
            fmt_comp(
                analysis_results.get("gen_vs_nongen_realbench_fp16"),
                "Non-Gen vs Generated (Common RealBench FP16)",
                ["NonGen", "Gen"],
            )
        )

        lines.append("\n\n--- LoFTR Filtering Analysis (Avg over RealBench FP16 Non-Gen Scenes) ---")
        loftr = analysis_results.get("loftr_filtering")
        if isinstance(loftr, dict) and loftr:
            metric_data = defaultdict(dict)
            rate_keys = []
            for k, v in loftr.items():
                scores = {m: score for m, score in v.items() if not m.startswith("_count_")}
                if scores:
                    metric_data[k] = scores
                    rate_keys.append(k)

            if not rate_keys:
                lines.append("N/A (No valid results after filtering)")
            else:
                try:
                    sorted_rates = sorted(rate_keys, key=lambda k: int(re.search(r"\d+", k).group()))
                except:
                    sorted_rates = sorted(rate_keys)  # Fallback sort
                df_raw = pd.DataFrame(metric_data)
                order = [m for m in METRICS_CONFIG.keys() if m in df_raw.index]
                df = df_raw.reindex(index=order, columns=sorted_rates).dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    lines.append("N/A (No valid results for specified metrics/rates)")
                else:
                    lines.append("Average score when keeping only top N images based on LoFTR rank:")
                    lines.append(df.to_string())
        else:
            lines.append(f"{loftr}")  # Append N/A message or error string

        lines.append("\n" + "=" * 80 + "\nMetric Interpretation:")
        higher = [m for m in self.metrics_to_run if m in METRICS_CONFIG and METRICS_CONFIG[m][1]]
        lower = [m for m in self.metrics_to_run if m in METRICS_CONFIG and not METRICS_CONFIG[m][1]]
        if higher:
            lines.append(f"  Higher is Better: {', '.join(higher)}")
        if lower:
            lines.append(f"  Lower is Better: {', '.join(lower)}")
        lines.append("=" * 80)
        return "\n".join(lines)

    def run(self):
        """Main execution flow."""
        self.discover_folders()
        if not self.discovered_folders:
            print("\nNo suitable result folders found. Exiting.")
            return

        self.run_all_metrics()  # Run metric calculations

        # Run LoFTR only if relevant folders exist and script is available
        if self.loftr_script_path and any(f.name.startswith("RealBench") for f in self.discovered_folders):
            self.run_loftr_ranking()
        elif not self.loftr_script_path:
            print("\nSkipping LoFTR ranking: Script path invalid or not found.")
        else:
            print("\nSkipping LoFTR ranking: No 'RealBench' folders discovered.")

        analysis = self.analyze_results()  # Analyze results from cache
        report = self.format_results(analysis)  # Format report
        print("\n" + report)  # Print report to console

        if self.output_file:  # Save report to file
            print(f"\nSaving report to {self.output_file}...")
            try:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_file, "w", encoding="utf-8") as f:
                    f.write(report)
                print("Report saved successfully.")
            except OSError as e:
                print(f"Error saving report to {self.output_file}: {e}")


# --- Command Line Interface ---
if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows/macOS frozen executables

    parser = argparse.ArgumentParser(
        description="Run image quality and correspondence benchmarks on generated image sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_base_dir", required=True, help="Parent directory containing scene result folders.")
    parser.add_argument("--cache_dir", required=True, help="Directory for cached results.")
    parser.add_argument("--output_file", default="benchmark_report.txt", help="Path to save the summary report.")
    parser.add_argument("--realbench_dataset_dir", help="Base directory of the RealBench dataset.")
    parser.add_argument("--custom_dataset_dir", help="Base directory of the Custom dataset.")
    parser.add_argument(
        "--num_images", type=int, default=DEFAULT_NUM_IMAGES, help="Expected images per scene (0.png to N-1.png)."
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=list(METRICS_CONFIG.keys()), help="Metrics to run (default: all configured)."
    )
    parser.add_argument(
        "--force_recalc", nargs="+", help="Force recalc for metric(s) or 'all' or 'loftr'. Case-insensitive."
    )
    # Expect loftr script relative to this benchmark script
    parser.add_argument(
        "--loftr_script_path",
        default="benchmark/loftr_ranking.py",
        help="Path to loftr_ranking.py relative to this script.",
    )

    args = parser.parse_args()

    # --- Initial Setup & Validation ---
    try:
        print(f"Detected {os.cpu_count()} CPU cores.")
    except NotImplementedError:
        print("Could not detect CPU cores.")
    if not Path(args.results_base_dir).is_dir():
        parser.error(f"Results base directory not found: {args.results_base_dir}")
    if not args.realbench_dataset_dir and not args.custom_dataset_dir:
        parser.error("At least one dataset directory must be specified.")

    # Ensure cache directories exist
    try:
        cache_p = Path(args.cache_dir).resolve()
        cache_p.mkdir(parents=True, exist_ok=True)
        psc_p = cache_p / PER_IMAGE_CACHE_BASE
        psc_p.mkdir(parents=True, exist_ok=True)
        metrics_to_setup = args.metrics or METRICS_CONFIG.keys()
        for name in metrics_to_setup:
            if name in METRICS_CONFIG:
                masked = name in ["PSNR", "SSIM", "LPIPS"]
                sub_dir_name = name.lower() + ("_masked" if masked else "")
                (psc_p / sub_dir_name).mkdir(parents=True, exist_ok=True)
        print(f"Cache directories ensured under {cache_p}")
    except OSError as e:
        parser.error(f"Failed to create cache directories: {e}")

    # --- Run Benchmarks ---
    start_time = time.time()
    exit_code = 0
    try:
        runner = BenchmarkRunner(args)
        runner.run()
    except Exception as e:
        print(f"\n--- FATAL ERROR DURING BENCHMARK RUN ---", flush=True)
        print(f"Error Type: {type(e).__name__}", flush=True)
        print(f"Error Details: {e}", flush=True)
        print("\n--- Traceback ---", flush=True)
        traceback.print_exc()
        print("--- BENCHMARK SCRIPT TERMINATED DUE TO ERROR ---", flush=True)
        exit_code = 1
    finally:
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

    print("\nBenchmarking script finished.")
    sys.exit(exit_code)
