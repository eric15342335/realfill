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

# --- Configuration & Constants ---
# ... (Keep METRICS_CONFIG, LOFTR_FILTER_RATES, etc. as before) ...
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
# ... (find_gt_mask_paths, count_result_images, parse_final_score, load_json_cache, save_json_cache, get_scene_key remain the same) ...
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
    base_path = dataset_base_dir / benchmark_type / scene_number / "target"
    gt_path, mask_path = base_path / "gt.png", base_path / "mask.png"
    if not gt_path.is_file() or not mask_path.is_file():
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
        print(f"Warning: Could not count images in {folder_path}: {e}")
        return 0


def parse_final_score(stdout_str):
    for line in stdout_str.splitlines():
        if line.startswith("FINAL_SCORE:"):
            score_part = line.split(":", 1)[1]
            if score_part == "ERROR":
                return None
            try:
                return float(score_part)
            except ValueError:
                print(f"Warning: Could not parse score: {line}")
                return None
    return None


def load_json_cache(file_path):
    if not file_path or not Path(file_path).is_file():
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, TypeError) as e:
        print(f"Warn: Cache load fail {file_path}: {e}")
        return None


def save_json_cache(data, file_path):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except (OSError, TypeError) as e:
        print(f"Error saving JSON cache {file_path}: {e}")


def get_scene_key(folder_name):
    match = re.match(r"^(RealBench|Custom)-(\d+)(-results.*)?$", folder_name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


# --- Function to run metric script (Corrected definition for multiprocessing) ---
def run_metric_script_parallel(
    metric_name, script_path_str, gt_path_str, mask_path_str, results_dir_str, cache_dir_str, num_images
):
    """Wrapper to run a metric script, designed for multiprocessing pool."""
    # No need to unpack args_tuple anymore, arguments are passed directly
    script_path = Path(script_path_str)
    results_dir = Path(results_dir_str)
    cache_dir = Path(cache_dir_str)

    command = [
        "python",
        str(script_path),
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
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=720)
        if process.returncode != 0:
            print(
                f"MP_ERROR: {metric_name} failed for {results_dir.name} (RC: {process.returncode}). Stderr: ...{process.stderr.strip()[-500:]}"
            )
            return metric_name, None
        score = parse_final_score(process.stdout)
        return metric_name, score
    except subprocess.TimeoutExpired:
        print(f"MP_ERROR: Timeout expired for {metric_name} on {results_dir.name}")
        return metric_name, None
    except Exception as e:
        # Print exception type for better debugging
        print(f"MP_ERROR: Exception ({type(e).__name__}) for {metric_name} on {results_dir.name}: {e}")
        return metric_name, None


# --- Main Orchestration Class ---
class BenchmarkRunner:
    def __init__(self, args):
        # ... (Initialization remains the same) ...
        self.args = args
        self.base_results_dir = Path(args.results_base_dir)
        self.dataset_dirs_map = {}
        if args.realbench_dataset_dir:
            rb_path = Path(args.realbench_dataset_dir)
            if rb_path.is_dir():
                self.dataset_dirs_map["RealBench"] = rb_path
            else:
                print(f"Warn: RealBench dataset dir not found: {rb_path}")
        if args.custom_dataset_dir:
            cu_path = Path(args.custom_dataset_dir)
            if cu_path.is_dir():
                self.dataset_dirs_map["Custom"] = cu_path
            else:
                print(f"Warn: Custom dataset dir not found: {cu_path}")
        if not self.dataset_dirs_map:
            raise ValueError("No valid dataset dirs found.")
        self.cache_dir = Path(args.cache_dir)
        self.output_file = Path(args.output_file) if args.output_file else None
        self.num_images = args.num_images
        self.force_recalc_metrics = [m.lower() for m in (args.force_recalc or [])]
        self.metrics_to_run = args.metrics or list(METRICS_CONFIG.keys())
        self.loftr_script_path = Path(args.loftr_script_path) if args.loftr_script_path else None
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
        print(f"Force: {self.force_recalc_metrics}")
        # Determine pool size
        try:
            self.cpu_cores = os.cpu_count()
            # Limit pool size based on cores, maybe add a safety margin if desired
            self.pool_size = max(1, self.cpu_cores)  # Use at least 1, up to available cores
            print(f"Using multiprocessing pool size: {self.pool_size}")
        except NotImplementedError:
            print("Could not detect CPU cores, using pool size 1.")
            self.pool_size = 1

    # ... (discover_folders, load_master_cache, save_master_cache, check_folder_contents remain the same) ...
    def discover_folders(self):
        print(f"\nScanning results folders in {self.base_results_dir}...")
        self.discovered_folders = []
        potential, skip_map, skip_gt = 0, 0, 0
        if not self.base_results_dir.is_dir():
            print(f"Error: Base dir not found.")
            return
        for item in self.base_results_dir.iterdir():
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
        self.discovered_folders.sort(key=lambda p: p.name)
        print(f"Scan complete. Found {potential} potential *results* folders.")
        print(f" - Added {len(self.discovered_folders)} with dataset & GT/Mask.")
        if skip_map > 0:
            print(f" - Skipped {skip_map} (unmapped dataset).")
        if skip_gt > 0:
            print(f" - Skipped {skip_gt} (missing GT/Mask).")

    def load_master_cache(self):
        print(f"Loading master cache: {self.master_cache_path}")
        cached = load_json_cache(self.master_cache_path)
        self.master_results = defaultdict(dict)
        if cached and isinstance(cached, dict):
            for k, v in cached.items():
                if isinstance(v, dict):
                    self.master_results[k] = v
            print(f"Loaded {len(self.master_results)} entries.")
        else:
            print("No valid master cache found/empty.")

    def save_master_cache(self):
        print(f"Saving master cache: {self.master_cache_path}")
        save_json_cache(dict(self.master_results), self.master_cache_path)

    def check_folder_contents(self, fp: Path) -> bool:
        if not fp.is_dir():
            return False
        try:
            return all((fp / f"{i}.png").is_file() for i in range(self.num_images))
        except Exception as e:
            tqdm.write(f"Error checking {fp}: {e}")
            return False

    def run_all_metrics(self):
        """Runs metrics, potentially in parallel for each folder."""
        if not self.discovered_folders:
            print("No result folders discovered.")
            return
        print(f"\n--- Running Metrics ({self.pool_size} parallel processes per folder) ---")
        self.load_master_cache()
        processed, skipped_inc = 0, 0

        # Create the pool *outside* the loop for efficiency
        # Use try/finally to ensure pool cleanup
        pool = multiprocessing.Pool(processes=self.pool_size)
        try:
            for fp in tqdm(self.discovered_folders, desc="Processing Folders", unit="folder"):
                fname = fp.name
                gt, mask = find_gt_mask_paths(fp, self.dataset_dirs_map)
                if not gt or not mask:
                    tqdm.write(f"Skip {fname}: GT/Mask missing.")
                    continue
                if not self.check_folder_contents(fp):
                    actual = count_result_images(fp)
                    tqdm.write(f"Skip {fname}: Found {actual}/{self.num_images} images.")
                    skipped_inc += 1
                    if fname in self.master_results:
                        del self.master_results[fname]
                    continue

                processed += 1
                if fname not in self.master_results:
                    self.master_results[fname] = {}

                tasks = []
                metrics_to_run_this_folder = []
                for metric in self.metrics_to_run:
                    if metric not in METRICS_CONFIG:
                        tqdm.write(f"Warn: Metric '{metric}' invalid.")
                        continue
                    script_name, _ = METRICS_CONFIG[metric]
                    script_path = Path(__file__).parent / "benchmark" / script_name
                    if not script_path.is_file():
                        tqdm.write(f"Warn: Script {script_path} missing.")
                        continue
                    force = "all" in self.force_recalc_metrics or metric.lower() in self.force_recalc_metrics
                    # Check master cache *before* adding to parallel tasks
                    if (
                        not force
                        and metric in self.master_results.get(fname, {})
                        and self.master_results[fname][metric] is not None
                    ):
                        # tqdm.write(f"Cache hit for {metric} on {fname}") # Optional debug
                        continue  # Skip valid cached entry

                    # If not cached or forced, prepare task arguments
                    metrics_to_run_this_folder.append(metric)
                    tasks.append((metric, str(script_path), gt, mask, str(fp), str(self.cache_dir), self.num_images))

                if not tasks:  # Skip folder if all metrics were cached
                    # tqdm.write(f"All metrics cached for {fname}") # Optional debug
                    continue

                # Run tasks in parallel for the current folder
                try:
                    # Use starmap to pass multiple arguments from the tuples in tasks
                    results = pool.starmap(run_metric_script_parallel, tasks)
                    # Update master_results with scores from parallel execution
                    for metric_name_result, score_result in results:
                        if metric_name_result:  # Should always be true here
                            self.master_results[fname][metric_name_result] = score_result
                except Exception as pool_exc:
                    tqdm.write(f"ERROR during parallel processing for folder {fname}: {pool_exc}")
                    # Mark all attempted metrics as None for this folder on error
                    for metric_name_task in metrics_to_run_this_folder:
                        self.master_results[fname][metric_name_task] = None

        finally:
            pool.close()  # Prevent new tasks
            pool.join()  # Wait for current tasks to complete

        self.save_master_cache()
        print("\n--- Metric Execution Finished ---")
        print(f"Folders processed: {processed}")
        if skipped_inc > 0:
            print(f"Folders skipped (incomplete): {skipped_inc}")

    # ... (load_per_image_results, run_loftr_ranking, load_loftr_ranks remain the same) ...
    def load_per_image_results(self, fname, metric):
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
        if not self.loftr_script_path or not self.loftr_script_path.is_file():
            print("LoFTR script skip.")
            return
        print("\n--- Running LoFTR Ranking ---")
        folders = [
            f
            for f in self.discovered_folders
            if f.name.startswith("RealBench") and "gen" not in f.name and "fp32" not in f.name
        ]
        print(f"Found {len(folders)} potential folders.")
        ranked, skip_ex, skip_ref, errors = 0, 0, 0, 0
        for fp in tqdm(folders, desc="LoFTR Ranking", unit="folder"):
            out_j = fp / LOFTR_RANKING_FILENAME
            force = "all" in self.force_recalc_metrics or "loftr" in self.force_recalc_metrics
            if out_j.is_file() and not force:
                skip_ex += 1
                continue
            gt, _ = find_gt_mask_paths(fp, self.dataset_dirs_map)
            if not gt:
                continue
            ref_d = Path(gt).parent.parent / "ref"
            if not ref_d.is_dir():
                tqdm.write(f"Warn: Ref dir '{ref_d}' missing {fp.name}.")
                skip_ref += 1
                continue
            if force:
                tqdm.write(f"Force LoFTR {fp.name}...")
            cmd = [
                "python",
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
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
                if proc.returncode != 0:
                    tqdm.write(f"Err LoFTR {fp.name} (RC:{proc.returncode}).")
                    stderr = proc.stderr.strip()[-500:]
                    if stderr:
                        tqdm.write(f"Stderr: ...{stderr}")
                    errors += 1
                else:
                    ranked += 1
            except subprocess.TimeoutExpired:
                tqdm.write(f"Timeout LoFTR {fp.name}")
                errors += 1
            except Exception as e:
                tqdm.write(f"Excp LoFTR {fp.name}: {e}")
                errors += 1
        print("--- LoFTR Ranking Finished ---")
        if ranked > 0:
            print(f"Ran/updated LoFTR ranks for {ranked}.")
        if skip_ex > 0:
            print(f"Skipped {skip_ex} (ranks existed).")
        if skip_ref > 0:
            print(f"Skipped {skip_ref} (missing ref dir).")
        if errors > 0:
            print(f"Errors LoFTR {errors}.")

    def load_loftr_ranks(self):
        if not self.loftr_script_path or not self.loftr_script_path.is_file():
            print("Skip LoFTR load: script path invalid.")
            return
        print("Loading LoFTR ranks...")
        self.loftr_ranks = defaultdict(list)
        folders = [
            f
            for f in self.discovered_folders
            if f.name.startswith("RealBench") and "gen" not in f.name and "fp32" not in f.name
        ]
        loaded, missing = 0, 0
        for fp in folders:
            rank_f = fp / LOFTR_RANKING_FILENAME
            data = load_json_cache(rank_f)
            if data and isinstance(data, dict) and "ranking" in data and isinstance(data["ranking"], list):
                fnames = [item["filename"] for item in data["ranking"] if isinstance(item, dict) and "filename" in item]
                if fnames:
                    self.loftr_ranks[fp.name] = fnames
                    loaded += 1
                else:
                    missing += 1
            else:
                missing += 1
        print(f"Loaded LoFTR ranks for {loaded}.")
        if missing > 0:
            print(f"LoFTR ranks missing/invalid for {missing}.")

    # ... (analyze_results, _calculate_average, format_results remain the same as the previous version with the bug fix) ...
    def analyze_results(self):
        if not self.master_results:
            print("No metrics loaded.")
            return {}
        print("\n--- Analyzing Results ---")
        analysis = {}
        valid = list(self.master_results.keys())
        rb_f16_ng = [f for f in valid if f.startswith("RealBench") and "fp32" not in f and "gen" not in f]
        cu_f16_ng = [f for f in valid if f.startswith("Custom") and "fp32" not in f and "gen" not in f]
        analysis["overall_realbench_fp16_nongen"] = self._calculate_average(rb_f16_ng)
        analysis["overall_custom"] = self._calculate_average(cu_f16_ng)
        f16_rb = {get_scene_key(f): f for f in rb_f16_ng if get_scene_key(f)}
        f32_rb = {
            get_scene_key(f): f
            for f in valid
            if f.startswith("RealBench") and "fp32" in f and "gen" not in f and get_scene_key(f)
        }
        com_rb = sorted([k for k in f16_rb if k in f32_rb])
        f16c_rb = [f16_rb[k] for k in com_rb]
        f32c_rb = [f32_rb[k] for k in com_rb]
        analysis["fp16_vs_fp32_realbench"] = (
            {
                "common_scenes": com_rb,
                "fp16_avg": self._calculate_average(f16c_rb),
                "fp32_avg": self._calculate_average(f32c_rb),
            }
            if com_rb
            else "N/A RealBench FP16/32"
        )
        f16_cu = {get_scene_key(f): f for f in cu_f16_ng if get_scene_key(f)}
        f32_cu = {
            get_scene_key(f): f
            for f in valid
            if f.startswith("Custom") and "fp32" in f and "gen" not in f and get_scene_key(f)
        }
        com_cu = sorted([k for k in f16_cu if k in f32_cu])
        f16c_cu = [f16_cu[k] for k in com_cu]
        f32c_cu = [f32_cu[k] for k in com_cu]
        analysis["fp16_vs_fp32_custom"] = (
            {
                "common_scenes": com_cu,
                "fp16_avg": self._calculate_average(f16c_cu),
                "fp32_avg": self._calculate_average(f32c_cu),
            }
            if com_cu
            else "N/A Custom FP16/32"
        )
        ng_rb_f16 = f16_rb
        g_rb_f16 = {
            get_scene_key(f): f
            for f in valid
            if f.startswith("RealBench") and "gen" in f and "fp32" not in f and get_scene_key(f)
        }
        com_g_rb = sorted([k for k in ng_rb_f16 if k in g_rb_f16])
        ngc_rb = [ng_rb_f16[k] for k in com_g_rb]
        gc_rb = [g_rb_f16[k] for k in com_g_rb]
        analysis["gen_vs_nongen_realbench_fp16"] = (
            {
                "common_scenes": com_g_rb,
                "nongen_avg": self._calculate_average(ngc_rb),
                "gen_avg": self._calculate_average(gc_rb),
            }
            if com_g_rb
            else "N/A RealBench FP16 Gen/NonGen"
        )

        run_loftr = self.loftr_script_path and self.loftr_script_path.is_file()
        if run_loftr:
            self.load_loftr_ranks()
        else:
            print("Skip LoFTR filter: script missing.")
        loftr_results = defaultdict(lambda: defaultdict(dict))
        base_loftr = rb_f16_ng
        if not run_loftr:
            analysis["loftr_filtering"] = "Skipped (script)"
        elif not base_loftr:
            analysis["loftr_filtering"] = "Skipped (no base)"
        elif not self.loftr_ranks:
            analysis["loftr_filtering"] = "Skipped (no ranks)"
        else:
            print(f"LoFTR filtering analysis on {len(base_loftr)} scenes.")
            missing_ct = 0
            for rate in LOFTR_FILTER_RATES:
                keep = max(1, int(round(self.num_images * (1.0 - rate))))
                key = f"{int(rate*100)}% Filtered"
                missing_f = False
                for metric in self.metrics_to_run:
                    scores = []
                    count = 0
                    for fname in base_loftr:
                        img_sc = self.load_per_image_results(fname, metric)
                        ranks = self.loftr_ranks.get(fname)
                        if img_sc is None or ranks is None:
                            if ranks is None:
                                missing_f = True
                                continue
                        valid_sc = [img_sc[f] for f in ranks if f in img_sc and img_sc[f] is not None]
                        top = valid_sc[:keep]
                        if top:
                            scores.append(np.mean(top))
                            count += 1
                    if scores:
                        overall = np.mean(scores)
                        loftr_results[key][metric] = overall
                        loftr_results[key][f"_count_{metric}"] = count
                if missing_f:
                    missing_ct += 1  # Count unique scenes missing ranks
            unique_miss = sum(1 for fn in base_loftr if self.loftr_ranks.get(fn) is None)
            if unique_miss > 0:
                print(f"Note: LoFTR ranks/scores missing for {unique_miss} scenes.")
            analysis["loftr_filtering"] = dict(loftr_results)
        print("--- Analysis Finished ---")
        return analysis

    def _calculate_average(self, folder_list):
        if not folder_list:
            return {"_counts": {m: 0 for m in self.metrics_to_run}}
        avg_s, counts = defaultdict(list), defaultdict(int)
        for fname in folder_list:
            if fname in self.master_results:
                for m, s in self.master_results[fname].items():
                    if s is not None and m in self.metrics_to_run:
                        try:
                            avg_s[m].append(float(s))
                            counts[m] += 1
                        except:
                            pass
        final = {m: np.mean(s) for m, s in avg_s.items() if s}
        final["_counts"] = {m: counts.get(m, 0) for m in self.metrics_to_run}
        return final

    def format_results(self, analysis_results):
        lines = []
        pd.set_option("display.precision", 4)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        lines.append("=" * 80)
        lines.append(" Benchmark Results Summary ".center(80))
        lines.append("=" * 80)
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Base Dir: {self.base_results_dir}")
        lines.append(f"Folders Found: {len(self.discovered_folders)}")
        lines.append(f"Metrics: {', '.join(self.metrics_to_run)}")
        lines.append("-" * 80)

        def fmt_avg(data, title):
            sub_lines = [f"\n--- {title} ---"]
            if not isinstance(data, dict) or not data:
                sub_lines.append("N/A")
                return sub_lines
            counts = data.pop("_counts", {})
            if not data:
                sub_lines.append("N/A (no results)")
                data["_counts"] = counts
                return sub_lines
            df = pd.DataFrame([data]).T
            df.columns = ["Avg"]
            df["N"] = df.index.map(lambda m: counts.get(m, 0))
            order = [m for m in METRICS_CONFIG.keys() if m in df.index]
            df = df.reindex(order).dropna(how="all")
            if df.empty:
                sub_lines.append("N/A (no results)")
            else:
                sub_lines.append(df.to_string())
            data["_counts"] = counts
            return sub_lines

        lines.extend(
            fmt_avg(analysis_results.get("overall_realbench_fp16_nongen", {}), "Overall Avg (RealBench FP16 Non-Gen)")
        )
        lines.extend(fmt_avg(analysis_results.get("overall_custom", {}), "Overall Avg (Custom FP16 Non-Gen)"))

        def fmt_comp(data, title, labels):
            sub_lines = [f"\n\n--- {title} ---"]
            if not isinstance(data, dict):
                sub_lines.append(f"{data}")
                return sub_lines
            com = data.get("common_scenes", [])
            avg1 = data.get(f"{labels[0].lower()}_avg", {})
            avg2 = data.get(f"{labels[1].lower()}_avg", {})
            sub_lines.append(f"Common Scenes ({len(com)}): {', '.join(com)}")
            cts1 = avg1.pop("_counts", {})
            cts2 = avg2.pop("_counts", {})
            if not avg1 and not avg2:
                sub_lines.append("N/A (no results)")
                return sub_lines
            df = pd.DataFrame({labels[0]: avg1, labels[1]: avg2})
            df[f"N_{labels[0]}"] = df.index.map(lambda x: cts1.get(x, 0))
            df[f"N_{labels[1]}"] = df.index.map(lambda x: cts2.get(x, 0))
            order = [m for m in METRICS_CONFIG.keys() if m in df.index]
            df = df.reindex(order).dropna(how="all", subset=labels)
            if df.empty:
                sub_lines.append("N/A (no results)")
            else:
                sub_lines.append(df.to_string())
            avg1["_counts"] = cts1
            avg2["_counts"] = cts2
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
        loftr = analysis_results.get("loftr_filtering")
        lines.append("\n\n--- LoFTR Filtering (RealBench FP16 Non-Gen) ---")
        if isinstance(loftr, dict) and loftr:
            try:
                rates = sorted(loftr.keys(), key=lambda x: int(re.search(r"\d+", x).group()))
            except:
                rates = sorted(loftr.keys())
            metric_data = {
                r: {m: score for m, score in loftr[r].items() if not m.startswith("_count_")} for r in rates
            }  # Corrected logic
            df_raw = pd.DataFrame(metric_data)
            order = [m for m in METRICS_CONFIG.keys() if m in df_raw.index]
            df = df_raw.reindex(index=order, columns=rates).dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                lines.append("N/A (no results)")
            else:
                lines.append(df.to_string())
        else:
            lines.append(f"{loftr}")
        lines.append("\n" + "=" * 80 + "\nMetric Direction:")
        for name, (_, higher_better) in METRICS_CONFIG.items():
            if name in self.metrics_to_run:
                lines.append(f"  {name}: {'Higher' if higher_better else 'Lower'} is better")
        lines.append("=" * 80)
        return "\n".join(lines)

    def run(self):
        self.discover_folders()
        if not self.discovered_folders:
            print("No folders found.")
            return
        self.run_all_metrics()  # Now uses multiprocessing pool internally
        analysis = self.analyze_results()
        report = self.format_results(analysis)
        print("\n" + report)
        if self.output_file:
            print(f"\nSaving report to {self.output_file}...")
            try:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_file, "w", encoding="utf-8") as f:
                    f.write(report)
                print("Report saved.")
            except OSError as e:
                print(f"Error saving report: {e}")


# --- Command Line Interface ---
if __name__ == "__main__":
    # Required for multiprocessing spawn method on some OS (like Windows)
    # Ensure this is only called when script is run directly
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Run RealFill benchmarks.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ... (Argument parsing remains the same) ...
    parser.add_argument("--results_base_dir", required=True, help="Parent dir containing *results* folders.")
    parser.add_argument("--cache_dir", required=True, help="Cache directory.")
    parser.add_argument("--realbench_dataset_dir", help="RealBench dataset base directory.")
    parser.add_argument("--custom_dataset_dir", help="Custom dataset base directory.")
    parser.add_argument("--output_file", default="benchmark_report.txt", help="Report output file path.")
    parser.add_argument(
        "--num_images", type=int, default=DEFAULT_NUM_IMAGES, help="Expected images per scene (0..N-1)."
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=list(METRICS_CONFIG.keys()), help="Metrics to run (default: all)."
    )
    parser.add_argument("--force_recalc", nargs="+", help="Force recalc for metric(s) or 'all'. Case-insensitive.")
    parser.add_argument("--loftr_script_path", default="loftr_ranking.py", help="Path to loftr_ranking.py.")

    args = parser.parse_args()

    try:
        print(f"Detected {os.cpu_count()} CPU cores.")
    except:
        print("Could not detect CPU cores.")
    if not Path(args.results_base_dir).is_dir():
        parser.error(f"Base dir not found: {args.results_base_dir}")
    if not args.realbench_dataset_dir and not args.custom_dataset_dir:
        parser.error("Need dataset dir.")
    if args.loftr_script_path and not Path(args.loftr_script_path).is_file():
        print(f"WARN: LoFTR script missing: {args.loftr_script_path}")
    try:
        cache_p = Path(args.cache_dir)
        cache_p.mkdir(parents=True, exist_ok=True)
        psc_p = cache_p / PER_IMAGE_CACHE_BASE
        psc_p.mkdir(parents=True, exist_ok=True)
        metrics = args.metrics or METRICS_CONFIG.keys()
        for name in metrics:
            masked = name in ["PSNR", "SSIM", "LPIPS"]
            sub = name.lower() + "_masked" if masked else name.lower()
            (psc_p / sub).mkdir(parents=True, exist_ok=True)
        print(f"Cache directories ensured under {cache_p}")
    except OSError as e:
        parser.error(f"Failed to create cache dirs: {e}")

    start = time.time()
    try:
        runner = BenchmarkRunner(args)
        runner.run()
    except Exception as e:
        print(f"\n--- ERROR DURING BENCHMARK RUN ---", flush=True)
        print(f"Error: {e}", flush=True)
        import traceback

        print("\n--- Traceback ---", flush=True)
        traceback.print_exc()
        print("--- BENCHMARK SCRIPT TERMINATED ---", flush=True)
        exit(1)
    finally:
        print(f"\nTotal execution time: {time.time() - start:.2f} seconds.")
    print("\nBenchmarking script finished.")
