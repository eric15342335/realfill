# benchmarks.py

import argparse
import json
import multiprocessing
import multiprocessing.queues
import os
import re
import subprocess
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console, Group

# Rich library imports for enhanced terminal UI
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

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
def find_gt_mask_paths(
    results_dir_path: Path, dataset_dirs_map: dict
) -> tuple[str | None, str | None]:
    """Finds Ground Truth (GT) and Mask paths based on a results folder name."""
    dir_name = results_dir_path.name
    # Regex to match folder names like "RealBench-0-results" or "Custom-123"
    match = re.match(r"^(RealBench|Custom)-(\d+)(-results.*)?$", dir_name)
    if not match:
        return None, None

    benchmark_type, scene_number_str = match.group(1), match.group(2)
    dataset_base_dir = dataset_dirs_map.get(benchmark_type)

    if not dataset_base_dir or not Path(dataset_base_dir).is_dir():
        # This print is okay, as it happens before any Rich Live display usually
        print(
            f"Warning: Dataset base directory for '{benchmark_type}' not found or invalid: {dataset_base_dir}"
        )
        return None, None

    # Construct path based on expected structure: dataset_base_dir / BenchmarkType / SceneNumber / target / gt.png (or mask.png)
    base_path = Path(dataset_base_dir) / benchmark_type / scene_number_str / "target"
    gt_path = base_path / "gt.png"
    mask_path = base_path / "mask.png"

    if not gt_path.is_file() or not mask_path.is_file():
        # Fallback or detailed logging if paths are not found
        # print(f"Debug: Checked GT '{gt_path}' and Mask '{mask_path}' - Not found.")
        return None, None
    return str(gt_path), str(mask_path)


def count_result_images(folder_path: Path) -> int:
    """Counts the number of image files (e.g., '0.png', '1.png') in a folder."""
    if not folder_path.is_dir():
        return 0
    image_count = 0
    try:
        for item in folder_path.iterdir():
            # Check if the file is a .png and its stem is purely digits
            if item.is_file() and item.suffix.lower() == ".png" and item.stem.isdigit():
                image_count += 1
        return image_count
    except OSError as e:
        # This print is okay, generally called before Rich Live context
        print(f"Warning: Could not count images in {folder_path}: {e}")
        return 0


def parse_final_score(stdout_str: str) -> float | None:
    """Parses the 'FINAL_SCORE:' line from a metric script's standard output."""
    if not isinstance(stdout_str, str):
        return None
    for line in stdout_str.splitlines():
        if line.startswith("FINAL_SCORE:"):
            score_part = line.split(":", 1)[1].strip()
            if score_part == "ERROR":  # Explicit error reported by script
                return None
            try:
                return float(score_part)
            except ValueError:
                # This print is from within a worker's subprocess call, will appear on worker's stdout.
                # Rich will attempt to draw around it.
                error_print_func = tqdm.write if sys.stdout.isatty() else print
                error_print_func(f"Warning: Could not parse score from FINAL_SCORE line: '{line}'")
                return None
    return None  # FINAL_SCORE line not found


def load_json_cache(file_path: Path | str) -> dict | None:
    """Safely loads JSON data from a file."""
    resolved_path = Path(file_path)
    if not resolved_path.is_file():
        return None
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, TypeError) as e:
        # This print is from the main process, usually before Rich Live.
        print(f"Warning: Cache load failed for {resolved_path}: {e}")
        return None


def save_json_cache(data: dict, file_path: Path | str):
    """Safely saves data to a JSON file, handling numpy types for serialization."""
    resolved_path = Path(file_path)
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_data = convert_numpy_types(data)  # Ensure data is JSON serializable
        with open(resolved_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)
    except (OSError, TypeError) as e:
        # This print is from the main process.
        print(f"Error saving JSON cache to {resolved_path}: {e}")
        if isinstance(e, TypeError):
            # Log a snippet of the data that caused the TypeError for debugging
            print(f"Problematic data snippet (first 500 chars): {str(data)[:500]}")


def convert_numpy_types(obj):
    """Recursively converts numpy data types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(
        obj,
        (
            np.integer,
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):  # More comprehensive integer check
        return int(obj)
    elif isinstance(
        obj, (np.floating, np.float16, np.float32, np.float64)
    ):  # More comprehensive float check
        if np.isnan(obj):
            return None  # Represent NaN as null in JSON, as JSON standard does not support NaN
        elif np.isinf(obj):
            # Represent Inf as null or a very large number string if contextually appropriate. Null is safer.
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())  # Convert numpy arrays to lists
    elif isinstance(obj, (np.bool_, bool)):  # Handles numpy bool_ and python bool
        return bool(obj)
    elif isinstance(obj, np.void):  # For structured arrays or void types
        return None
    return obj  # Return the object itself if not a recognized numpy type


def get_scene_key(folder_name: str) -> str | None:
    """Extracts a scene key (e.g., 'RealBench-0') from a folder name."""
    match = re.match(r"^(RealBench|Custom)-(\d+)(-results.*)?$", folder_name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def run_metric_script_parallel(
    metric_name: str,
    script_filename: str,
    gt_path: str,
    mask_path: str,
    results_dir_str: str,
    cache_dir_str: str,
    num_images: int,
) -> tuple[str, float | None, str]:
    """
    Wrapper to run a single metric script as a subprocess.
    Returns (metric_name, score, folder_name).
    This function is executed within a worker process.
    """
    results_path = Path(results_dir_str)
    folder_name = results_path.name

    # Determine the directory of the metric scripts ('benchmark' subdirectory)
    # Assumes this script (benchmarks.py) is in the parent directory of 'benchmark/'
    current_script_dir = Path(__file__).parent
    benchmark_scripts_dir = current_script_dir / "benchmark"
    absolute_script_path = benchmark_scripts_dir / script_filename

    # tqdm.write is used for worker messages to minimize interference with Rich in the main process.
    # If not in a TTY, it falls back to print.
    worker_log_func = tqdm.write if sys.stdout.isatty() else print

    if not absolute_script_path.is_file():
        worker_log_func(
            f"WORKER_ERROR ({metric_name} on {folder_name}): Script not found at {absolute_script_path}"
        )
        return metric_name, None, folder_name

    command = [
        sys.executable,  # Use the same Python interpreter
        str(absolute_script_path),
        "--gt_path",
        gt_path,
        "--mask_path",
        mask_path,
        "--results_dir",
        str(results_path),
        "--cache_dir",
        cache_dir_str,
        "--num_images",
        str(num_images),
    ]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=720,  # 12-minute timeout per metric script
            cwd=benchmark_scripts_dir,  # Run script from its directory to handle relative paths within it
            encoding="utf-8",
            errors="replace",
        )
        stdout_content = process.stdout or ""
        stderr_content = process.stderr or ""

        if process.returncode != 0:
            stderr_suffix = stderr_content.strip()[-500:]  # Get last 500 chars of stderr
            worker_log_func(
                f"\nWORKER_ERROR ({metric_name} on {folder_name}, RC:{process.returncode}): Subprocess failed.\n"
                # f"  Command: {' '.join(command)}\n" # Potentially too verbose for regular logging
                f"  Script Dir: {benchmark_scripts_dir}\n"
                f"  Stderr (last 500 chars): ...{stderr_suffix}\n"
            )
            return metric_name, None, folder_name

        score = parse_final_score(stdout_content)
        if (
            score is None and "FINAL_SCORE:" in stdout_content and "ERROR" not in stdout_content
        ):  # Check if parsing failed but line was present
            stdout_suffix = stdout_content.strip()[-500:]
            worker_log_func(
                f"\nWORKER_WARN ({metric_name} on {folder_name}, RC:0): Score parsing failed.\n"
                f"  Stdout (last 500 chars): ...{stdout_suffix}\n"
            )
        elif score is None and "FINAL_SCORE:" not in stdout_content:  # FINAL_SCORE line missing
            worker_log_func(
                f"\nWORKER_WARN ({metric_name} on {folder_name}, RC:0): FINAL_SCORE line missing in output.\n"
            )

        return metric_name, score, folder_name

    except subprocess.TimeoutExpired:
        worker_log_func(
            f"\nWORKER_ERROR ({metric_name} on {folder_name}): Timeout expired for metric script.\n"
        )
        return metric_name, None, folder_name
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        worker_log_func(
            f"\nWORKER_CRITICAL ({metric_name} on {folder_name}): Exception during script execution: {type(e).__name__} - {e}\n"
        )
        traceback.print_exc(file=sys.stderr)  # Print full traceback to worker's stderr
        return metric_name, None, folder_name


def execute_metric_tasks_for_worker(
    metric_name_arg: str,
    script_filename_arg: str,
    tasks_for_this_metric: list,  # List of tuples (gt, mask, results_dir, cache_dir, num_images)
    results_queue: multiprocessing.Queue,
):
    """
    Worker function executed by a dedicated process for a single metric.
    It processes all folder tasks for this specific metric and sends results/progress via queue.
    Does NOT create its own Rich/tqdm progress bar for display.
    """
    for task_details_tuple in tasks_for_this_metric:
        # Unpack arguments for run_metric_script_parallel
        # task_details_tuple is (gt_path, mask_path, results_dir_str, cache_dir_str, num_images_for_scene)

        # For error reporting if run_metric_script_parallel itself crashes badly before returning folder_name
        current_folder_name_for_error_context = Path(task_details_tuple[2]).name

        try:
            # Prepend metric_name and script_filename to the task_details_tuple
            full_args_for_metric_script = (
                metric_name_arg,
                script_filename_arg,
            ) + task_details_tuple

            metric_name_result, score_result, folder_name_result = run_metric_script_parallel(
                *full_args_for_metric_script
            )

            # Send the actual result (score could be None if metric errored)
            results_queue.put(("RESULT", metric_name_result, score_result, folder_name_result))

        except Exception as e:
            # This is a fallback for critical errors if run_metric_script_parallel itself raises an unhandled exception
            # (which it shouldn't, given its own try-except blocks, but defense-in-depth).
            # This print goes to the worker's stderr. Rich in main process will try to draw around it.
            print(
                f"\nWORKER_UNHANDLED_CRASH ({metric_name_arg} on {current_folder_name_for_error_context}): "
                f"Unhandled exception in 'run_metric_script_parallel' call: {type(e).__name__} - {e}\n",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            # Send an error result for this task
            results_queue.put(
                ("RESULT", metric_name_arg, None, current_folder_name_for_error_context)
            )

        # Send a progress update message after each task is attempted (successfully or not)
        # The main process will use this to advance the Rich progress bar for this specific metric.
        results_queue.put(("PROGRESS_TICK", metric_name_arg))


class BenchmarkRunner:
    def __init__(self, cli_args: argparse.Namespace):
        self.args = cli_args
        self.base_results_dir = Path(cli_args.results_base_dir).resolve()
        self.dataset_dirs_map = (
            {}
        )  # Stores mapping like {"RealBench": Path(...), "Custom": Path(...)}

        # Validate and store dataset paths from arguments
        if cli_args.realbench_dataset_dir:
            rb_path = Path(cli_args.realbench_dataset_dir).resolve()
            if rb_path.is_dir():
                self.dataset_dirs_map["RealBench"] = rb_path
            else:
                print(f"Warning: RealBench dataset directory not found: {rb_path}")
        if cli_args.custom_dataset_dir:
            cu_path = Path(cli_args.custom_dataset_dir).resolve()
            if cu_path.is_dir():
                self.dataset_dirs_map["Custom"] = cu_path
            else:
                print(f"Warning: Custom dataset directory not found: {cu_path}")

        if not self.dataset_dirs_map:
            # This error will be caught by the main execution block if raised here
            raise ValueError(
                "No valid dataset directories were provided or found. Please specify --realbench_dataset_dir or --custom_dataset_dir."
            )

        self.cache_dir = Path(cli_args.cache_dir).resolve()
        self.output_file_path = (
            Path(cli_args.output_file).resolve() if cli_args.output_file else None
        )
        self.num_images_per_scene = cli_args.num_images

        # Normalize force_recalc list to lowercase for case-insensitive matching
        self.force_recalc_metrics_list = [
            metric.lower() for metric in (cli_args.force_recalc or [])
        ]

        # Determine metrics to run: from args or all from METRICS_CONFIG
        if cli_args.metrics:
            self.metrics_to_run_list = cli_args.metrics
        else:  # Default to all configured metrics if --metrics is not specified
            self.metrics_to_run_list = list(METRICS_CONFIG.keys())

        # Resolve LoFTR script path relative to this benchmark script's location
        if cli_args.loftr_script_path:
            # Path(__file__).parent gives the directory of the current script (benchmarks.py)
            candidate_loftr_path = Path(__file__).parent / cli_args.loftr_script_path
            if candidate_loftr_path.is_file():
                self.loftr_script_path = candidate_loftr_path.resolve()
            else:
                print(
                    f"Warning: LoFTR script '{cli_args.loftr_script_path}' (resolved to '{candidate_loftr_path}') "
                    "not found. LoFTR analysis will be skipped."
                )
                self.loftr_script_path = None
        else:
            # Should not happen if default is set in argparse, but good for robustness
            print("Warning: LoFTR script path not specified. LoFTR analysis will be skipped.")
            self.loftr_script_path = None

        self.master_cache_file = self.cache_dir / MASTER_CACHE_FILENAME
        self.per_image_cache_root_dir = self.cache_dir / PER_IMAGE_CACHE_BASE

        self.discovered_result_folders = []  # List of Path objects for valid result folders
        self.master_results_data = defaultdict(dict)  # Stores {folder_name: {metric: score}}
        self.per_image_scores_cache = defaultdict(
            lambda: defaultdict(dict)
        )  # In-memory cache for per-image scores
        self.loftr_ranking_data = defaultdict(
            list
        )  # Stores {folder_name: [ranked_image_filenames]}

        # Initial console output (before Rich Live typically starts)
        print("Benchmark Runner Initialized:")
        print(f"  Results Base Directory: {self.base_results_dir}")
        print(f"  Dataset Mappings: {self.dataset_dirs_map}")
        print(f"  Cache Directory: {self.cache_dir}")
        print(f"  Metrics to Evaluate: {', '.join(self.metrics_to_run_list)}")
        print(
            f"  Force Recalculate: {', '.join(self.force_recalc_metrics_list) if self.force_recalc_metrics_list else 'None'}"
        )
        if self.loftr_script_path:
            print(f"  LoFTR Script: {self.loftr_script_path}")
        if self.output_file_path:
            print(f"  Report Output File: {self.output_file_path}")

        try:
            cpu_core_count = os.cpu_count()
            print(f"  System Info: Detected {cpu_core_count} CPU cores.")
        except NotImplementedError:
            print("  System Info: Could not detect CPU core count.")
            cpu_core_count = None  # Unused for Rich strategy but informative

    def discover_folders(self):
        """Discovers result folders matching the expected pattern and GT/Mask availability."""
        print(f"\nScanning for result folders in {self.base_results_dir}...")
        self.discovered_result_folders = []  # Reset if called multiple times
        potential_folders_count = 0
        skipped_due_to_mapping = 0
        skipped_due_to_gt_mask = 0

        if not self.base_results_dir.is_dir():
            print(f"Error: Base results directory not found: {self.base_results_dir}")
            return

        # Use tqdm for this initial scan as it's single-threaded and can be long
        # Disabling bar if not TTY for cleaner logs
        folder_iterator = sorted(self.base_results_dir.iterdir())  # Sort for predictable order
        pbar_disabled = not sys.stdout.isatty()
        for item_path in tqdm(
            folder_iterator, desc="Scanning Result Folders", unit="folder", disable=pbar_disabled
        ):
            if item_path.is_dir() and re.match(
                r"^(RealBench|Custom)-\d+(-results.*)?$", item_path.name
            ):
                potential_folders_count += 1
                # Extract benchmark type (RealBench or Custom) to check against dataset_dirs_map
                type_match = re.match(r"^(RealBench|Custom)", item_path.name)
                if type_match:
                    benchmark_type_from_folder = type_match.group(1)
                    if benchmark_type_from_folder in self.dataset_dirs_map:
                        # Check for corresponding GT and Mask files
                        gt_file, mask_file = find_gt_mask_paths(item_path, self.dataset_dirs_map)
                        if gt_file and mask_file:
                            self.discovered_result_folders.append(item_path)
                        else:
                            # Use tqdm.write for messages during tqdm loop to avoid breaking the bar
                            if not pbar_disabled:
                                tqdm.write(f"  Skipping '{item_path.name}': Missing GT/Mask files.")
                            skipped_due_to_gt_mask += 1
                    else:
                        if not pbar_disabled:
                            tqdm.write(
                                f"  Skipping '{item_path.name}': Dataset type '{benchmark_type_from_folder}' not mapped in dataset_dirs_map."
                            )
                        skipped_due_to_mapping += 1

        # Summary after scan
        print(f"Folder scan complete. Found {potential_folders_count} potential result folders.")
        print(
            f"  - Added {len(self.discovered_result_folders)} folders with valid dataset mapping & GT/Mask files."
        )
        if skipped_due_to_mapping > 0:
            print(
                f"  - Skipped {skipped_due_to_mapping} folders (dataset type not specified or mapped)."
            )
        if skipped_due_to_gt_mask > 0:
            print(
                f"  - Skipped {skipped_due_to_gt_mask} folders (missing corresponding GT/Mask files)."
            )

    def load_master_cache(self):
        """Loads the master results cache file from disk."""
        print(f"Loading master results cache: {self.master_cache_file}")
        cached_data = load_json_cache(self.master_cache_file)

        self.master_results_data = defaultdict(dict)  # Reset before loading
        if cached_data and isinstance(cached_data, dict):
            loaded_entries_count = 0
            for folder_name, metrics_dict in cached_data.items():
                if isinstance(metrics_dict, dict):
                    self.master_results_data[folder_name] = metrics_dict
                    loaded_entries_count += 1
                else:
                    print(
                        f"Warning: Invalid cache entry format for folder '{folder_name}'. Skipping."
                    )
            print(f"Loaded {loaded_entries_count} folder entries from master cache.")
        else:
            print("No valid master cache found or cache file is empty/corrupted.")

    def save_master_cache(self):
        """Saves the current master results to the cache file."""
        # Convert defaultdict to dict for saving, as defaultdict might not be ideal for JSON structure
        data_to_save = dict(self.master_results_data)
        print(f"Saving master results cache ({len(data_to_save)} entries)...")
        save_json_cache(data_to_save, self.master_cache_file)
        print(f"Master cache saved to: {self.master_cache_file}")

    def check_folder_contents(self, folder_path: Path) -> bool:
        """Checks if the folder contains the expected number of result images (e.g., 0.png to N-1.png)."""
        if not folder_path.is_dir():
            return False
        try:
            # Check for existence of each numbered image file
            for i in range(self.num_images_per_scene):
                if not (folder_path / f"{i}.png").is_file():
                    return False  # Missing at least one expected image
            return True  # All expected images found
        except Exception as e:
            # This print occurs during the initial scan, before Rich Live
            print(f"Error while checking contents of folder {folder_path}: {e}")
            return False

    def run_all_metrics(self):
        """
        Runs metric calculations for all discovered and valid folders.
        Uses Rich library for live progress display.
        """
        if not self.discovered_result_folders:
            print("No result folders discovered to run metrics on. Skipping metric calculation.")
            return

        # Initialize Rich Console for consistent output
        # All prints from this method onwards should ideally use rich_console.print()
        # if they are intended to interact correctly with the Live display.
        rich_console = Console()
        rich_console.print(
            "\n--- Preparing Metric Tasks for Parallel Execution ---", style="bold cyan"
        )
        self.load_master_cache()  # This method uses standard print, okay before Live starts

        skipped_incomplete_folders = 0
        # tasks_by_metric: dict where key is metric_name, value is list of task_detail_tuples
        tasks_to_run_by_metric = defaultdict(list)
        folders_processed_count = 0  # Folders that are complete and will be processed
        folders_requiring_metric_tasks = set()  # Folders that have at least one metric to calculate

        # Create a mutable copy of metrics to run, as it might be pruned
        # if scripts are missing or metrics are invalid.
        # Use valid_metrics_for_run to track metrics that actually have scripts.
        valid_metrics_for_run = set(self.metrics_to_run_list)

        rich_console.print(
            "Checking folder contents and identifying required metric calculations..."
        )
        benchmark_scripts_dir_base = Path(__file__).parent / "benchmark"

        # This initial folder check still uses tqdm because it's a straightforward,
        # single-threaded preparation step before the complex multiprocessing with Rich Live.
        # tqdm.write is used for messages within this loop.
        pbar_disabled = not sys.stdout.isatty()
        for folder_path in tqdm(
            self.discovered_result_folders,
            desc="Validating Folders & Cache",
            unit="folder",
            disable=pbar_disabled,
        ):
            folder_name = folder_path.name
            ground_truth_path, mask_path = find_gt_mask_paths(folder_path, self.dataset_dirs_map)

            if not ground_truth_path or not mask_path:
                # This should ideally not happen if discover_folders worked correctly
                tqdm.write(
                    f"Warning: Skipping '{folder_name}': GT/Mask path became invalid post-discovery."
                )
                continue

            if not self.check_folder_contents(folder_path):
                actual_image_count = count_result_images(
                    folder_path
                )  # Recount for accurate message
                tqdm.write(
                    f"Skipping '{folder_name}': Incomplete ({actual_image_count}/{self.num_images_per_scene} images)."
                )
                skipped_incomplete_folders += 1
                if folder_name in self.master_results_data:
                    tqdm.write(
                        f"  - Removing stale cache entry for incomplete folder '{folder_name}'."
                    )
                    del self.master_results_data[folder_name]
                continue

            folders_processed_count += 1
            if folder_name not in self.master_results_data:
                self.master_results_data[folder_name] = {}  # Initialize if new

            # Iterate over a copy of valid_metrics_for_run for safe removal during iteration
            for metric_name in list(valid_metrics_for_run):
                if metric_name not in METRICS_CONFIG:
                    tqdm.write(
                        f"Warning: Metric '{metric_name}' is not in METRICS_CONFIG. Removing from current run."
                    )
                    valid_metrics_for_run.discard(metric_name)
                    if metric_name in self.metrics_to_run_list:
                        self.metrics_to_run_list.remove(metric_name)
                    continue

                script_filename, _ = METRICS_CONFIG[metric_name]
                absolute_script_path = benchmark_scripts_dir_base / script_filename
                if not absolute_script_path.is_file():
                    tqdm.write(
                        f"Warning: Script '{absolute_script_path}' missing for metric '{metric_name}'. "
                        f"This metric will be skipped for all folders."
                    )
                    valid_metrics_for_run.discard(
                        metric_name
                    )  # Remove from consideration for this run
                    if metric_name in self.metrics_to_run_list:
                        self.metrics_to_run_list.remove(metric_name)
                    continue  # Skip this metric for this folder and subsequent ones

                # Determine if recalculation is needed
                force_recalculation = (
                    "all" in self.force_recalc_metrics_list
                    or metric_name.lower() in self.force_recalc_metrics_list
                )

                score_is_cached_and_valid = (
                    metric_name in self.master_results_data.get(folder_name, {})
                    and self.master_results_data[folder_name][metric_name] is not None
                )

                if force_recalculation or not score_is_cached_and_valid:
                    task_details_tuple = (
                        ground_truth_path,
                        mask_path,
                        str(folder_path),
                        str(self.cache_dir),
                        self.num_images_per_scene,
                    )
                    tasks_to_run_by_metric[metric_name].append(task_details_tuple)
                    folders_requiring_metric_tasks.add(folder_name)
                    # Pre-initialize or invalidate stale cache entry for this metric and folder
                    self.master_results_data[folder_name][metric_name] = None

        # Filter out metrics that ended up with no tasks after folder/cache checks
        # or were removed because their scripts were missing.
        active_tasks_by_metric = {
            metric: tasks
            for metric, tasks in tasks_to_run_by_metric.items()
            if tasks and metric in valid_metrics_for_run
        }
        total_individual_metric_folder_tasks = sum(
            len(tasks) for tasks in active_tasks_by_metric.values()
        )

        if not total_individual_metric_folder_tasks:
            rich_console.print(
                "\nNo metric calculations required (all results cached, or folders incomplete/skipped, or no valid metrics)."
            )
            self.save_master_cache()  # Save in case stale entries were removed
            rich_console.print("\n--- Metric Execution Phase Skipped ---", style="bold yellow")
            rich_console.print(
                f"Folders eligible for processing (complete): {folders_processed_count}"
            )
            if skipped_incomplete_folders > 0:
                rich_console.print(
                    f"Folders skipped due to incompleteness: {skipped_incomplete_folders}"
                )
            rich_console.print(f"Metric-folder tasks needing calculation: 0")
            return

        num_dedicated_metric_workers = len(active_tasks_by_metric)
        rich_console.print(
            f"\n--- Launching {total_individual_metric_folder_tasks} Metric-Folder Tasks "
            f"using {num_dedicated_metric_workers} Dedicated Metric Worker Processes ---",
            style="bold cyan",
        )

        final_metrics_being_executed = sorted(list(active_tasks_by_metric.keys()))
        rich_console.print(
            f"Metrics involved in this run: {', '.join(final_metrics_being_executed)}"
        )
        rich_console.print(
            f"Results will be updated/stored for {len(folders_requiring_metric_tasks)} folders."
        )

        # --- Multiprocessing and Rich Live Display Setup ---
        # Manager().Queue() is generally more robust for complex objects or across different OS
        results_queue = multiprocessing.Manager().Queue()
        worker_processes_list = []

        # Define Rich Progress objects
        # Overall progress bar for all individual metric-folder tasks
        overall_progress_display = Progress(
            TextColumn("Overall Progress:"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TextColumn("({task.completed} of {task.total} tasks)"),
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            SpinnerColumn(spinner_name="dots"),
            console=rich_console,
            transient=False,  # Keep this bar visible after completion
        )
        overall_task_id = overall_progress_display.add_task(
            "Calculating all metrics", total=total_individual_metric_folder_tasks, start=False
        )

        # Group for individual metric progress bars
        metric_specific_progress_display = Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("({task.completed} of {task.total} folders)"),
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            console=rich_console,
            transient=True,  # Individual metric bars can disappear when done
        )

        # Map metric names to their Rich Task IDs
        metric_task_ids_map = {}
        for metric_name_key, tasks_list_for_metric in active_tasks_by_metric.items():
            task_id = metric_specific_progress_display.add_task(
                description=f"Metric: {metric_name_key:<10}",  # Pad for alignment
                total=len(tasks_list_for_metric),
                start=False,  # Will be started when worker process launches
            )
            metric_task_ids_map[metric_name_key] = task_id

        # Group the progress displays for Rich Live
        # Any other Rich renderables (like Tables or Panels for logs) could be added here.
        live_display_group = Group(metric_specific_progress_display, overall_progress_display)

        # Start the Live display context
        # redirect_stdout and redirect_stderr are False by default. This means prints from
        # worker processes (like those from tqdm.write in run_metric_script_parallel)
        # will print directly to terminal, and Rich Live will redraw around them.
        # Setting them to True can capture worker output but might have performance implications
        # or require careful handling if workers produce a lot of output.
        with Live(
            live_display_group,
            console=rich_console,
            refresh_per_second=12,
            vertical_overflow="visible",
        ) as live:
            overall_progress_display.start_task(overall_task_id)  # Start the overall counter

            # Launch worker processes
            for metric_name_to_run, tasks_list in active_tasks_by_metric.items():
                if metric_name_to_run in metric_task_ids_map:  # Ensure task ID exists
                    metric_specific_progress_display.start_task(
                        metric_task_ids_map[metric_name_to_run]
                    )

                script_filename_for_metric, _ = METRICS_CONFIG[metric_name_to_run]
                worker_args = (
                    metric_name_to_run,
                    script_filename_for_metric,
                    tasks_list,
                    results_queue,
                )
                process = multiprocessing.Process(
                    target=execute_metric_tasks_for_worker, args=worker_args
                )
                worker_processes_list.append(process)
                process.start()

            live.console.print("[green]All metric worker processes launched.[/green]")

            # Collect results from the queue
            num_results_collected = 0
            active_workers_exist = True

            while (
                num_results_collected < total_individual_metric_folder_tasks
                and active_workers_exist
            ):
                try:
                    # Timeout allows the Live display to refresh and check worker status
                    message_from_worker = results_queue.get(timeout=0.5)  # seconds

                    msg_type, metric_name_from_msg, *payload_data = message_from_worker

                    if msg_type == "PROGRESS_TICK":
                        if metric_name_from_msg in metric_task_ids_map:
                            metric_specific_progress_display.advance(
                                metric_task_ids_map[metric_name_from_msg], 1
                            )
                        # Note: Overall progress is advanced when a 'RESULT' is processed.

                    elif msg_type == "RESULT":
                        score_value, folder_name_from_msg = payload_data
                        if folder_name_from_msg and metric_name_from_msg:
                            # Ensure folder entry exists (should, from earlier prep)
                            if folder_name_from_msg not in self.master_results_data:
                                live.console.print(
                                    f"[yellow]Warning: Result for unexpected folder '{folder_name_from_msg}'. Initializing entry.[/yellow]"
                                )
                                self.master_results_data[folder_name_from_msg] = {}

                            # Store the result if the metric is still considered valid for this run
                            if metric_name_from_msg in valid_metrics_for_run:
                                self.master_results_data[folder_name_from_msg][
                                    metric_name_from_msg
                                ] = score_value

                        num_results_collected += 1
                        overall_progress_display.advance(overall_task_id, 1)

                    # Could add handling for other message types, e.g., explicit log messages from workers
                    # elif msg_type == 'WORKER_LOG':
                    #     log_level, log_message = payload_data
                    #     live.console.print(f"Worker ({metric_name_from_msg}): {log_message}", style=log_level)

                # Expected if queue is empty with timeout
                except multiprocessing.queues.Empty:
                    pass
                except Exception as e:  # Handle unexpected errors during queue processing
                    live.console.print(
                        f"[bold red]Error processing message from worker queue: {e}[/bold red]"
                    )
                    # Log traceback to stderr to avoid interfering with Rich display too much
                    traceback.print_exc(file=sys.stderr)

                # Check if all worker processes have exited if we haven't collected all results yet
                if num_results_collected < total_individual_metric_folder_tasks:
                    active_workers_exist = any(p.is_alive() for p in worker_processes_list)
                    if not active_workers_exist:
                        live.console.print(
                            "[bold yellow]Warning: All worker processes have exited, "
                            "but not all expected results were collected. "
                            "Check for errors in worker logs (printed above Rich UI or in stderr).[/bold yellow]"
                        )
                        # Update overall progress to reflect actual collected items if short
                        if overall_progress_display.tasks[0].completed < num_results_collected:
                            overall_progress_display.update(
                                overall_task_id, completed=num_results_collected
                            )
                        break  # Exit collection loop

            # After collection loop (either completed or broken due to workers finishing early)
            # Ensure overall progress reflects the final count
            overall_progress_display.update(overall_task_id, completed=num_results_collected)
            if num_results_collected < total_individual_metric_folder_tasks:
                overall_progress_display.update(
                    overall_task_id, description="Calculating all metrics (Run Incomplete)"
                )
            else:
                overall_progress_display.update(
                    overall_task_id, description="Calculating all metrics (Completed)"
                )

            live.console.print(
                "[green]Result collection phase complete. Waiting for worker processes to join...[/green]"
            )
            for i, process_to_join in enumerate(worker_processes_list):
                process_to_join.join(timeout=10)  # Give a generous timeout for clean exit
                if process_to_join.is_alive():
                    live.console.print(
                        f"[yellow]Warning: Worker process {process_to_join.pid} (task {i}) did not join cleanly. Terminating.[/yellow]"
                    )
                    process_to_join.terminate()  # Force terminate if stuck
                    process_to_join.join()  # Wait for termination

            # Stop individual metric progress tasks if they are not transient or to ensure they are marked done
            for task_id_val in metric_task_ids_map.values():
                if not metric_specific_progress_display.tasks[task_id_val].finished:
                    metric_specific_progress_display.stop_task(task_id_val)
            # Stop the entire metric_specific_progress_display if it's not needed anymore
            metric_specific_progress_display.stop()

            # Overall progress is not transient, so its task will remain. We can stop it.
            if not overall_progress_display.tasks[0].finished:
                overall_progress_display.stop_task(overall_task_id)
            # overall_progress_display itself is not stopped to keep it visible as a summary.

        # --- End of Rich Live context ---

        # Final messages after Rich Live context has ended
        rich_console.print(
            "[bold green]All dedicated metric worker processes have completed processing.[/bold green]"
        )
        self.save_master_cache()  # Save all collected results (including None for errors)

        rich_console.print("\n--- Metric Execution Summary ---", style="bold cyan")
        rich_console.print(f"Folders eligible for processing (complete): {folders_processed_count}")
        if skipped_incomplete_folders > 0:
            rich_console.print(
                f"Folders skipped due to incompleteness: {skipped_incomplete_folders}"
            )
        rich_console.print(
            f"Folders requiring metric calculations this run: {len(folders_requiring_metric_tasks)}"
        )
        rich_console.print(
            f"Total individual metric-folder tasks processed: {num_results_collected} "
            f"(out of {total_individual_metric_folder_tasks} identified)."
        )

    def load_per_image_results(self, folder_name: str, metric_name: str) -> dict | None:
        """
        Loads per-image scores for a given folder and metric, used for LoFTR analysis.
        Caches results in self.per_image_scores_cache to avoid redundant file I/O.
        """
        # Check in-memory cache first
        if (
            folder_name in self.per_image_scores_cache
            and metric_name in self.per_image_scores_cache[folder_name]
        ):
            return self.per_image_scores_cache[folder_name][metric_name]

        # Determine subdirectory for per-image cache based on metric type (masked or not)
        # This matches the naming convention used in the __main__ block for cache setup.
        is_masked_metric = metric_name in ["PSNR", "SSIM", "LPIPS"]  # Example masked metrics
        cache_subdir_name = metric_name.lower() + ("_masked" if is_masked_metric else "")

        per_image_cache_file = (
            self.per_image_cache_root_dir / cache_subdir_name / f"{folder_name}.json"
        )

        data = load_json_cache(per_image_cache_file)  # Uses standard print for errors
        if (
            data
            and isinstance(data, dict)
            and "per_image" in data
            and isinstance(data["per_image"], dict)
        ):
            # Store in in-memory cache for future calls
            self.per_image_scores_cache[folder_name][metric_name] = data["per_image"]
            return data["per_image"]

        # print(f"Debug: Per-image scores not found or invalid for {metric_name} in {folder_name} at {per_image_cache_file}")
        return None

    def run_loftr_ranking(self):
        """
        Runs the LoFTR ranking script for suitable RealBench folders.
        This method uses tqdm for its own progress as it's a sequential operation.
        """
        if not self.loftr_script_path:
            print("\nLoFTR ranking skipped: LoFTR script path is invalid or script not found.")
            return

        print("\n--- Running LoFTR Ranking ---", flush=True)  # flush in case of subsequent tqdm

        # Filter folders suitable for LoFTR: RealBench, non-gen, non-fp32, and complete
        folders_for_loftr_ranking = [
            folder_path
            for folder_path in self.discovered_result_folders
            if folder_path.name.startswith("RealBench")
            and "gen" not in folder_path.name.lower()  # case-insensitive check for "gen"
            and "fp32" not in folder_path.name.lower()  # case-insensitive check for "fp32"
            and self.check_folder_contents(folder_path)  # Ensure folder is complete
        ]

        if not folders_for_loftr_ranking:
            print("No suitable RealBench folders found for LoFTR ranking.")
            return

        print(f"Found {len(folders_for_loftr_ranking)} suitable folders for LoFTR ranking.")

        num_ranked_successfully = 0
        num_skipped_existing = 0
        num_skipped_missing_ref = 0
        num_errors = 0

        force_loftr_recalc = (
            "all" in self.force_recalc_metrics_list or "loftr" in self.force_recalc_metrics_list
        )

        print("Executing LoFTR ranking script for each suitable folder (this may take time)...")
        # tqdm for LoFTR processing loop
        pbar_disabled = not sys.stdout.isatty()
        for folder_path_for_loftr in tqdm(
            folders_for_loftr_ranking,
            desc="LoFTR Ranking",
            unit="folder",
            leave=False,
            disable=pbar_disabled,
        ):
            output_json_path = folder_path_for_loftr / LOFTR_RANKING_FILENAME

            if output_json_path.is_file() and not force_loftr_recalc:
                num_skipped_existing += 1
                continue

            gt_path_str, _ = find_gt_mask_paths(folder_path_for_loftr, self.dataset_dirs_map)
            if not gt_path_str:
                tqdm.write(
                    f"Warning: No Ground Truth path found for '{folder_path_for_loftr.name}', skipping LoFTR."
                )
                num_skipped_missing_ref += 1
                continue

            # LoFTR expects a 'ref' directory at the same level as 'target'
            reference_images_dir = Path(gt_path_str).parent.parent / "ref"
            if not reference_images_dir.is_dir():
                tqdm.write(
                    f"Warning: Reference image directory '{reference_images_dir}' missing for "
                    f"'{folder_path_for_loftr.name}', skipping LoFTR."
                )
                num_skipped_missing_ref += 1
                continue

            if (
                force_loftr_recalc and output_json_path.is_file()
            ):  # If forcing, message that it's re-running
                tqdm.write(f"Forcing LoFTR re-ranking for {folder_path_for_loftr.name}...")

            command_to_run_loftr = [
                sys.executable,
                str(self.loftr_script_path),
                "--source-dir",
                str(folder_path_for_loftr),
                "--ref-dir",
                str(reference_images_dir),
                "--rank-only",  # Assuming loftr_ranking.py supports this flag
                "--ranking-output-file",
                LOFTR_RANKING_FILENAME,  # Script should save to its source-dir
            ]

            try:
                process_loftr = subprocess.run(
                    command_to_run_loftr,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                    encoding="utf-8",
                    errors="replace",  # 10-minute timeout for LoFTR
                )
                if process_loftr.returncode != 0:
                    stderr_info = process_loftr.stderr or ""
                    tqdm.write(
                        f"\nERROR during LoFTR ranking for '{folder_path_for_loftr.name}' "
                        f"(RC:{process_loftr.returncode}). Stderr (last 500 chars): ...{stderr_info.strip()[-500:]}\n"
                    )
                    num_errors += 1
                else:
                    # Check if the output file was actually created by the script
                    if output_json_path.is_file():
                        num_ranked_successfully += 1
                    else:
                        tqdm.write(
                            f"\nERROR LoFTR for '{folder_path_for_loftr.name}': Script ran (RC:0) but output file '{LOFTR_RANKING_FILENAME}' is missing.\n"
                        )
                        num_errors += 1
            except subprocess.TimeoutExpired:
                tqdm.write(
                    f"\nERROR: Timeout expired during LoFTR ranking for '{folder_path_for_loftr.name}'.\n"
                )
                num_errors += 1
            except Exception as e:
                tqdm.write(
                    f"\nERROR: An unexpected exception occurred during LoFTR ranking for '{folder_path_for_loftr.name}': {e}\n"
                )
                num_errors += 1

        # Summary after LoFTR ranking loop
        print("--- LoFTR Ranking Finished ---")
        if num_ranked_successfully > 0:
            print(f"Successfully ran/updated LoFTR ranks for {num_ranked_successfully} folders.")
        if num_skipped_existing > 0:
            print(
                f"Skipped {num_skipped_existing} folders (LoFTR ranks already existed and no force recalc)."
            )
        if num_skipped_missing_ref > 0:
            print(
                f"Skipped {num_skipped_missing_ref} folders (missing reference directory or GT path)."
            )
        if num_errors > 0:
            print(
                f"Encountered errors during LoFTR ranking for {num_errors} folders. Check logs above."
            )

    def load_loftr_ranks(self):
        """Loads LoFTR ranking results from JSON files into self.loftr_ranking_data."""
        if not self.loftr_script_path:  # Check if LoFTR was even available for ranking
            print("Skipping LoFTR rank loading: LoFTR script was not available.")
            self.loftr_ranking_data = defaultdict(list)  # Ensure it's empty
            return

        print("Loading LoFTR ranking files...")
        self.loftr_ranking_data = defaultdict(list)  # Reset before loading

        # Consider only folders that were deemed suitable for LoFTR ranking initially
        folders_to_check_for_ranks = [
            folder_path
            for folder_path in self.discovered_result_folders
            if folder_path.name.startswith("RealBench")
            and "gen" not in folder_path.name.lower()
            and "fp32" not in folder_path.name.lower()
            and self.check_folder_contents(folder_path)  # Only load for complete folders
        ]

        num_ranks_loaded = 0
        num_missing_or_invalid = 0

        if not folders_to_check_for_ranks:
            print("No suitable folders found from which to load LoFTR ranks.")
            return

        for folder_path in folders_to_check_for_ranks:
            rank_file_path = folder_path / LOFTR_RANKING_FILENAME
            rank_data_json = load_json_cache(rank_file_path)  # Uses standard print for errors

            if (
                rank_data_json
                and isinstance(rank_data_json, dict)
                and "ranking" in rank_data_json
                and isinstance(rank_data_json["ranking"], list)
            ):

                # Validate the structure of the ranking list (list of dicts with 'filename')
                ranked_filenames = [
                    item["filename"]
                    for item in rank_data_json["ranking"]
                    if isinstance(item, dict)
                    and "filename" in item
                    and isinstance(item["filename"], str)
                ]

                # Further validate filenames (e.g., "0.png", "1.png")
                if ranked_filenames and all(
                    re.match(r"^\d+\.png$", fname) for fname in ranked_filenames
                ):
                    self.loftr_ranking_data[folder_path.name] = ranked_filenames
                    num_ranks_loaded += 1
                else:
                    if ranked_filenames:  # Only warn if list was populated but filenames were bad
                        print(
                            f"Warning: Invalid filenames found in LoFTR rank file: {rank_file_path}"
                        )
                    num_missing_or_invalid += 1
            else:
                # print(f"Debug: LoFTR rank file missing or invalid for {folder_path.name} at {rank_file_path}")
                num_missing_or_invalid += 1

        print("Finished loading LoFTR ranks.")
        if num_ranks_loaded > 0:
            print(f"Successfully loaded LoFTR ranks for {num_ranks_loaded} folders.")
        if num_missing_or_invalid > 0:
            # This count includes files not found and files with invalid format.
            print(
                f"LoFTR ranks were missing or in an invalid format for {num_missing_or_invalid} folders."
            )

    def analyze_results(self) -> dict:
        """Performs comparative analysis on the collected master results."""
        if not self.master_results_data:
            print("Analysis skipped: No metric results available in master_results_data.")
            return {}

        print("\n--- Analyzing Metric Results ---", flush=True)
        analysis_summary = {}

        # Get a list of folder names for which results were successfully collected (non-None for at least one metric)
        # For simplicity, using all keys from master_results_data, assuming _calculate_average handles None scores.
        valid_folder_names_with_results = list(self.master_results_data.keys())

        # Define groups for analysis based on folder naming conventions
        realbench_fp16_nongen_folders = [
            name
            for name in valid_folder_names_with_results
            if name.startswith("RealBench")
            and "fp32" not in name.lower()
            and "gen" not in name.lower()
        ]
        custom_fp16_nongen_folders = [
            name
            for name in valid_folder_names_with_results
            if name.startswith("Custom")
            and "fp32" not in name.lower()
            and "gen" not in name.lower()
        ]

        # --- Overall Averages ---
        analysis_summary["overall_realbench_fp16_nongen"] = self._calculate_average(
            realbench_fp16_nongen_folders
        )
        analysis_summary["overall_custom_fp16_nongen"] = self._calculate_average(
            custom_fp16_nongen_folders
        )

        # --- FP16 vs FP32 Comparisons (RealBench) ---
        # Map scene keys to full folder names for fp16 and fp32 variants
        fp16_realbench_map = {
            get_scene_key(name): name
            for name in realbench_fp16_nongen_folders
            if get_scene_key(name)
        }
        fp32_realbench_map = {
            get_scene_key(name): name
            for name in valid_folder_names_with_results
            if name.startswith("RealBench")
            and "fp32" in name.lower()
            and "gen" not in name.lower()
            and get_scene_key(name)
        }
        common_realbench_fp_scenes = sorted(
            [key for key in fp16_realbench_map if key in fp32_realbench_map]
        )
        if common_realbench_fp_scenes:
            analysis_summary["fp16_vs_fp32_realbench"] = {
                "common_scenes": common_realbench_fp_scenes,
                "fp16_avg": self._calculate_average(
                    [fp16_realbench_map[key] for key in common_realbench_fp_scenes]
                ),
                "fp32_avg": self._calculate_average(
                    [fp32_realbench_map[key] for key in common_realbench_fp_scenes]
                ),
            }
        else:
            analysis_summary["fp16_vs_fp32_realbench"] = (
                "N/A (No common RealBench FP16/FP32 non-generated scenes found for comparison)"
            )

        # --- FP16 vs FP32 Comparisons (Custom) ---
        fp16_custom_map = {
            get_scene_key(name): name for name in custom_fp16_nongen_folders if get_scene_key(name)
        }
        fp32_custom_map = {
            get_scene_key(name): name
            for name in valid_folder_names_with_results
            if name.startswith("Custom")
            and "fp32" in name.lower()
            and "gen" not in name.lower()
            and get_scene_key(name)
        }
        common_custom_fp_scenes = sorted([key for key in fp16_custom_map if key in fp32_custom_map])
        if common_custom_fp_scenes:
            analysis_summary["fp16_vs_fp32_custom"] = {
                "common_scenes": common_custom_fp_scenes,
                "fp16_avg": self._calculate_average(
                    [fp16_custom_map[key] for key in common_custom_fp_scenes]
                ),
                "fp32_avg": self._calculate_average(
                    [fp32_custom_map[key] for key in common_custom_fp_scenes]
                ),
            }
        else:
            analysis_summary["fp16_vs_fp32_custom"] = (
                "N/A (No common Custom FP16/FP32 non-generated scenes found for comparison)"
            )

        # --- Non-Generated vs Generated Comparisons (RealBench FP16) ---
        generated_realbench_fp16_map = {
            get_scene_key(name): name
            for name in valid_folder_names_with_results
            if name.startswith("RealBench")
            and "gen" in name.lower()
            and "fp32" not in name.lower()
            and get_scene_key(name)
        }
        common_realbench_gen_scenes = sorted(
            [key for key in fp16_realbench_map if key in generated_realbench_fp16_map]
        )
        if common_realbench_gen_scenes:
            analysis_summary["gen_vs_nongen_realbench_fp16"] = {
                "common_scenes": common_realbench_gen_scenes,
                "nongen_avg": self._calculate_average(
                    [fp16_realbench_map[key] for key in common_realbench_gen_scenes]
                ),
                "gen_avg": self._calculate_average(
                    [generated_realbench_fp16_map[key] for key in common_realbench_gen_scenes]
                ),
            }
        else:
            analysis_summary["gen_vs_nongen_realbench_fp16"] = (
                "N/A (No common RealBench FP16 generated/non-generated scenes found)"
            )

        # --- LoFTR Filtering Analysis ---
        if self.loftr_script_path:  # Only proceed if LoFTR functionality was available
            self.load_loftr_ranks()  # Ensure latest ranks are loaded before this analysis
        else:
            # If LoFTR script was not available, analysis["loftr_filtering"] should indicate this.
            print("LoFTR Filtering analysis skipped: LoFTR script was not available.")
            analysis_summary["loftr_filtering"] = "Skipped (LoFTR script unavailable)"

        if (
            self.loftr_script_path
        ):  # Double-check, as load_loftr_ranks might also set it to unavailable
            loftr_filtered_results = defaultdict(lambda: defaultdict(dict))
            # Base folders for LoFTR analysis: RealBench FP16 Non-Gen that are complete
            # Use the list of folder names directly
            base_folders_for_loftr_analysis = [
                name
                for name in realbench_fp16_nongen_folders
                if self.check_folder_contents(
                    self.base_results_dir / name
                )  # Ensure folder is complete
            ]

            if not base_folders_for_loftr_analysis:
                analysis_summary["loftr_filtering"] = (
                    "Skipped (No complete base RealBench FP16 non-gen results for LoFTR filtering)"
                )
            elif not self.loftr_ranking_data:  # Check if any ranks were actually loaded
                analysis_summary["loftr_filtering"] = (
                    "Skipped (No LoFTR ranking data loaded for analysis)"
                )
            else:
                print(
                    f"Performing LoFTR filtering analysis on {len(base_folders_for_loftr_analysis)} RealBench FP16 non-gen scenes..."
                )
                notes_on_missing_data = (
                    set()
                )  # To collect notes about missing per-image scores or ranks

                for filter_rate in LOFTR_FILTER_RATES:
                    num_images_to_keep = max(
                        1, int(round(self.num_images_per_scene * (1.0 - filter_rate)))
                    )
                    rate_description_key = (
                        f"{int(filter_rate*100)}% Filtered (Top {num_images_to_keep} images)"
                    )

                    # Iterate through metrics that were specified for the run
                    for metric_name_for_loftr in self.metrics_to_run_list:
                        scores_for_current_metric_at_rate = []
                        num_scenes_contributing = 0
                        for folder_name_str in base_folders_for_loftr_analysis:
                            per_image_scores_for_folder = self.load_per_image_results(
                                folder_name_str, metric_name_for_loftr
                            )
                            ranked_image_filenames = self.loftr_ranking_data.get(folder_name_str)

                            if per_image_scores_for_folder is None:
                                notes_on_missing_data.add(
                                    f"{folder_name_str} ({metric_name_for_loftr} per-image scores missing)"
                                )
                                continue
                            if ranked_image_filenames is None:
                                notes_on_missing_data.add(
                                    f"{folder_name_str} (LoFTR rank data missing)"
                                )
                                continue

                            # Get scores for the top N ranked images, ensuring they exist and are valid
                            valid_scores_from_top_ranked = []
                            for img_filename in ranked_image_filenames[
                                :num_images_to_keep
                            ]:  # Slice to top N
                                if img_filename in per_image_scores_for_folder:
                                    score_val = per_image_scores_for_folder[img_filename]
                                    if (
                                        score_val is not None
                                    ):  # Ensure score is not None (e.g. from failed metric on one image)
                                        try:
                                            valid_scores_from_top_ranked.append(float(score_val))
                                        except (ValueError, TypeError):
                                            notes_on_missing_data.add(
                                                f"{folder_name_str} ({metric_name_for_loftr}, img {img_filename} has invalid score: '{score_val}')"
                                            )
                                    # else: score for this specific image was None, skip it
                                # else: ranked image filename not found in scores dict, skip it

                            if valid_scores_from_top_ranked:
                                scores_for_current_metric_at_rate.append(
                                    np.mean(valid_scores_from_top_ranked)
                                )
                                num_scenes_contributing += 1

                        if scores_for_current_metric_at_rate:  # If any scenes contributed
                            loftr_filtered_results[rate_description_key][metric_name_for_loftr] = (
                                np.mean(scores_for_current_metric_at_rate)
                            )
                            # Store count of scenes that contributed to this average
                            loftr_filtered_results[rate_description_key][
                                f"_count_{metric_name_for_loftr}"
                            ] = num_scenes_contributing

                if notes_on_missing_data:
                    print(
                        "Note: LoFTR filtering analysis encountered missing data points (displaying up to 5 unique notes):"
                    )
                    for idx, note in enumerate(sorted(list(notes_on_missing_data))):
                        if idx < 5:
                            print(f"  - {note}")
                        elif idx == 5:
                            print("  - ... and possibly others.")
                            break

                if loftr_filtered_results:
                    analysis_summary["loftr_filtering"] = dict(loftr_filtered_results)
                else:  # No results after filtering, possibly due to missing data
                    analysis_summary["loftr_filtering"] = (
                        "N/A (No results obtained after LoFTR filtering, possibly due to missing per-image scores or ranks)"
                    )

        print("--- Metric Result Analysis Finished ---")
        return analysis_summary

    def _calculate_average(self, folder_name_list: list[str]) -> dict:
        """
        Calculates average scores for a given list of folder names.
        Averages are calculated per metric, based on scores in self.master_results_data.
        """
        if not folder_name_list:
            # Return counts for all metrics that were intended to run, even if list is empty
            return {"_counts": {metric: 0 for metric in self.metrics_to_run_list}}

        # Stores lists of scores for each metric: {metric_name: [score1, score2, ...]}
        scores_by_metric = defaultdict(list)
        # Stores counts of valid scores for each metric: {metric_name: count}
        counts_by_metric = defaultdict(int)

        # Use self.metrics_to_run_list to define the scope of metrics for averaging.
        # These are the metrics the user intended to run (or all by default).
        metrics_in_scope_for_averaging = set(self.metrics_to_run_list)

        for folder_name in folder_name_list:
            if folder_name in self.master_results_data:
                for metric_name, score_value in self.master_results_data[folder_name].items():
                    # Consider only metrics in the current run's scope and with valid (non-None) scores
                    if metric_name in metrics_in_scope_for_averaging and score_value is not None:
                        try:
                            scores_by_metric[metric_name].append(float(score_value))
                            counts_by_metric[metric_name] += 1
                        except (ValueError, TypeError):
                            # This print would occur during analysis phase, after Rich Live.
                            # print(f"Warning: Could not convert score '{score_value}' to float for metric '{metric_name}' "
                            #       f"in folder '{folder_name}' during averaging. Skipping this score.")
                            pass  # Silently ignore scores that cannot be converted to float

        # Calculate mean only for metrics that had at least one valid score
        final_average_scores = {
            metric: np.mean(score_list)
            for metric, score_list in scores_by_metric.items()
            if score_list
        }

        # Ensure counts are reported for all metrics that were in scope for this run,
        # defaulting to 0 if no valid scores were found for a metric in this folder_list.
        final_counts_for_report = {
            metric: counts_by_metric.get(metric, 0) for metric in self.metrics_to_run_list
        }

        # Combine averages and counts into the output dictionary
        # Start with averages, then add the counts dictionary under "_counts" key
        analysis_output = {**final_average_scores}
        analysis_output["_counts"] = final_counts_for_report

        # Convert any numpy types in the result to native Python types for JSON serializability if needed later
        return convert_numpy_types(analysis_output)

    def format_results(self, analysis_results_dict: dict) -> str:
        """Formats the analysis results dictionary into a human-readable string report."""
        report_lines = []

        # Pandas display options for neatly formatted tables in the report
        pd.set_option("display.precision", 4)  # Display floats with 4 decimal places
        pd.set_option("display.max_rows", None)  # Display all rows
        pd.set_option("display.max_columns", None)  # Display all columns
        pd.set_option("display.width", 1000)  # Wider display for tables

        report_lines.append("=" * 80)
        report_lines.append(" Benchmark Results Summary ".center(80, "="))
        report_lines.append("=" * 80)
        report_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Base Results Directory: {self.base_results_dir}")
        report_lines.append(
            f"Folders Discovered for Processing: {len(self.discovered_result_folders)}"
        )

        # self.metrics_to_run_list might have been pruned if scripts were missing.
        # This list reflects metrics actively considered during the run.
        metrics_evaluated_str = ", ".join(sorted(list(set(self.metrics_to_run_list))))
        report_lines.append(
            f"Metrics Evaluated in This Run: {metrics_evaluated_str if metrics_evaluated_str else 'None'}"
        )
        report_lines.append("-" * 80)

        # Helper function to format a section of average scores
        def format_average_section(data_for_section: dict, section_title: str) -> list[str]:
            section_lines = [f"\n--- {section_title} ---"]
            if not isinstance(data_for_section, dict) or not data_for_section:
                section_lines.append("N/A (No data available for this section)")
                return section_lines

            counts_info = data_for_section.get("_counts", {})
            # Scores are all other keys in data_for_section
            scores_data = {k: v for k, v in data_for_section.items() if k != "_counts"}

            if not scores_data:  # No actual score data, only counts might exist
                section_lines.append("N/A (No valid results for any metrics in this section)")
                # Optionally, list counts if they exist and are informative:
                # if counts_info: section_lines.append(f"  Scene counts attempted: {counts_info}")
                return section_lines

            # Create DataFrame from scores for pretty printing
            df_scores = pd.DataFrame([scores_data]).T  # Transpose to have metrics as rows
            df_scores.columns = ["Average Score"]
            # Add counts for each metric, defaulting to 0 if a metric isn't in counts_info
            df_scores["Num Scenes (N)"] = df_scores.index.map(
                lambda metric_name: counts_info.get(metric_name, 0)
            )

            # Order rows by METRICS_CONFIG definition, then any others found
            metrics_order = [m_key for m_key in METRICS_CONFIG.keys() if m_key in df_scores.index]
            # Add any metrics present in df_scores but not in METRICS_CONFIG (should not happen if data is clean)
            remaining_metrics_order = [
                m_key for m_key in df_scores.index if m_key not in metrics_order
            ]
            final_metrics_order = metrics_order + remaining_metrics_order

            df_scores = df_scores.reindex(final_metrics_order).dropna(
                subset=["Average Score"], how="all"
            )

            if df_scores.empty:
                section_lines.append(
                    "N/A (No valid results for specified metrics after ordering/filtering)"
                )
            else:
                section_lines.append(df_scores.to_string())
            return section_lines

        report_lines.extend(
            format_average_section(
                analysis_results_dict.get("overall_realbench_fp16_nongen", {}),
                "Overall Average Scores (RealBench FP16 Non-Generated)",
            )
        )
        report_lines.extend(
            format_average_section(
                analysis_results_dict.get("overall_custom_fp16_nongen", {}),
                "Overall Average Scores (Custom FP16 Non-Generated)",
            )
        )

        # Helper function to format a section comparing two groups
        def format_comparison_section(
            comparison_data: dict, section_title: str, group_labels: list[str]
        ) -> list[str]:
            section_lines = [f"\n\n--- {section_title} ---"]
            if not isinstance(
                comparison_data, dict
            ):  # Handles "N/A (No common scenes)" string case
                section_lines.append(str(comparison_data))
                return section_lines

            common_scenes = comparison_data.get("common_scenes", [])
            avg_group1_data = comparison_data.get(f"{group_labels[0].lower()}_avg", {})
            avg_group2_data = comparison_data.get(f"{group_labels[1].lower()}_avg", {})

            section_lines.append(
                f"Comparison based on {len(common_scenes)} common scene(s): "
                f"{', '.join(common_scenes) if common_scenes else 'None'}"
            )

            counts_group1 = avg_group1_data.pop("_counts", {})  # Remove and store counts
            counts_group2 = avg_group2_data.pop("_counts", {})

            # Check if there's any actual score data beyond counts for either group
            if not any(k != "_counts" for k in avg_group1_data.keys()) and not any(
                k != "_counts" for k in avg_group2_data.keys()
            ):
                section_lines.append(
                    "N/A (No valid results for comparison in either group for any metrics)"
                )
                return section_lines

            df_comparison = pd.DataFrame(
                {
                    f"{group_labels[0]} Avg Score": avg_group1_data,
                    f"{group_labels[1]} Avg Score": avg_group2_data,
                }
            )
            df_comparison[f"N ({group_labels[0]})"] = df_comparison.index.map(
                lambda metric_name: counts_group1.get(metric_name, 0)
            )
            df_comparison[f"N ({group_labels[1]})"] = df_comparison.index.map(
                lambda metric_name: counts_group2.get(metric_name, 0)
            )

            metrics_order = [
                m_key for m_key in METRICS_CONFIG.keys() if m_key in df_comparison.index
            ]
            remaining_metrics_order = [
                m_key for m_key in df_comparison.index if m_key not in metrics_order
            ]
            final_metrics_order = metrics_order + remaining_metrics_order

            # Drop rows where both group averages are NaN (i.e., no data for that metric in either group)
            df_comparison = df_comparison.reindex(final_metrics_order).dropna(
                how="all", subset=[f"{group_labels[0]} Avg Score", f"{group_labels[1]} Avg Score"]
            )

            section_lines.append(
                df_comparison.to_string()
                if not df_comparison.empty
                else "N/A (No valid results for specified metrics in comparison)"
            )
            return section_lines

        report_lines.extend(
            format_comparison_section(
                analysis_results_dict.get("fp16_vs_fp32_realbench"),
                "Comparison: FP16 vs FP32 (Common RealBench Non-Generated Scenes)",
                ["FP16", "FP32"],
            )
        )
        report_lines.extend(
            format_comparison_section(
                analysis_results_dict.get("fp16_vs_fp32_custom"),
                "Comparison: FP16 vs FP32 (Common Custom Non-Generated Scenes)",
                ["FP16", "FP32"],
            )
        )
        report_lines.extend(
            format_comparison_section(
                analysis_results_dict.get("gen_vs_nongen_realbench_fp16"),
                "Comparison: Non-Generated vs Generated (Common RealBench FP16 Scenes)",
                ["NonGen", "Gen"],
            )
        )

        report_lines.append(
            "\n\n--- LoFTR Filtering Analysis (Average Scores over RealBench FP16 Non-Generated Scenes) ---"
        )
        loftr_analysis_data_dict = analysis_results_dict.get("loftr_filtering")
        if isinstance(loftr_analysis_data_dict, dict) and loftr_analysis_data_dict:
            # Prepare data for DataFrame: keys are filter rates (columns), values are dicts of metric scores (rows)
            data_for_loftr_df = defaultdict(dict)
            all_filter_rate_keys = []  # To store column names for ordering

            for rate_desc_key, metric_scores_map_at_rate in loftr_analysis_data_dict.items():
                # Filter out internal '_count' entries, keep only actual metric scores
                actual_metric_scores = {
                    metric_key: score_val
                    for metric_key, score_val in metric_scores_map_at_rate.items()
                    if not metric_key.startswith("_count_")
                }
                if actual_metric_scores:  # Only add if there are actual scores for this rate
                    data_for_loftr_df[rate_desc_key] = actual_metric_scores
                    all_filter_rate_keys.append(rate_desc_key)

            if not all_filter_rate_keys:  # No data after filtering out counts
                report_lines.append("N/A (No valid LoFTR filtered results after processing counts)")
            else:
                # Try to sort columns (filter rates) numerically by the percentage in their description
                try:
                    # Extracts the first number (percentage) from the rate key for sorting
                    sorted_filter_rate_keys = sorted(
                        all_filter_rate_keys,
                        key=lambda k_str: int(re.search(r"\d+", k_str).group()),
                    )
                except (AttributeError, ValueError):  # Fallback if regex or int conversion fails
                    sorted_filter_rate_keys = sorted(all_filter_rate_keys)

                # Create DataFrame with metrics as rows and sorted filter rates as columns
                df_loftr_raw = pd.DataFrame(data_for_loftr_df)

                # Reorder rows (metrics) according to METRICS_CONFIG, then columns by sorted_filter_rate_keys
                metrics_order_for_loftr = [
                    m_key for m_key in METRICS_CONFIG.keys() if m_key in df_loftr_raw.index
                ]
                remaining_loftr_metrics = [
                    m_key for m_key in df_loftr_raw.index if m_key not in metrics_order_for_loftr
                ]
                final_loftr_metrics_order = metrics_order_for_loftr + remaining_loftr_metrics

                # Apply both row and column order, drop fully NaN rows/columns
                df_loftr_styled = (
                    df_loftr_raw.reindex(
                        index=final_loftr_metrics_order, columns=sorted_filter_rate_keys
                    )
                    .dropna(how="all", axis=0)
                    .dropna(how="all", axis=1)
                )

                if df_loftr_styled.empty:
                    report_lines.append(
                        "N/A (No valid results for specified metrics/rates after LoFTR filtering and ordering)"
                    )
                else:
                    report_lines.append(
                        "Average metric scores when keeping only top N images based on LoFTR rank:"
                    )
                    report_lines.append(df_loftr_styled.to_string())
        else:  # Handles cases where loftr_analysis_data_dict is a string like "Skipped..." or "N/A..."
            report_lines.append(str(loftr_analysis_data_dict))

        report_lines.append("\n" + "=" * 80 + "\nMetric Interpretation Guide:")
        # Use self.metrics_to_run_list to reflect only metrics the user intended for this run
        # (or all configured if self.metrics_to_run_list was empty/None initially, though it defaults to all).
        metrics_for_interpretation_guide = (
            self.metrics_to_run_list if self.metrics_to_run_list else list(METRICS_CONFIG.keys())
        )

        higher_is_better_metrics = [
            m_name
            for m_name in metrics_for_interpretation_guide
            if m_name in METRICS_CONFIG
            and METRICS_CONFIG[m_name][
                1
            ]  # METRICS_CONFIG[m_name][1] is the "higher is better" boolean
        ]
        lower_is_better_metrics = [
            m_name
            for m_name in metrics_for_interpretation_guide
            if m_name in METRICS_CONFIG and not METRICS_CONFIG[m_name][1]
        ]

        if higher_is_better_metrics:
            report_lines.append(
                f"  Higher is Better for: {', '.join(sorted(higher_is_better_metrics))}"
            )
        if lower_is_better_metrics:
            report_lines.append(
                f"  Lower is Better for: {', '.join(sorted(lower_is_better_metrics))}"
            )

        if (
            not higher_is_better_metrics
            and not lower_is_better_metrics
            and metrics_for_interpretation_guide
        ):
            report_lines.append(
                "  (Interpretation for some metrics may be missing if not in METRICS_CONFIG or if none were run)"
            )
        elif not metrics_for_interpretation_guide:
            report_lines.append(
                "  (No metrics were specified or successfully run for interpretation guidance)"
            )
        report_lines.append("=" * 80)
        return "\n".join(report_lines)

    def run(self):
        """Main execution flow for the benchmarking process."""
        self.discover_folders()  # Uses tqdm and standard print
        if not self.discovered_result_folders:
            print(
                "\nNo suitable result folders found matching criteria. Exiting benchmark process."
            )
            return

        # Core metric calculation using Rich for UI
        self.run_all_metrics()

        # LoFTR ranking (if applicable) runs after main metrics, uses tqdm and print
        # Check if LoFTR script is available and if any RealBench folders were discovered
        # (run_loftr_ranking itself also checks for suitable folders)
        if self.loftr_script_path and any(
            folder.name.startswith("RealBench") for folder in self.discovered_result_folders
        ):
            self.run_loftr_ranking()
        elif not self.loftr_script_path:
            print(
                "\nSkipping LoFTR ranking: LoFTR script path was not configured or script not found."
            )
        else:  # LoFTR script exists, but no RealBench folders were found by discover_folders
            print(
                "\nSkipping LoFTR ranking: No 'RealBench' type folders were discovered to run LoFTR on."
            )

        # Analysis and reporting (uses standard print)
        analysis_data = self.analyze_results()
        report_string = self.format_results(analysis_data)

        print("\n" + report_string)  # Print the full report to console

        if self.output_file_path:  # Save report to file if path is provided
            print(f"\nSaving benchmark report to {self.output_file_path}...")
            try:
                self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_file_path, "w", encoding="utf-8") as f:
                    f.write(report_string)
                print("Benchmark report saved successfully.")
            except OSError as e:
                print(f"Error: Could not save benchmark report to {self.output_file_path}: {e}")


# --- Command Line Interface Setup and Main Execution ---
if __name__ == "__main__":
    # Essential for PyInstaller or cx_Freeze to work correctly with multiprocessing on Windows/macOS
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Run image quality and correspondence benchmarks on generated image sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help message
    )
    parser.add_argument(
        "--results_base_dir",
        required=True,
        type=str,
        help="Parent directory containing all scene result folders (e.g., 'RealBench-0-results', 'Custom-1-results').",
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        type=str,
        help="Directory for storing cached metric results (master cache and per-image caches).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="benchmark_report.txt",
        help="Path to save the final summary report text file.",
    )
    parser.add_argument(
        "--realbench_dataset_dir",
        type=str,
        default=None,  # Default to None if not provided
        help="Base directory of the RealBench dataset (containing GT/Masks).",
    )
    parser.add_argument(
        "--custom_dataset_dir",
        type=str,
        default=None,  # Default to None if not provided
        help="Base directory of any Custom dataset (containing GT/Masks).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Expected number of images (0.png to N-1.png) per scene result folder.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRICS_CONFIG.keys()),
        default=list(METRICS_CONFIG.keys()),
        help="List of specific metrics to run. If not provided, all configured metrics will be run.",
    )
    parser.add_argument(
        "--force_recalc",
        nargs="+",
        type=str,
        default=[],
        help="List of metrics to force recalculation for, or 'all' to force all, or 'loftr' for LoFTR. Case-insensitive.",
    )
    parser.add_argument(
        "--loftr_script_path",
        type=str,
        default="benchmark/loftr_ranking.py",
        help="Path to the 'loftr_ranking.py' script, relative to this benchmark script's location.",
    )

    parsed_args = parser.parse_args()

    # --- Initial Setup & Validation (before creating BenchmarkRunner instance) ---
    # These prints are standard and occur before Rich Live UI starts.
    try:
        cpu_count = os.cpu_count()
        print(f"System Info: Detected {cpu_count} CPU cores (for informational purposes).")
    except NotImplementedError:
        print("System Info: Could not detect CPU core count.")

    if not Path(parsed_args.results_base_dir).is_dir():
        parser.error(f"Results base directory not found: {parsed_args.results_base_dir}")
    if not parsed_args.realbench_dataset_dir and not parsed_args.custom_dataset_dir:
        parser.error(
            "At least one dataset directory must be specified via --realbench_dataset_dir or --custom_dataset_dir."
        )

    # Ensure cache directories exist, including subdirectories for per-image scores
    try:
        main_cache_path = Path(parsed_args.cache_dir).resolve()
        main_cache_path.mkdir(parents=True, exist_ok=True)

        per_image_root_cache = main_cache_path / PER_IMAGE_CACHE_BASE
        per_image_root_cache.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for per-image caches based on METRICS_CONFIG conventions
        metrics_for_cache_setup = (
            parsed_args.metrics if parsed_args.metrics else METRICS_CONFIG.keys()
        )
        for metric_name_for_cache in metrics_for_cache_setup:
            if metric_name_for_cache in METRICS_CONFIG:
                # Determine subdirectory name (e.g., "psnr_masked", "dreamsim")
                # This logic should match load_per_image_results
                is_metric_masked_type = metric_name_for_cache in [
                    "PSNR",
                    "SSIM",
                    "LPIPS",
                ]  # As per original logic
                sub_dir_cache_name = metric_name_for_cache.lower() + (
                    "_masked" if is_metric_masked_type else ""
                )
                (per_image_root_cache / sub_dir_cache_name).mkdir(parents=True, exist_ok=True)
        print(f"Cache directories ensured under {main_cache_path}")
    except OSError as e:
        parser.error(f"Failed to create necessary cache directories: {e}")

    # --- Run Benchmarks ---
    benchmark_start_time = time.time()
    application_exit_code = 0  # Default to success
    try:
        runner_instance = BenchmarkRunner(parsed_args)
        runner_instance.run()
    except ValueError as ve:  # Catch specific configuration errors from BenchmarkRunner init
        print(f"\n--- CONFIGURATION ERROR ---", flush=True)
        print(f"Error Details: {ve}", flush=True)
        print("Please check your command line arguments and dataset paths.")
        print("--- BENCHMARK SCRIPT TERMINATED ---", flush=True)
        application_exit_code = 2  # Use a different exit code for configuration errors
    except Exception as e:
        # Catch any other unhandled exceptions during the benchmark run
        print(f"\n--- FATAL ERROR DURING BENCHMARK EXECUTION ---", flush=True)
        print(f"Error Type: {type(e).__name__}", flush=True)
        print(f"Error Details: {e}", flush=True)
        print("\n--- Traceback ---", flush=True)
        traceback.print_exc()  # Print full traceback
        print("--- BENCHMARK SCRIPT TERMINATED DUE TO UNHANDLED ERROR ---", flush=True)
        application_exit_code = 1  # Standard error exit code
    finally:
        benchmark_end_time = time.time()
        total_duration_seconds = benchmark_end_time - benchmark_start_time
        print(f"\nTotal benchmark script execution time: {total_duration_seconds:.2f} seconds.")

    print("\nBenchmarking script has finished.")
    sys.exit(application_exit_code)
