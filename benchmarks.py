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

# --- Configuration & Constants ---
# Metrics: Name -> (Script Name, Higher is Better?)
# Note: DreamSim/LPIPS are distances (lower is better)
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

# LoFTR Filtering Rates (Percentage to filter *out*)
LOFTR_FILTER_RATES = [0.0, 0.25, 0.50, 0.75]  # 0%, 25%, 50%, 75%

DEFAULT_NUM_IMAGES = 16  # Standard number of images per result set
MASTER_CACHE_FILENAME = "master_results_cache.json"
PER_IMAGE_CACHE_BASE = "per_scene_cache"
LOFTR_RANKING_FILENAME = "loftr_ranking_scores.json"  # Output from loftr_ranking.py --rank-only

# --- Helper Functions ---


def find_gt_mask_paths(results_dir_path, dataset_dirs_map):
    """
    Find corresponding GT and Mask paths for a results directory,
    using a map of benchmark types to their dataset locations.

    Args:
        results_dir_path (Path): Path object for the results directory.
        dataset_dirs_map (dict): Dictionary mapping benchmark type (str, e.g., "RealBench")
                                 to its base dataset directory (Path).

    Returns:
        tuple: (str path to gt, str path to mask) or (None, None) if not found.
    """
    dir_name = results_dir_path.name
    # Examples: RealBench-15-results, Custom-35-results-fp32, RealBench-24-results-generated
    match = re.match(r"^(RealBench|Custom)-(\d+)(-results.*)?$", dir_name)
    if not match:
        # Try matching model folders if needed
        match = re.match(r"^(RealBench|Custom)-(\d+)(-model.*)?$", dir_name)
        if not match:
            # print(f"Warning: Could not parse benchmark/number from directory name: {dir_name}")
            return None, None

    benchmark_type = match.group(1)  # "RealBench" or "Custom"
    scene_number = match.group(2)  # "15", "35", "24"

    # Look up the base directory for this benchmark type
    dataset_base_dir = dataset_dirs_map.get(benchmark_type)
    if not dataset_base_dir:
        print(
            f"Warning: No dataset directory defined for benchmark type '{benchmark_type}' in map. Skipping {dir_name}"
        )
        return None, None
    if not dataset_base_dir.is_dir():
        print(
            f"Warning: Specified dataset directory for '{benchmark_type}' not found: {dataset_base_dir}. Skipping {dir_name}"
        )
        return None, None

    # Construct the path within the specific dataset directory
    # Assumes structure: <dataset_base_dir>/<benchmark_type>/<scene_number>/target/...
    # Note: We use benchmark_type *again* here because the folders inside
    # realfill_data_release_full are named RealBench/ and Custom/,
    # and inside jensen_images it might be Custom/ as well.
    base_path = dataset_base_dir / benchmark_type / scene_number / "target"

    gt_path = base_path / "gt.png"
    mask_path = base_path / "mask.png"

    # Check if files actually exist
    if not gt_path.is_file():
        # print(f"Warning: Ground truth file not found at {gt_path}") # Can be noisy
        return None, None
    if not mask_path.is_file():
        # print(f"Warning: Mask file not found at {mask_path}")
        return None, None

    return str(gt_path), str(mask_path)


def parse_final_score(stdout_str):
    """Extracts the score from the 'FINAL_SCORE:...' line."""
    for line in stdout_str.splitlines():
        if line.startswith("FINAL_SCORE:"):
            score_part = line.split(":", 1)[1]
            if score_part == "ERROR":
                return None
            try:
                return float(score_part)
            except ValueError:
                print(f"Warning: Could not parse score from line: {line}")
                return None
    # print(f"Warning: FINAL_SCORE line not found in output:\n{stdout_str}")
    return None


def load_json_cache(file_path):
    """Safely loads a JSON file."""
    if not file_path or not Path(file_path).is_file():
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Error loading JSON cache {file_path}: {e}")
        return None


def save_json_cache(data, file_path):
    """Saves data to a JSON file."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except OSError as e:
        print(f"Error saving JSON cache {file_path}: {e}")


def get_scene_key(folder_name):
    """Extracts the base scene identifier (e.g., RealBench-15) from a folder name."""
    match = re.match(r"^(RealBench|Custom)-(\d+)", folder_name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


# --- Main Orchestration Class ---


class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.base_results_dir = Path(args.results_base_dir)

        # Store the mapping of dataset types to their locations
        self.dataset_dirs_map = {}
        if args.realbench_dataset_dir:
            self.dataset_dirs_map["RealBench"] = Path(args.realbench_dataset_dir)
        if args.custom_dataset_dir:
            self.dataset_dirs_map["Custom"] = Path(args.custom_dataset_dir)
        if not self.dataset_dirs_map:
            raise ValueError(
                "At least one dataset directory (--realbench_dataset_dir or --custom_dataset_dir) must be provided."
            )

        self.cache_dir = Path(args.cache_dir)
        self.output_file = Path(args.output_file) if args.output_file else None
        self.num_images = args.num_images
        self.force_recalc_metrics = args.force_recalc or []  # Ensure it's a list
        self.metrics_to_run = args.metrics or list(METRICS_CONFIG.keys())
        self.loftr_script_path = Path(args.loftr_script_path)

        self.master_cache_path = self.cache_dir / MASTER_CACHE_FILENAME
        self.per_image_cache_dir = self.cache_dir / PER_IMAGE_CACHE_BASE

        # Results storage
        self.discovered_folders = []
        self.master_results = defaultdict(dict)  # {folder_name: {metric: score}}
        self.per_image_scores = defaultdict(lambda: defaultdict(dict))  # {folder: {metric: {img_name: score}}}
        self.loftr_ranks = defaultdict(list)  # {folder_name: [img_name_rank1, img_name_rank2,...]}

        print("Benchmark Runner Initialized.")
        print(f"Results Base Dir: {self.base_results_dir}")
        print(f"Dataset Directories Map: {self.dataset_dirs_map}")
        print(f"Cache Dir: {self.cache_dir}")
        print(f"Metrics to run: {self.metrics_to_run}")

    def discover_folders(self):
        """Scan results base dir for relevant folders."""
        print(f"\nScanning for result folders in {self.base_results_dir}...")
        self.discovered_folders = []
        if not self.base_results_dir.is_dir():
            print(f"Error: Results base directory not found: {self.base_results_dir}")
            return

        for item in self.base_results_dir.iterdir():
            if item.is_dir() and re.match(r"^(RealBench|Custom)-\d+(-results.*)?$", item.name):
                # Check if we have a dataset mapping for this type
                benchmark_type = re.match(r"^(RealBench|Custom)", item.name).group(1)
                if benchmark_type in self.dataset_dirs_map:
                    self.discovered_folders.append(item)
                else:
                    print(
                        f"Skipping folder {item.name} as no dataset directory was provided for type '{benchmark_type}'."
                    )

        self.discovered_folders.sort(key=lambda p: p.name)  # Sort for consistent order
        print(f"Found {len(self.discovered_folders)} relevant result folders with mapped datasets.")

    def load_master_cache(self):
        """Load existing average scores from master cache."""
        print(f"Loading master cache: {self.master_cache_path}")
        cached_data = load_json_cache(self.master_cache_path)
        if cached_data and isinstance(cached_data, dict):
            self.master_results = defaultdict(dict, cached_data)
            print(f"Loaded {len(self.master_results)} entries from master cache.")
        else:
            print("No valid master cache found or cache empty.")
            self.master_results = defaultdict(dict)

    def save_master_cache(self):
        """Save current average scores to master cache."""
        print(f"Saving master cache: {self.master_cache_path}")
        save_json_cache(self.master_results, self.master_cache_path)

    def run_metric_script(self, metric_name, script_path, gt_path, mask_path, results_dir):
        """Executes a single metric script using subprocess."""
        command = [
            "python",
            str(script_path),
            "--gt_path",
            str(gt_path),
            "--mask_path",
            str(mask_path),
            "--results_dir",
            str(results_dir),
            "--cache_dir",
            str(self.cache_dir),  # Pass the main cache dir
            "--num_images",
            str(self.num_images),
        ]
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
            if process.returncode != 0:
                print(f"Error running {metric_name} script for {results_dir.name}.")
                print("Stderr:", process.stderr)
                # print("Stdout:", process.stdout) # Often less useful than stderr
                return None

            score = parse_final_score(process.stdout)
            if score is None:
                print(f"Could not parse score for {metric_name} on {results_dir.name}.")
                print("Stdout:", process.stdout)
            return score

        except subprocess.TimeoutExpired:
            print(f"Timeout expired while running {metric_name} script for {results_dir.name}")
            return None
        except Exception as e:
            print(f"Exception running {metric_name} script for {results_dir.name}: {e}")
            return None

    def run_all_metrics(self):
        """Run metric scripts for all discovered folders."""
        if not self.discovered_folders:
            print("No folders discovered. Cannot run metrics.")
            return

        print("\n--- Running Metrics ---")
        self.load_master_cache()  # Load existing results first

        for folder_path in tqdm(self.discovered_folders, desc="Processing Folders"):
            folder_name = folder_path.name
            # Use the dataset map here
            gt_path, mask_path = find_gt_mask_paths(folder_path, self.dataset_dirs_map)

            if not gt_path or not mask_path:
                # find_gt_mask_paths already printed a warning if needed
                print(f"Skipping metrics for folder {folder_name}: Could not find GT/Mask files.")
                continue

            if folder_name not in self.master_results:
                self.master_results[folder_name] = {}

            for metric_name in self.metrics_to_run:
                script_name, _ = METRICS_CONFIG[metric_name]
                metric_script_path = Path(__file__).parent / "benchmark" / script_name

                if not metric_script_path.is_file():
                    print(f"Warning: Metric script not found: {metric_script_path}. Skipping {metric_name}.")
                    continue

                # Check cache and force_recalc
                force_this = "all" in self.force_recalc_metrics or metric_name.lower() in self.force_recalc_metrics

                if (
                    not force_this
                    and metric_name in self.master_results[folder_name]
                    and self.master_results[folder_name][metric_name] is not None
                ):
                    # print(f"Skipping {metric_name} for {folder_name} (cached in master).")
                    continue  # Skip if valid score already exists and not forcing

                if force_this:
                    print(f"Force recalculating {metric_name} for {folder_name}...")

                avg_score = self.run_metric_script(metric_name, metric_script_path, gt_path, mask_path, folder_path)

                self.master_results[folder_name][metric_name] = avg_score  # Store score or None if failed
                if avg_score is None:
                    print(f"Failed to get score for {metric_name} on {folder_name}")

        self.save_master_cache()  # Save at the end
        print("--- Metric Execution Finished ---")

    def load_per_image_results(self, folder_name, metric_name):
        """Load per-image scores from the specific metric's cache file."""
        if folder_name in self.per_image_scores and metric_name in self.per_image_scores[folder_name]:
            return self.per_image_scores[folder_name][metric_name]  # Already loaded

        # Determine cache subfolder (masked or not)
        is_masked = metric_name in ["PSNR", "SSIM", "LPIPS"]
        cache_subfolder = metric_name.lower() + "_masked" if is_masked else metric_name.lower()
        cache_file = self.per_image_cache_dir / cache_subfolder / f"{folder_name}.json"

        cache_data = load_json_cache(cache_file)
        if cache_data and "per_image" in cache_data:
            self.per_image_scores[folder_name][metric_name] = cache_data["per_image"]
            return cache_data["per_image"]
        else:
            return None

    def run_loftr_ranking(self):
        """Runs LoFTR ranking script for relevant folders if needed for filtering."""
        if not self.loftr_script_path.is_file():
            print("LoFTR script not found. Skipping LoFTR ranking.")
            return

        print("\n--- Running LoFTR Ranking ---")
        # Rank standard results folders for which we have datasets
        folders_to_rank = [f for f in self.discovered_folders if "generated" not in f.name and "fp32" not in f.name]
        print(f"Will attempt LoFTR ranking for {len(folders_to_rank)} folders.")

        for folder_path in tqdm(folders_to_rank, desc="LoFTR Ranking"):
            folder_name = folder_path.name
            loftr_output_json = folder_path / LOFTR_RANKING_FILENAME

            # Find GT path to determine the reference dir location
            gt_path, _ = find_gt_mask_paths(folder_path, self.dataset_dirs_map)
            if not gt_path:
                print(f"Skipping LoFTR for {folder_name}: Cannot determine base path (no GT found).")
                continue

            # Determine the reference directory used during training/copying
            # This assumes the 'ref' folder exists within the scene's structure in the dataset dir
            scene_base_path = Path(gt_path).parent.parent  # e.g., .../RealBench/15 or .../Custom/35
            ref_dir = scene_base_path / "ref"

            if not ref_dir.is_dir():
                print(
                    f"Warning: Reference directory '{ref_dir}' not found for LoFTR ranking of {folder_name}. LoFTR needs references. Skipping."
                )
                continue

            # Ensure source dir (results folder) exists
            if not folder_path.is_dir():
                print(f"ERROR: Source directory {folder_path} does not exist for LoFTR ranking. Skipping.")
                continue

            command = [
                "python",
                str(self.loftr_script_path),
                "--source-dir",
                str(folder_path),
                "--ref-dir",
                str(ref_dir),
                "--rank-only",
            ]
            try:
                # print(f"\nRunning LoFTR for {folder_name}...")
                process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
                if process.returncode != 0:
                    print(f"Error running LoFTR script for {folder_name}.")
                # else:
                #    print(f"LoFTR ranking finished for {folder_name}.") # Can be verbose

            except subprocess.TimeoutExpired:
                print(f"Timeout expired running LoFTR script for {folder_name}")
            except Exception as e:
                print(f"Exception running LoFTR script for {folder_name}: {e}")

        print("--- LoFTR Ranking Finished ---")

    def load_loftr_ranks(self):
        """Load the results of the LoFTR ranking runs."""
        print("Loading LoFTR ranking results...")
        self.loftr_ranks = defaultdict(list)  # Clear previous loads if any
        folders_ranked = [f for f in self.discovered_folders if "generated" not in f.name and "fp32" not in f.name]
        loaded_count = 0
        for folder_path in folders_ranked:
            folder_name = folder_path.name
            rank_file = folder_path / LOFTR_RANKING_FILENAME
            rank_data = load_json_cache(rank_file)
            if rank_data and "ranking" in rank_data and isinstance(rank_data["ranking"], list):
                self.loftr_ranks[folder_name] = [item["filename"] for item in rank_data["ranking"]]
                loaded_count += 1
        print(f"Loaded LoFTR ranks for {loaded_count} folders.")

    def analyze_results(self):
        """Perform comparisons and aggregations."""
        if not self.master_results:
            print("No metric results available for analysis.")
            return {}

        print("\n--- Analyzing Results ---")
        analysis = {}

        # Filter master results to only include folders for which datasets were found/processed
        valid_folders = list(self.master_results.keys())

        # --- 1. Overall Averages ---
        realbench_folders = [f for f in valid_folders if f.startswith("RealBench")]
        custom_folders = [f for f in valid_folders if f.startswith("Custom")]

        analysis["overall_realbench"] = self._calculate_average(realbench_folders)
        analysis["overall_custom"] = self._calculate_average(custom_folders)

        # --- 2. FP16 vs FP32 (RealBench only, for simplicity) ---
        fp16_map = {get_scene_key(f): f for f in realbench_folders if "fp32" not in f and "generated" not in f}
        fp32_map = {get_scene_key(f): f for f in realbench_folders if "fp32" in f}
        common_scenes_fp_keys = sorted([key for key in fp16_map if key in fp32_map])  # Sort keys

        fp16_common = [fp16_map[key] for key in common_scenes_fp_keys]
        fp32_common = [fp32_map[key] for key in common_scenes_fp_keys]

        analysis["fp16_vs_fp32"] = (
            {
                "common_scenes": common_scenes_fp_keys,
                "fp16_avg": self._calculate_average(fp16_common),
                "fp32_avg": self._calculate_average(fp32_common),
            }
            if common_scenes_fp_keys
            else "No common FP16/FP32 scenes found."
        )

        # --- 3. Generated vs Non-Generated (RealBench only) ---
        # Base = fp16 non-gen
        non_gen_map = {get_scene_key(f): f for f in realbench_folders if "generated" not in f and "fp32" not in f}
        gen_map = {get_scene_key(f): f for f in realbench_folders if "generated" in f}
        common_scenes_gen_keys = sorted([key for key in non_gen_map if key in gen_map])  # Sort keys

        non_gen_common = [non_gen_map[key] for key in common_scenes_gen_keys]
        gen_common = [gen_map[key] for key in common_scenes_gen_keys]

        analysis["gen_vs_nongen"] = (
            {
                "common_scenes": common_scenes_gen_keys,
                "nongen_avg": self._calculate_average(non_gen_common),
                "gen_avg": self._calculate_average(gen_common),
            }
            if common_scenes_gen_keys
            else "No common Generated/Non-Generated scenes found."
        )

        # --- 4. LoFTR Filtering Analysis (RealBench FP16 Non-Gen only) ---
        # Run LoFTR only if needed for analysis and script exists
        run_loftr = any("loftr" in str(a).lower() for a in analysis.keys()) or True  # Assume needed for now
        if run_loftr and self.loftr_script_path.is_file():
            self.run_loftr_ranking()  # Ensure ranks are generated
            self.load_loftr_ranks()  # Load ranks into self.loftr_ranks
        elif run_loftr:
            print("Skipping LoFTR analysis as script was not found.")

        loftr_analysis = defaultdict(lambda: defaultdict(dict))  # {rate: {metric: avg_score}}
        # Use the common non-gen scenes if available, otherwise all non-gen fp16
        base_folders_for_loftr = non_gen_common if common_scenes_gen_keys else list(non_gen_map.values())

        if not base_folders_for_loftr or not self.loftr_ranks:
            print("No base folders or LoFTR ranks available for LoFTR filtering analysis.")
            analysis["loftr_filtering"] = "Skipped (no base folders or ranks)"
        else:
            print(f"Performing LoFTR filtering analysis on {len(base_folders_for_loftr)} base scenes.")
            for rate in LOFTR_FILTER_RATES:
                num_to_keep = max(1, int(self.num_images * (1.0 - rate)))
                rate_key = f"{int(rate*100)}%"

                for metric_name in self.metrics_to_run:
                    metric_scores_for_rate = []
                    processed_folders_count = 0
                    for folder_name in base_folders_for_loftr:
                        img_scores = self.load_per_image_results(folder_name, metric_name)
                        ranked_filenames = self.loftr_ranks.get(folder_name)

                        if img_scores is None or ranked_filenames is None:
                            continue

                        # Get scores for top N valid images
                        valid_scores = [
                            img_scores[f] for f in ranked_filenames if f in img_scores and img_scores[f] is not None
                        ]
                        top_scores = valid_scores[:num_to_keep]

                        if not top_scores:
                            continue

                        avg_score_filtered = np.mean(top_scores)
                        metric_scores_for_rate.append(avg_score_filtered)
                        processed_folders_count += 1

                    if metric_scores_for_rate:
                        overall_avg_for_rate_metric = np.mean(metric_scores_for_rate)
                        loftr_analysis[rate_key][metric_name] = overall_avg_for_rate_metric

            analysis["loftr_filtering"] = dict(loftr_analysis)

        print("--- Analysis Finished ---")
        return analysis

    def _calculate_average(self, folder_list):
        """Helper to calculate average scores across a list of folders."""
        if not folder_list:
            return {}
        avg_scores = defaultdict(list)
        valid_folders_per_metric = defaultdict(int)

        for folder_name in folder_list:
            if folder_name in self.master_results:
                for metric, score in self.master_results[folder_name].items():
                    if score is not None and metric in self.metrics_to_run:
                        avg_scores[metric].append(score)
                        valid_folders_per_metric[metric] += 1

        final_avg = {}
        for metric, scores in avg_scores.items():
            if scores:
                final_avg[metric] = np.mean(scores)
        # Add counts for context
        final_avg["_counts"] = {m: valid_folders_per_metric.get(m, 0) for m in self.metrics_to_run if m in final_avg}
        return final_avg

    def format_results(self, analysis_results):
        """Formats the analysis results into strings/tables."""
        output_lines = []
        pd.set_option("display.precision", 4)  # Set pandas precision

        output_lines.append("=" * 80)
        output_lines.append("                          Benchmark Results Summary")
        output_lines.append("=" * 80)
        output_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append(f"Results Base: {self.base_results_dir}")
        output_lines.append(f"Total Folders Found: {len(self.discovered_folders)}")
        output_lines.append(f"Metrics Run: {', '.join(self.metrics_to_run)}")
        output_lines.append("-" * 80)

        def format_avg_table(data, title):
            if not data or "_counts" not in data:
                return [f"\n--- {title} ---", "No data available."]
            counts = data.pop("_counts", {})
            if not data:
                return [f"\n--- {title} ---", "No data available."]
            df = pd.DataFrame([data]).T
            df.columns = ["Average"]
            df["Count"] = df.index.map(lambda x: counts.get(x, 0))
            return [f"\n--- {title} ---", df.to_string()]

        output_lines.extend(
            format_avg_table(analysis_results.get("overall_realbench", {}), "Overall Average (RealBench)")
        )
        output_lines.extend(format_avg_table(analysis_results.get("overall_custom", {}), "Overall Average (Custom)"))

        fp_comp = analysis_results.get("fp16_vs_fp32")
        output_lines.append("\n\n--- FP16 vs FP32 Comparison (Common RealBench Scenes) ---")
        if isinstance(fp_comp, dict):
            output_lines.append(
                f"Common Scenes ({len(fp_comp['common_scenes'])}): {', '.join(fp_comp['common_scenes'])}"
            )
            fp16_counts = fp_comp["fp16_avg"].pop("_counts", {})
            fp32_counts = fp_comp["fp32_avg"].pop("_counts", {})
            df_fp = pd.DataFrame({"FP16": fp_comp["fp16_avg"], "FP32": fp_comp["fp32_avg"]})
            df_fp["Count_FP16"] = df_fp.index.map(lambda x: fp16_counts.get(x, 0))
            df_fp["Count_FP32"] = df_fp.index.map(lambda x: fp32_counts.get(x, 0))
            output_lines.append(df_fp.to_string())
        else:
            output_lines.append(f"{fp_comp}")

        gen_comp = analysis_results.get("gen_vs_nongen")
        output_lines.append("\n\n--- Generated vs Non-Generated Comparison (Common RealBench Scenes) ---")
        if isinstance(gen_comp, dict):
            output_lines.append(
                f"Common Scenes ({len(gen_comp['common_scenes'])}): {', '.join(gen_comp['common_scenes'])}"
            )
            nongen_counts = gen_comp["nongen_avg"].pop("_counts", {})
            gen_counts = gen_comp["gen_avg"].pop("_counts", {})
            df_gen = pd.DataFrame({"Non-Generated": gen_comp["nongen_avg"], "Generated (LoFTR)": gen_comp["gen_avg"]})
            df_gen["Count_NonGen"] = df_gen.index.map(lambda x: nongen_counts.get(x, 0))
            df_gen["Count_Gen"] = df_gen.index.map(lambda x: gen_counts.get(x, 0))
            output_lines.append(df_gen.to_string())
        else:
            output_lines.append(f"{gen_comp}")

        loftr_res = analysis_results.get("loftr_filtering")
        output_lines.append("\n\n--- LoFTR Filtering Analysis (RealBench Non-Gen FP16 Common Scenes) ---")
        if isinstance(loftr_res, dict) and loftr_res:
            # Ensure columns are sorted by filter rate percentage
            sorted_rates = sorted(loftr_res.keys(), key=lambda x: int(x.replace("%", "")))
            df_loftr_raw = pd.DataFrame(loftr_res)[sorted_rates]  # Rates as columns
            df_loftr = df_loftr_raw.reindex(list(METRICS_CONFIG.keys())).dropna(how="all")  # Metrics as rows, sorted
            output_lines.append(df_loftr.to_string())
        else:
            output_lines.append(f"{loftr_res}")

        output_lines.append("\n" + "=" * 80)
        output_lines.append("\nMetric Direction:")
        for name, (_, higher_is_better) in METRICS_CONFIG.items():
            if name in self.metrics_to_run:  # Only show direction for metrics that were run
                output_lines.append(f"  {name}: {'Higher' if higher_is_better else 'Lower'} is better")
        output_lines.append("=" * 80)

        return "\n".join(output_lines)

    def run(self):
        """Main execution flow."""
        self.discover_folders()
        self.run_all_metrics()
        analysis = self.analyze_results()
        report = self.format_results(analysis)

        print("\n" + report)

        if self.output_file:
            print(f"\nSaving report to {self.output_file}...")
            try:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_file, "w") as f:
                    f.write(report)
                print("Report saved.")
            except OSError as e:
                print(f"Error saving report: {e}")


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and analyze RealFill benchmarks.")
    # Required Paths
    parser.add_argument(
        "--results_base_dir",
        required=True,
        help="Path to the parent directory containing all result folders (e.g., /content/drive/MyDrive/RealFill).",
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Path to the directory for storing benchmark cache files (e.g., /content/drive/MyDrive/RealFill/benchmark_cache).",
    )
    # Dataset Paths (At least one required)
    parser.add_argument(
        "--realbench_dataset_dir",
        help="Path to the base directory for RealBench dataset (e.g., /content/realfill_data_release_full).",
    )
    parser.add_argument(
        "--custom_dataset_dir", help="Path to the base directory for Custom dataset (e.g., /content/jensen_images)."
    )

    # Optional Paths & Config
    parser.add_argument("--output_file", help="Optional path to save the final formatted report.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images per scene to process (default: {DEFAULT_NUM_IMAGES}).",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=list(METRICS_CONFIG.keys()), help="Specify which metrics to run (default: all)."
    )
    parser.add_argument(
        "--force_recalc", nargs="+", help="Force recalculation for specified metrics (or 'all') ignoring cache."
    )
    parser.add_argument("--loftr_script_path", default="loftr_ranking.py", help="Path to the loftr_ranking.py script.")

    args = parser.parse_args()

    # --- Validations ---
    if not Path(args.results_base_dir).is_dir():
        parser.error(f"Results base directory not found: {args.results_base_dir}")
    if not args.realbench_dataset_dir and not args.custom_dataset_dir:
        parser.error(
            "You must provide at least one dataset directory using --realbench_dataset_dir or --custom_dataset_dir."
        )
    if args.realbench_dataset_dir and not Path(args.realbench_dataset_dir).is_dir():
        print(f"WARNING: RealBench dataset directory not found: {args.realbench_dataset_dir}")
        # Allow running without it if only Custom is needed
    if args.custom_dataset_dir and not Path(args.custom_dataset_dir).is_dir():
        print(f"WARNING: Custom dataset directory not found: {args.custom_dataset_dir}")
        # Allow running without it if only RealBench is needed
    if not Path(args.loftr_script_path).is_file():
        print(f"WARNING: LoFTR script not found at {args.loftr_script_path}. LoFTR filtering analysis will be skipped.")

    # Ensure cache directory exists
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.cache_dir) / PER_IMAGE_CACHE_BASE).mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(args)
    runner.run()

    print("\nBenchmarking script finished.")
