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

    benchmark_type = match.group(1)
    scene_number = match.group(2)
    dataset_base_dir = dataset_dirs_map.get(benchmark_type)
    if not dataset_base_dir or not dataset_base_dir.is_dir():
        return None, None

    base_path = dataset_base_dir / benchmark_type / scene_number / "target"
    gt_path = base_path / "gt.png"
    mask_path = base_path / "mask.png"

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
                print(f"Warning: Could not parse score from line: {line}")
                return None
    return None


def load_json_cache(file_path):
    if not file_path or not Path(file_path).is_file():
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, TypeError) as e:
        print(f"Warning: Error loading JSON cache {file_path}: {e}")
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


# --- Main Orchestration Class ---


class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.base_results_dir = Path(args.results_base_dir)
        self.dataset_dirs_map = {}
        if args.realbench_dataset_dir:
            rb_path = Path(args.realbench_dataset_dir)
            if rb_path.is_dir():
                self.dataset_dirs_map["RealBench"] = rb_path
            else:
                print(f"Warning: Provided RealBench dataset directory not found: {rb_path}")
        if args.custom_dataset_dir:
            custom_path = Path(args.custom_dataset_dir)
            if custom_path.is_dir():
                self.dataset_dirs_map["Custom"] = custom_path
            else:
                print(f"Warning: Provided Custom dataset directory not found: {custom_path}")
        if not self.dataset_dirs_map:
            raise ValueError(
                "No valid dataset directories found. Provide paths via --realbench_dataset_dir or --custom_dataset_dir."
            )

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
        print(f"Results Base Dir: {self.base_results_dir}")
        print(f"Dataset Directories Map: {self.dataset_dirs_map}")
        print(f"Cache Dir: {self.cache_dir}")
        print(f"Metrics to run: {self.metrics_to_run}")
        print(f"Force Recalc: {self.force_recalc_metrics}")

    def discover_folders(self):
        print(f"\nScanning for result folders in {self.base_results_dir}...")
        self.discovered_folders = []
        if not self.base_results_dir.is_dir():
            print(f"Error: Results base directory not found: {self.base_results_dir}")
            return
        potential, skipped_map, skipped_gt = 0, 0, 0
        for item in self.base_results_dir.iterdir():
            if item.is_dir() and re.match(r"^(RealBench|Custom)-\d+(-results.*)?$", item.name):
                potential += 1
                match = re.match(r"^(RealBench|Custom)", item.name)
                if match:
                    b_type = match.group(1)
                    if b_type in self.dataset_dirs_map:
                        gt_path, mask_path = find_gt_mask_paths(item, self.dataset_dirs_map)
                        if gt_path and mask_path:
                            self.discovered_folders.append(item)
                        else:
                            skipped_gt += 1
                    else:
                        skipped_map += 1
        self.discovered_folders.sort(key=lambda p: p.name)
        print(f"Scan complete. Found {potential} potential *results* folders.")
        print(f" - Added {len(self.discovered_folders)} folders with mapped datasets and GT/Mask files.")
        if skipped_map > 0:
            print(f" - Skipped {skipped_map} folders due to unmapped dataset type.")
        if skipped_gt > 0:
            print(f" - Skipped {skipped_gt} folders due to missing GT or Mask files.")

    def load_master_cache(self):
        print(f"Loading master cache: {self.master_cache_path}")
        cached_data = load_json_cache(self.master_cache_path)
        if cached_data and isinstance(cached_data, dict):
            self.master_results = defaultdict(dict)
            for k, v in cached_data.items():
                if isinstance(v, dict):
                    self.master_results[k] = v
            print(f"Loaded {len(self.master_results)} entries from master cache.")
        else:
            print("No valid master cache found or cache empty.")
            self.master_results = defaultdict(dict)

    def save_master_cache(self):
        print(f"Saving master cache: {self.master_cache_path}")
        save_json_cache(dict(self.master_results), self.master_cache_path)

    def check_folder_contents(self, folder_path: Path) -> bool:
        if not folder_path.is_dir():
            return False
        try:
            for i in range(self.num_images):
                if not (folder_path / f"{i}.png").is_file():
                    return False
            return True
        except Exception as e:
            tqdm.write(f"Error checking contents of {folder_path}: {e}")
            return False

    def run_metric_script(self, metric_name, script_path, gt_path, mask_path, results_dir):
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
            str(self.cache_dir),
            "--num_images",
            str(self.num_images),
        ]
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
            if process.returncode != 0:
                tqdm.write(f"Error running {metric_name} script for {results_dir.name} (RC: {process.returncode}).")
                stderr = process.stderr.strip()[-500:]
                if stderr:
                    tqdm.write(f"Stderr: ...{stderr}")
                return None
            return parse_final_score(process.stdout)
        except subprocess.TimeoutExpired:
            tqdm.write(f"Timeout running {metric_name} for {results_dir.name}")
            return None
        except Exception as e:
            tqdm.write(f"Exception running {metric_name} for {results_dir.name}: {e}")
            return None

    def run_all_metrics(self):
        if not self.discovered_folders:
            print("No result folders discovered.")
            return
        print(f"\n--- Running Metrics for {len(self.discovered_folders)} Folders ---")
        self.load_master_cache()
        processed, skipped_incomplete = 0, 0
        for folder_path in tqdm(self.discovered_folders, desc="Processing Folders", unit="folder"):
            folder_name = folder_path.name
            gt_path, mask_path = find_gt_mask_paths(folder_path, self.dataset_dirs_map)
            if not gt_path or not mask_path:
                tqdm.write(f"Skipping {folder_name}: GT/Mask missing (unexpected).")
                continue

            if not self.check_folder_contents(folder_path):
                actual_count = count_result_images(folder_path)
                tqdm.write(f"Skipping {folder_name}: Found {actual_count}/{self.num_images} expected result images.")
                skipped_incomplete += 1
                if folder_name in self.master_results:
                    del self.master_results[folder_name]
                continue

            processed += 1
            if folder_name not in self.master_results:
                self.master_results[folder_name] = {}
            for metric_name in self.metrics_to_run:
                if metric_name not in METRICS_CONFIG:
                    tqdm.write(f"Warn: Metric '{metric_name}' invalid.")
                    continue
                script_name, _ = METRICS_CONFIG[metric_name]
                script_path = Path(__file__).parent / "benchmark" / script_name
                if not script_path.is_file():
                    tqdm.write(f"Warn: Script {script_path} missing.")
                    continue
                force = "all" in self.force_recalc_metrics or metric_name.lower() in self.force_recalc_metrics
                if (
                    not force
                    and metric_name in self.master_results.get(folder_name, {})
                    and self.master_results[folder_name][metric_name] is not None
                ):
                    continue  # Skip valid cached entry if not forcing
                avg_score = self.run_metric_script(metric_name, script_path, gt_path, mask_path, folder_path)
                self.master_results[folder_name][metric_name] = avg_score  # Store result (even None)

        self.save_master_cache()
        print("\n--- Metric Execution Finished ---")
        print(f"Folders processed for metrics: {processed}")
        if skipped_incomplete > 0:
            print(f"Folders skipped due to incomplete results: {skipped_incomplete}")

    def load_per_image_results(self, folder_name, metric_name):
        if folder_name in self.per_image_scores and metric_name in self.per_image_scores[folder_name]:
            return self.per_image_scores[folder_name][metric_name]
        is_masked = metric_name in ["PSNR", "SSIM", "LPIPS"]
        subfolder = metric_name.lower() + "_masked" if is_masked else metric_name.lower()
        cache_file = self.per_image_cache_dir / subfolder / f"{folder_name}.json"
        cache_data = load_json_cache(cache_file)
        if (
            cache_data
            and isinstance(cache_data, dict)
            and "per_image" in cache_data
            and isinstance(cache_data["per_image"], dict)
        ):
            self.per_image_scores[folder_name][metric_name] = cache_data["per_image"]
            return cache_data["per_image"]
        return None

    def run_loftr_ranking(self):
        if not self.loftr_script_path or not self.loftr_script_path.is_file():
            print("LoFTR script not found/set. Skipping LoFTR ranking.")
            return
        print("\n--- Running LoFTR Ranking ---")
        folders = [
            f
            for f in self.discovered_folders
            if f.name.startswith("RealBench") and "generated" not in f.name and "fp32" not in f.name
        ]
        print(f"Found {len(folders)} potential folders for LoFTR ranking.")
        ranked, skipped_exist, skipped_ref, errors = 0, 0, 0, 0
        for folder_path in tqdm(folders, desc="LoFTR Ranking", unit="folder"):
            out_json = folder_path / LOFTR_RANKING_FILENAME
            force = "all" in self.force_recalc_metrics or "loftr" in self.force_recalc_metrics
            if out_json.is_file() and not force:
                skipped_exist += 1
                continue
            gt_path, _ = find_gt_mask_paths(folder_path, self.dataset_dirs_map)
            if not gt_path:
                continue
            ref_dir = Path(gt_path).parent.parent / "ref"
            if not ref_dir.is_dir():
                tqdm.write(f"Warn: Ref dir '{ref_dir}' missing for {folder_path.name}.")
                skipped_ref += 1
                continue
            if force:
                tqdm.write(f"Force running LoFTR ranking for {folder_path.name}...")
            cmd = [
                "python",
                str(self.loftr_script_path),
                "--source-dir",
                str(folder_path),
                "--ref-dir",
                str(ref_dir),
                "--rank-only",
                "--ranking-output-file",
                LOFTR_RANKING_FILENAME,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
                if proc.returncode != 0:
                    tqdm.write(f"Error LoFTR {folder_path.name} (RC:{proc.returncode}).")
                    stderr = proc.stderr.strip()[-500:]
                    if stderr:
                        tqdm.write(f"Stderr: ...{stderr}")
                    errors += 1
                else:
                    ranked += 1
            except subprocess.TimeoutExpired:
                tqdm.write(f"Timeout LoFTR {folder_path.name}")
                errors += 1
            except Exception as e:
                tqdm.write(f"Exception LoFTR {folder_path.name}: {e}")
                errors += 1
        print("--- LoFTR Ranking Finished ---")
        if ranked > 0:
            print(f"Ran/updated LoFTR ranks for {ranked} folders.")
        if skipped_exist > 0:
            print(f"Skipped {skipped_exist} folders (ranks existed).")
        if skipped_ref > 0:
            print(f"Skipped {skipped_ref} folders (missing ref dir).")
        if errors > 0:
            print(f"Errors during LoFTR ranking for {errors} folders.")

    def load_loftr_ranks(self):
        if not self.loftr_script_path or not self.loftr_script_path.is_file():
            print("Skipping LoFTR rank loading: script path invalid.")
            return
        print("Loading LoFTR ranking results...")
        self.loftr_ranks = defaultdict(list)
        folders_attempted = [
            f
            for f in self.discovered_folders
            if f.name.startswith("RealBench") and "gen" not in f.name and "fp32" not in f.name
        ]
        loaded, missing = 0, 0
        for folder_path in folders_attempted:
            rank_file = folder_path / LOFTR_RANKING_FILENAME
            rank_data = load_json_cache(rank_file)
            if (
                rank_data
                and isinstance(rank_data, dict)
                and "ranking" in rank_data
                and isinstance(rank_data["ranking"], list)
            ):
                fnames = [
                    item["filename"] for item in rank_data["ranking"] if isinstance(item, dict) and "filename" in item
                ]
                if fnames:
                    self.loftr_ranks[folder_path.name] = fnames
                    loaded += 1
                else:
                    missing += 1
            else:
                missing += 1
        print(f"Loaded LoFTR ranks for {loaded} folders.")
        if missing > 0:
            print(f"LoFTR rank files missing/invalid for {missing} folders.")

    def analyze_results(self):
        if not self.master_results:
            print("No metric results loaded.")
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
            else "No common RealBench FP16/FP32 non-gen."
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
            else "No common Custom FP16/FP32 non-gen."
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
            else "No common RealBench FP16 Gen/Non-Gen."
        )

        run_loftr = self.loftr_script_path and self.loftr_script_path.is_file()
        if run_loftr:
            self.load_loftr_ranks()
        else:
            print("Skipping LoFTR filtering analysis: script not found.")
        loftr_analysis = defaultdict(lambda: defaultdict(dict))
        base_loftr = rb_f16_ng
        if not run_loftr:
            analysis["loftr_filtering"] = "Skipped (script not found)"
        elif not base_loftr:
            analysis["loftr_filtering"] = "Skipped (no baseline folders)"
        elif not self.loftr_ranks:
            analysis["loftr_filtering"] = "Skipped (no ranks loaded)"
        else:
            print(f"Performing LoFTR filtering analysis on {len(base_loftr)} scenes.")
            missing_scenes_count = 0
            for rate in LOFTR_FILTER_RATES:
                keep = max(1, int(round(self.num_images * (1.0 - rate))))
                key = f"{int(rate*100)}% Filtered"
                missing_flag = False
                for metric in self.metrics_to_run:
                    scores = []
                    count = 0
                    for fname in base_loftr:
                        img_scores = self.load_per_image_results(fname, metric)
                        ranks = self.loftr_ranks.get(fname)
                        if img_scores is None or ranks is None:
                            if ranks is None:
                                missing_flag = True
                                continue
                        valid_scores = [img_scores[f] for f in ranks if f in img_scores and img_scores[f] is not None]
                        top = valid_scores[:keep]
                        if top:
                            scores.append(np.mean(top))
                            count += 1
                    if scores:
                        overall = np.mean(scores)
                        loftr_analysis[key][metric] = overall
                        loftr_analysis[key][f"_count_{metric}"] = count
                if missing_flag:
                    missing_scenes_count += 1  # Count unique scenes missing ranks
            unique_missing = sum(1 for fn in base_loftr if self.loftr_ranks.get(fn) is None)
            if unique_missing > 0:
                print(f"Note: LoFTR ranks/scores missing for {unique_missing} scenes.")
            analysis["loftr_filtering"] = dict(loftr_analysis)
        print("--- Analysis Finished ---")
        return analysis

    def _calculate_average(self, folder_list):
        if not folder_list:
            return {"_counts": {m: 0 for m in self.metrics_to_run}}
        avg_s, counts = defaultdict(list), defaultdict(int)
        for fname in folder_list:
            if fname in self.master_results:
                for metric, score in self.master_results[fname].items():
                    if score is not None and metric in self.metrics_to_run:
                        try:
                            avg_s[metric].append(float(score))
                            counts[metric] += 1
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
        lines.append(f"Results Base: {self.base_results_dir}")
        lines.append(f"Folders Found: {len(self.discovered_folders)}")
        lines.append(f"Metrics Run: {', '.join(self.metrics_to_run)}")
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
            data = {r: {m: d.get(m) for m, d in loftr[r].items() if not m.startswith("_count_")} for r in rates}
            df_raw = pd.DataFrame(data)
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
            print("No suitable folders found.")
            return
        self.run_all_metrics()
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
    parser = argparse.ArgumentParser(
        description="Run RealFill benchmarks.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        parser.error("Need --realbench_dataset_dir or --custom_dataset_dir.")
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
