import subprocess
import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob
import argparse  # For command-line arguments
from pathlib import Path  # For modern path manipulation
import multiprocessing  # For freeze_support

# --- Default Configuration (can be overridden by CLI args) ---
# These are relative path names or default values used if not specified via CLI
DEFAULT_RESULTS_DIR_NAME = "realfill_results"
DEFAULT_DATA_DIR_NAME = os.path.join(DEFAULT_RESULTS_DIR_NAME, "realfill_data_release_full")
# Cache dir is often inside results, e.g., realfill_results/benchmark_cache
DEFAULT_CACHE_DIR_NAME_FRAGMENT = "benchmark_cache"
DEFAULT_BENCHMARK_SCRIPT_DIR_NAME = "benchmark"
DEFAULT_PLOT_OUTPUT_DIR_NAME = "."

# --- Fixed Configurations ---
METRIC_SCRIPTS_AVAILABLE = [  # List of all known metric scripts
    "clip_metric.py",
    "dino_metric.py",
    "dreamsim_metric.py",
    "lpips_metric.py",
    "psnr_metric.py",
    "ssim_metric.py",
]

METRIC_INFO = {
    "clip": {"display_name": "CLIP Score", "higher_is_better": True, "unit": ""},
    "dino": {"display_name": "DINO Score", "higher_is_better": True, "unit": ""},
    "dreamsim": {"display_name": "DreamSim Score", "higher_is_better": True, "unit": ""},
    "lpips": {"display_name": "LPIPS", "higher_is_better": False, "unit": ""},
    "psnr": {"display_name": "PSNR", "higher_is_better": True, "unit": "dB"},
    "ssim": {"display_name": "SSIM", "higher_is_better": True, "unit": ""},
}

SCORE_REGEX = re.compile(r"FINAL_SCORE:\s*([\d.]+)")


def parse_score_from_output(output_text):
    """Parses the FINAL_SCORE from the script's stdout."""
    match = SCORE_REGEX.search(output_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            # Using tqdm.write ensures it plays nice with active progress bars
            tqdm.write(f"Warning: Could not parse score from '{match.group(1)}'")
            return None
    return None


def run_metric_for_variant(metric_script_name, variant_name, path_config):
    """
    Runs a single metric script for a single result variant.
    'path_config' is an object containing resolved absolute paths.
    """
    metric_script_path = path_config.benchmark_script_dir_abs / metric_script_name
    results_dir_path_for_variant = path_config.results_base_dir_abs / variant_name

    if not results_dir_path_for_variant.exists():
        error_msg = f"Error: Results directory not found: {results_dir_path_for_variant} for variant {variant_name}"
        return None, error_msg

    command = [
        sys.executable,  # Use the current Python interpreter (handles venv)
        str(metric_script_path),
        "--gt_path",
        str(path_config.gt_full_path_abs),
        "--mask_path",
        str(path_config.mask_full_path_abs),
        "--results_dir",
        str(results_dir_path_for_variant),
        "--cache_dir",
        str(path_config.cache_dir_abs),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(path_config.realfill_root_abs),  # Run from the project root
            encoding="utf-8",
            errors="replace",
        )

        output_text = process.stdout + process.stderr
        score = parse_score_from_output(output_text)

        if process.returncode != 0:
            error_message = f"Error running {metric_script_name} for {variant_name}.\n"
            error_message += f"  Command: {' '.join(command)}\n"
            error_message += f"  Return Code: {process.returncode}\n"
            error_log = process.stderr.strip() if process.stderr.strip() else process.stdout.strip()
            error_message += f"  Output: {error_log[:1000]}"  # Limit output length
            return score, error_message

        if score is None:
            no_score_message = (
                f"No FINAL_SCORE found in output for {metric_script_name} on {variant_name}.\n"
            )
            no_score_message += f"  Stdout: {process.stdout.strip()[:500]}\n  Stderr: {process.stderr.strip()[:500]}"
            return None, no_score_message

        return score, None
    except FileNotFoundError:
        return (
            None,
            f"Error: Python interpreter or script {metric_script_path} not found. Check paths and venv.",
        )
    except Exception as e:
        return (
            None,
            f"Exception during subprocess for {metric_script_name} on {variant_name}: {e}",
        )


def process_metric_type(metric_script_name, pbar_position, path_config, result_variants_list):
    """
    Processes all variants for a single metric type.
    'path_config' contains paths.
    'result_variants_list' is the list of variant folder names.
    """
    metric_results = {}
    errors = []
    metric_key_name = metric_script_name.split(".")[0]

    for variant_name in tqdm(
        result_variants_list,
        desc=f"{metric_key_name:<18}",
        leave=False,
        position=pbar_position,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ):
        score, error_msg = run_metric_for_variant(metric_script_name, variant_name, path_config)
        metric_results[variant_name] = score
        if error_msg:
            errors.append(error_msg)
            if score is None:  # Critical error, print immediately if no score was parsed
                tqdm.write(f"\n{error_msg}")
    return metric_script_name, metric_results, errors


def main_logic(args_cli):
    """
    Main logic for running benchmarks and visualization, using parsed CLI arguments.
    """
    # --- Path Setup using pathlib from CLI arguments ---
    realfill_root_abs = args_cli.realfill_root.resolve()

    data_dir_abs = realfill_root_abs / args_cli.data_dir_name
    results_base_dir_abs = realfill_root_abs / args_cli.results_dir_name
    cache_dir_abs = realfill_root_abs / args_cli.cache_dir_name  # Full path from root
    benchmark_script_dir_abs = realfill_root_abs / args_cli.benchmark_script_dir_name
    plot_output_dir_abs = realfill_root_abs / args_cli.plot_output_dir_name

    gt_full_path_abs = (
        data_dir_abs / "RealBench" / str(args_cli.benchmark_number) / "target" / "gt.png"
    )
    mask_full_path_abs = (
        data_dir_abs / "RealBench" / str(args_cli.benchmark_number) / "target" / "mask.png"
    )

    # Simple namespace-like class to bundle path configurations
    class PathConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    path_cfg = PathConfig(
        realfill_root_abs=realfill_root_abs,
        benchmark_script_dir_abs=benchmark_script_dir_abs,
        results_base_dir_abs=results_base_dir_abs,
        gt_full_path_abs=gt_full_path_abs,
        mask_full_path_abs=mask_full_path_abs,
        cache_dir_abs=cache_dir_abs,
        benchmark_number=args_cli.benchmark_number,
    )

    # --- Ensure necessary directories exist ---
    for p_dir in [results_base_dir_abs, cache_dir_abs, plot_output_dir_abs]:
        p_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover result variants ---
    result_variants_glob_pattern = str(
        results_base_dir_abs / f"RealBench-{args_cli.benchmark_number}-*"
    )
    result_variants_list = sorted(
        [Path(p).name for p in glob.glob(result_variants_glob_pattern) if Path(p).is_dir()]
    )

    print(f"Running benchmark evaluation for RealBench-{args_cli.benchmark_number}")
    print(f"  Project Root: {realfill_root_abs}")
    print(f"  Data Source Dir: {data_dir_abs}")
    print(f"  Results Base Dir: {results_base_dir_abs}")
    print(f"  Cache Dir: {cache_dir_abs}")
    print(f"  Metric Scripts Dir: {benchmark_script_dir_abs}")
    print(f"  Plot Output Dir: {plot_output_dir_abs}")

    if not gt_full_path_abs.is_file() or not mask_full_path_abs.is_file():
        print(f"Error: GT path ({gt_full_path_abs}) or Mask path ({mask_full_path_abs}) not found!")
        print("Please check --data_dir_name and --benchmark_number.")
        return

    print(f"Found {len(result_variants_list)} result variants to process: {result_variants_list}")
    if not result_variants_list:
        print(f"No variants found matching pattern '{result_variants_glob_pattern}'. Exiting.")
        return
    print(f"Ensure your Python venv is activated and all dependencies are installed.")

    all_scores = {}
    all_errors = []
    metrics_to_execute = args_cli.metrics_to_run

    with ThreadPoolExecutor(max_workers=len(metrics_to_execute)) as executor:
        futures = []
        # Start inner progress bars from position 1
        for idx, script_name in enumerate(metrics_to_execute):
            futures.append(
                executor.submit(
                    process_metric_type, script_name, idx + 1, path_cfg, result_variants_list
                )
            )

        # Outer tqdm for the metric types themselves
        for future in tqdm(
            as_completed(futures),
            total=len(metrics_to_execute),
            desc="Overall Progress   ",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ):
            try:
                metric_script_name, metric_results, errors_for_metric = future.result()
                metric_key_name = metric_script_name.replace("_metric.py", "")
                all_scores[metric_key_name] = metric_results
                if errors_for_metric:
                    all_errors.extend(errors_for_metric)
            except Exception as e:
                tqdm.write(f"\nError processing future for a metric: {e}")  # Use tqdm.write
                all_errors.append(f"Future processing error: {str(e)}")

    tqdm.write("\n\n--- Summary of Scores ---")  # Use tqdm.write to clear progress bars correctly

    df_data = {}
    # Ensure metrics appear in the order they were run, if scores exist
    for metric_key in [m.replace("_metric.py", "") for m in metrics_to_execute]:
        if metric_key in all_scores:
            df_data[metric_key] = {}
            for variant_name, score in all_scores[metric_key].items():
                df_data[metric_key][variant_name] = score

    df = pd.DataFrame.from_dict(df_data, orient="index")

    if not df.empty:
        # Columns (variants) were already sorted when result_variants_list was created
        df = df[
            result_variants_list
        ]  # Ensure column order matches discovery order (already sorted)

    print(df.to_string())  # Print to console directly

    if all_errors:
        print("\n--- Errors and Warnings Summary ---")
        printed_errors_summary = set()
        for i, err_full_msg in enumerate(all_errors):
            err_summary = err_full_msg.splitlines()[0]
            if err_summary not in printed_errors_summary:
                is_no_score_error = "No FINAL_SCORE found" in err_summary
                score_was_parsed_despite_error = False
                if is_no_score_error:
                    try:  # Try to determine if a score was still parsed (e.g., from combined output)
                        parts = err_summary.replace(".", "").split(" ")
                        metric_name_in_err_script = parts[6]
                        variant_name_in_err = parts[8]
                        metric_key_from_err = metric_name_in_err_script.replace("_metric.py", "")
                        if (
                            all_scores.get(metric_key_from_err, {}).get(variant_name_in_err)
                            is not None
                        ):
                            score_was_parsed_despite_error = True
                    except (IndexError, KeyError):
                        pass

                if not (is_no_score_error and score_was_parsed_despite_error):
                    print(f"{i+1}. {err_full_msg}\n")
                    printed_errors_summary.add(err_summary)

    # --- Visualization ---
    if args_cli.skip_plot:
        print("\nSkipping plot generation as per --skip_plot flag.")
        return

    if not df.empty:
        if df.isnull().all().all():
            print(
                "\nNo scores available to plot. All metrics might have failed or returned no score."
            )
            return

        # Apply a professional plot style
        plt.style.use("seaborn-v0_8")

        # Plot metrics in the order they were specified / run
        metrics_in_df_order = [
            m.replace("_metric.py", "")
            for m in metrics_to_execute
            if m.replace("_metric.py", "") in df.index
        ]
        metrics_to_plot_final = [m for m in metrics_in_df_order if not df.loc[m].isnull().all()]

        if not metrics_to_plot_final:
            print("\nNo valid scores found for any configured metrics to plot.")
            return

        num_metrics_to_plot = len(metrics_to_plot_final)

        # --- Consistent Color Mapping for Variants ---
        variant_to_color_map = {}
        if result_variants_list:
            num_variants = len(result_variants_list)
            if num_variants <= 10:  # 'tab10' is great for up to 10 categories
                cmap = plt.get_cmap("tab10")
                variant_base_colors = [cmap(i) for i in range(num_variants)]
            else:  # For more variants, use a sequential map like viridis, but pick distinct points
                cmap = plt.get_cmap("viridis")
                # Sample from 0 to 0.9 to avoid very light colors at the end of viridis
                variant_base_colors = [cmap(i) for i in np.linspace(0, 0.9, num_variants)]

            variant_to_color_map = {
                name: color for name, color in zip(result_variants_list, variant_base_colors)
            }
        # --- End Color Mapping ---

        fig_width = 10  # Slightly wider for potentially rotated labels
        # Adjusted height per subplot for a more balanced look
        fig_height = max(fig_width * 4, num_metrics_to_plot * 8)
        fig, axes = plt.subplots(
            nrows=num_metrics_to_plot, ncols=1, figsize=(fig_width, fig_height), squeeze=False
        )
        axes = axes.flatten()

        for i, metric_key_name in enumerate(metrics_to_plot_final):
            ax = axes[i]
            metric_scores = df.loc[metric_key_name, result_variants_list]

            # Get colors for current variants in their specific order for this plot
            bar_colors_for_plot = [
                variant_to_color_map.get(name, "grey") for name in metric_scores.index
            ]

            metric_scores.plot(
                kind="bar",
                ax=ax,
                width=0.2,
                color=bar_colors_for_plot,
                edgecolor="black",  # Add edgecolor for definition
                linewidth=0.7,
            )

            info = METRIC_INFO.get(
                metric_key_name,
                {"display_name": metric_key_name.upper(), "higher_is_better": None, "unit": ""},
            )
            title_text = info["display_name"]
            if info["unit"]:
                title_text += f" ({info['unit']})"

            optimality_text = ""
            if info["higher_is_better"] is not None:
                arrow = "↑" if info["higher_is_better"] else "↓"
                # Using a more distinct arrow symbol if preferred: ▲ or ▼
                # arrow = "▲" if info["higher_is_better"] else "▼"
                optimality_text = f" ({arrow} is better)"

            # Slightly bolder subplot titles
            ax.set_title(
                title_text + optimality_text, fontsize=15, loc="left", fontweight="semibold"
            )
            ax.set_ylabel("Score", fontsize=13)  # Slightly larger y-label
            ax.tick_params(axis="x", labelsize=11)  # Slightly larger tick labels
            ax.tick_params(axis="y", labelsize=11)

            if result_variants_list:
                ax.set_xticks(range(len(result_variants_list)))
                # Adjust rotation if labels are long, 40-45 deg can be better
                ax.set_xticklabels(result_variants_list, rotation=40, ha="right", fontsize=11)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            ax.set_xlabel("")  # Individual x-labels cleared, common one added later
            # Grid is usually handled well by seaborn styles, but can be customized further if needed
            # ax.grid(axis="y", linestyle="--", alpha=0.7) # Kept for consistency if style doesn't provide satisfying grid

            # Annotations
            for p_idx, p in enumerate(ax.patches):
                score_val = p.get_height()
                if pd.isna(score_val):
                    continue
                ax.annotate(
                    f"{score_val:.3f}",
                    (p.get_x() + p.get_width() / 2.0, score_val),
                    ha="center",
                    va="bottom" if score_val >= 0 else "top",
                    xytext=(0, 6 if score_val >= 0 else -15),  # Slightly increased offset
                    textcoords="offset points",
                    fontsize=9,  # Slightly larger annotation font
                    color="dimgray",  # Softer color for annotations
                    fontweight="medium",
                )

        if num_metrics_to_plot > 0 and result_variants_list:
            axes[-1].set_xlabel(
                "Result Variants", fontsize=15, fontweight="semibold"
            )  # Bolder common x-label

        # Adjust layout: rect provides [left, bottom, right, top]
        # May need to adjust 'bottom' if x-labels (rotated) take more space
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        # Add hspace for more vertical separation between subplots if tight_layout isn't enough
        fig.subplots_adjust(hspace=0.5)

        fig.suptitle(
            f"Metric Comparison for RealBench-{args_cli.benchmark_number} Variants",
            fontsize=20,  # Larger main title
            fontweight="bold",
            y=0.99,  # Ensure it's close to the top edge specified in tight_layout
        )

        plot_filename = (
            plot_output_dir_abs / f"RealBench-{args_cli.benchmark_number}-metrics_comparison.png"
        )
        plt.savefig(plot_filename, dpi=150)  # Save with a decent DPI
        print(f"\nComparison plot saved as {plot_filename}")
        if args_cli.show_plot:
            plt.show()
    else:
        print("\nNo data to plot (DataFrame was empty or contained no scores).")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Important for PyInstaller/frozen executables

    parser = argparse.ArgumentParser(
        description="Run evaluation metrics for RealFill benchmarks and visualize results. Assumes a specific project directory structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows defaults in help message
    )
    parser.add_argument(
        "--benchmark_number",
        type=int,
        required=True,
        help="Identifier for the RealBench benchmark set (e.g., 24).",
    )
    parser.add_argument(
        "--realfill_root",
        type=Path,
        default=Path(".").resolve(),
        help="Root directory of the RealFill project. Defaults to the current directory.",
    )
    parser.add_argument(
        "--data_dir_name",
        type=str,
        default=DEFAULT_DATA_DIR_NAME,
        help="Name of the directory (relative to realfill_root) containing datasets (e.g., 'realfill_data_release_full').",
    )
    parser.add_argument(
        "--results_dir_name",
        type=str,
        default=DEFAULT_RESULTS_DIR_NAME,
        help="Name of the directory (relative to realfill_root) containing result variant subfolders.",
    )
    parser.add_argument(
        "--cache_dir_name",
        type=str,
        default=str(
            Path(DEFAULT_RESULTS_DIR_NAME) / DEFAULT_CACHE_DIR_NAME_FRAGMENT
        ),  # e.g., realfill_results/benchmark_cache
        help="Name of the directory (relative to realfill_root) for metric script caches.",
    )
    parser.add_argument(
        "--benchmark_script_dir_name",
        type=str,
        default=DEFAULT_BENCHMARK_SCRIPT_DIR_NAME,
        help="Name of the directory (relative to realfill_root) containing metric calculation scripts (e.g., 'benchmark').",
    )
    parser.add_argument(
        "--plot_output_dir_name",
        type=str,
        default=DEFAULT_PLOT_OUTPUT_DIR_NAME,
        help="Name of the directory (relative to realfill_root) where comparison plots will be saved.",
    )
    parser.add_argument(
        "--metrics_to_run",
        nargs="+",
        default=METRIC_SCRIPTS_AVAILABLE,
        choices=METRIC_SCRIPTS_AVAILABLE,
        metavar="SCRIPT_FILENAME",
        help=f"List of metric script filenames to run. Default is all. Choices: {', '.join(METRIC_SCRIPTS_AVAILABLE)}",
    )
    parser.add_argument("--skip_plot", action="store_true", help="If set, skip plot generation.")
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="If set, display the plot after saving (in addition to saving it).",
    )

    args_cli = parser.parse_args()

    # --- Basic Validation for Critical Paths ---
    # Validate that the benchmark script directory and specified metric scripts exist
    resolved_benchmark_script_dir = args_cli.realfill_root / args_cli.benchmark_script_dir_name
    if not resolved_benchmark_script_dir.is_dir():
        parser.error(
            f"Benchmark script directory not found: {resolved_benchmark_script_dir}\nPlease check --realfill_root and --benchmark_script_dir_name."
        )

    for script_file in args_cli.metrics_to_run:
        if not (resolved_benchmark_script_dir / script_file).is_file():
            parser.error(
                f"Specified metric script '{script_file}' not found in {resolved_benchmark_script_dir}."
            )

    # --- Call the main logic ---
    main_logic(args_cli)
