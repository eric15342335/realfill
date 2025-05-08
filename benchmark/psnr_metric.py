# benchmark/psnr_metric.py

# --- Lightweight Imports First ---
import argparse
import json
import os
from pathlib import Path
import time
import sys
import math  # For log10

# --- Constants (Lightweight) ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"
METRIC_NAME = "psnr_masked"  # Explicitly denote that it's masked
# TARGET_SIZE is handled in the heavy part if needed, not fixed for early cache logic
PSNR_TARGET_SIZE = (512, 512)  # Default, can be overridden if made a CLI arg

# --- Lightweight Helper Functions for Cache Handling ---


def get_cache_path_light(
    cache_dir_str: str, results_dir_name_str: str, metric_name_str: str
) -> Path:
    return (
        Path(cache_dir_str) / "per_scene_cache" / metric_name_str / f"{results_dir_name_str}.json"
    )


def get_modification_time_light(file_path_str: str) -> float:
    try:
        return os.path.getmtime(file_path_str)
    except OSError:
        return 0.0


def load_cache_light(
    cache_file_path: Path,
    gt_mtime_current: float,
    mask_mtime_current: float,
    expected_cache_version: str,
) -> dict | None:
    if not cache_file_path.is_file():
        return None
    try:
        with open(cache_file_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        metadata = cache_data.get("metadata")
        if (
            not isinstance(cache_data, dict)
            or "average" not in cache_data
            or not isinstance(metadata, dict)
            or metadata.get("cache_version") != expected_cache_version
        ):
            return None

        if gt_mtime_current and metadata.get("gt_mtime") != gt_mtime_current:
            return None
        # Mask mtime IS critical for masked PSNR
        if mask_mtime_current and metadata.get("mask_mtime") != mask_mtime_current:
            return None

        if cache_data.get("average") is None:  # Check for a valid cached score
            return None

        return cache_data
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Masked PSNR with early cache exit.")
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to the ground truth image."
    )
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory containing result images.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to the base directory for caching results.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of result images per scene (default: {DEFAULT_NUM_IMAGES}).",
    )
    # Add --target_width and --target_height if you want to make resizing configurable via CLI
    # For now, using a fixed PSNR_TARGET_SIZE for the heavy path.
    args = parser.parse_args()

    current_results_dir_name = Path(args.results_dir).name
    cache_file = get_cache_path_light(args.cache_dir, current_results_dir_name, METRIC_NAME)
    current_gt_mtime = get_modification_time_light(args.gt_path)
    current_mask_mtime = get_modification_time_light(args.mask_path)

    cached_data = load_cache_light(cache_file, current_gt_mtime, current_mask_mtime, CACHE_VERSION)

    if cached_data:
        cached_avg_score = cached_data.get("average")
        if cached_avg_score is not None:
            print(f"FINAL_SCORE:{cached_avg_score:.8f}")
            sys.exit(0)

    try:
        import cv2
        import numpy as np
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error: Missing heavy libraries for PSNR calculation (cv2, numpy): {e}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    # --- Heavy Path Helper Functions ---

    def calculate_psnr_masked_heavy(
        ground_truth_np: np.ndarray, generated_np: np.ndarray, mask_np: np.ndarray
    ) -> float | None:
        """Calculates PSNR for masked regions. Assumes images are numpy arrays [H,W,C] or [H,W]."""
        if (
            ground_truth_np.shape != generated_np.shape
            or ground_truth_np.shape[:2] != mask_np.shape[:2]
        ):
            return None

        mask_bool = mask_np > 0
        if not np.any(mask_bool):
            return 0.0  # Or None if preferred for empty mask scenario

        gt_masked = ground_truth_np[mask_bool]
        gen_masked = generated_np[mask_bool]

        if gt_masked.size == 0:
            return 0.0

        mse = np.mean((gt_masked.astype(np.float64) - gen_masked.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0  # Infinite PSNR for identical images in mask
        if mse < 1.0e-10:
            mse = 1.0e-10  # Prevent log(0)

        max_pixel = 255.0
        try:
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        except ValueError:  # Should be rare with mse checks
            return None
        return psnr

    def save_cache_heavy(
        cache_file_path: Path,
        data_to_save: dict,
        gt_mtime: float,
        mask_mtime: float,
        current_cache_version: str,
    ):
        try:
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            data_to_save["metadata"] = {
                "timestamp": time.time(),
                "gt_mtime": gt_mtime,
                "mask_mtime": mask_mtime,
                "cache_version": current_cache_version,
            }
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e_save:
            print(f"Error saving PSNR cache (heavy path) to {cache_file_path}: {e_save}")

    # --- Main Calculation Logic (Heavy Path) ---
    final_average_score = None
    try:
        gt_image_raw = cv2.imread(args.gt_path)
        mask_image_raw = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)

        if gt_image_raw is None or mask_image_raw is None:
            print(
                f"Error loading GT/Mask for PSNR (heavy path). GT: {args.gt_path}, Mask: {args.mask_path}"
            )
            print("FINAL_SCORE:ERROR")
            sys.exit(1)

        # Resize GT and Mask (if target_size is defined for PSNR)
        # Using PSNR_TARGET_SIZE constant for this version
        gt_image_processed = cv2.resize(
            gt_image_raw, PSNR_TARGET_SIZE, interpolation=cv2.INTER_LINEAR
        )
        mask_image_processed = cv2.resize(
            mask_image_raw, PSNR_TARGET_SIZE, interpolation=cv2.INTER_NEAREST
        )

    except Exception as e_prep:
        print(f"Error during GT/Mask preparation for PSNR (heavy path): {e_prep}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    results_dir_obj = Path(args.results_dir)
    total_psnr_val = 0.0
    valid_image_count_val = 0
    per_image_scores_dict = {}

    image_paths_to_process = []
    for i in range(args.num_images):
        img_p = results_dir_obj / f"{i}.png"
        if img_p.is_file():
            image_paths_to_process.append(img_p)
        else:
            per_image_scores_dict[f"{i}.png"] = None

    if not image_paths_to_process and args.num_images > 0:
        save_cache_heavy(
            cache_file,
            {"average": None, "per_image": per_image_scores_dict, "count": 0},
            current_gt_mtime,
            current_mask_mtime,
            CACHE_VERSION,
        )
        print("FINAL_SCORE:ERROR")  # Or 0.0 if appropriate for no images
        sys.exit(0)

    for result_img_path_obj in tqdm(
        image_paths_to_process,
        desc=f"PSNR Processing {current_results_dir_name}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        try:
            gen_image_raw = cv2.imread(str(result_img_path_obj))
            if gen_image_raw is None:
                per_image_scores_dict[result_img_path_obj.name] = None
                continue

            # Resize generated image to match processed GT/Mask
            gen_image_processed = cv2.resize(
                gen_image_raw, PSNR_TARGET_SIZE, interpolation=cv2.INTER_LINEAR
            )

            psnr_value = calculate_psnr_masked_heavy(
                gt_image_processed, gen_image_processed, mask_image_processed
            )

            if psnr_value is not None:
                total_psnr_val += psnr_value
                per_image_scores_dict[result_img_path_obj.name] = psnr_value
                valid_image_count_val += 1
            else:
                per_image_scores_dict[result_img_path_obj.name] = None
        except Exception:
            per_image_scores_dict[result_img_path_obj.name] = None

    if valid_image_count_val > 0:
        final_average_score = total_psnr_val / valid_image_count_val
        cache_data_to_save = {
            "average": final_average_score,
            "per_image": per_image_scores_dict,
            "count": valid_image_count_val,
        }
        save_cache_heavy(
            cache_file, cache_data_to_save, current_gt_mtime, current_mask_mtime, CACHE_VERSION
        )
    else:
        save_cache_heavy(
            cache_file,
            {"average": None, "per_image": per_image_scores_dict, "count": 0},
            current_gt_mtime,
            current_mask_mtime,
            CACHE_VERSION,
        )

    if final_average_score is not None:
        # Higher PSNR is better.
        print(f"FINAL_SCORE:{final_average_score:.8f}")
    else:
        print("FINAL_SCORE:ERROR")
