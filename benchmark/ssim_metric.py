# benchmark/ssim_metric.py

# --- Lightweight Imports First ---
import argparse
import json
import os
from pathlib import Path
import time
import sys

# math is not directly used by SSIM usually, but good to have if variations are considered

# --- Constants (Lightweight) ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"
METRIC_NAME = "ssim_masked"  # Explicitly denote that it's masked
SSIM_TARGET_SIZE = (512, 512)  # Default, can be made configurable

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
        if (
            mask_mtime_current and metadata.get("mask_mtime") != mask_mtime_current
        ):  # Mask is critical
            return None

        if cache_data.get("average") is None:
            return None

        return cache_data
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Masked SSIM with early cache exit.")
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
        print(f"Error: Missing heavy libraries for SSIM calculation (cv2, numpy): {e}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    # --- Heavy Path Helper Functions ---

    def ssim_single_channel_heavy(
        img1_ch_np: np.ndarray,
        img2_ch_np: np.ndarray,
        mask_np: np.ndarray,
        C1=(0.01 * 255) ** 2,
        C2=(0.03 * 255) ** 2,
    ) -> float | None:
        """Calculates SSIM for a single channel, masked."""
        img1_ch_np = img1_ch_np.astype(np.float64)
        img2_ch_np = img2_ch_np.astype(np.float64)
        mask_bool = mask_np > 0

        if not np.any(mask_bool):
            return 0.0

        kernel_size = 11
        sigma = 1.5
        window = cv2.getGaussianKernel(kernel_size, sigma)
        window = np.outer(window, window.transpose())

        mu1 = cv2.filter2D(img1_ch_np, -1, window, borderType=cv2.BORDER_REPLICATE)
        mu2 = cv2.filter2D(img2_ch_np, -1, window, borderType=cv2.BORDER_REPLICATE)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
        sigma1_sq = (
            cv2.filter2D(img1_ch_np**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
        )
        sigma2_sq = (
            cv2.filter2D(img2_ch_np**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
        )
        sigma12 = (
            cv2.filter2D(img1_ch_np * img2_ch_np, -1, window, borderType=cv2.BORDER_REPLICATE)
            - mu1_mu2
        )

        sigma1_sq = np.maximum(0.0, sigma1_sq)
        sigma2_sq = np.maximum(0.0, sigma2_sq)

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        ssim_masked_values = ssim_map[mask_bool]
        if ssim_masked_values.size == 0:
            return 0.0

        mean_ssim = np.mean(ssim_masked_values)
        return np.clip(mean_ssim, 0.0, 1.0)

    def calculate_ssim_masked_heavy(
        img1_color_np: np.ndarray, img2_color_np: np.ndarray, mask_np: np.ndarray
    ) -> float | None:
        """Calculates average SSIM over color channels for the masked region."""
        if (
            img1_color_np.shape != img2_color_np.shape
            or img1_color_np.shape[:2] != mask_np.shape[:2]
        ):
            return None

        if img1_color_np.ndim == 2:  # Grayscale
            return ssim_single_channel_heavy(img1_color_np, img2_color_np, mask_np)
        elif img1_color_np.ndim == 3 and img1_color_np.shape[2] == 3:  # Color
            ssims = []
            for i in range(3):  # B, G, R channels
                channel_ssim = ssim_single_channel_heavy(
                    img1_color_np[:, :, i], img2_color_np[:, :, i], mask_np
                )
                if channel_ssim is None:
                    return None  # Propagate error
                ssims.append(channel_ssim)
            return np.mean(ssims) if ssims else None
        return None

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
            print(f"Error saving SSIM cache (heavy path) to {cache_file_path}: {e_save}")

    # --- Main Calculation Logic (Heavy Path) ---
    final_average_score = None
    try:
        gt_image_raw = cv2.imread(args.gt_path)  # BGR
        mask_image_raw = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_image_raw is None or mask_image_raw is None:
            print(
                f"Error loading GT/Mask for SSIM (heavy path). GT: {args.gt_path}, Mask: {args.mask_path}"
            )
            print("FINAL_SCORE:ERROR")
            sys.exit(1)

        gt_image_processed = cv2.resize(
            gt_image_raw, SSIM_TARGET_SIZE, interpolation=cv2.INTER_LINEAR
        )
        mask_image_processed = cv2.resize(
            mask_image_raw, SSIM_TARGET_SIZE, interpolation=cv2.INTER_NEAREST
        )
    except Exception as e_prep:
        print(f"Error during GT/Mask preparation for SSIM (heavy path): {e_prep}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    results_dir_obj = Path(args.results_dir)
    total_ssim_val = 0.0
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
        print("FINAL_SCORE:ERROR")  # Or 0.0
        sys.exit(0)

    for result_img_path_obj in tqdm(
        image_paths_to_process,
        desc=f"SSIM Processing {current_results_dir_name}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        try:
            gen_image_raw = cv2.imread(str(result_img_path_obj))  # BGR
            if gen_image_raw is None:
                per_image_scores_dict[result_img_path_obj.name] = None
                continue

            gen_image_processed = cv2.resize(
                gen_image_raw, SSIM_TARGET_SIZE, interpolation=cv2.INTER_LINEAR
            )
            ssim_value = calculate_ssim_masked_heavy(
                gt_image_processed, gen_image_processed, mask_image_processed
            )

            if ssim_value is not None:
                total_ssim_val += ssim_value
                per_image_scores_dict[result_img_path_obj.name] = ssim_value
                valid_image_count_val += 1
            else:
                per_image_scores_dict[result_img_path_obj.name] = None
        except Exception:
            per_image_scores_dict[result_img_path_obj.name] = None

    if valid_image_count_val > 0:
        final_average_score = total_ssim_val / valid_image_count_val
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
        # Higher SSIM is better (0 to 1).
        print(f"FINAL_SCORE:{final_average_score:.8f}")
    else:
        print("FINAL_SCORE:ERROR")
