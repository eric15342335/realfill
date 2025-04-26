# benchmark/psnr_metric.py

import math
import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import time

# --- Constants ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"
TARGET_SIZE = (512, 512)  # Resize dimension used in the original script

# --- Helper Functions ---


def get_modification_time(file_path):
    """Gets the modification time of a file."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0


def calculate_psnr_masked(ground_truth, generated, mask):
    """
    Calculates PSNR only for the masked regions defined by the mask.
    Assumes images are numpy arrays [H, W, C] or [H, W] with values [0, 255].
    Mask is a numpy array [H, W] where >0 indicates the region of interest.
    """
    if ground_truth.shape != generated.shape:
        # Basic check, could be more robust
        print(
            f"Warning: Shape mismatch between GT {ground_truth.shape} and Generated {generated.shape}. Skipping PSNR."
        )
        return None  # Indicate error
    if ground_truth.shape[:2] != mask.shape[:2]:
        print(
            f"Warning: Shape mismatch between Image {ground_truth.shape[:2]} and Mask {mask.shape[:2]}. Skipping PSNR."
        )
        return None

    mask_bool = mask > 0  # Boolean mask for masked regions

    # Ensure mask covers some area
    if not np.any(mask_bool):
        # print("Warning: Mask is empty. Cannot calculate masked PSNR.")
        return 0.0  # Or perhaps None, though 0 is plausible if no area changed

    gt_masked = ground_truth[mask_bool]
    gen_masked = generated[mask_bool]

    if gt_masked.size == 0:  # Should be covered by np.any above, but double-check
        # print("Warning: No pixels selected by mask. Cannot calculate masked PSNR.")
        return 0.0

    # Calculate MSE on the masked region
    mse = np.mean((gt_masked.astype(np.float64) - gen_masked.astype(np.float64)) ** 2)

    if mse == 0:
        # Images are identical in the masked region
        return 100.0  # Often used to represent infinite PSNR
    if mse < 1.0e-10:
        # Handle potential floating point inaccuracies close to zero
        mse = 1.0e-10

    max_pixel = 255.0
    try:
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    except ValueError:
        # Handle cases where sqrt(mse) might be invalid (should be caught by mse checks)
        print(f"Warning: math domain error during PSNR calculation (MSE: {mse}).")
        return None  # Indicate error
    return psnr


def get_cache_path(cache_dir, results_dir_name):
    """Constructs the path for the cache file."""
    return Path(cache_dir) / "per_scene_cache" / "psnr_masked" / f"{results_dir_name}.json"  # Use _masked suffix


def load_cache(cache_file, gt_mtime_current, mask_mtime_current):
    """Loads results from cache if valid."""
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        # Validation
        if (
            not isinstance(cache_data, dict)
            or "average" not in cache_data
            or "per_image" not in cache_data
            or "metadata" not in cache_data
            or cache_data.get("metadata", {}).get("cache_version") != CACHE_VERSION
        ):
            print(f"Cache file {cache_file} format invalid or version mismatch. Recalculating.")
            return None

        # Check modification times - Mask is CRITICAL here
        if gt_mtime_current and cache_data["metadata"].get("gt_mtime") != gt_mtime_current:
            print(f"Ground truth mtime changed for {cache_file.name}. Recalculating.")
            return None
        if mask_mtime_current and cache_data["metadata"].get("mask_mtime") != mask_mtime_current:
            print(f"Mask mtime changed for {cache_file.name}. Recalculating.")
            return None

        print(f"Cache hit for {cache_file.name}")
        return cache_data
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error loading cache file {cache_file}: {e}. Recalculating.")
        return None


def save_cache(cache_file, data, gt_mtime, mask_mtime):
    """Saves results to the cache file."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data["metadata"] = {
            "timestamp": time.time(),
            "gt_mtime": gt_mtime,
            "mask_mtime": mask_mtime,  # Save mask mtime
            "cache_version": CACHE_VERSION,
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4)
        # print(f"Saved cache to {cache_file}")
    except Exception as e:
        print(f"Error saving cache to {cache_file}: {e}")


# --- Main Function ---
def calculate_scene_psnr(
    gt_path_str, mask_path_str, results_dir_str, cache_dir_str, num_images=DEFAULT_NUM_IMAGES, target_size=TARGET_SIZE
):
    """
    Calculates the average masked PSNR for a given scene.

    Args:
        gt_path_str (str): Path to the ground truth image.
        mask_path_str (str): Path to the mask image.
        results_dir_str (str): Path to the directory containing result images (0.png, 1.png, ...).
        cache_dir_str (str): Path to the base directory for caching.
        num_images (int): Number of result images to process.
        target_size (tuple): Target (width, height) to resize images, or None to skip resize.

    Returns:
        float: The average masked PSNR score (higher is better), or None on error.
    """
    gt_path = Path(gt_path_str)
    mask_path = Path(mask_path_str)
    results_dir = Path(results_dir_str)
    cache_dir = Path(cache_dir_str)

    if not gt_path.is_file():
        print(f"Error: Ground truth image not found at {gt_path}")
        return None
    if not mask_path.is_file():
        print(f"Error: Mask image not found at {mask_path}")
        return None
    if not results_dir.is_dir():
        print(f"Error: Results directory not found at {results_dir}")
        return None

    results_dir_name = results_dir.name
    cache_file = get_cache_path(cache_dir, results_dir_name)

    # --- Load GT and Mask (Load once) ---
    try:
        gt_image = cv2.imread(str(gt_path))
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_image is None:
            print(f"Error: Failed to load ground truth image: {gt_path}")
            return None
        if mask_image is None:
            print(f"Error: Failed to load mask image: {mask_path}")
            return None

        # Resize if target_size is specified
        if target_size:
            gt_image = cv2.resize(gt_image, target_size, interpolation=cv2.INTER_LINEAR)  # Use quality interpolation
            # IMPORTANT: Use nearest neighbor for mask to avoid introducing gray values
            mask_image = cv2.resize(mask_image, target_size, interpolation=cv2.INTER_NEAREST)

    except Exception as e:
        print(f"Error loading/resizing GT or Mask: {e}")
        return None

    # --- Cache Check ---
    gt_mtime = get_modification_time(gt_path)
    mask_mtime = get_modification_time(mask_path)
    cached_results = load_cache(cache_file, gt_mtime, mask_mtime)
    if cached_results:
        return cached_results.get("average")

    # --- Calculation ---
    print(f"Calculating Masked PSNR for scene: {results_dir_name}")
    total_psnr = 0.0
    valid_image_count = 0
    per_image_scores = {}

    image_files = sorted([p for p in results_dir.glob("*.png") if p.stem.isdigit()], key=lambda x: int(x.stem))
    image_files = image_files[:num_images]

    if not image_files:
        print(f"Warning: No valid result images (0..{num_images-1}.png) found in {results_dir}")
        return 0.0  # Or None

    for i in tqdm(range(num_images), desc=f"PSNR Processing {results_dir_name}", leave=False):
        result_img_name = f"{i}.png"
        result_img_path = results_dir / result_img_name

        if result_img_path.is_file():
            try:
                gen_image = cv2.imread(str(result_img_path))
                if gen_image is None:
                    print(f"Warning: Failed to load generated image {result_img_name}. Skipping.")
                    per_image_scores[result_img_name] = None
                    continue

                # Resize if needed
                if target_size:
                    # Ensure generated image matches GT size for comparison
                    if gen_image.shape[:2] != target_size[::-1]:  # OpenCV size is (W, H)
                        gen_image = cv2.resize(gen_image, target_size, interpolation=cv2.INTER_LINEAR)

                psnr_value = calculate_psnr_masked(gt_image, gen_image, mask_image)

                if psnr_value is not None:
                    total_psnr += psnr_value
                    per_image_scores[result_img_name] = psnr_value
                    valid_image_count += 1
                else:
                    per_image_scores[result_img_name] = None  # Mark error
                    # print(f"Skipping PSNR for {result_img_name} due to calculation error.")

            except Exception as e:
                print(f"Error processing image {result_img_name}: {e}")
                per_image_scores[result_img_name] = None
        # else:
        #     print(f"Warning: Result image {result_img_name} not found.")
        #     per_image_scores[result_img_name] = None

    if valid_image_count == 0:
        print(f"Error: No valid result images processed for Masked PSNR in {results_dir}")
        return None

    average_psnr = total_psnr / valid_image_count

    # --- Save to Cache ---
    cache_data = {"average": average_psnr, "per_image": per_image_scores, "count": valid_image_count}
    save_cache(cache_file, cache_data, gt_mtime, mask_mtime)

    return average_psnr


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Masked PSNR for a RealFill result scene.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth image.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image.")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to the directory containing result images."
    )
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to the base directory for caching results.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of result images per scene (default: {DEFAULT_NUM_IMAGES}).",
    )
    # Optional: Add argument to disable resizing or change size
    # parser.add_argument("--no_resize", action='store_true', help="Disable resizing to target size.")
    # parser.add_argument("--target_size", type=int, nargs=2, default=[TARGET_SIZE[0], TARGET_SIZE[1]], help="Target size (W H) for resizing.")

    args = parser.parse_args()

    # Parse target_size if needed
    current_target_size = TARGET_SIZE  # Use default for now
    # if args.no_resize:
    #     current_target_size = None
    # else:
    #     current_target_size = tuple(args.target_size)

    # Ensure cache subdirectories exist
    psnr_cache_dir = Path(args.cache_dir) / "per_scene_cache" / "psnr_masked"
    psnr_cache_dir.mkdir(parents=True, exist_ok=True)

    avg_score = calculate_scene_psnr(
        args.gt_path, args.mask_path, args.results_dir, args.cache_dir, args.num_images, target_size=current_target_size
    )

    # Remember: Higher PSNR is better
    if avg_score is not None:
        print(f"\nAverage Masked PSNR for {Path(args.results_dir).name}: {avg_score:.2f} dB")
        print(f"FINAL_SCORE:{avg_score:.8f}")
    else:
        print(f"\nFailed to calculate Masked PSNR for {Path(args.results_dir).name}")
        print("FINAL_SCORE:ERROR")
