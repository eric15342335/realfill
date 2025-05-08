# benchmark/ssim_metric.py

import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import time
import math  # Keep for potential future use, though original SSIM doesn't use log

# --- Constants ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"
TARGET_SIZE = (512, 512)  # Resize dimension used in the original script

# --- SSIM Calculation Logic (Adapted from original SSIM.py) ---


def ssim_filled(img1, img2, mask, C1=(0.01 * 255) ** 2, C2=(0.03 * 255) ** 2):
    """
    Calculates SSIM only for the filled-in regions defined by the mask.
    Adapted from the user's original SSIM.py.
    Assumes img1, img2 are numpy arrays [H, W] (grayscale) [0, 255].
    Mask is numpy array [H, W] where >0 indicates filled region.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask_bool = mask > 0  # Boolean mask for filled regions

    if not np.any(mask_bool):
        # print("Warning: Mask is empty for SSIM calculation.")
        return 0.0  # No region to compare

    # Using OpenCV's filter2D for windowed calculations is generally faster
    # Use a Gaussian window as standard for SSIM
    kernel_size = 11
    sigma = 1.5
    window = cv2.getGaussianKernel(kernel_size, sigma)
    window = np.outer(window, window.transpose())  # 2D Gaussian window

    # --- Calculate means ---
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)

    # --- Calculate variances and covariance ---
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2

    # Clamp negative variances to zero (can happen due to floating point errors)
    sigma1_sq = np.maximum(0.0, sigma1_sq)
    sigma2_sq = np.maximum(0.0, sigma2_sq)

    # --- Calculate SSIM map ---
    # Numerator: (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    # Denominator: (mu1^2 + mu2^2 + C1) * (sigma1^2 + sigma2^2 + C2)
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    # --- Average SSIM over the masked region ---
    ssim_masked_values = ssim_map[mask_bool]

    if ssim_masked_values.size == 0:
        # print("Warning: No valid SSIM values in masked region.")
        return 0.0

    mean_ssim = np.mean(ssim_masked_values)

    # Clamp result to [0, 1] range (SSIM can sometimes slightly exceed 1 due to numerics)
    return np.clip(mean_ssim, 0.0, 1.0)


def calculate_ssim_masked(img1_color, img2_color, mask):
    """
    Calculates average SSIM over color channels for the masked region.
    img1_color, img2_color: Numpy arrays [H, W, C], BGR format [0, 255]
    mask: Numpy array [H, W], grayscale [0, 255]
    """
    if img1_color.shape != img2_color.shape or img1_color.shape[:2] != mask.shape[:2]:
        print(
            f"Warning: Shape mismatch in SSIM input. GT: {img1_color.shape}, Gen: {img2_color.shape}, Mask: {mask.shape}"
        )
        return None  # Indicate error

    if img1_color.ndim == 2:  # Grayscale case
        return ssim_filled(img1_color, img2_color, mask)
    elif img1_color.ndim == 3 and img1_color.shape[2] == 3:  # Color image
        ssims = []
        for i in range(3):  # Iterate over B, G, R channels
            channel_ssim = ssim_filled(img1_color[:, :, i], img2_color[:, :, i], mask)
            if channel_ssim is None:  # Propagate error if single channel fails
                return None
            ssims.append(channel_ssim)
        return np.mean(ssims)
    else:
        print(f"Warning: Unsupported image dimensions for SSIM: {img1_color.shape}")
        return None


# --- Helper Functions (Caching etc.) ---


def get_modification_time(file_path):
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0


def get_cache_path(cache_dir, results_dir_name):
    return Path(cache_dir) / "per_scene_cache" / "ssim_masked" / f"{results_dir_name}.json"


def load_cache(cache_file, gt_mtime_current, mask_mtime_current):
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        if (
            not isinstance(cache_data, dict)
            or "average" not in cache_data
            or "per_image" not in cache_data
            or "metadata" not in cache_data
            or cache_data.get("metadata", {}).get("cache_version") != CACHE_VERSION
        ):
            print(f"Cache file {cache_file} format invalid or version mismatch. Recalculating.")
            return None
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
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data["metadata"] = {
            "timestamp": time.time(),
            "gt_mtime": gt_mtime,
            "mask_mtime": mask_mtime,
            "cache_version": CACHE_VERSION,
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving cache to {cache_file}: {e}")


# --- Main Function ---
def calculate_scene_ssim(
    gt_path_str,
    mask_path_str,
    results_dir_str,
    cache_dir_str,
    num_images=DEFAULT_NUM_IMAGES,
    target_size=TARGET_SIZE,
):
    """
    Calculates the average masked SSIM for a given scene.

    Args:
        gt_path_str (str): Path to the ground truth image.
        mask_path_str (str): Path to the mask image.
        results_dir_str (str): Path to the directory containing result images.
        cache_dir_str (str): Path to the base directory for caching.
        num_images (int): Number of result images to process.
        target_size (tuple): Target (width, height) for resizing, or None.

    Returns:
        float: The average masked SSIM score (higher is better, 0-1), or None on error.
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

        if target_size:
            gt_image = cv2.resize(gt_image, target_size, interpolation=cv2.INTER_LINEAR)
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
    print(f"Calculating Masked SSIM for scene: {results_dir_name}")
    total_ssim = 0.0
    valid_image_count = 0
    per_image_scores = {}

    image_files = sorted(
        [p for p in results_dir.glob("*.png") if p.stem.isdigit()], key=lambda x: int(x.stem)
    )
    image_files = image_files[:num_images]

    if not image_files:
        print(f"Warning: No valid result images (0..{num_images-1}.png) found in {results_dir}")
        return 0.0  # SSIM defaults to 0 if no images

    for i in tqdm(range(num_images), desc=f"SSIM Processing {results_dir_name}", leave=False):
        result_img_name = f"{i}.png"
        result_img_path = results_dir / result_img_name

        if result_img_path.is_file():
            try:
                gen_image = cv2.imread(str(result_img_path))
                if gen_image is None:
                    print(f"Warning: Failed to load generated image {result_img_name}. Skipping.")
                    per_image_scores[result_img_name] = None
                    continue

                if target_size:
                    if gen_image.shape[:2] != target_size[::-1]:
                        gen_image = cv2.resize(
                            gen_image, target_size, interpolation=cv2.INTER_LINEAR
                        )

                ssim_value = calculate_ssim_masked(gt_image, gen_image, mask_image)

                if ssim_value is not None:
                    total_ssim += ssim_value
                    per_image_scores[result_img_name] = ssim_value
                    valid_image_count += 1
                else:
                    per_image_scores[result_img_name] = None  # Mark error
                    # print(f"Skipping SSIM for {result_img_name} due to calculation error.")

            except Exception as e:
                print(f"Error processing image {result_img_name} for SSIM: {e}")
                per_image_scores[result_img_name] = None
        # else:
        #     print(f"Warning: Result image {result_img_name} not found.")
        #     per_image_scores[result_img_name] = None

    if valid_image_count == 0:
        print(f"Error: No valid result images processed for Masked SSIM in {results_dir}")
        return None

    average_ssim = total_ssim / valid_image_count

    # --- Save to Cache ---
    cache_data = {
        "average": average_ssim,
        "per_image": per_image_scores,
        "count": valid_image_count,
    }
    save_cache(cache_file, cache_data, gt_mtime, mask_mtime)

    return average_ssim


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Masked SSIM for a RealFill result scene."
    )
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
    # Add args for resizing control if needed

    args = parser.parse_args()

    current_target_size = TARGET_SIZE
    # Update current_target_size based on args if resizing control is added

    # Ensure cache subdirectories exist
    ssim_cache_dir = Path(args.cache_dir) / "per_scene_cache" / "ssim_masked"
    ssim_cache_dir.mkdir(parents=True, exist_ok=True)

    avg_score = calculate_scene_ssim(
        args.gt_path,
        args.mask_path,
        args.results_dir,
        args.cache_dir,
        args.num_images,
        target_size=current_target_size,
    )

    # Remember: Higher SSIM is better (0 to 1)
    if avg_score is not None:
        print(f"\nAverage Masked SSIM for {Path(args.results_dir).name}: {avg_score:.4f}")
        print(f"FINAL_SCORE:{avg_score:.8f}")
    else:
        print(f"\nFailed to calculate Masked SSIM for {Path(args.results_dir).name}")
        print("FINAL_SCORE:ERROR")
