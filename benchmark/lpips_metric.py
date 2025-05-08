# benchmark/lpips_metric.py

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import time
import traceback  # Added for better error logging

# Attempt LPIPS import
try:
    import lpips

    lpips_available = True
except ImportError:
    print("Error: lpips library not found. Please install it: pip install lpips")
    lpips = None
    lpips_available = False


# --- Constants ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"
TARGET_SIZE = (512, 512)  # Resize dimension used in the original script
LPIPS_NET = "alex"  # Or 'vgg'

# --- Model Variables (Initialize to None) ---
loss_fn_lpips = None
device = None
model_loaded = False  # Flag to track loading state

# --- Transforms (Define globally) ---
# Converts numpy HWC [0,255] to torch CHW [0,1]
img_to_tensor = transforms.ToTensor()


def normalize_tensor(tensor):
    """Normalizes a tensor from [0,1] to [-1,1] as expected by LPIPS."""
    return tensor * 2.0 - 1.0


# --- Helper Functions ---


def _load_model_if_needed():
    """Loads the LPIPS model once, only when needed."""
    global loss_fn_lpips, device, model_loaded, lpips_available
    if model_loaded:
        return True  # Already loaded successfully

    if not lpips_available:
        print("LPIPS library not available, cannot load model.")
        return False

    if loss_fn_lpips is not None and device is not None:
        model_loaded = True  # Already loaded
        return True

    print(f"Attempting to load LPIPS model (net={LPIPS_NET})...")
    try:
        # Initialize LPIPS model
        _loss_fn_lpips = lpips.LPIPS(net=LPIPS_NET)  # Use AlexNet by default
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _loss_fn_lpips.to(_device)
        _loss_fn_lpips.eval()  # Set to evaluation mode

        # Assign to global variables only after successful loading
        loss_fn_lpips = _loss_fn_lpips
        device = _device
        model_loaded = True
        print(f"LPIPS model loaded successfully on {device}.")
        return True
    except Exception as e:
        print(f"FATAL Error loading LPIPS model: {e}")
        traceback.print_exc()
        # Ensure globals remain None on failure
        loss_fn_lpips = None
        device = None
        model_loaded = False
        return False


def get_modification_time(file_path):
    """Gets the modification time of a file."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0


@torch.no_grad()  # Essential for inference
def calculate_lpips_masked(gt_tensor_norm, gen_tensor_norm, mask_tensor, model, device):
    """
    Calculates LPIPS distance focused on the masked regions.
    Assumes input tensors are normalized to [-1, 1] and have batch dim [1, C, H, W].
    Mask tensor is [1, 1, H, W] with values in [0, 1].
    """
    # Safeguard checks
    if not model or not device:
        print("LPIPS model or device not available during distance calculation.")
        return None

    if (
        gt_tensor_norm.shape != gen_tensor_norm.shape
        or gt_tensor_norm.shape[2:] != mask_tensor.shape[2:]
    ):
        print("Warning: Tensor shape mismatch. Skipping LPIPS calculation.")
        return None

    mask_tensor = mask_tensor.to(device)
    masked_gt = gt_tensor_norm * mask_tensor
    masked_gen = gen_tensor_norm * mask_tensor

    try:
        distance = model(masked_gt, masked_gen)
        return distance.item()
    except Exception as e:
        print(f"Error during LPIPS forward pass: {e}")
        # traceback.print_exc() # Optional
        return None


def get_cache_path(cache_dir, results_dir_name):
    """Constructs the path for the cache file."""
    return (
        Path(cache_dir) / "per_scene_cache" / "lpips_masked" / f"{results_dir_name}.json"
    )  # Use _masked suffix


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

        # Check modification times - Mask is CRITICAL
        if gt_mtime_current and cache_data["metadata"].get("gt_mtime") != gt_mtime_current:
            print(f"Ground truth mtime changed for {cache_file.name}. Recalculating.")
            return None
        if mask_mtime_current and cache_data["metadata"].get("mask_mtime") != mask_mtime_current:
            print(f"Mask mtime changed for {cache_file.name}. Recalculating.")
            return None

        # print(f"Cache hit for {cache_file.name}") # Moved to main function
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
def calculate_scene_lpips(
    gt_path_str,
    mask_path_str,
    results_dir_str,
    cache_dir_str,
    num_images=DEFAULT_NUM_IMAGES,
    target_size=TARGET_SIZE,
):
    """
    Calculates the average masked LPIPS distance for a given scene.
    Loads the model only if needed.
    """
    gt_path = Path(gt_path_str)
    mask_path = Path(mask_path_str)
    results_dir = Path(results_dir_str)
    cache_dir = Path(cache_dir_str)

    # --- Input Validation ---
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

    # --- Load GT and Mask, Preprocess (Do this before cache check for mtime) ---
    try:
        gt_image_bgr = cv2.imread(str(gt_path))
        mask_image_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_image_bgr is None:
            print(f"Error: Failed to load ground truth image: {gt_path}")
            return None
        if mask_image_gray is None:
            print(f"Error: Failed to load mask image: {mask_path}")
            return None

        gt_image_rgb = cv2.cvtColor(gt_image_bgr, cv2.COLOR_BGR2RGB)

        if target_size:
            gt_image_rgb = cv2.resize(gt_image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            mask_image_gray = cv2.resize(
                mask_image_gray, target_size, interpolation=cv2.INTER_NEAREST
            )

        # Tensors are needed for calculation, but can be created later if needed
        # We only need paths and mtimes for cache check initially

    except Exception as e:
        print(f"Error loading/preprocessing GT or Mask base images: {e}")
        return None

    # --- Cache Check ---
    gt_mtime = get_modification_time(gt_path)
    mask_mtime = get_modification_time(mask_path)
    cached_results = load_cache(cache_file, gt_mtime, mask_mtime)
    if cached_results:
        print(f"Cache hit for {results_dir_name}")
        return cached_results.get("average")

    # --- Model Loading (Only if needed) ---
    if not _load_model_if_needed():
        print("Error: Failed to load LPIPS model or library unavailable. Cannot proceed.")
        return None

    # --- Prepare Tensors (Now that we know we need them) ---
    try:
        # Convert GT to tensors and normalize
        gt_tensor = img_to_tensor(gt_image_rgb)  # [C, H, W], [0, 1]
        # Ensure device is valid before using .to(device)
        if device is None:
            print("Error: LPIPS device is None after model loading attempt.")
            return None
        gt_tensor_norm = (
            normalize_tensor(gt_tensor).unsqueeze(0).to(device)
        )  # [1, C, H, W], [-1, 1]

        # Convert mask to tensor [1, 1, H, W], [0, 1]
        mask_tensor = (
            img_to_tensor(mask_image_gray.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        )
        mask_tensor = (mask_tensor > 0.5).float()
    except Exception as e:
        print(f"Error creating tensors from GT/Mask: {e}")
        return None

    # --- Calculation ---
    print(f"Calculating Masked LPIPS for scene: {results_dir_name}")
    total_lpips = 0.0
    valid_image_count = 0
    per_image_scores = {}

    image_files = sorted(
        [p for p in results_dir.glob("*.png") if p.stem.isdigit()], key=lambda x: int(x.stem)
    )
    image_files = image_files[:num_images]

    if not image_files:
        print(f"Warning: No valid result images (0..{num_images-1}.png) found in {results_dir}")
        save_cache(cache_file, {"average": None, "per_image": {}, "count": 0}, gt_mtime, mask_mtime)
        return None  # Return None as 0.0 is a valid LPIPS score

    # Pass loaded model and device to helper
    for i in tqdm(range(num_images), desc=f"LPIPS Processing {results_dir_name}", leave=False):
        result_img_name = f"{i}.png"
        result_img_path = results_dir / result_img_name

        if result_img_path.is_file():
            try:
                gen_image_bgr = cv2.imread(str(result_img_path))
                if gen_image_bgr is None:
                    print(f"Warning: Failed to load generated image {result_img_name}. Skipping.")
                    per_image_scores[result_img_name] = None
                    continue

                gen_image_rgb = cv2.cvtColor(gen_image_bgr, cv2.COLOR_BGR2RGB)

                if target_size:
                    if gen_image_rgb.shape[:2] != target_size[::-1]:
                        gen_image_rgb = cv2.resize(
                            gen_image_rgb, target_size, interpolation=cv2.INTER_LINEAR
                        )

                gen_tensor = img_to_tensor(gen_image_rgb)
                gen_tensor_norm = normalize_tensor(gen_tensor).unsqueeze(0).to(device)

                lpips_value = calculate_lpips_masked(
                    gt_tensor_norm, gen_tensor_norm, mask_tensor, loss_fn_lpips, device
                )

                if lpips_value is not None:
                    total_lpips += lpips_value
                    per_image_scores[result_img_name] = lpips_value
                    valid_image_count += 1
                else:
                    per_image_scores[result_img_name] = None  # Mark error

            except Exception as e:
                print(f"Error processing image {result_img_name} for LPIPS: {e}")
                per_image_scores[result_img_name] = None
        else:
            # print(f"Warning: Result image {result_img_name} not found.")
            per_image_scores[result_img_name] = None

    if valid_image_count == 0:
        print(f"Error: No valid result images processed for Masked LPIPS in {results_dir}")
        save_cache(
            cache_file,
            {"average": None, "per_image": per_image_scores, "count": 0},
            gt_mtime,
            mask_mtime,
        )
        return None

    average_lpips = total_lpips / valid_image_count

    # --- Save to Cache ---
    cache_data = {
        "average": average_lpips,
        "per_image": per_image_scores,
        "count": valid_image_count,
    }
    save_cache(cache_file, cache_data, gt_mtime, mask_mtime)

    return average_lpips


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Masked LPIPS distance for a RealFill result scene."
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

    args = parser.parse_args()

    if not lpips_available:
        print("Exiting because LPIPS library is not installed.")
        print("FINAL_SCORE:ERROR")
        exit(1)

    current_target_size = TARGET_SIZE

    # Ensure cache subdirectories exist
    lpips_cache_dir = Path(args.cache_dir) / "per_scene_cache" / "lpips_masked"
    lpips_cache_dir.mkdir(parents=True, exist_ok=True)

    # calculate_scene_lpips handles model loading internally
    avg_score = calculate_scene_lpips(
        args.gt_path,
        args.mask_path,
        args.results_dir,
        args.cache_dir,
        args.num_images,
        target_size=current_target_size,
    )

    # Remember: Lower LPIPS is better
    if avg_score is not None:
        print(
            f"\nAverage Masked LPIPS ({LPIPS_NET}) for {Path(args.results_dir).name}: {avg_score:.4f}"
        )
        print(f"FINAL_SCORE:{avg_score:.8f}")
    else:
        print(f"\nFailed to calculate Masked LPIPS for {Path(args.results_dir).name}")
        print("FINAL_SCORE:ERROR")
