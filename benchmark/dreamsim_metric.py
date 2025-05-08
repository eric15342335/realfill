# benchmark/dreamsim_metric.py

import torch
import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import traceback  # Import traceback for better error printing

# Ensure the dreamsim library is installed: pip install dreamsim
try:
    from dreamsim import dreamsim as dreamsim_model_loader

    dreamsim_available = True
except ImportError:
    print("Error: dreamsim library not found. Please install it: pip install dreamsim")
    dreamsim_model_loader = None
    dreamsim_available = False

# --- Constants ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.1"  # Keep incremented cache version

# --- Model Variables (Initialize to None) ---
model = None
preprocess = None
device = None
model_loaded = False  # Flag to track loading state

# --- Helper Functions ---


def _load_model_if_needed():
    """Loads the DreamSim model once, only when needed."""
    global model, preprocess, device, model_loaded, dreamsim_available
    if model_loaded:
        return True  # Already loaded successfully

    if not dreamsim_available:
        print("DreamSim library not available, cannot load model.")
        return False

    if model is not None and preprocess is not None and device is not None:
        model_loaded = True  # Already loaded
        return True

    print("Attempting to load DreamSim model...")
    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dreamsim() returns model, preprocess
        _model, _preprocess = dreamsim_model_loader(pretrained=True, device=_device)
        _model.eval()  # Set to evaluation mode

        # Assign to global variables only after successful loading
        model = _model
        preprocess = _preprocess
        device = _device
        model_loaded = True
        print(f"DreamSim model loaded successfully on {device}.")
        return True
    except Exception as e:
        print(f"FATAL Error loading DreamSim model: {e}")
        traceback.print_exc()
        # Ensure globals remain None on failure
        model = None
        preprocess = None
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
def calculate_dreamsim_distance(img_path1, img_path2, model, preprocess, device):
    """Calculates DreamSim distance between two images."""
    # Safeguard checks
    if not model or not preprocess or not device:
        print("DreamSim model/preprocess/device not available during distance calculation.")
        return None  # Use None to indicate error

    try:
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")

        # Preprocess images
        processed_img1 = preprocess(image1).to(device)
        processed_img2 = preprocess(image2).to(device)

        # --- Verify shape AFTER preprocess ---
        expected_shape = torch.Size([1, 3, 224, 224])  # Assuming DreamSim standard
        if processed_img1.shape != expected_shape or processed_img2.shape != expected_shape:
            print(
                f"ERROR: Unexpected shape after preprocess. img1: {processed_img1.shape}, img2: {processed_img2.shape}"
            )
            return None  # Error out

        batched_img1 = processed_img1
        batched_img2 = processed_img2

        # --- Model call and output handling ---
        output = model(batched_img1, batched_img2)

        if isinstance(output, (tuple, list)):
            distance_tensor = output[0] if len(output) > 0 else None
        elif torch.is_tensor(output):
            distance_tensor = output
        else:
            distance_tensor = None

        if not torch.is_tensor(distance_tensor):
            print(
                f"Error: DreamSim model returned unexpected output or couldn't extract tensor. Type: {type(output)}, Raw: {str(output)[:100]}"
            )
            return None

        # Return the scalar distance value
        return distance_tensor.item()

    except FileNotFoundError:
        print(f"Warning: Could not find image file: {img_path1} or {img_path2}")
        return None
    except Exception as e:
        print(
            f"Error calculating DreamSim distance between {Path(img_path1).name} and {Path(img_path2).name}: {e}"
        )
        traceback.print_exc()  # Print full traceback
        return None


def get_cache_path(cache_dir, results_dir_name):
    """Constructs the path for the cache file."""
    return Path(cache_dir) / "per_scene_cache" / "dreamsim" / f"{results_dir_name}.json"


def load_cache(cache_file, gt_mtime_current, mask_mtime_current):
    """Loads results from cache if valid."""
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        # Validation - check cache version
        if (
            not isinstance(cache_data, dict)
            or "average" not in cache_data
            or "per_image" not in cache_data
            or "metadata" not in cache_data
            or cache_data.get("metadata", {}).get("cache_version") != CACHE_VERSION
        ):
            print(
                f"Cache file {cache_file} format invalid or version mismatch (Expected: {CACHE_VERSION}). Recalculating."
            )
            return None

        if gt_mtime_current and cache_data["metadata"].get("gt_mtime") != gt_mtime_current:
            print(f"Ground truth mtime changed for {cache_file.name}. Recalculating.")
            return None
        # Mask not used, but check for consistency if needed
        # if mask_mtime_current and cache_data["metadata"].get("mask_mtime") != mask_mtime_current:
        #     print(f"Mask mtime changed for {cache_file.name}. Recalculating.")
        #     return None

        # Don't print cache hit here, main function will
        # print(f"Cache hit for {cache_file.name}") # Removed
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
            "mask_mtime": mask_mtime,
            "cache_version": CACHE_VERSION,  # Save current cache version
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving cache to {cache_file}: {e}")


# --- Main Function ---
def calculate_scene_dreamsim(
    gt_path_str, mask_path_str, results_dir_str, cache_dir_str, num_images=DEFAULT_NUM_IMAGES
):
    """
    Calculates the average DreamSim distance for a given scene.
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
    if not results_dir.is_dir():
        print(f"Error: Results directory not found at {results_dir}")
        return None

    results_dir_name = results_dir.name
    cache_file = get_cache_path(cache_dir, results_dir_name)

    # --- Cache Check ---
    gt_mtime = get_modification_time(gt_path)
    mask_mtime = get_modification_time(mask_path)
    cached_results = load_cache(cache_file, gt_mtime, mask_mtime)
    if cached_results:
        # Check if cached average is valid (not None)
        avg_cache = cached_results.get("average")
        if avg_cache is not None:
            print(f"Cache hit for {results_dir_name}")  # Print hit here
            return avg_cache
        else:
            print(f"Cached average for {results_dir_name} was None. Recalculating.")
            # Proceed to recalculate

    # --- Model Loading (Only if needed) ---
    if not _load_model_if_needed():
        print("Error: Failed to load DreamSim model or library unavailable. Cannot proceed.")
        return None

    # --- Calculation ---
    print(f"Calculating DreamSim for scene: {results_dir_name}")
    total_distance = 0.0
    valid_image_count = 0
    per_image_scores = {}  # Store distances here

    image_files = sorted(
        [p for p in results_dir.glob("*.png") if p.stem.isdigit()], key=lambda x: int(x.stem)
    )
    image_files = image_files[:num_images]

    if not image_files:
        print(f"Warning: No valid result images (0..{num_images-1}.png) found in {results_dir}")
        save_cache(cache_file, {"average": None, "per_image": {}, "count": 0}, gt_mtime, mask_mtime)
        return None

    # Pass loaded model, preprocess, device to helper
    for i in tqdm(range(num_images), desc=f"DreamSim Processing {results_dir_name}", leave=False):
        result_img_name = f"{i}.png"
        result_img_path = results_dir / result_img_name

        if result_img_path.is_file():
            distance = calculate_dreamsim_distance(
                gt_path, result_img_path, model, preprocess, device
            )
            if distance is not None:  # Check calculation success
                total_distance += distance
                per_image_scores[result_img_name] = distance  # Store distance
                valid_image_count += 1
            else:
                per_image_scores[result_img_name] = None  # Mark error
        else:
            per_image_scores[result_img_name] = None

    if valid_image_count == 0:
        print(f"Error: No valid result images processed for DreamSim distance in {results_dir}")
        save_cache(
            cache_file,
            {"average": None, "per_image": per_image_scores, "count": 0},
            gt_mtime,
            mask_mtime,
        )
        return None

    average_distance = total_distance / valid_image_count

    # --- Save to Cache ---
    cache_data = {
        "average": average_distance,
        "per_image": per_image_scores,
        "count": valid_image_count,
    }
    save_cache(cache_file, cache_data, gt_mtime, mask_mtime)

    return average_distance


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate DreamSim distance for a RealFill result scene."
    )
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to the ground truth image."
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the mask image (for cache validation).",
    )
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

    # Check if library is available *before* trying to calculate
    if not dreamsim_available:
        print("Exiting because DreamSim library is not installed.")
        print("FINAL_SCORE:ERROR")
        exit(1)

    # Ensure cache subdirectories exist
    dreamsim_cache_dir = Path(args.cache_dir) / "per_scene_cache" / "dreamsim"
    dreamsim_cache_dir.mkdir(parents=True, exist_ok=True)

    # calculate_scene_dreamsim handles model loading internally
    avg_score = calculate_scene_dreamsim(
        args.gt_path, args.mask_path, args.results_dir, args.cache_dir, args.num_images
    )

    # Remember: Lower DreamSim distance is better
    if avg_score is not None:
        print(f"\nAverage DreamSim Distance for {Path(args.results_dir).name}: {avg_score:.4f}")
        print(f"FINAL_SCORE:{avg_score:.8f}")
    else:
        print(f"\nFailed to calculate DreamSim distance for {Path(args.results_dir).name}")
        print("FINAL_SCORE:ERROR")  # Print error if calculation failed (e.g., model load failed)
