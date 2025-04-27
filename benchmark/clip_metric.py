# benchmark/clip_metric.py

import torch
import argparse
import json
import os
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import time
import traceback # Added for better error logging

# --- Constants ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"  # Increment if caching logic changes

# --- Model Variables (Initialize to None) ---
model = None
processor = None
device = None
model_loaded = False # Flag to track loading state

# --- Helper Functions ---

def _load_model_if_needed():
    """Loads the CLIP model and processor once, only when needed."""
    global model, processor, device, model_loaded
    if model_loaded:
        return True # Already loaded successfully

    if model is not None and processor is not None and device is not None:
        model_loaded = True # Already loaded (perhaps by another call, unlikely here but safe)
        return True

    print("Attempting to load CLIP model and processor...")
    try:
        # Consider using a specific version if reproducibility is critical
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()  # Set to evaluation mode

        # Assign to global variables only after successful loading
        model = _model
        processor = _processor
        device = _device
        model_loaded = True
        print(f"CLIP model loaded successfully on {device}.")
        return True
    except Exception as e:
        print(f"FATAL Error loading CLIP model: {e}")
        traceback.print_exc()
        # Ensure globals remain None on failure
        model = None
        processor = None
        device = None
        model_loaded = False
        return False


def get_modification_time(file_path):
    """Gets the modification time of a file."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0  # File likely doesn't exist


def calculate_clip_similarity(img_path1, img_path2, model, processor, device):
    """Calculates CLIP cosine similarity between two images."""
    # This check remains as a safeguard within the loop
    if not model or not processor:
        print("CLIP model/processor not available during similarity calculation.")
        return 0.0

    try:
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")

        # Ensure processor and device are valid before use
        if processor is None or device is None:
             print("CLIP processor or device is None during similarity calculation.")
             return 0.0

        inputs1 = processor(images=image1, return_tensors="pt", padding=True, truncation=True).to(device)
        inputs2 = processor(images=image2, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            features1 = model.get_image_features(**inputs1)
            features2 = model.get_image_features(**inputs2)

            # Normalize features
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(features1, features2)
            return similarity.item()

    except FileNotFoundError:
        print(f"Warning: Could not find image file: {img_path1} or {img_path2}")
        return 0.0  # Or handle as an error
    except Exception as e:
        print(f"Error calculating CLIP similarity between {Path(img_path1).name} and {Path(img_path2).name}: {e}")
        # traceback.print_exc() # Optional: more detail
        return 0.0  # Or raise


def get_cache_path(cache_dir, results_dir_name):
    """Constructs the path for the cache file."""
    return Path(cache_dir) / "per_scene_cache" / "clip" / f"{results_dir_name}.json"


def load_cache(cache_file, gt_mtime_current, mask_mtime_current):
    """Loads results from cache if valid."""
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        # Basic validation
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
        # Mask mtime check isn't strictly needed for CLIP full image, but good practice
        # if mask_mtime_current and cache_data["metadata"].get("mask_mtime") != mask_mtime_current:
        #     print(f"Mask mtime changed for {cache_file.name}. Recalculating.")
        #     return None

        print(f"Cache hit for {cache_file.name}")
        return cache_data
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error loading cache file {cache_file}: {e}. Recalculating.")
        return None


def save_cache(cache_file, data, gt_mtime, mask_mtime):
    """Saves results to the cache file."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Add metadata
        data["metadata"] = {
            "timestamp": time.time(),
            "gt_mtime": gt_mtime,
            "mask_mtime": mask_mtime,
            "cache_version": CACHE_VERSION,
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4)
        # print(f"Saved cache to {cache_file}")
    except Exception as e:
        print(f"Error saving cache to {cache_file}: {e}")


# --- Main Function ---
def calculate_scene_clip(gt_path_str, mask_path_str, results_dir_str, cache_dir_str, num_images=DEFAULT_NUM_IMAGES):
    """
    Calculates the average CLIP similarity for a given scene (results directory).
    Loads the model only if needed.
    """
    gt_path = Path(gt_path_str)
    mask_path = Path(mask_path_str)  # Included for consistency
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
        return cached_results.get("average")  # Return cached average

    # --- Model Loading (Only if needed) ---
    if not _load_model_if_needed():
        print("Error: Failed to load CLIP model. Cannot proceed.")
        return None # Model loading failed

    # --- Calculation ---
    print(f"Calculating CLIP for scene: {results_dir_name}")
    total_similarity = 0.0
    valid_image_count = 0
    per_image_scores = {}

    image_files = sorted([p for p in results_dir.glob("*.png") if p.stem.isdigit()], key=lambda x: int(x.stem))
    image_files = image_files[:num_images]  # Limit to the first num_images

    if not image_files:
        print(f"Warning: No valid result images (0..{num_images-1}.png) found in {results_dir}")
        # Save empty cache to prevent recalculation attempts on invalid folders
        save_cache(cache_file, {"average": 0.0, "per_image": {}, "count": 0}, gt_mtime, mask_mtime)
        return 0.0 # Or None, depending on desired behavior for empty folders

    # Use tqdm for progress bar
    # Pass the now loaded model, processor, device to the helper
    for i in tqdm(range(num_images), desc=f"CLIP Processing {results_dir_name}", leave=False):
        result_img_name = f"{i}.png"
        result_img_path = results_dir / result_img_name

        if result_img_path.is_file():
            similarity = calculate_clip_similarity(gt_path, result_img_path, model, processor, device)
            if similarity is not None:  # Check if calculation was successful
                total_similarity += similarity
                per_image_scores[result_img_name] = similarity
                valid_image_count += 1
            else: # Handle calculation error if needed
                 print(f"Skipping image {result_img_name} due to calculation error (returned None).")
                 per_image_scores[result_img_name] = None # Mark as error in per-image results

        else: # Optional: Warn about missing specific image numbers
            # print(f"Warning: Result image {result_img_name} not found in {results_dir}")
            per_image_scores[result_img_name] = None # Mark as missing

    if valid_image_count == 0:
        print(f"Error: No valid result images processed for CLIP similarity in {results_dir}")
        save_cache(cache_file, {"average": None, "per_image": per_image_scores, "count": 0}, gt_mtime, mask_mtime)
        return None

    average_similarity = total_similarity / valid_image_count

    # --- Save to Cache ---
    cache_data = {"average": average_similarity, "per_image": per_image_scores, "count": valid_image_count}
    save_cache(cache_file, cache_data, gt_mtime, mask_mtime)

    return average_similarity


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP similarity for a RealFill result scene.")
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to the ground truth image (e.g., .../target/gt.png)."
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the mask image (e.g., .../target/mask.png). Needed for cache validation.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory containing result images (0.png, 1.png...).",
    )
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to the base directory for caching results.")
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of result images per scene to process (default: {DEFAULT_NUM_IMAGES}).",
    )

    args = parser.parse_args()

    # Ensure cache subdirectories exist
    clip_cache_dir = Path(args.cache_dir) / "per_scene_cache" / "clip"
    clip_cache_dir.mkdir(parents=True, exist_ok=True)

    # calculate_scene_clip will handle model loading internally if needed
    avg_score = calculate_scene_clip(args.gt_path, args.mask_path, args.results_dir, args.cache_dir, args.num_images)

    if avg_score is not None:
        print(f"\nAverage CLIP Similarity for {Path(args.results_dir).name}: {avg_score:.4f}")
        # Output average score to stdout for the orchestrator script
        print(f"FINAL_SCORE:{avg_score:.8f}")  # Use a unique prefix
    else:
        print(f"\nFailed to calculate CLIP similarity for {Path(args.results_dir).name}")
        # Indicate failure to the orchestrator
        print("FINAL_SCORE:ERROR")