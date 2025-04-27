# benchmark/dino_metric.py

import torch
import argparse
import json
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from transformers import ViTModel
from tqdm import tqdm
import time
import traceback # Added for better error logging

# --- Constants ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"  # Increment if caching logic changes

# --- DINO Transforms (Define globally, they are lightweight) ---
try:
    # Standard DINOv1/v2 transforms (ViT-S/16 from facebook/dino-vits16 expects 224)
    T = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # Follows DINO paper
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    print("DINO transforms defined.")
except Exception as e:
    print(f"Error defining transforms: {e}")
    T = None

# --- Model Variables (Initialize to None) ---
model = None
device = None
model_loaded = False # Flag to track loading state

# --- Helper Functions ---

def _load_model_if_needed():
    """Loads the DINO model once, only when needed."""
    global model, device, model_loaded
    if model_loaded:
        return True # Already loaded successfully

    if model is not None and device is not None:
        model_loaded = True # Should be caught by flag, but safe check
        return True

    print("Attempting to load DINO model (facebook/dino-vits16)...")
    try:
        # Using ViT-S/16 as in the original script
        _model = ViTModel.from_pretrained("facebook/dino-vits16")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()  # Set to evaluation mode

        # Assign to global variables only after successful loading
        model = _model
        device = _device
        model_loaded = True
        print(f"DINO model loaded successfully on {device}.")
        return True
    except Exception as e:
        print(f"FATAL Error loading DINO model: {e}")
        traceback.print_exc()
        # Ensure globals remain None on failure
        model = None
        device = None
        model_loaded = False
        return False


def get_modification_time(file_path):
    """Gets the modification time of a file."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0


def calculate_dino_similarity(img_path1, img_path2, model, transform, device):
    """Calculates DINO CLS token cosine similarity between two images."""
    # Safeguard checks inside the loop
    if not model or not transform:
        print("DINO model/transform not available during similarity calculation.")
        return 0.0

    try:
        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        # Apply transforms
        t_img1 = transform(img1)
        t_img2 = transform(img2)

        # Add batch dimension
        inputs = torch.stack([t_img1, t_img2]).to(device)  # Batch size = 2

        with torch.no_grad():
            outputs = model(inputs)

        # Extract CLS tokens
        last_hidden_states = outputs.last_hidden_state
        emb_img1 = last_hidden_states[0, 0]  # CLS token for image 1
        emb_img2 = last_hidden_states[1, 0]  # CLS token for image 2

        # Calculate cosine similarity
        similarity = F.cosine_similarity(emb_img1, emb_img2, dim=0)
        return similarity.item()

    except FileNotFoundError:
        print(f"Warning: Could not find image file: {img_path1} or {img_path2}")
        return 0.0
    except RuntimeError as e:
        print(f"RuntimeError during DINO calculation for {Path(img_path1).name}/{Path(img_path2).name}: {e}")
        return 0.0
    except Exception as e:
        print(f"Error calculating DINO similarity between {Path(img_path1).name} and {Path(img_path2).name}: {e}")
        # traceback.print_exc() # Optional
        return 0.0


def get_cache_path(cache_dir, results_dir_name):
    """Constructs the path for the cache file."""
    return Path(cache_dir) / "per_scene_cache" / "dino" / f"{results_dir_name}.json"


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

        if gt_mtime_current and cache_data["metadata"].get("gt_mtime") != gt_mtime_current:
            print(f"Ground truth mtime changed for {cache_file.name}. Recalculating.")
            return None
        # No mask used in calculation, but keep check for consistency if desired
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
def calculate_scene_dino(gt_path_str, mask_path_str, results_dir_str, cache_dir_str, num_images=DEFAULT_NUM_IMAGES):
    """
    Calculates the average DINO CLS token similarity for a given scene.
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
    if T is None: # Check if transforms loaded
        print("Error: DINO transforms failed to define. Cannot proceed.")
        return None

    results_dir_name = results_dir.name
    cache_file = get_cache_path(cache_dir, results_dir_name)

    # --- Cache Check ---
    gt_mtime = get_modification_time(gt_path)
    mask_mtime = get_modification_time(mask_path)
    cached_results = load_cache(cache_file, gt_mtime, mask_mtime)
    if cached_results:
        return cached_results.get("average")

    # --- Model Loading (Only if needed) ---
    if not _load_model_if_needed():
        print("Error: Failed to load DINO model. Cannot proceed.")
        return None

    # --- Calculation ---
    print(f"Calculating DINO for scene: {results_dir_name}")
    total_similarity = 0.0
    valid_image_count = 0
    per_image_scores = {}

    image_files = sorted([p for p in results_dir.glob("*.png") if p.stem.isdigit()], key=lambda x: int(x.stem))
    image_files = image_files[:num_images]

    if not image_files:
        print(f"Warning: No valid result images (0..{num_images-1}.png) found in {results_dir}")
        save_cache(cache_file, {"average": 0.0, "per_image": {}, "count": 0}, gt_mtime, mask_mtime)
        return 0.0

    # Pass loaded model, T, device to helper
    for i in tqdm(range(num_images), desc=f"DINO Processing {results_dir_name}", leave=False):
        result_img_name = f"{i}.png"
        result_img_path = results_dir / result_img_name

        if result_img_path.is_file():
            similarity = calculate_dino_similarity(gt_path, result_img_path, model, T, device)
            if similarity is not None:  # Check for successful calculation
                total_similarity += similarity
                per_image_scores[result_img_name] = similarity
                valid_image_count += 1
            else:
                 print(f"Skipping DINO for {result_img_name} due to error (returned None).")
                 per_image_scores[result_img_name] = None # Mark error

        else:
            # print(f"Warning: Result image {result_img_name} not found in {results_dir}")
            per_image_scores[result_img_name] = None

    if valid_image_count == 0:
        print(f"Error: No valid result images processed for DINO similarity in {results_dir}")
        save_cache(cache_file, {"average": None, "per_image": per_image_scores, "count": 0}, gt_mtime, mask_mtime)
        return None

    average_similarity = total_similarity / valid_image_count

    # --- Save to Cache ---
    cache_data = {"average": average_similarity, "per_image": per_image_scores, "count": valid_image_count}
    save_cache(cache_file, cache_data, gt_mtime, mask_mtime)

    return average_similarity


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DINO CLS similarity for a RealFill result scene.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth image.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image (for cache validation).")
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

    args = parser.parse_args()

    # Ensure cache subdirectories exist
    dino_cache_dir = Path(args.cache_dir) / "per_scene_cache" / "dino"
    dino_cache_dir.mkdir(parents=True, exist_ok=True)

    # calculate_scene_dino will handle model loading internally
    avg_score = calculate_scene_dino(args.gt_path, args.mask_path, args.results_dir, args.cache_dir, args.num_images)

    if avg_score is not None:
        print(f"\nAverage DINO Similarity for {Path(args.results_dir).name}: {avg_score:.4f}")
        print(f"FINAL_SCORE:{avg_score:.8f}")
    else:
        print(f"\nFailed to calculate DINO similarity for {Path(args.results_dir).name}")
        print("FINAL_SCORE:ERROR")