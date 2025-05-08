# benchmark/clip_metric.py

# --- Lightweight Imports First ---
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# --- Constants (Lightweight) ---
DEFAULT_NUM_IMAGES = 16
CACHE_VERSION = "1.0"  # Version for the cache structure
METRIC_NAME = "clip"  # Specific to this metric, used for cache path


# --- Lightweight Helper Functions for Cache Handling ---
def get_cache_path_light(
    cache_dir_str: str, results_dir_name_str: str, metric_name_str: str
) -> Path:
    """Constructs the cache file path using only basic types."""
    return (
        Path(cache_dir_str) / "per_scene_cache" / metric_name_str / f"{results_dir_name_str}.json"
    )


def get_modification_time_light(file_path_str: str) -> float:
    """Gets file modification time; returns 0 if file not found."""
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
    """
    Loads and validates cache using only lightweight operations.
    Returns cache data if valid and contains a non-None 'average' score, otherwise None.
    """
    if not cache_file_path.is_file():
        return None
    try:
        with open(cache_file_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # Validate cache structure and version
        metadata = cache_data.get("metadata")
        if (
            not isinstance(cache_data, dict)
            or "average" not in cache_data
            or not isinstance(metadata, dict)
            or metadata.get("cache_version") != expected_cache_version
        ):
            return None  # Invalid format or version

        # Validate modification times
        if gt_mtime_current and metadata.get("gt_mtime") != gt_mtime_current:
            return None  # GT changed
        # CLIP doesn't use mask for calculation, but mtime check can be kept for consistency
        # if your general caching policy relies on it.
        if mask_mtime_current and metadata.get("mask_mtime") != mask_mtime_current:
            return None  # Mask changed

        # Ensure 'average' score exists and is not None (indicating a successful previous run)
        if cache_data.get("average") is None:
            return None  # Previous run might have errored or had no valid images

        return cache_data
    except (json.JSONDecodeError, KeyError, OSError):
        return None  # Error during cache loading or validation


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP similarity with early cache exit.")
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to the ground truth image."
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the mask image (for mtime validation).",
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

    # --- Perform Early Cache Check ---
    current_results_dir_name = Path(args.results_dir).name
    cache_file = get_cache_path_light(args.cache_dir, current_results_dir_name, METRIC_NAME)
    current_gt_mtime = get_modification_time_light(args.gt_path)
    current_mask_mtime = get_modification_time_light(args.mask_path)

    cached_data = load_cache_light(cache_file, current_gt_mtime, current_mask_mtime, CACHE_VERSION)

    if cached_data:
        cached_avg_score = cached_data.get("average")
        if cached_avg_score is not None:
            # print(f"CLIP: Early cache hit for {current_results_dir_name}. Score: {cached_avg_score:.8f}") # For debugging
            print(f"FINAL_SCORE:{cached_avg_score:.8f}")
            sys.exit(0)

    # --- If NO early cache hit, proceed with HEAVY imports and full logic ---
    # print(f"CLIP: Cache miss or invalid for {current_results_dir_name}. Proceeding with full calculation.") # For debugging

    try:
        import torch
        from PIL import Image
        from tqdm import tqdm
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        print(f"Error: Missing heavy libraries for CLIP calculation: {e}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    # --- Global Variables for Model (initialized for the heavy computation part) ---
    heavy_model = None
    heavy_processor = None
    heavy_device = None
    heavy_model_loaded = False

    def _load_model_if_needed_heavy():
        """Loads the CLIP model and processor if not already loaded."""
        global heavy_model, heavy_processor, heavy_device, heavy_model_loaded
        if heavy_model_loaded:
            return True
        try:
            # print("CLIP: Loading CLIP model (heavy path)...") # For debugging
            heavy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            heavy_model = (
                CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(heavy_device).eval()
            )
            heavy_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            heavy_model_loaded = True
            # print(f"CLIP: Model loaded on {heavy_device} (heavy path).") # For debugging
            return True
        except Exception as e_load:
            print(f"FATAL Error loading CLIP model (heavy path): {e_load}")
            traceback.print_exc()
            heavy_model, heavy_processor, heavy_device, heavy_model_loaded = None, None, None, False
            return False

    @torch.no_grad()
    def calculate_clip_similarity_heavy(img_path1_str: str, img_path2_str: str) -> float | None:
        """Calculates CLIP cosine similarity between two images using loaded heavy models."""
        if not heavy_model_loaded:
            # print("CLIP model/processor not available for similarity calculation (heavy path).") # For debugging
            return None
        try:
            image1 = Image.open(img_path1_str).convert("RGB")
            image2 = Image.open(img_path2_str).convert("RGB")

            inputs1 = heavy_processor(
                images=image1, return_tensors="pt", padding=True, truncation=True
            ).to(heavy_device)
            inputs2 = heavy_processor(
                images=image2, return_tensors="pt", padding=True, truncation=True
            ).to(heavy_device)

            features1 = heavy_model.get_image_features(**inputs1)
            features2 = heavy_model.get_image_features(**inputs2)

            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            similarity = torch.nn.functional.cosine_similarity(features1, features2)
            return similarity.item()
        except FileNotFoundError:
            # print(f"Warning: Could not find image file (heavy path): {img_path1_str} or {img_path2_str}") # For debugging
            return None
        except Exception as e_calc:
            # print(f"Error calculating CLIP similarity (heavy path) between {Path(img_path1_str).name} and {Path(img_path2_str).name}: {e_calc}") # For debugging
            return None

    def save_cache_heavy(
        cache_file_path: Path,
        data_to_save: dict,
        gt_mtime: float,
        mask_mtime: float,
        current_cache_version: str,
    ):
        """Saves computed results to the cache file."""
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
            # print(f"CLIP: Saved cache to {cache_file_path} (heavy path).") # For debugging
        except Exception as e_save:
            print(f"Error saving cache (heavy path) to {cache_file_path}: {e_save}")

    # --- Main Calculation Logic (Heavy Path) ---
    final_average_score = None
    if not _load_model_if_needed_heavy():
        print("Error: Failed to load CLIP model (heavy path). Cannot proceed.")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    # print(f"CLIP: Calculating for scene {current_results_dir_name} (heavy path).") # For debugging
    gt_path_obj = Path(args.gt_path)
    results_dir_obj = Path(args.results_dir)

    total_similarity_val = 0.0
    valid_image_count_val = 0
    per_image_scores_dict = {}

    # Limit to the first num_images, ensure they are digits
    image_paths_to_process = []
    for i in range(args.num_images):
        img_p = results_dir_obj / f"{i}.png"
        if img_p.is_file():  # Check if 0.png, 1.png etc. exist
            image_paths_to_process.append(img_p)
        else:  # Image not found, still record as None for per_image
            per_image_scores_dict[f"{i}.png"] = None

    if not image_paths_to_process and args.num_images > 0:
        # print(f"Warning: No images (0..N-1.png) found in {results_dir_obj} for CLIP calc.") # For debugging
        # Save an empty/error cache entry if no images were found to process
        save_cache_heavy(
            cache_file,
            {"average": None, "per_image": per_image_scores_dict, "count": 0},
            current_gt_mtime,
            current_mask_mtime,
            CACHE_VERSION,
        )
        print("FINAL_SCORE:ERROR")  # Or a specific score like 0.0 if that's preferred for no images
        sys.exit(0)

    for result_img_path_obj in tqdm(
        image_paths_to_process,
        desc=f"CLIP Processing {current_results_dir_name}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        similarity = calculate_clip_similarity_heavy(str(gt_path_obj), str(result_img_path_obj))
        if similarity is not None:
            total_similarity_val += similarity
            per_image_scores_dict[result_img_path_obj.name] = similarity
            valid_image_count_val += 1
        else:
            per_image_scores_dict[result_img_path_obj.name] = None  # Record error for this image

    if valid_image_count_val > 0:
        final_average_score = total_similarity_val / valid_image_count_val
        cache_data_to_save = {
            "average": final_average_score,
            "per_image": per_image_scores_dict,
            "count": valid_image_count_val,
        }
        save_cache_heavy(
            cache_file, cache_data_to_save, current_gt_mtime, current_mask_mtime, CACHE_VERSION
        )
    else:  # No valid images were successfully processed
        # print(f"Warning: No images successfully processed for CLIP in {results_dir_obj}.") # For debugging
        # Save cache with None average if all image calculations failed or no images processed
        save_cache_heavy(
            cache_file,
            {"average": None, "per_image": per_image_scores_dict, "count": 0},
            current_gt_mtime,
            current_mask_mtime,
            CACHE_VERSION,
        )

    # --- Output Final Score ---
    if final_average_score is not None:
        # print(f"\nAverage CLIP Similarity for {current_results_dir_name}: {final_average_score:.4f}") # For debugging
        print(f"FINAL_SCORE:{final_average_score:.8f}")
    else:
        # print(f"\nFailed to calculate CLIP similarity for {current_results_dir_name} (heavy path).") # For debugging
        print("FINAL_SCORE:ERROR")
