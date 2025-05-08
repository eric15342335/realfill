# benchmark/dino_metric.py

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
METRIC_NAME = "dino"  # Specific to this metric, used for cache path

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
        # DINO doesn't use mask for calculation, but mtime check kept for consistency.
        if mask_mtime_current and metadata.get("mask_mtime") != mask_mtime_current:
            return None

        if cache_data.get("average") is None:
            return None

        return cache_data
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate DINO CLS token similarity with early cache exit."
    )
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
        import torch
        from PIL import Image
        from torch.nn import functional as F_torch  # Alias to avoid conflict if F is used elsewhere
        from torchvision import transforms
        from tqdm import tqdm
        from transformers import ViTModel
    except ImportError as e:
        print(f"Error: Missing heavy libraries for DINO calculation: {e}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    # --- Global Variables for Model and Transforms (Heavy Path) ---
    heavy_model = None
    heavy_device = None
    heavy_model_loaded = False
    dino_transform = None  # Will be initialized in _load_model_if_needed_heavy

    def _load_model_if_needed_heavy():
        global heavy_model, heavy_device, heavy_model_loaded, dino_transform
        if heavy_model_loaded:
            return True
        try:
            heavy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            heavy_model = ViTModel.from_pretrained("facebook/dino-vits16").to(heavy_device).eval()

            # Standard DINOv1/v2 transforms
            dino_transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            heavy_model_loaded = True
            return True
        except Exception as e_load:
            print(f"FATAL Error loading DINO model or defining transforms (heavy path): {e_load}")
            traceback.print_exc()
            heavy_model, heavy_device, heavy_model_loaded, dino_transform = None, None, False, None
            return False

    @torch.no_grad()
    def calculate_dino_similarity_heavy(img_path1_str: str, img_path2_str: str) -> float | None:
        if not heavy_model_loaded or dino_transform is None:
            return None
        try:
            img1 = Image.open(img_path1_str).convert("RGB")
            img2 = Image.open(img_path2_str).convert("RGB")

            t_img1 = dino_transform(img1)
            t_img2 = dino_transform(img2)
            inputs = torch.stack([t_img1, t_img2]).to(heavy_device)
            outputs = heavy_model(inputs)
            last_hidden_states = outputs.last_hidden_state
            emb_img1 = last_hidden_states[0, 0]  # CLS token for image 1
            emb_img2 = last_hidden_states[1, 0]  # CLS token for image 2
            similarity = F_torch.cosine_similarity(emb_img1, emb_img2, dim=0)
            return similarity.item()
        except FileNotFoundError:
            return None
        except Exception:
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
            print(f"Error saving cache (heavy path) to {cache_file_path}: {e_save}")

    # --- Main Calculation Logic (Heavy Path) ---
    final_average_score = None
    if not _load_model_if_needed_heavy():
        print("Error: Failed to load DINO model (heavy path). Cannot proceed.")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    gt_path_obj = Path(args.gt_path)
    results_dir_obj = Path(args.results_dir)
    total_similarity_val = 0.0
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
        print("FINAL_SCORE:ERROR")
        sys.exit(0)

    for result_img_path_obj in tqdm(
        image_paths_to_process,
        desc=f"DINO Processing {current_results_dir_name}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        similarity = calculate_dino_similarity_heavy(str(gt_path_obj), str(result_img_path_obj))
        if similarity is not None:
            total_similarity_val += similarity
            per_image_scores_dict[result_img_path_obj.name] = similarity
            valid_image_count_val += 1
        else:
            per_image_scores_dict[result_img_path_obj.name] = None

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
    else:
        save_cache_heavy(
            cache_file,
            {"average": None, "per_image": per_image_scores_dict, "count": 0},
            current_gt_mtime,
            current_mask_mtime,
            CACHE_VERSION,
        )

    if final_average_score is not None:
        print(f"FINAL_SCORE:{final_average_score:.8f}")
    else:
        print("FINAL_SCORE:ERROR")
