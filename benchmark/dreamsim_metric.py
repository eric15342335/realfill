# benchmark/dreamsim_metric.py

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
CACHE_VERSION = "1.1"  # Keep incremented if cache structure for DreamSim changes
METRIC_NAME = "dreamsim"  # Specific to this metric

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
        # DreamSim doesn't use mask for calculation, but mtime check kept for consistency.
        if mask_mtime_current and metadata.get("mask_mtime") != mask_mtime_current:
            return None

        if cache_data.get("average") is None:  # Important: ensure a score was actually cached
            return None

        return cache_data
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate DreamSim distance with early cache exit."
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
        from tqdm import tqdm

        # Attempt to import dreamsim specific library
        try:
            from dreamsim import dreamsim as dreamsim_model_loader

            dreamsim_library_available = True
        except ImportError:
            print("Error: dreamsim library not found. Please install it: pip install dreamsim")
            dreamsim_library_available = False
            dreamsim_model_loader = None  # Ensure it's None
            # Exit here if the library is mandatory for the heavy path
            print("FINAL_SCORE:ERROR")
            sys.exit(1)

    except ImportError as e:  # Catch import errors for torch, PIL, tqdm
        print(f"Error: Missing general heavy libraries for DreamSim calculation: {e}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    # --- Global Variables for Model (Heavy Path) ---
    heavy_model = None
    heavy_preprocess = None
    heavy_device = None
    heavy_model_loaded = False

    def _load_model_if_needed_heavy():
        global heavy_model, heavy_preprocess, heavy_device, heavy_model_loaded, dreamsim_library_available
        if heavy_model_loaded:
            return True
        if not dreamsim_library_available or dreamsim_model_loader is None:  # Double check
            return False
        try:
            heavy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # dreamsim_model_loader returns model, preprocess
            model_instance, preprocess_instance = dreamsim_model_loader(
                pretrained=True, device=heavy_device
            )
            heavy_model = model_instance.eval()
            heavy_preprocess = preprocess_instance
            heavy_model_loaded = True
            return True
        except Exception as e_load:
            print(f"FATAL Error loading DreamSim model (heavy path): {e_load}")
            traceback.print_exc()
            heavy_model, heavy_preprocess, heavy_device, heavy_model_loaded = (
                None,
                None,
                None,
                False,
            )
            return False

    @torch.no_grad()
    def calculate_dreamsim_distance_heavy(img_path1_str: str, img_path2_str: str) -> float | None:
        if not heavy_model_loaded or heavy_preprocess is None:
            return None
        try:
            image1 = Image.open(img_path1_str).convert("RGB")
            image2 = Image.open(img_path2_str).convert("RGB")
            processed_img1 = heavy_preprocess(image1).to(heavy_device)  # expects [1,C,H,W]
            processed_img2 = heavy_preprocess(image2).to(heavy_device)  # expects [1,C,H,W]

            # DreamSim model expects batched inputs [B, C, H, W]
            # The preprocess should already return it in the correct shape with B=1
            output = heavy_model(processed_img1, processed_img2)

            distance_tensor = None
            if isinstance(output, (tuple, list)) and len(output) > 0:
                distance_tensor = output[0]
            elif torch.is_tensor(output):
                distance_tensor = output

            if not torch.is_tensor(distance_tensor):
                return None
            return distance_tensor.item()

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
        print("Error: Failed to load DreamSim model (heavy path). Cannot proceed.")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    gt_path_obj = Path(args.gt_path)
    results_dir_obj = Path(args.results_dir)
    total_distance_val = 0.0
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
        desc=f"DreamSim Processing {current_results_dir_name}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        distance = calculate_dreamsim_distance_heavy(str(gt_path_obj), str(result_img_path_obj))
        if distance is not None:
            total_distance_val += distance
            per_image_scores_dict[result_img_path_obj.name] = distance
            valid_image_count_val += 1
        else:
            per_image_scores_dict[result_img_path_obj.name] = None

    if valid_image_count_val > 0:
        final_average_score = total_distance_val / valid_image_count_val
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
        # Lower DreamSim distance is better.
        print(f"FINAL_SCORE:{final_average_score:.8f}")
    else:
        print("FINAL_SCORE:ERROR")
