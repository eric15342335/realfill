# benchmark/lpips_metric.py

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
CACHE_VERSION = "1.0"
METRIC_NAME = "lpips_masked"  # Explicitly denote that it's masked
LPIPS_NET_TYPE = "alex"  # Or "vgg", affects model loaded and potentially results

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
        # Mask mtime IS critical for masked LPIPS
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
        description="Calculate Masked LPIPS distance with early cache exit."
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
    # TARGET_SIZE is fixed for LPIPS internal processing via OpenCV, not a CLI arg for this script version
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
        import torch
        import torchvision.transforms as T_vision  # Alias for torchvision.transforms
        from tqdm import tqdm

        try:
            import lpips

            lpips_library_available = True
        except ImportError:
            print("Error: lpips library not found. Please install it: pip install lpips")
            lpips_library_available = False
            lpips = None
            print("FINAL_SCORE:ERROR")
            sys.exit(1)
    except ImportError as e:
        print(f"Error: Missing general heavy libraries for LPIPS calculation: {e}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    # --- Global Variables for Model (Heavy Path) ---
    heavy_loss_fn_lpips = None
    heavy_device = None
    heavy_model_loaded = False
    # Fixed target size for LPIPS preprocessing
    LPIPS_TARGET_SIZE = (512, 512)

    def _load_model_if_needed_heavy():
        global heavy_loss_fn_lpips, heavy_device, heavy_model_loaded, lpips_library_available
        if heavy_model_loaded:
            return True
        if not lpips_library_available or lpips is None:
            return False
        try:
            heavy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            heavy_loss_fn_lpips = lpips.LPIPS(net=LPIPS_NET_TYPE).to(heavy_device).eval()
            heavy_model_loaded = True
            return True
        except Exception as e_load:
            print(f"FATAL Error loading LPIPS model (heavy path): {e_load}")
            traceback.print_exc()
            heavy_loss_fn_lpips, heavy_device, heavy_model_loaded = None, None, False
            return False

    # Preprocessing functions for heavy path
    img_to_tensor_heavy = T_vision.ToTensor()  # HWC uint8 [0,255] -> CHW float [0,1]

    def normalize_tensor_heavy(tensor_chw_float_01: torch.Tensor) -> torch.Tensor:
        """Normalizes a CHW tensor from [0,1] to [-1,1] as expected by LPIPS."""
        return tensor_chw_float_01 * 2.0 - 1.0

    @torch.no_grad()
    def calculate_lpips_masked_heavy(
        gt_tensor_norm_bchw: torch.Tensor,
        gen_tensor_norm_bchw: torch.Tensor,
        mask_tensor_b1hw: torch.Tensor,
    ) -> float | None:
        if not heavy_model_loaded:
            return None
        try:
            # Mask tensors should be on the same device
            masked_gt = gt_tensor_norm_bchw * mask_tensor_b1hw.to(heavy_device)
            masked_gen = gen_tensor_norm_bchw * mask_tensor_b1hw.to(heavy_device)
            distance = heavy_loss_fn_lpips(masked_gt, masked_gen)
            return distance.item()
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
        print("Error: Failed to load LPIPS model (heavy path). Cannot proceed.")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    try:
        # Load GT and Mask images (once for the scene)
        gt_image_bgr = cv2.imread(args.gt_path)
        mask_image_gray = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_image_bgr is None or mask_image_gray is None:
            print(
                f"Error loading GT/Mask for LPIPS (heavy path). GT: {args.gt_path}, Mask: {args.mask_path}"
            )
            print("FINAL_SCORE:ERROR")
            sys.exit(1)

        gt_image_rgb = cv2.cvtColor(gt_image_bgr, cv2.COLOR_BGR2RGB)

        # Resize GT and Mask
        gt_image_resized_rgb = cv2.resize(
            gt_image_rgb, LPIPS_TARGET_SIZE, interpolation=cv2.INTER_LINEAR
        )
        mask_image_resized_gray = cv2.resize(
            mask_image_gray, LPIPS_TARGET_SIZE, interpolation=cv2.INTER_NEAREST
        )

        # Convert GT to normalized tensor [1, C, H, W], range [-1, 1]
        gt_tensor_chw_01 = img_to_tensor_heavy(gt_image_resized_rgb)
        gt_tensor_norm_bchw = normalize_tensor_heavy(gt_tensor_chw_01).unsqueeze(0).to(heavy_device)

        # Convert Mask to tensor [1, 1, H, W], range [0, 1] (binary)
        mask_tensor_1hw_01 = img_to_tensor_heavy(mask_image_resized_gray.astype(np.float32) / 255.0)
        mask_tensor_b1hw = (
            (mask_tensor_1hw_01 > 0.5).float().unsqueeze(0).to(heavy_device)
        )  # Ensure it's binary and on device

    except Exception as e_prep:
        print(f"Error during GT/Mask preparation for LPIPS (heavy path): {e_prep}")
        print("FINAL_SCORE:ERROR")
        sys.exit(1)

    results_dir_obj = Path(args.results_dir)
    total_lpips_val = 0.0
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
        desc=f"LPIPS Processing {current_results_dir_name}",
        leave=False,
        disable=not sys.stdout.isatty(),
    ):
        try:
            gen_image_bgr = cv2.imread(str(result_img_path_obj))
            if gen_image_bgr is None:
                per_image_scores_dict[result_img_path_obj.name] = None
                continue

            gen_image_rgb = cv2.cvtColor(gen_image_bgr, cv2.COLOR_BGR2RGB)
            gen_image_resized_rgb = cv2.resize(
                gen_image_rgb, LPIPS_TARGET_SIZE, interpolation=cv2.INTER_LINEAR
            )

            gen_tensor_chw_01 = img_to_tensor_heavy(gen_image_resized_rgb)
            gen_tensor_norm_bchw = (
                normalize_tensor_heavy(gen_tensor_chw_01).unsqueeze(0).to(heavy_device)
            )

            lpips_value = calculate_lpips_masked_heavy(
                gt_tensor_norm_bchw, gen_tensor_norm_bchw, mask_tensor_b1hw
            )

            if lpips_value is not None:
                total_lpips_val += lpips_value
                per_image_scores_dict[result_img_path_obj.name] = lpips_value
                valid_image_count_val += 1
            else:
                per_image_scores_dict[result_img_path_obj.name] = None
        except Exception:  # Catch errors during individual image processing
            per_image_scores_dict[result_img_path_obj.name] = None

    if valid_image_count_val > 0:
        final_average_score = total_lpips_val / valid_image_count_val
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
        # Lower LPIPS is better.
        print(f"FINAL_SCORE:{final_average_score:.8f}")
    else:
        print("FINAL_SCORE:ERROR")
