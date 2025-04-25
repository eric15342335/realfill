"""
Command-line script to select and copy the best candidate image regions
(crops) from a source directory to a target reference directory based on
LoFTR matching against existing reference images. Includes disk caching.
"""

import os
import json
import torch
import kornia as K
import kornia.feature as KF
from pathlib import Path
import glob
import re
import shutil
import argparse
from PIL import Image
import numpy as np

# --- Default Configuration ---
DEFAULT_LOFTR_MODEL = 'outdoor'
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_CACHE_FILENAME = '.loftr_match_cache.json'
DEFAULT_PATCH_SIZE = 64  # Size of the image patches to extract and compare
DEFAULT_MIN_MATCHES_FOR_CROP = 5  # Minimum LoFTR matches to consider a crop
DEFAULT_OVERLAP_THRESHOLD = 0.5  # Threshold for merging overlapping crops

class LoFTRMatcher:
    """Handles LoFTR model loading, image comparison, and caching."""
    def __init__(self, model_type=DEFAULT_LOFTR_MODEL, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, device=None):
        self.config = {
            'model_type': model_type,
            'confidence_threshold': confidence_threshold
        }
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cache = {} # In-memory cache
        print(f"[LoFTRMatcher] Initialized. Device: {self.device}, Confidence Threshold: {self.config['confidence_threshold']}")

    def _load_model(self):
        """Loads the LoFTR model lazily."""
        if self.model is None:
            print(f"[LoFTRMatcher] Loading LoFTR model ({self.config['model_type']})...")
            try:
                self.model = KF.LoFTR(pretrained=self.config['model_type']).to(self.device).eval()
                print("[LoFTRMatcher] LoFTR model loaded.")
            except Exception as e:
                print(f"[LoFTRMatcher] ERROR: Failed to load LoFTR model: {e}")
                raise

    def _load_image_tensor(self, img_path):
        """Loads an image path into a Kornia-compatible tensor."""
        try:
            # Load grayscale [0,1] float32 tensor [1, 1, H, W]
            img_tensor = K.io.load_image(img_path, K.io.ImageLoadType.GRAY32, device=self.device)[None, ...]
            return img_tensor
        except Exception as e:
            print(f"[LoFTRMatcher] ERROR: Failed to load image {img_path}: {e}")
            return None

    def _get_cache_key(self, path1, path2):
        """Generates a cache key using paths, mtimes, and config."""
        try:
            mtime1 = os.path.getmtime(path1)
            mtime2 = os.path.getmtime(path2)
            sorted_paths = tuple(sorted((path1, path2)))
            # Key incorporates files, modification times, and confidence setting
            key_tuple = (sorted_paths[0], mtime1, sorted_paths[1], mtime2, self.config['confidence_threshold'])
            return key_tuple
        except FileNotFoundError:
            return None # Can't cache if a file is missing

    def load_disk_cache(self, cache_file_path):
        """Loads comparison results from a JSON disk cache if it exists."""
        self.cache = {}
        if cache_file_path and os.path.exists(cache_file_path):
            print(f"[LoFTRMatcher] Loading disk cache: {cache_file_path}")
            try:
                with open(cache_file_path, 'r') as f:
                    loaded_cache_list = json.load(f)
                    # Convert list keys from JSON back to tuples
                    self.cache = {tuple(item[0]): item[1] for item in loaded_cache_list}
                print(f"[LoFTRMatcher] Loaded {len(self.cache)} items.")
            except Exception as e:
                print(f"[LoFTRMatcher] WARNING: Failed to load disk cache ({e}). Starting fresh.")
                self.cache = {}
        else:
            print("[LoFTRMatcher] No disk cache found/specified. Starting fresh.")

    def save_disk_cache(self, cache_file_path):
        """Saves the in-memory cache to a JSON disk file."""
        if cache_file_path and self.cache: # Only save if path provided and cache not empty
            print(f"[LoFTRMatcher] Saving cache ({len(self.cache)} items) to: {cache_file_path}")
            try:
                # Convert tuple keys to lists for JSON compatibility
                cache_list = [[list(key), value] for key, value in self.cache.items()]
                Path(cache_file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file_path, 'w') as f:
                    json.dump(cache_list, f) # Save compact JSON
                print("[LoFTRMatcher] Cache saved.")
            except Exception as e:
                print(f"[LoFTRMatcher] ERROR: Failed to save disk cache: {e}")

    @torch.no_grad() # Essential for inference
    def get_matches(self, path1, path2):
        """Compares two images using LoFTR, returns confident match data. Uses cache."""
        cache_key = self._get_cache_key(path1, path2)
        if cache_key is None: return None # File missing

        if cache_key in self.cache:
            return self.cache[cache_key] # Return cached result

        # --- Cache miss: Perform LoFTR comparison ---
        self._load_model() # Ensure model is loaded
        if self.model is None: return None

        img_tensor1 = self._load_image_tensor(path1)
        img_tensor2 = self._load_image_tensor(path2)
        if img_tensor1 is None or img_tensor2 is None: return None # Image loading error

        try:
            input_dict = {"image0": img_tensor1, "image1": img_tensor2}
            correspondences = self.model(input_dict)
            conf = correspondences['confidence']
            pts0 = correspondences['keypoints0'][conf > self.config['confidence_threshold']]
            pts1 = correspondences['keypoints1'][conf > self.config['confidence_threshold']]
            match_data = {'num_matches': len(pts0), 'points0': pts0.cpu().numpy().tolist(), 'points1': pts1.cpu().numpy().tolist()}
            self.cache[cache_key] = match_data # Store result in cache
            return match_data
        except Exception as e:
            # Print more specific error during inference
            print(f"\n[LoFTRMatcher] ERROR during inference between {os.path.basename(path1)} & {os.path.basename(path2)}: {e}")
            return None # Indicate error

def extract_patches(image_path, patch_size):
    """Extracts non-overlapping patches from an image."""
    img = Image.open(image_path).convert('L')
    width, height = img.size
    patches = []
    locations = []
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            patches.append(patch)
            locations.append(box)
    return patches, locations

def compare_candidate_to_references(matcher, reference_paths, candidate_path, patch_size, min_matches):
    """Compares patches of a candidate image to all reference images."""
    candidate_patches, patch_locations = extract_patches(candidate_path, patch_size)
    best_matching_crops = []

    for i, candidate_patch in enumerate(candidate_patches):
        best_match_count = 0
        best_match_points = {'points0': [], 'points1': []}
        for ref_path in reference_paths:
            # Save the patch temporarily for LoFTR input
            temp_patch_path = f"_temp_patch_{i}.png"
            candidate_patch.save(temp_patch_path)
            match_data = matcher.get_matches(ref_path, temp_patch_path)
            os.remove(temp_patch_path)

            if match_data and match_data['num_matches'] > best_match_count:
                best_match_count = match_data['num_matches']
                best_match_points = {'points0': np.array(match_data['points0']), 'points1': np.array(match_data['points1'])}

        if best_match_count >= min_matches and best_match_points['points1'].size > 0:
            min_x = int(np.min(best_match_points['points1'][:, 0]))
            min_y = int(np.min(best_match_points['points1'][:, 1]))
            max_x = int(np.max(best_match_points['points1'][:, 0]))
            max_y = int(np.max(best_match_points['points1'][:, 1]))
            # Ensure the bounding box is within the patch boundaries
            patch_x, patch_y, _, _ = patch_locations[i]
            crop_bbox = (
                max(min_x, 0),
                max(min_y, 0),
                min(max_x, patch_size),
                min(max_y, patch_size)
            )
            # Adjust crop bbox to the original image coordinates
            original_crop_bbox = (
                crop_bbox[0] + patch_x,
                crop_bbox[1] + patch_y,
                crop_bbox[2] + patch_x,
                crop_bbox[3] + patch_y
            )
            best_matching_crops.append({'location': original_crop_bbox, 'matches': best_match_count, 'candidate_path': candidate_path})

    return best_matching_crops

def merge_overlapping_crops(crops, overlap_threshold):
    """Merges overlapping bounding boxes."""
    if not crops:
        return []

    merged_crops = []
    sorted_crops = sorted(crops, key=lambda x: x['matches'], reverse=True)

    while sorted_crops:
        current_crop = sorted_crops.pop(0)
        merged = False
        for existing_merged_crop in merged_crops:
            if calculate_iou(current_crop['location'], existing_merged_crop['location']) > overlap_threshold:
                # Merge the boxes (take the union)
                min_x = min(current_crop['location'][0], existing_merged_crop['location'][0])
                min_y = min(current_crop['location'][1], existing_merged_crop['location'][1])
                max_x = max(current_crop['location'][2], existing_merged_crop['location'][2])
                max_y = max(current_crop['location'][3], existing_merged_crop['location'][3])
                existing_merged_crop['location'] = (min_x, min_y, max_x, max_y)
                existing_merged_crop['matches'] = max(existing_merged_crop['matches'], current_crop['matches']) # Keep the best match count
                merged = True
                break
        if not merged:
            merged_crops.append(current_crop)

    return merged_crops

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    intersection_min_x = max(x1_min, x2_min)
    intersection_min_y = max(y1_min, y2_min)
    intersection_max_x = min(x1_max, x2_max)
    intersection_max_y = min(y1_max, y2_max)

    intersection_area = max(0, intersection_max_x - intersection_min_x) * max(0, intersection_max_y - intersection_min_y)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area

def process_candidate_image(matcher, reference_paths, candidate_path, patch_size, min_matches, overlap_threshold, output_dir):
    """Processes a single candidate image to extract and save best matching crops."""
    print(f"[Processor] Processing candidate: {os.path.basename(candidate_path)}")
    best_matching_crops = compare_candidate_to_references(matcher, reference_paths, candidate_path, patch_size, min_matches)
    merged_crops = merge_overlapping_crops(best_matching_crops, overlap_threshold)

    original_image = Image.open(candidate_path)
    copied_crops_count = 0

    for i, crop_info in enumerate(merged_crops):
        location = crop_info['location']
        crop = original_image.crop(location)
        base_filename = os.path.splitext(os.path.basename(candidate_path))[0]
        output_filename = f"{base_filename}_crop_{i+1}_matches_{crop_info['matches']}.png"
        output_path = os.path.join(output_dir, output_filename)
        try:
            crop.save(output_path)
            copied_crops_count += 1
            print(f"  [Processor] Saved crop {i+1} with {crop_info['matches']} matches to {output_filename}")
        except Exception as e:
            print(f"  [Processor] ERROR saving crop: {e}")

    return copied_crops_count

# --- Main CLI Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and copy best matching regions from candidate images using LoFTR.")
    parser.add_argument("--source-dir", required=True, help="Directory containing candidate images (e.g., 0.png).")
    parser.add_argument("--ref-dir", required=True, help="Directory with original refs.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the selected and cropped regions.")
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help=f"LoFTR confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD}).")
    parser.add_argument("--model-type", default=DEFAULT_LOFTR_MODEL, choices=['indoor', 'outdoor'], help=f"LoFTR pretrained model type (default: {DEFAULT_LOFTR_MODEL}).")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help=f"Size of image patches to compare (default: {DEFAULT_PATCH_SIZE}).")
    parser.add_argument("--min-matches", type=int, default=DEFAULT_MIN_MATCHES_FOR_CROP, help=f"Minimum LoFTR matches to consider a crop (default: {DEFAULT_MIN_MATCHES_FOR_CROP}).")
    parser.add_argument("--overlap-threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD, help=f"Threshold for merging overlapping crops (default: {DEFAULT_OVERLAP_THRESHOLD}).")

    args = parser.parse_args()

    print("-" * 30)
    print(f"[Main] Starting LoFTR region selection and copy process.")
    print(f"[Main] Args: {vars(args)}") # Print parsed arguments
    print("-" * 30)

    # --- Validate Paths ---
    source_dir_path = Path(args.source_dir)
    ref_dir_path = Path(args.ref_dir)
    output_dir_path = Path(args.output_dir)

    if not source_dir_path.is_dir():
        print(f"[Main] ERROR: Source directory not found: {source_dir_path}")
        exit(1)
    if not ref_dir_path.is_dir():
        print(f"[Main] ERROR: Reference directory not found: {ref_dir_path}")
        exit(1)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Saving selected crops to: {output_dir_path}")

    # --- Find Existing Refs ---
    original_ref_paths = sorted(glob.glob(str(ref_dir_path / '*.png')))
    if not original_ref_paths:
        print("
