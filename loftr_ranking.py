"""
Command-line script to select and copy the best candidate images from a
source directory to a target reference directory based on LoFTR matching
against existing reference images. Includes disk caching.
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

# --- Default Configuration ---
DEFAULT_LOFTR_MODEL = 'outdoor'
DEFAULT_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_CACHE_FILENAME = '.loftr_match_cache.json'

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
        """Compares two images using LoFTR, returns confident match count. Uses cache."""
        cache_key = self._get_cache_key(path1, path2)
        if cache_key is None: return -1 # File missing

        if cache_key in self.cache:
            return self.cache[cache_key] # Return cached result

        # --- Cache miss: Perform LoFTR comparison ---
        self._load_model() # Ensure model is loaded
        if self.model is None: return -1

        img_tensor1 = self._load_image_tensor(path1)
        img_tensor2 = self._load_image_tensor(path2)
        if img_tensor1 is None or img_tensor2 is None: return -1 # Image loading error

        try:
            input_dict = {"image0": img_tensor1, "image1": img_tensor2}
            correspondences = self.model(input_dict)
            conf = correspondences['confidence']
            # Count matches where confidence exceeds the threshold
            num_confident_matches = torch.sum(conf > self.config['confidence_threshold']).item()
            self.cache[cache_key] = num_confident_matches # Store result in cache
            return num_confident_matches
        except Exception as e:
            # Print more specific error during inference
            print(f"\n[LoFTRMatcher] ERROR during inference between {os.path.basename(path1)} & {os.path.basename(path2)}: {e}")
            return -1 # Indicate error


# --- Ranking Function (Keep as before) ---
def rank_candidates_by_references(matcher, reference_paths, candidate_paths, cache_file_path):
    """Ranks candidates based on summed LoFTR matches against all references."""
    matcher.load_disk_cache(cache_file_path) # Load cache before starting

    print(f"[Ranker] Ranking {len(candidate_paths)} candidates against {len(reference_paths)} references...")
    candidate_scores = []
    total_comparisons = len(candidate_paths) * len(reference_paths)
    processed_count = 0

    # Iterate through each candidate image
    for cand_path in candidate_paths:
        total_matches_for_candidate = 0
        # Compare the current candidate against all reference images
        for ref_path in reference_paths:
            matches = matcher.get_matches(ref_path, cand_path)
            processed_count += 1
            if matches > 0: # Only sum successful comparisons
                total_matches_for_candidate += matches
            # Simple progress indicator
            if processed_count % 50 == 0: print('.', end='', flush=True)
            if processed_count % (50 * 80) == 0: print(f' ({processed_count}/{total_comparisons})') # Newline occasionally

        candidate_scores.append((total_matches_for_candidate, cand_path))

    print(f"\n[Ranker] Completed {processed_count}/{total_comparisons} comparisons.")
    # Sort by score (number of matches) in descending order
    candidate_scores.sort(key=lambda x: x[0], reverse=True)
    matcher.save_disk_cache(cache_file_path) # Save updated cache
    return candidate_scores


# --- Main CLI Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank and select candidate images using LoFTR.")
    parser.add_argument("--source-dir", required=True, help="Directory containing candidate images (e.g., 0.png).")
    parser.add_argument("--ref-dir", required=True, help="Directory with original refs and where selected candidates are copied.")
    parser.add_argument("--target-count", type=int, required=True, help="Desired total number of reference images.")
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help=f"LoFTR confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD}).")
    parser.add_argument("--model-type", default=DEFAULT_LOFTR_MODEL, choices=['indoor', 'outdoor'], help=f"LoFTR pretrained model type (default: {DEFAULT_LOFTR_MODEL}).")

    args = parser.parse_args()

    print("-" * 30)
    print(f"[Main] Starting LoFTR selection and copy process.")
    print(f"[Main] Args: {vars(args)}") # Print parsed arguments
    print("-" * 30)

    # --- Validate Paths ---
    # Use Path objects internally for consistency, but accept strings as args
    source_dir_path = Path(args.source_dir)
    ref_dir_path = Path(args.ref_dir)
    if not source_dir_path.is_dir():
        print(f"[Main] ERROR: Source directory not found: {source_dir_path}")
        exit(1)
    if not ref_dir_path.is_dir():
        print(f"[Main] ERROR: Target reference directory not found: {ref_dir_path}")
        exit(1)

    # --- Find Existing Refs ---
    original_ref_paths = sorted(glob.glob(str(ref_dir_path / '*.png')))
    current_ref_count = len(original_ref_paths)
    print(f"[Main] Found {current_ref_count} existing reference images in '{ref_dir_path}'.")

    if not original_ref_paths:
        print("[Main] WARNING: No existing reference images found. Cannot perform LoFTR ranking.")
        exit(0) # Exit gracefully if no refs to compare against

    # --- Calculate Needed ---
    num_needed = args.target_count - current_ref_count
    if num_needed <= 0:
        print("[Main] Target reference count already met. No candidates needed.")
        exit(0)
    print(f"[Main] Need to select and copy {num_needed} new candidate(s).")

    # --- Find Candidates ---
    candidate_paths = []
    try:
        # List directory contents (filenames as strings)
        all_files_in_source = os.listdir(source_dir_path)
        # Filter for filenames matching the pattern "N.png"
        potential_filenames = [fname for fname in all_files_in_source if re.match(r"^\d+\.png$", fname)]
        # Sort filenames numerically based on the stem (name without extension)
        potential_filenames.sort(key=lambda fname: int(os.path.splitext(fname)[0]))
        # Construct full, absolute paths as strings
        candidate_paths = [os.path.abspath(os.path.join(args.source_dir, fname)) for fname in potential_filenames]
    except Exception as e:
        print(f"[Main] ERROR: Could not list or process candidate files in {args.source_dir}: {e}")
        exit(1)

    if not candidate_paths:
        print(f"[Main] ERROR: No candidate images (e.g., 0.png) found in {args.source_dir}.")
        exit(1)
    print(f"[Main] Found {len(candidate_paths)} potential candidates.")

    # --- Rank Candidates ---
    loftr_matcher = LoFTRMatcher(model_type=args.model_type, confidence_threshold=args.conf_threshold)
    cache_file_path = str(source_dir_path / DEFAULT_CACHE_FILENAME) # Use Path object to join
    ranked_scores = rank_candidates_by_references(loftr_matcher, original_ref_paths, candidate_paths, cache_file_path)

    if not ranked_scores:
        print("[Main] ERROR: LoFTR ranking returned no results.")
        exit(1)

    print(f"\n[Main] LoFTR Ranking results (Top {min(5, len(ranked_scores))}):")
    for i, (score, path) in enumerate(ranked_scores[:5]):
        print(f"  Rank {i+1}: Score={score}, File={os.path.basename(path)}")

    # --- Select Top N ---
    num_to_select = min(num_needed, len(ranked_scores))
    selected_candidate_paths = [path for score, path in ranked_scores[:num_to_select]]

    if num_to_select <= 0:
        print("[Main] No candidates needed or selected.")
        exit(0)

    print(f"\n[Main] Selected top {len(selected_candidate_paths)} candidates for copying.")

    # --- Perform Copy ---
    print(f"[Main] Copying selected candidates to '{ref_dir_path}'...")
    # Calculate next available index based on CURRENT content of ref_dir
    all_current_ref_files = glob.glob(str(ref_dir_path / '*.png'))
    max_existing_num = -1
    for img_path in all_current_ref_files:
        basename = os.path.basename(img_path)
        match = re.match(r"^(\d+)\.png$", basename)
        if match:
            max_existing_num = max(max_existing_num, int(match.group(1)))
    next_available_index = max_existing_num + 1
    print(f"[Main] Starting copy index: {next_available_index}")

    copied_count = 0
    for i, src_path in enumerate(selected_candidate_paths):
        dst_filename = f"{next_available_index + i}.png"
        dst_path = ref_dir_path / dst_filename # Use Path object for destination

        if dst_path.exists():
            print(f"[Main] WARNING: Destination path '{dst_path}' already exists! Skipping.")
            continue
        try:
            shutil.copy2(src_path, str(dst_path)) # copy2 needs string path for dest
            print(f"  Copied '{os.path.basename(src_path)}' to '{dst_path}'")
            copied_count += 1
        except Exception as e:
            print(f"[Main] ERROR copying '{src_path}' to '{dst_path}': {e}")

    print(f"\n[Main] Finished copying. Added {copied_count} new images.")
    final_ref_images = glob.glob(str(ref_dir_path / '*.png'))
    print(f"[Main] Reference directory '{ref_dir_path}' now contains {len(final_ref_images)} PNG images.")
    print("-" * 30)
