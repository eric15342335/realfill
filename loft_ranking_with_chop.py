# --- Perform Copy (AMENDED to include chopping) ---
    print(f"[Main] Processing and chopping selected candidates for '{ref_dir_path}'...")
    # Calculate next available index
    all_current_ref_files = glob.glob(str(ref_dir_path / '*.png'))
    max_existing_num = -1
    for img_path in all_current_ref_files:
        basename = os.path.basename(img_path)
        match = re.match(r"^(\d+)\.png$", basename)
        if match:
            max_existing_num = max(max_existing_num, int(match.group(1)))
    next_available_index = max_existing_num + 1
    print(f"[Main] Starting chopped image index: {next_available_index}")

    processed_count = 0
    for i, src_path in enumerate(selected_candidate_paths):
        print(f"[Main] Analyzing '{os.path.basename(src_path)}' for high-match regions...")

        high_match_regions = []
        for ref_path in original_ref_paths:
            match_data = loftr_matcher.get_matches(ref_path, src_path)
            if match_data and match_data['num_matches'] > 0:
                # This is a simplified way to get "regions". A more robust
                # approach would involve clustering or other spatial analysis
                # of the keypoints. Here, we'll just consider the bounding
                # box of all matched keypoints as a region.

                if match_data['points0'] and match_data['points1']:
                    points_cand = np.array(match_data['points1']) # Points in the candidate image
                    if points_cand.size > 0:
                        min_x = int(np.min(points_cand[:, 0]))
                        min_y = int(np.min(points_cand[:, 1]))
                        max_x = int(np.max(points_cand[:, 0]))
                        max_y = int(np.max(points_cand[:, 1]))
                        high_match_regions.append(((min_x, min_y, max_x, max_y), match_data['num_matches']))
                else:
                    print(f"[Main] No matches found between '{os.path.basename(src_path)}' and '{os.path.basename(ref_path)}'.")

        if high_match_regions:
            # Sort regions by number of matches (descending)
            high_match_regions.sort(key=lambda item: item[1], reverse=True)

            # For this severely amended version, let's just take the region with the most matches
            best_region, num_matches = high_match_regions[0]
            img = Image.open(src_path)
            cropped_img = img.crop(best_region)

            dst_filename = f"{next_available_index + processed_count}_chopped.png"
            dst_path = ref_dir_path / dst_filename

            try:
                cropped_img.save(str(dst_path))
                print(f"  [Main] Saved chopped region (from '{os.path.basename(src_path)}') to '{dst_path}' with {num_matches} matches.")
                processed_count += 1
            except Exception as e:
                print(f"[Main] ERROR saving chopped image: {e}")
        else:
            print(f"[Main] No significant matching regions found in '{os.path.basename(src_path)}'.")

    print(f"\n[Main] Finished processing and chopping. Added {processed_count} new images.")
    final_ref_images = glob.glob(str(ref_dir_path / '*.png'))
    print(f"[Main] Reference directory '{ref_dir_path}' now contains {len(final_ref_images)} PNG images.")
    print("-" * 30)
