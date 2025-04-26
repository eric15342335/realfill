from math import log10, sqrt
import cv2
import numpy as np
import os

def PSNR_inpainted(ground_truth, inpainted, mask):
    """Calculates PSNR only for the inpainted regions defined by the mask."""
    mask = mask > 0  # Consider only non-zero mask values as inpainted regions
    gt_inpainted = ground_truth[mask]
    inpaint_filled = inpainted[mask]

    if gt_inpainted.size == 0:
        return 0  # No inpainted regions to compare

    mse = np.mean((gt_inpainted.astype(np.float64) - inpaint_filled.astype(np.float64)) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def main(m):
    base_path = r"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench"
    gt_path = os.path.join(base_path, str(m), "target", "gt.png")
    mask_path = os.path.join(base_path, str(m), "target", "mask.png")
    results_path = os.path.join(base_path, str(m), "results")

    try:
        ground_truth = cv2.imread(gt_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth is None or mask is None:
            print(f"Warning: Could not read ground truth or mask for folder {m}. Skipping.")
            return 0
    except FileNotFoundError:
        print(f"Warning: Ground truth or mask not found for folder {m}. Skipping.")
        return 0

    ground_truth = cv2.resize(ground_truth, (512, 512))
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) # Resize mask with nearest neighbor

    total_psnr = 0
    count = 0
    for i in range(16):
        inpainted_path = os.path.join(results_path, f"{i}.png")
        inpainted = cv2.imread(inpainted_path, cv2.IMREAD_COLOR)
        if inpainted is not None:
            psnr_value = PSNR_inpainted(ground_truth, inpainted, mask)
            total_psnr += psnr_value
            count += 1
        else:
            print(f"Warning: Could not read inpainted image {i}.png in folder {m}.")

    if count > 0:
        avg_psnr = total_psnr / count
        print(f"PSNR value for folder {m} (inpainted regions) is {avg_psnr} dB")
        return avg_psnr
    else:
        print(f"No valid inpainted images found for folder {m}.")
        return 0

m_limit = 22
total_avg_psnr = 0
valid_folders = 0
for m in range(m_limit):
    avg_psnr_folder = main(m)
    if avg_psnr_folder > 0:
        total_avg_psnr += avg_psnr_folder
        valid_folders += 1

if valid_folders > 0:
    overall_avg_psnr = total_avg_psnr / valid_folders
    print(f"Average PSNR value (inpainted regions) across {valid_folders} folders is {overall_avg_psnr} dB")
else:
    print("No valid folders found to calculate the average PSNR.")