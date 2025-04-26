import math
import numpy as np
import cv2
import os

def ssim_filled(img1, img2, mask):
    """Calculates SSIM only for the filled-in regions defined by the mask."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask_bool = mask > 0  # Boolean mask for filled regions

    img1_filled = img1[mask_bool]
    img2_filled = img2[mask_bool]

    if img1_filled.size == 0:
        return 0  # No filled regions to compare

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # Apply windowing only to the filled regions (this is a simplification)
    # A more accurate approach might involve applying the full SSIM window
    # and then averaging only over the masked region. However, due to the
    # local nature of the SSIM calculation, this direct masking can provide
    # a reasonable estimate focused on the filled area.

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_filled = mu1[mask_bool]
    mu2_filled = mu2[mask_bool]
    mu1_sq = mu1_filled**2
    mu2_sq = mu2_filled**2
    mu1_mu2 = mu1_filled * mu2_filled
    sigma1_sq_full = cv2.filter2D(img1**2, -1, window) - mu1**2
    sigma2_sq_full = cv2.filter2D(img2**2, -1, window) - mu2**2
    sigma12_full = cv2.filter2D(img1 * img2, -1, window) - mu1 * mu2
    sigma1_sq_filled = sigma1_sq_full[mask_bool]
    sigma2_sq_filled = sigma2_sq_full[mask_bool]
    sigma12_filled = sigma12_full[mask_bool]

    ssim_map_filled = ((2 * mu1_mu2 + C1) * (2 * sigma12_filled + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq_filled + sigma2_sq_filled + C2))
    return ssim_map_filled.mean()


def calculate_ssim_filled(img1, img2, mask):
    '''calculate SSIM for filled regions
    the same outputs as MATLAB's for the filled regions
    img1, img2: [0, 255]
    mask: grayscale image where non-zero indicates filled region
    '''
    if not img1.shape == img2.shape or not img1.shape[:2] == mask.shape:
        raise ValueError('Input images and mask must have compatible dimensions.')
    if img1.ndim == 2:
        return ssim_filled(img1, img2, mask)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_filled(img1[:,:,i], img2[:,:,i], mask))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_filled(np.squeeze(img1), np.squeeze(img2), mask)
        else:
            raise ValueError('Wrong input image dimensions.')

def main(m):
    base_path = r"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench"
    gt_path = os.path.join(base_path, str(m), "target", "gt.png")
    mask_path = os.path.join(base_path, str(m), "target", "mask.png")
    results_path = os.path.join(base_path, str(m), "results")

    try:
        original = cv2.imread(gt_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if original is None or mask is None:
            print(f"Warning: Could not read ground truth or mask for folder {m}. Skipping.")
            return 0
    except FileNotFoundError:
        print(f"Warning: Ground truth or mask not found for folder {m}. Skipping.")
        return 0

    original = cv2.resize(original, (512, 512))
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    total_ssim = 0
    count = 0
    for i in range(16):
        compressed_path = os.path.join(results_path, f"{i}.png")
        compressed = cv2.imread(compressed_path, cv2.IMREAD_COLOR)
        if compressed is not None:
            ssim_value = calculate_ssim_filled(original, compressed, mask)
            total_ssim += ssim_value
            count += 1
        else:
            print(f"Warning: Could not read compressed image {i}.png in folder {m}.")

    if count > 0:
        avg_ssim = total_ssim / count
        print(f"SSIM value for folder {m} (filled regions) is {avg_ssim} ")
        return avg_ssim
    else:
        print(f"No valid compressed images found for folder {m}.")
        return 0

m_limit = 22
total_avg_ssim = 0
valid_folders = 0
for m in range(m_limit):
    avg_ssim_folder = main(m)
    if avg_ssim_folder > 0:
        total_avg_ssim += avg_ssim_folder
        valid_folders += 1

if valid_folders > 0:
    overall_avg_ssim = total_avg_ssim / valid_folders
    print(f"Average SSIM value (filled regions) across {valid_folders} folders is {overall_avg_ssim} ")
else:
    print("No valid folders found to calculate the average SSIM.")