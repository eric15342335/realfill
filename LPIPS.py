import lpips
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

transform = transforms.ToTensor()

def LPIPS_filled(img0, img1, mask):
    """Calculates LPIPS only for the filled-in regions defined by the mask."""
    mask_tensor = transform(mask.astype(np.float32) / 255.0).unsqueeze(0)
    masked_img0 = img0 * mask_tensor
    masked_img1 = img1 * mask_tensor
    d = loss_fn_alex(masked_img0, masked_img1)
    return d.item()

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

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (512, 512))
    original_tensor = transform(original).unsqueeze(0)
    original_tensor = (original_tensor * 2) - 1

    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    total_lpips = 0
    count = 0
    for i in range(16):
        compressed_path = os.path.join(results_path, f"{i}.png")
        compressed = cv2.imread(compressed_path, 1)
        if compressed is not None:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
            compressed = cv2.resize(compressed, (512, 512))
            compressed_tensor = transform(compressed).unsqueeze(0)
            compressed_tensor = (compressed_tensor * 2) - 1
            lpips_value = LPIPS_filled(original_tensor, compressed_tensor, mask_resized)
            total_lpips += lpips_value
            count += 1
        else:
            print(f"Warning: Could not read compressed image {i}.png in folder {m}.")

    if count > 0:
        avg_lpips = total_lpips / count
        print(f"LPIPS value for folder {m} (filled regions) is {avg_lpips}")
        return avg_lpips
    else:
        print(f"No valid compressed images found for folder {m}.")
        return 0

m_limit = 22
total_avg_lpips = 0
valid_folders = 0
for h in range(m_limit):
    avg_lpips_folder = main(h)
    if avg_lpips_folder is not None:
        total_avg_lpips += avg_lpips_folder
        valid_folders += 1

if valid_folders > 0:
    overall_avg_lpips = total_avg_lpips / valid_folders
    print(f"Average LPIPS value (filled regions) across {valid_folders} folders is {overall_avg_lpips}")
else:
    print("No valid folders found to calculate the average LPIPS.")