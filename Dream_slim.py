from dreamsim import dreamsim
from PIL import Image
import cv2 
import numpy as np 
import torch
import os

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = dreamsim(pretrained=True, device=device)

def dream(img1, img2, mask):
	mask[mask == 0] = 254
	mask[mask == 255] = 0
	mask[mask == 254] = 255
	distance = model(img1*mask, img2*mask)
	return distance

def main(m):
	try:
		original  = preprocess(Image.open(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench-gen\{m}\target\gt.png")).to(device)
		mask = preprocess(Image.open(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench-gen\{m}\target\mask.png")).to(device)
		if original is None or mask is None:
			print (f"Warning: Could not read ground truth or mask for folder {m}. Skipping.")
			return 0
	except FileNotFoundError:
		print(f"Warning: Ground truth not found or mask not found for folder {m}. Skipping.")
		return 0
	
	value = 0
	count = 0
	for i in range(16):
		try:
			result = preprocess(Image.open(rf"\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench-gen\{m}\results-gen\{i}.png")).to(device)
			if result is None:
				print (f"Warning: Could not read ground truth for folder {m}. Skipping.")
		except  FileNotFoundError:
			print(f"Warning: Ground truth not found for folder {m}. Skipping.")
		value += dream(original, result, mask)
		count += 1
	if count > 0:
		print(f"Dreamsim value for folder{m} is {value/count}")
		return value/count
	else:
		print(f"No valid generated images found for folder {m}.")
		return 0

total = 0
m_limit = 30
validfolders = 0
for m in range(m_limit):
	avg_dream = main(m)
	if avg_dream != 0:
		total += avg_dream
		validfolders += 1
if validfolders > 0:
    overall_avg_lpips = total / validfolders
    print(f"Average LPIPS value (filled regions) across {validfolders} folders is {overall_avg_lpips}")
else:
    print("No valid folders found to calculate the average LPIPS.")
