from dreamsim import dreamsim
from PIL import Image
import cv2 
import numpy as np 
import torch

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = dreamsim(pretrained=True, device=device)

def dream(img1, img2):
	distance = model(img1, img2)
	return distance

def main(m): 
	original = preprocess(Image.open(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\target\gt.png")).to(device)
	value = 0
	for i in range(16):
		compressed= preprocess(Image.open(rf"\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\results\{i}.png")).to(device)
		value += dream(original, compressed) 
	print(f"Dreamsim value is {value/16} dB") 
	return value/16
	

m = 22
ans = 0
for m in range(m):
	ans += main(m)
print(f"Average Dreamsim value is {ans/(22)} dB")