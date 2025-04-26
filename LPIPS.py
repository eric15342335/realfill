import lpips
import cv2 
import numpy as np 
import torch
import torchvision.transforms as transforms
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

transform = transforms.ToTensor()
def LPIPS(img0, img1):
    d=  loss_fn_alex(img0, img1)
    return d

def main(m): 
	original = cv2.imread(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\target\gt.png") 
	original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
	original = cv2.resize(original, (512, 512))
	original = transform(original)
	original = (original * 2) - 1
	value = 0
	for i in range(16):
		compressed = cv2.imread(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\results\{i}.png", 1) 
		compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
		compressed = transform(compressed)
		compressed = (compressed * 2) - 1
		value += LPIPS(original, compressed) 
	print(f"LPIPS value is {value/16} dB") 
	return value/16
	

m = 22
ans = 0
for h in range(m):
	ans += main(h)
print(f"Average LPIPS value is {ans/(22)} dB")


