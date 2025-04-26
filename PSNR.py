from math import log10, sqrt 
import cv2 
import numpy as np 

def PSNR(original, compressed): 
	mse = np.mean((original - compressed) ** 2) 
	if(mse == 0): # MSE is zero means no noise is present in the signal . 
				# Therefore PSNR have no importance. 
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	return psnr 

def main(m): 
	try:
		original = cv2.imread(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\target\gt.png") 
	except FileExistsError:
		return 0
	
	original = cv2.resize(original, (512, 512))
	value = 0
	for i in range(16):
		compressed = cv2.imread(rf"\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\results\{i}.png", 1) 
		value += PSNR(original, compressed) 
	print(f"PSNR value is {value/16} dB") 
	return value/16
	

m = 22
ans = 0
for m in range(m):
	ans += main(m)
print(f"Average PSNR value is {ans/(22)} dB")
