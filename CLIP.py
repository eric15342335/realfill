from torchmetrics.multimodal.clip_score import CLIPScore
import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def main(m): 
	original = Image.open(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\target\gt.png") 
	original = processor(images=original, return_tensors="pt")
	value = 0
	for i in range(16):
		compressed = Image.open(rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench\{m}\results\{i}.png")
		compressed = processor(images=compressed, return_tensors="pt")

		with torch.no_grad():
			original_features = model.get_image_features(**original)
			compressed_features = model.get_image_features(**compressed)
			
			original_features = original_features / original_features.norm(dim=-1, keepdim=True)
			compressed_features = compressed_features / compressed_features.norm(dim=-1, keepdim=True)
			
		score = torch.nn.functional.cosine_similarity(original_features, compressed_features)
		value += score
	print(f"CLIP value is {value/16} dB") 
	return value/16
	

m = 22
ans = 0
for m in range(m):
	ans += main(m)
print(f"Average CLIP value is {ans/(22)} dB")