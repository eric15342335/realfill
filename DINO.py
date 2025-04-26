import torch
from torchvision import transforms
from torch.nn import functional as F
from transformers import ViTModel
from PIL import Image

# DINO Transforms
T = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
model = ViTModel.from_pretrained('facebook/dino-vits16')

def DINO(img1, img2):
    img1 = T(img1)
    img2 = T(img2)
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    images = [img1, img2]
    inputs = torch.stack(images)  # Batch size = 2
    with torch.no_grad():
        outputs = model(inputs)
    last_hidden_states = outputs.last_hidden_state
    emb_img1, emb_img2 = last_hidden_states[0, 0], last_hidden_states[1, 0]  # cls tokens
    metric = F.cosine_similarity(emb_img1, emb_img2, dim=0)
    return metric.item()

def main(dir_idx): 
    count = 0
    original_path = rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench-gen\{dir_idx}\target\gt.png"
    try:
        original = Image.open(original_path)
    except FileNotFoundError:
        print(f"Error: Missing original image at {original_path}")
        return 0
    original = original.resize((512, 512))  # Resizing image
    value = 0
    for i in range(16):
        compressed_path = rf"C:\Users\Jensen\Documents\APAI3010\Project\realfill_data_release_full\RealBench-gen\{dir_idx}\results-gen\{i}.png"
        try:
            compressed = Image.open(compressed_path)
        except FileNotFoundError:
            print(f"Warning: Missing compressed image at {compressed_path}")
            continue
        try:
            value += DINO(original, compressed)
            count += 1
        except RuntimeError as e:
            print(f"Error at iteration {i} in RealBench-{dir_idx}: {e}")
        continue

    average_dino = value / count
    print(f"DINO value for RealBench-{dir_idx}: {average_dino:.2f} dB") 
    return average_dino

# Main script: Loop over multiple directories
num_limit = 30
total_ans = 0
validfolders = 0

for dir_idx in range(num_limit):
    avgdino = main(dir_idx)
    total_ans += avgdino
    if avgdino != 0:
        validfolders += 1

average_dino_score = total_ans / validfolders
print(f"Average DINO value across all directories: {average_dino_score:.2f} dB")

