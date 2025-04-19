import argparse
import os

import torch
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--validation_image",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation image",
)
parser.add_argument(
    "--validation_mask",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation mask",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible inference.")

args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    generator = None 

    expected_size = (512, 512)

    # create & load model
    weights_dtype = torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=weights_dtype,
        revision=None
    )

    pipe.unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet", revision=None, torch_dtype=weights_dtype
    )
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder", revision=None, torch_dtype=weights_dtype
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    image = Image.open(args.validation_image)
    mask_image = Image.open(args.validation_mask)

    print(f"Original image size: {image.size}")
    print(f"Original mask size: {mask_image.size}")
    if image.size != expected_size:
        print(f"Resizing image to {expected_size}...")
        image = image.resize(expected_size, Image.Resampling.LANCZOS)
    if mask_image.size != expected_size:
        print(f"Resizing mask to {expected_size}...")
        mask_image = mask_image.resize(expected_size, Image.Resampling.NEAREST)

    erode_kernel = ImageFilter.MaxFilter(3)
    mask_image = mask_image.filter(erode_kernel)

    blur_kernel = ImageFilter.BoxBlur(1)
    mask_image = mask_image.filter(blur_kernel)

    for idx in range(16):
        result = pipe(
            prompt="a photo of sks", image=image, mask_image=mask_image, 
            num_inference_steps=200, guidance_scale=1, generator=generator, 
            height=expected_size[1], width=expected_size[0],
        ).images[0]
        
        result = Image.composite(result, image, mask_image)
        result.save(f"{args.output_dir}/{idx}.png")

    print("Inference complete.")
    del pipe
    torch.cuda.empty_cache()
