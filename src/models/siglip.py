

import os
import torch

from ..config import device

def load_siglip_model(model_path):
    from transformers import AutoModel, AutoProcessor

    print(f"Loading SigLIP model from {model_path}...")
    model = AutoModel.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor




@torch.no_grad()
def get_siglip_image_feature(image, model, processor):
    try:
        inputs = processor(images=[image], return_tensors="pt").to(device)
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeds = extract_tensor(image_embeds)
        return F.normalize(image_embeds, p=2, dim=-1).cpu()
    except Exception as e:
        print("SigLIP image error:", e)
        return None


@torch.no_grad()
def get_siglip_text_feature(text, model, processor):
    try:
        inputs = processor(
            text=[text], padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        text_embeds = model.get_text_features(**inputs)
        text_embeds = extract_tensor(text_embeds)
        return F.normalize(text_embeds, p=2, dim=-1).cpu()
    except Exception as e:
        print("SigLIP text error:", e)
        return None
