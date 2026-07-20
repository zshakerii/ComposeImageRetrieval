import os
import torch
import clip

from ..config import device

def load_clip_model(model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def get_clip_image_feature(image, model, preprocess):
    x = preprocess(image).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    return F.normalize(feat, dim=-1).cpu()


@torch.no_grad()
def get_clip_text_feature(text, model):
    tokens = clip.tokenize([text], truncate=True).to(device)
    feat = model.encode_text(tokens)
    return F.normalize(feat, dim=-1).cpu()

