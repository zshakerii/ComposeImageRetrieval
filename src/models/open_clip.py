import os
import torch
import open_clip
import torch.nn.functional as F


from ..config import DEFAULT_OPEN_CLIP_MODEL, DEFAULT_OPEN_CLIP_PRETRAINED, device


def load_open_clip_model(
        model_name=DEFAULT_OPEN_CLIP_MODEL,
        pretrained=DEFAULT_OPEN_CLIP_PRETRAINED,  # اینجا مسیر local فایل بده
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained  # مثلاً: "/path/to/model.pt"
    )

    # 1. ساخت توکنایزر مخصوص مدل
    tokenizer = open_clip.get_tokenizer(model_name)

    # 2. برگرداندن هر سه مقدار
    return model, preprocess, tokenizer

@torch.no_grad()
def get_open_clip_image_feature(image, model, preprocess):
    x = preprocess(image).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    return F.normalize(feat, dim=-1).cpu()


@torch.no_grad()
def get_open_clip_text_feature(text, model, tokenizer):
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    return F.normalize(feat, dim=-1).cpu()

