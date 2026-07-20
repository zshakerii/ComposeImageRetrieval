import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import device
from .io_utils import load_image
from .models.features import extract_tensor


def build_clip_image_cache(image_ids, image_folder, clip_model, preprocess, dataset_type=None):
    image_features = {}
    image_features_raw = {}

    for img_id in tqdm(image_ids, desc="Extracting CLIP image features"):
        img = load_image(image_folder, img_id, dataset_type)
        if img is None:
            continue
        try:
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = clip_model.encode_image(x)
            image_features_raw[img_id] = feat.cpu()
            image_features[img_id] = F.normalize(feat, dim=-1).cpu()
        except Exception:
            continue

    return image_features, image_features_raw


def build_open_clip_image_cache(image_ids, image_folder, model, preprocess, dataset_type=None):
    """Build normalized image feature cache for OpenCLIP (independent from OpenAI CLIP)."""
    cache = {}

    for img_id in tqdm(image_ids, desc="Extracting OpenCLIP image features"):
        img = load_image(image_folder, img_id, dataset_type)
        if img is None:
            continue
        try:
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(x)
            cache[img_id] = F.normalize(feat, dim=-1).cpu()
        except Exception:
            continue

    return cache


def build_generic_image_cache(image_ids, image_folder, feature_fn, dataset_type=None):
    cache = {}
    for img_id in tqdm(image_ids, desc="Extracting image features"):
        img = load_image(image_folder, img_id, dataset_type)
        if img is None:
            continue
        try:
            feat = feature_fn(img)
            if feat is not None:
                cache[img_id] = feat
        except Exception:
            continue
    return cache


@torch.no_grad()
def build_siglip_batch_cache(image_ids, image_folder, model, processor,
                             dataset_type=None, batch_size=16):
    image_feature_cache = {}
    image_ids = list(image_ids)

    for start in tqdm(range(0, len(image_ids), batch_size),
                      desc="SigLIP image cache (batched)"):
        batch_ids = image_ids[start:start + batch_size]

        images, valid_ids = [], []
        for img_id in batch_ids:
            img = load_image(image_folder, img_id, dataset_type)
            if img is not None:
                images.append(img)
                valid_ids.append(img_id)

        if not images:
            continue

        try:
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = extract_tensor(feats)
            feats = F.normalize(feats, p=2, dim=-1).cpu()  # [B, D]

            for i, img_id in enumerate(valid_ids):
                image_feature_cache[img_id] = feats[i].unsqueeze(0)  # [1, D]
        except Exception as e:
            print("❌ SigLIP batch error:", e)
            continue

    return image_feature_cache


@torch.no_grad()
def build_qwen_batch_cache(image_ids, image_folder, model, processor, batch_size=8, dataset_type=None):
    cache = {}
    valid_imgs = []

    for img_id in image_ids:
        img = load_image(image_folder, img_id, dataset_type)
        if img is not None:
            valid_imgs.append((img_id, img))

    for i in tqdm(range(0, len(valid_imgs), batch_size), desc="Qwen batch features"):
        batch = valid_imgs[i:i + batch_size]
        ids = [x[0] for x in batch]
        images = [x[1] for x in batch]
        texts = [" "] * len(images)

        inputs = processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(device)

        pixel_values = inputs["pixel_values"].to(model.dtype)
        kwargs = {}
        if "image_grid_thw" in inputs:
            kwargs["grid_thw"] = inputs["image_grid_thw"]

        outputs = model.visual(pixel_values, **kwargs)

        if hasattr(outputs, "hidden_states"):
            feats = outputs.hidden_states[-1]
        elif isinstance(outputs, tuple):
            feats = outputs[0]
        else:
            feats = outputs

        feats = feats.mean(dim=1)
        feats = F.normalize(feats, dim=-1).cpu()

        for img_id, feat in zip(ids, feats):
            cache[img_id] = feat.unsqueeze(0)

    return cache


def stack_feature_cache(image_feature_cache):
    ids = list(image_feature_cache.keys())
    feats = torch.cat([image_feature_cache[i] for i in ids], dim=0).float()
    feats = F.normalize(feats, dim=-1)
    return ids, feats
