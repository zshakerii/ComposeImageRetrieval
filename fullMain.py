
import os
import json
import argparse
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import clip
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
K_VALUES = [1, 5, 10,50]


# ===============================
# Dataset Parsers
# ===============================
def parse_cirr_sample(sample):
    reference_id = os.path.splitext(sample.get("reference", ""))[0]
    caption = sample.get("caption", "").strip()
    members = [os.path.splitext(m)[0] for m in sample.get("img_set", {}).get("members", [])]
    ref_index = sample.get("img_set", {}).get("reference_rank", 0)

    if not reference_id or not caption or not members or ref_index >= len(members):
        return None

    target_id = members[ref_index]
    return {
        "reference_id": reference_id,
        "caption": caption,
        "target_id": target_id,
        "members": members,
    }


def parse_circo_sample(sample):
    reference_id = sample.get("reference_img_id")
    caption = sample.get("relative_caption", "").strip()
    target_id = sample.get("target_img_id")
    gt_ids = sample.get("gt_img_ids", [])

    if reference_id is not None:
        reference_id = str(reference_id)
    if target_id is not None:
        target_id = str(target_id)
    gt_ids = [str(x) for x in gt_ids]

    if not reference_id or not caption or not target_id:
        return None

    return {
        "reference_id": reference_id,
        "caption": caption,
        "target_id": target_id,
        "members": gt_ids,
    }


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        parsed = [parse_circo_sample(s) for s in data["annotations"]]
        return "circo", [x for x in parsed if x is not None]

    #if isinstance(data, list):
       # parsed = [parse_cirr_sample(s) for s in data]
        #return "cirr", [x for x in parsed if x is not None]

    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]

        # CIRCO: دارای reference_img_id و target_img_id و gt_img_ids
        if "reference_img_id" in first_item and "target_img_id" in first_item:
            parsed = [parse_circo_sample(s) for s in data]
            return "circo", [x for x in parsed if x is not None]

        # CIRR: دارای reference و img_set و target_hard
        elif "reference" in first_item and "img_set" in first_item:
            parsed = [parse_cirr_sample(s) for s in data]
            return "cirr", [x for x in parsed if x is not None]


    raise ValueError("Unknown dataset format")


# ===============================
# Utilities
# ===============================
def init_results():
    return {
        "prec": {k: [] for k in K_VALUES},
        "rec": {k: [] for k in K_VALUES},
        "map": {5: [], 10: [],50:[]},
        "mrr": [],
    }

'''
def average_precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    score = 0.0
    num_hits = 0.0
    for i, img in enumerate(retrieved_k, start=1):
        if img in relevant:
            num_hits += 1.0
            score += num_hits / i
    if len(relevant) == 0:
        return 0.0
    return score / min(len(relevant), k)
'''
def average_precision_at_k(relevant, retrieved, k):
    relevant_set = set(relevant)  # O(1) lookup
    retrieved_k = retrieved[:k]
    score, num_hits = 0.0, 0.0
    for i, img in enumerate(retrieved_k, start=1):
        if img in relevant_set:
            num_hits += 1.0
            score += num_hits / i
    return 0.0 if not relevant else score / min(len(relevant), k)



def mean_reciprocal_rank(relevant, retrieved):
    for rank, img in enumerate(retrieved, start=1):
        if img in relevant:
            return 1.0 / rank
    return 0.0


def summarize_results(results):
    return {
        "mrr": float(np.mean(results["mrr"])) if results["mrr"] else 0.0,
        "map5": float(np.mean(results["map"][5])) if results["map"][5] else 0.0,
        "map10": float(np.mean(results["map"][10])) if results["map"][10] else 0.0,
        "map50": float(np.mean(results["map"][50])) if results["map"][50] else 0.0,        # اضافه شدن map10
        "prec1": float(np.mean(results["prec"][1])) if results["prec"][1] else 0.0,
        "prec5": float(np.mean(results["prec"][5])) if results["prec"][5] else 0.0,
        "prec10": float(np.mean(results["prec"][10])) if results["prec"][10] else 0.0,
        "prec50": float(np.mean(results["prec"][50])) if results["prec"][50] else 0.0,
        "rec1": float(np.mean(results["rec"][1])) if results["rec"][1] else 0.0,
        "rec5": float(np.mean(results["rec"][5])) if results["rec"][5] else 0.0,
        "rec10": float(np.mean(results["rec"][10])) if results["rec"][10] else 0.0,
        "rec50": float(np.mean(results["rec"][50])) if results["rec"][50] else 0.0,
    }


def find_image_path(image_folder, image_id, dataset_type=None):
    image_id = str(image_id)
    if image_id.isdigit():
        image_id = image_id.zfill(12)

    for ext in IMAGE_EXTS:
        path = os.path.join(image_folder, image_id + ext)
        if os.path.exists(path):
            return path
    return None


def load_image(image_folder, image_id,dataset_type=None):
    path = find_image_path(image_folder, image_id, dataset_type)
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_generated_captions(json_path):
    """
    بارگذاری caption های تولیدشده برای هر تصویر.

    فرمت‌های پشتیبانی‌شده:
      1) dict کلیددار با image_id:
         { "123": {"caption": "..."} }  یا  { "123": "..." }
      2) list از آبجکت‌ها:
         [ {"image_id": "123", "caption": "..."}, ... ]
    خروجی همیشه: { image_id(str): caption(str) }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = {}

    # حالت list
    if isinstance(data, list):
        # کلیدهای محتمل برای شناسه‌ی تصویر و متن
        id_keys = ("image_id", "img_id", "reference_img_id", "reference", "id")
        cap_keys = ("caption", "relative_caption", "generated_caption", "text")

        for entry in data:
            if not isinstance(entry, dict):
                continue

            img_id = None
            for k in id_keys:
                if k in entry and entry[k] is not None:
                    img_id = str(entry[k])
                    break
            if img_id is None:
                continue

            # حذف پسوند فایل اگر وجود داشت (هماهنگ با parse_cirr_sample)
            img_id = os.path.splitext(img_id)[0]

            caption = ""
            for k in cap_keys:
                if k in entry and entry[k]:
                    caption = entry[k]
                    break

            captions[img_id] = caption
        return captions


    elif isinstance(data, dict):

        for key, value in data.items():

            img_id = os.path.splitext(str(key))[0]

            if isinstance(value, dict):

                caption = ""

                for k in cap_keys:

                    if value.get(k):
                        caption = value[k]

                        break

                captions[img_id] = caption

            elif isinstance(value, str):

                captions[img_id] = value

            else:

                captions[img_id] = ""

        return captions

    raise ValueError(f"فرمت ناشناخته‌ی generated captions: {type(data)}")



# ===============================
# Model Loaders
# ===============================
def load_clip_model(model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def load_open_clip_model(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    model = model.to(device)
    model.eval()
    return model, preprocess

def load_searle_model(clip_model_name="ViT-B/32"):
    searle, encode_with_pseudo_tokens = torch.hub.load(
        repo_or_dir="miccunifi/SEARLE",
        source="github",
        model="searle",
        backbone=clip_model_name,
    )
    searle.to(device).eval()
    return searle, encode_with_pseudo_tokens


def load_blip_model(model_path=None):
    from transformers import BlipProcessor, BlipForImageTextRetrieval

    if model_path is None:
        model_path = "Salesforce/blip-itm-base-coco"

    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForImageTextRetrieval.from_pretrained(model_path).to(device)
    model.eval()

    # حذف embed_model اضافی - مدل اصلی کافی است
    return model, processor


def load_qwen_model(model_path,device):
    from transformers import AutoProcessor, AutoModel

    print(f"🔄 بارگذاری مدل Qwen از {model_path}...")

    try:
        # استفاده از AutoProcessor به جای Qwen2VLProcessor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            #device_map="auto",
            trust_remote_code=True
        )

        model.eval()

        print(f"✅ مدل Qwen بارگذاری شد (device: {device})")
        return model, processor

    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل Qwen: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_lava_model(hf_token=None):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_path = "llava-hf/llava-1.5-7b-hf"
    hf_token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    processor_kwargs = {}
    if hf_token:
        processor_kwargs["token"] = hf_token

    model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    model.eval()
    return model, processor


def load_siglip_model(model_path):
    print(f"Loading SigLIP model from {model_path}...")
    from transformers import AutoModel, AutoProcessor
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor

# ===============================
# Feature Extraction
# ===============================
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


@torch.no_grad()
def get_blip_image_feature(image, model, processor):
    """استخراج ویژگی تصویر با Blip-ITM"""
    try:
        dummy_text = "an image"

        inputs = processor(
            images=image,
            text=dummy_text,
            return_tensors="pt",
            padding=True
        ).to(device)

        # استخراج vision embeddings
        vision_outputs = model.vision_model(
            pixel_values=inputs['pixel_values'],
            return_dict=True
        )

        # گرفتن CLS token
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        return image_embeds.cpu()

    except Exception as e:
        print(f"❌ Blip image error: {e}")
        import traceback
        traceback.print_exc()
        return None


@torch.no_grad()
def get_blip_text_feature(caption, model, processor):
    """استخراج ویژگی متن با Blip-ITM"""
    try:
        dummy_image = Image.new("RGB", (384, 384), color="white")

        inputs = processor(
            images=dummy_image,
            text=caption,
            return_tensors="pt",
            padding=True
        ).to(device)

        # استخراج text embeddings
        text_outputs = model.text_encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )

        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return text_embeds.cpu()

    except Exception as e:
        print(f"❌ Blip text error: {e}")
        import traceback
        traceback.print_exc()
        return None

#@torch.no_grad()
@torch.inference_mode()
def get_qwen_image_feature(image, model, processor,device):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": " "},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # استخراج مقادیر مربوط به تصویر
        pixel_values = inputs['pixel_values'].to(model.dtype)

        # مدل‌های Qwen2-VL و Qwen3-VL معمولا به image_grid_thw نیاز دارند
        kwargs = {}
        if 'image_grid_thw' in inputs:
            kwargs['grid_thw'] = inputs['image_grid_thw']

        # استفاده از model.visual به جای model.vision_tower
        vision_outputs = model.visual(pixel_values, **kwargs)

        # خروجی در Qwen3 معمولا یک تنسور است. چک می‌کنیم که اگر آبجکت بود hidden_states را بگیریم
        if hasattr(vision_outputs, 'hidden_states'):
            image_features = vision_outputs.hidden_states[-1]
        elif isinstance(vision_outputs, tuple):
            image_features = vision_outputs[0]
        else:
            image_features = vision_outputs

        # میانگین‌گیری روی توکن‌های تصویر برای رسیدن به یک بردار واحد
        if image_features.dim() == 3:
            image_features = image_features.mean(dim=1)
        elif image_features.dim() == 2:
            # اگر ابعاد (تعداد پچ‌ها، سایز فیچر) باشد
            image_features = image_features.mean(dim=0, keepdim=True)

        return F.normalize(image_features, p=2, dim=-1).cpu()

    except Exception as e:
        import traceback
        print(f"Qwen image error: {e}")
        traceback.print_exc()
        return None


    '''try:
        # برای Qwen-VL معمولا ساختار پیام ضروری است
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},  # متن ساختگی برای جلوگیری از خطای پردازشگر
                ],
            }
        ]

        # آماده‌سازی ورودی‌ها با استفاده از template پردازشگر
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # استخراج ویژگی (بسته به معماری مدل، ممکن است از متد get_image_features استفاده شود)
            if hasattr(model, 'get_image_features'):
                image_features = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                # استخراج آخرین لایه پنهان
                image_features = outputs.hidden_states[-1].mean(dim=1)

        return image_features.cpu().numpy()

    except Exception as e:
        print(f"Qwen image error: {e}")
        return None
'''

@torch.no_grad()
def get_qwen_text_feature(text, model, processor,device):
    """استخراج ویژگی متن با Qwen-VL"""
    if processor is None or model is None:
        return None

    try:
        # پردازش متن
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # استخراج ویژگی از مدل زبان
        # توجه: برخی مدل‌ها متد get_text_features دارند، اما این روش عمومی‌تر است
        outputs = model.language_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )

        # استخراج embedding از آخرین لایه پنهان و گرفتن میانگین
        embedding = outputs.hidden_states[-1].mean(dim=1)

        # نرمال‌سازی ویژگی‌ها (بسیار مهم)
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.cpu()

    except Exception as e:
        import traceback
        print(f"Qwen text error: {e}")
        traceback.print_exc()
        return None

    '''
    """استخراج ویژگی متن با Qwen3-VL-Embedding"""
    if processor is None or model is None:
        return None

    try:
        # پردازش متن
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.language_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )

            # استخراج embedding از آخرین لایه
            embedding = outputs.hidden_states[-1].mean(dim=1)

            # نرمال‌سازی
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu()

    except Exception as e:
        print(f"Qwen text error: {e}")
        return None

'''


@torch.no_grad()
def get_lava_image_feature(image, model, processor):
    try:
        prompt = "USER: <image>\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        feat = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(feat.float(), dim=-1).cpu()
    except Exception:
        return None


@torch.no_grad()
def get_lava_text_feature(caption, model, processor):
    try:
        dummy_image = Image.new("RGB", (336, 336), color="white")
        prompt = f"USER: {caption}\nASSISTANT:"
        inputs = processor(text=prompt, images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        feat = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(feat.float(), dim=-1).cpu()
    except Exception:
        return None

def _extract_tensor(out):
    """خروجی مدل را به تنسور تبدیل می‌کند (سازگار با SigLIP2)."""
    if isinstance(out, torch.Tensor):
        return out
    # آبجکت‌هایی مثل BaseModelOutputWithPooling
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        # میانگین یا CLS بسته به معماری
        return out.last_hidden_state.mean(dim=1)
    raise TypeError(f"خروجی غیرمنتظره: {type(out)}")

@torch.no_grad()
def get_siglip_image_feature(image, model, processor):
    try:
        inputs = processor(images=[image], return_tensors="pt").to(device)
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeds = _extract_tensor(image_embeds)
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
        text_embeds = _extract_tensor(text_embeds)
        return F.normalize(text_embeds, p=2, dim=-1).cpu()
    except Exception as e:
        print("SigLIP text error:", e)
        return None

# ===============================
# Feature Cache Builders
# ===============================
def build_clip_image_cache(image_ids, image_folder, clip_model, preprocess, dataset_type=None):
    image_features = {}
    image_features_raw = {}

    for img_id in tqdm(image_ids, desc="Extracting CLIP image features"):
        img = load_image(image_folder, img_id,dataset_type)
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


def build_generic_image_cache(image_ids, image_folder, feature_fn, dataset_type=None):
    cache = {}
    for img_id in tqdm(image_ids, desc="Extracting image features"):
        img = load_image(image_folder, img_id,dataset_type)
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

        # فقط تصاویر معتبر را جمع کن و idهای متناظرشان را نگه دار
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
            feats = _extract_tensor(feats)
            feats = F.normalize(feats, p=2, dim=-1).cpu()        # [B, D]

            for i, img_id in enumerate(valid_ids):
                image_feature_cache[img_id] = feats[i].unsqueeze(0)  # [1, D]
        except Exception as e:
            print("❌ SigLIP batch error:", e)
            continue

    return image_feature_cache

@torch.no_grad()
def build_qwen_batch_cache(image_ids, image_folder, model, processor,
                           batch_size=8):

    cache = {}

    valid_imgs = []

    for img_id in image_ids:
        img = load_image(image_folder, img_id)
        if img is not None:
            valid_imgs.append((img_id, img))

    for i in tqdm(range(0, len(valid_imgs), batch_size),
                  desc="Qwen batch features"):

        batch = valid_imgs[i:i+batch_size]

        ids = [x[0] for x in batch]
        images = [x[1] for x in batch]

        texts = [" "] * len(images)

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
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


# ===============================
# Ranking
# ===============================
def rank_by_similarity(query_feat, gallery_feats):
    sims = torch.matmul(gallery_feats, query_feat.squeeze(0).float().T).squeeze(-1)
    ranked_idx = torch.argsort(sims, descending=True)
    return ranked_idx.cpu().tolist()


def update_metrics(results, ranked_ids, positives):
    n_pos = len(positives)
    for k in K_VALUES:
        top_k = ranked_ids[:k]
        hits = sum(1 for img in top_k if img in positives)
        results["prec"][k].append(hits / k if k else 0.0)
        results["rec"][k].append(hits / n_pos if n_pos else 0.0)

    for k in MAP_K_VALUES:
        results["map"][k].append(average_precision_at_k(positives, ranked_ids, k))
    results["mrr"].append(mean_reciprocal_rank(positives, ranked_ids))


# ===============================
# Evaluators
# ===============================
def evaluate_clip_alphas(data, clip_model, image_features_cache, gallery_ids, gallery_feats, alphas):
    text_cache = {}
    results_by_alpha = {}

    for item in tqdm(data, desc="Caching CLIP text features"):
        caption_full = "a photo of " + item["caption"]
        key = (item["reference_id"], caption_full)
        if key not in text_cache:
            text_cache[key] = get_clip_text_feature(caption_full, clip_model)

    ref_cache = {k: v for k, v in image_features_cache.items()}

    for alpha in alphas:
        results = init_results()
        total, skipped = 0, 0

        for item in tqdm(data, desc=f"CLIP alpha={alpha:.2f}"):
            reference_id = item["reference_id"]
            target_id = item["target_id"]
            positives = {target_id}
            caption_full = "a photo of " + item["caption"]

            text_feat = text_cache.get((reference_id, caption_full))
            ref_feat = ref_cache.get(reference_id)

            if text_feat is None or ref_feat is None:
                skipped += 1
                continue

            sims_txt = torch.matmul(gallery_feats, text_feat.squeeze(0).float().T)
            sims_ref = torch.matmul(gallery_feats, ref_feat.squeeze(0).float().T)
            sims = alpha * sims_txt + (1.0 - alpha) * sims_ref
            ranked_idx = torch.argsort(sims.squeeze(-1), descending=True).cpu().tolist()
            ranked_ids = [gallery_ids[i] for i in ranked_idx]

            update_metrics(results, ranked_ids, positives)
            total += 1

        results_by_alpha[alpha] = (results, total, skipped)

    return results_by_alpha


def evaluate_clip_beta(data, clip_model, image_features_cache,
                       gallery_ids, gallery_feats,
                       generated_captions, alphas, betas):
    """
    clip_beta:
      t_rel : caption نسبی (cap.rc2.test1)
      t_gen : caption تولیدشده‌ی تصویر مرجع (generated_test1_captions)
      r     : ویژگی تصویر مرجع

      q1 = beta*t_rel + (1-beta)*t_gen  ->  s1 = G·q1
      q2 = beta*r     + (1-beta)*t_gen  ->  s2 = G·q2
      s  = alpha*s1 + (1-alpha)*s2
    """
    rel_text_cache = {}
    gen_text_cache = {}

    for item in tqdm(data, desc="Caching clip_beta text features"):
        rel_caption = "a photo of " + item["caption"]
        if rel_caption not in rel_text_cache:
            rel_text_cache[rel_caption] = get_clip_text_feature(rel_caption, clip_model)

        ref_id = item["reference_id"]
        gen_cap = generated_captions.get(ref_id, "")
        if gen_cap and ref_id not in gen_text_cache:
            gen_text_cache[ref_id] = get_clip_text_feature("a photo of " + gen_cap, clip_model)

    ref_cache = {k: v for k, v in image_features_cache.items()}
    results_by_params = {}

    for beta in betas:
        for alpha in alphas:
            results = init_results()
            total, skipped, no_gen = 0, 0, 0

            for item in tqdm(data, desc=f"clip_beta a={alpha:.2f} b={beta:.2f}"):
                reference_id = item["reference_id"]
                positives = {item["target_id"]}
                rel_caption = "a photo of " + item["caption"]

                t_rel = rel_text_cache.get(rel_caption)
                t_gen = gen_text_cache.get(reference_id)
                r = ref_cache.get(reference_id)

                if t_rel is None or r is None:
                    skipped += 1
                    continue
                if t_gen is None:
                    # اگر caption تولیدشده نبود، فقط روی متن نسبی و تصویر تکیه می‌کنیم
                    no_gen += 1
                    q1 = t_rel
                    q2 = r
                else:
                    q1 = beta * t_rel + (1.0 - beta) * t_gen
                    q2 = beta * r + (1.0 - beta) * t_gen

                q1 = F.normalize(q1, dim=-1)
                q2 = F.normalize(q2, dim=-1)

                s1 = torch.matmul(gallery_feats, q1.squeeze(0).float().T)
                s2 = torch.matmul(gallery_feats, q2.squeeze(0).float().T)
                sims = alpha * s1 + (1.0 - alpha) * s2

                ranked_idx = torch.argsort(sims.squeeze(-1), descending=True).cpu().tolist()
                ranked_ids = [gallery_ids[i] for i in ranked_idx]

                update_metrics(results, ranked_ids, positives)
                total += 1

            results_by_params[(alpha, beta)] = (results, total, skipped)

    return results_by_params


def evaluate_searle(data, searle, encode_with_pseudo_tokens, clip_model,
                    image_features_cache, image_features_cache_raw, gallery_ids, gallery_feats):
    results = init_results()
    total, skipped = 0, 0

    for item in tqdm(data, desc="Searle"):
        reference_id = item["reference_id"]
        caption = item["caption"]
        target_id = item["target_id"]
        positives = {target_id}
        caption_full = "a photo of $" + caption

        ref_feat_raw = image_features_cache_raw.get(reference_id)
        if ref_feat_raw is None:
            skipped += 1
            continue

        try:
            with torch.no_grad():
                pseudo_tokens = searle(ref_feat_raw.to(device))
                tokenized = clip.tokenize([caption_full], truncate=True).to(device)
                text_feat = encode_with_pseudo_tokens(clip_model, tokenized, pseudo_tokens)
                text_feat = F.normalize(text_feat.float(), dim=-1).cpu()

            ranked_idx = rank_by_similarity(text_feat, gallery_feats)
            ranked_ids = [gallery_ids[i] for i in ranked_idx]
            update_metrics(results, ranked_ids, positives)
            total += 1
        except Exception:
            skipped += 1

    return results, total, skipped


def evaluate_generic(data, image_folder, image_features_cache, gallery_ids, gallery_feats,
                     get_ref_feature_fn, get_text_feature_fn, model_name="MODEL", alpha=0.5, dataset_type=None):
    results = init_results()
    total, skipped = 0, 0

    ref_cache = {}
    text_cache = {}

    for item in tqdm(data, desc=f"{model_name} eval"):
        reference_id = item["reference_id"]
        caption_full = "a photo of " + item["caption"]
        target_id = item["target_id"]
        positives = {target_id}

        if reference_id not in ref_cache:
            ref_img = load_image(image_folder, reference_id,dataset_type)
            ref_cache[reference_id] = None if ref_img is None else get_ref_feature_fn(ref_img)

        if caption_full not in text_cache:
            text_cache[caption_full] = get_text_feature_fn(caption_full)

        ref_feat = ref_cache[reference_id]
        text_feat = text_cache[caption_full]

        if ref_feat is None or text_feat is None:
            skipped += 1
            continue

        sims_txt = torch.matmul(gallery_feats, text_feat.squeeze(0).float().T)
        sims_ref = torch.matmul(gallery_feats, ref_feat.squeeze(0).float().T)
        sims = alpha * sims_txt + (1.0 - alpha) * sims_ref

        ranked_idx = torch.argsort(sims.squeeze(-1), descending=True).cpu().tolist()
        ranked_ids = [gallery_ids[i] for i in ranked_idx]

        update_metrics(results, ranked_ids, positives)
        total += 1

    return results, total, skipped



def _minmax_normalize(sims):
    """نرمال‌سازی per-query در بازه [0,1]"""
    s = sims.squeeze(-1)
    s_min = s.min()
    s_max = s.max()
    if (s_max - s_min) < 1e-8:
        return torch.zeros_like(s)
    return (s - s_min) / (s_max - s_min)


def evaluate_clip_alphas_separate(data, clip_model, image_features_cache,
                                  gallery_ids, gallery_feats, alphas):
    """
    نسخه‌ی جدید CLIP alpha:
    1) شباهت متن↔گالری و تصویرِ مرجع↔گالری جداگانه محاسبه می‌شود
    2) هر شباهت به‌صورت مستقل (per-query) نرمال می‌شود
    3) سپس با alpha ترکیب می‌شوند
    """
    text_cache = {}
    results_by_alpha = {}

    # کش کردن ویژگی متن (بدون باگ تایپی)
    for item in tqdm(data, desc="Caching CLIP text features (separate)"):
        caption_full = "a photo of " + item["caption"]
        key = (item["reference_id"], caption_full)
        if key not in text_cache:
            text_cache[key] = get_clip_text_feature(caption_full, clip_model)

    ref_cache = {k: v for k, v in image_features_cache.items()}

    for alpha in alphas:
        results = init_results()
        total, skipped = 0, 0

        for item in tqdm(data, desc=f"CLIP-sep alpha={alpha:.2f}"):
            reference_id = item["reference_id"]
            target_id = item["target_id"]
            positives = {target_id}
            caption_full = "a photo of " + item["caption"]

            text_feat = text_cache.get((reference_id, caption_full))
            ref_feat = ref_cache.get(reference_id)

            if text_feat is None or ref_feat is None:
                skipped += 1
                continue

            # شباهت‌ها جداگانه
            sims_txt = torch.matmul(gallery_feats, text_feat.squeeze(0).float().T)
            sims_ref = torch.matmul(gallery_feats, ref_feat.squeeze(0).float().T)

            # نرمال‌سازی مستقل هر شباهت
            sims_txt_n = _minmax_normalize(sims_txt)
            sims_ref_n = _minmax_normalize(sims_ref)

            # ترکیب بر اساس alpha
            sims = alpha * sims_txt_n + (1.0 - alpha) * sims_ref_n

            ranked_idx = torch.argsort(sims, descending=True).cpu().tolist()
            ranked_ids = [gallery_ids[i] for i in ranked_idx]

            update_metrics(results, ranked_ids, positives)
            total += 1

        results_by_alpha[alpha] = (results, total, skipped)

    return results_by_alpha

# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation for CIRR and CIRCO")
    parser.add_argument("--dataset", type=str, choices=["cirr", "circo"], required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--qwen_model_path", type=str, default=None,
                        help="مسیر لوکال مدل Qwen (اجباری برای qwen)")
    parser.add_argument("--blip_model_path", type=str, default=None,
                        help="مسیر لوکال مدل blip (اجباری برای blip)")
    parser.add_argument("--siglip_path", type=str, default=None,
                        help="Path to local SigLIP model")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["clip", "searle", "qwen", "blip", "lava","siglip","clip_sep","clip_beta", "open_clip"])
    parser.add_argument("--generated_captions_path", type=str, default=None,
                        help="مسیر json caption های تولیدشده (generated_test1_captions)")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])

    args = parser.parse_args()

    print(f"🔄 بارگذاری دیتاست {args.dataset.upper()}...")
    detected_dataset, data = load_dataset(args.json_path)
    if detected_dataset != args.dataset:
        raise ValueError(f"Dataset mismatch: arg={args.dataset}, detected={detected_dataset}")

    print(f"✅ تعداد {len(data)} نمونه بارگذاری شد.")

    print("🔄 بارگذاری مدل CLIP...")
    clip_model, preprocess = load_clip_model()

    print("🔄 بارگذاری مدل open_clip ...")
    clip_model, preprocess = load_open_clip_model()

    image_ids = set()
    for item in data:
        image_ids.add(item["reference_id"])
        image_ids.add(item["target_id"])
        for member in item.get("members", []):
            image_ids.add(member)

    dataset_type = args.dataset

    need_clip_cache = any(m in args.models for m in ("clip", "searle", "clip_sep", "clip_beta"))
    need_open_clip = "open_clip" in args.models

    clip_model = preprocess = None
    open_clip_model = open_clip_preprocess = None
    image_features_cache = image_features_cache_raw = None
    gallery_ids = gallery_feats = None

    if need_open_clip:
        print("🔄 بارگذاری مدل open_clip ...")
        open_clip_model, open_clip_preprocess = load_open_clip_model()
    if need_clip_cache:
        print("🔄 بارگذاری مدل CLIP...")
        clip_model, preprocess = load_clip_model()
        image_features_cache, image_features_cache_raw = build_clip_image_cache(
            image_ids, args.image_folder, clip_model, preprocess, dataset_type
        )
        gallery_ids, gallery_feats = stack_feature_cache(image_features_cache)
    else:
        print("⏭️  CLIP و Searle انتخاب نشده‌اند؛ کش CLIP ساخته نمی‌شود.")

    all_results = {}

    if "clip" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی CLIP")
        print("=" * 60)

        clip_results = evaluate_clip_alphas(
            data, clip_model, image_features_cache, gallery_ids, gallery_feats, args.alphas
        )
        for alpha, (results, total, skipped) in clip_results.items():
            all_results[f"CLIP alpha={alpha:.2f}"] = summarize_results(results)

    if "clip_sep" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی CLIP (Separate + Normalized)")
        print("=" * 60)

        clip_sep_results = evaluate_clip_alphas_separate(
            data, clip_model, image_features_cache,
            gallery_ids, gallery_feats, args.alphas
        )
        for alpha, (results, total, skipped) in clip_sep_results.items():
            all_results[f"CLIP-sep alpha={alpha:.2f}"] = summarize_results(results)

    if "clip_beta" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی CLIP-Beta")
        print("=" * 60)
        if not args.generated_captions_path:
            print("❌ برای clip_beta باید --generated_captions_path مشخص شود.")
        else:
            generated_captions = load_generated_captions(args.generated_captions_path)
            print(f"✅ تعداد {len(generated_captions)} caption تولیدشده بارگذاری شد.")

            beta_results = evaluate_clip_beta(
                data, clip_model, image_features_cache,
                gallery_ids, gallery_feats,
                generated_captions, args.alphas, args.betas
            )
            for (alpha, beta), (results, total, skipped) in beta_results.items():
                all_results[f"CLIP-beta a={alpha:.2f} b={beta:.2f}"] = summarize_results(results)

    if "searle" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی Searle")
        print("=" * 60)
        try:
            searle, encode_with_pseudo_tokens = load_searle_model()
            results, total, skipped = evaluate_searle(
                data, searle, encode_with_pseudo_tokens, clip_model,
                image_features_cache, image_features_cache_raw,
                gallery_ids, gallery_feats
            )
            all_results["Searle"] = summarize_results(results)
        except Exception as e:
            print(f"❌ خطا در ارزیابی Searle: {e}")
            traceback.print_exc()

    if "qwen" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی Qwen")
        print("=" * 60)
        try:
            model, processor = load_qwen_model(args.qwen_model_path, device)
            qwen_image_features = build_qwen_batch_cache(
                image_ids,
                args.image_folder,
                model,
                processor,
                batch_size=8
            )
            '''
            
            qwen_image_features = build_generic_image_cache(
                image_ids,
                args.image_folder,
                lambda img: get_qwen_image_feature(img, model, processor, device),
                dataset_type=dataset_type
            )
            '''
            print(f"✅ تعداد {len(qwen_image_features)} تصویر Qwen پردازش شد.")

            if qwen_image_features:
                q_ids, q_feats = stack_feature_cache(qwen_image_features)
                results, total, skipped = evaluate_generic(
                    data,
                    args.image_folder,
                    qwen_image_features,
                    q_ids,
                    q_feats,
                    lambda img: get_qwen_image_feature(img, model, processor,device),
                    lambda txt: get_qwen_text_feature(txt, model, processor,device),
                    model_name="Qwen",
                    alpha=0.5,
                    dataset_type=dataset_type
                )
                print(f"✅ Qwen: {total} نمونه ارزیابی شد، {skipped} نمونه رد شد.")
                all_results["Qwen"] = summarize_results(results)
            else:
                print("❌ هیچ ویژگی تصویری Qwen استخراج نشد!")
        except Exception as e:
            print(f"❌ خطا در ارزیابی Qwen: {e}")
            traceback.print_exc()

    if "blip" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی Blip")
        print("=" * 60)
        try:
            # حذف embed_model از خروجی
            model, processor = load_blip_model(args.blip_model_path)

            blip_image_features = build_generic_image_cache(
                image_ids,
                args.image_folder,
                lambda img: get_blip_image_feature(img, model, processor),  # حذف embed_model
                dataset_type=dataset_type
            )

            print(f"✅ تعداد {len(blip_image_features)} تصویر Blip پردازش شد.")

            if blip_image_features:
                b_ids, b_feats = stack_feature_cache(blip_image_features)

                results, total, skipped = evaluate_generic(
                    data,
                    args.image_folder,
                    blip_image_features,
                    b_ids,
                    b_feats,
                    lambda img: get_blip_image_feature(img, model, processor),  # حذف embed_model
                    lambda txt: get_blip_text_feature(txt, model, processor),  # حذف embed_model
                    model_name="Blip",
                    alpha=0.5,
                    dataset_type=dataset_type
                )

                print(f"✅ Blip: {total} نمونه ارزیابی شد، {skipped} نمونه رد شد.")
                all_results["Blip"] = summarize_results(results)
            else:
                print("⚠️ هیچ ویژگی تصویری برای Blip استخراج نشد.")
                all_results["Blip"] = {
                    "MRR": 0.0,
                    "mAP@5": 0.0,
                    "mAP@10": 0.0,
                    "Prec@1": 0.0,
                    "Prec@5": 0.0,
                    "Prec@10": 0.0,
                    "Rec@1": 0.0,
                    "Rec@5": 0.0,
                    "Rec@10": 0.0
                }

            # آزادسازی حافظه
            del model, processor
            if 'blip_image_features' in locals():
                del blip_image_features
            if 'b_ids' in locals():
                del b_ids, b_feats
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ خطا در ارزیابی Blip: {e}")
            import traceback
            traceback.print_exc()
            all_results["Blip"] = {
                "MRR": 0.0,
                "mAP@5": 0.0,
                "mAP@10": 0.0,
                "Prec@1": 0.0,
                "Prec@5": 0.0,
                "Prec@10": 0.0,
                "Rec@1": 0.0,
                "Rec@5": 0.0,
                "Rec@10": 0.0
            }

    if "siglip" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی SigLIP")
        print("=" * 60)
        try:
            model, processor = load_siglip_model(args.siglip_path)

            siglip_image_features = build_siglip_batch_cache(
                image_ids,
                args.image_folder,  # ← همان args.image_folder
                model,
                processor,
                dataset_type=dataset_type,
                batch_size=16,
            )
            print(f"✅ تعداد {len(siglip_image_features)} تصویر SigLIP پردازش شد.")

            if siglip_image_features:
                s_ids, s_feats = stack_feature_cache(siglip_image_features)

                results, total, skipped = evaluate_generic(
                    data,
                    args.image_folder,
                    siglip_image_features,
                    s_ids,
                    s_feats,
                    lambda img: get_siglip_image_feature(img, model, processor),
                    lambda txt: get_siglip_text_feature(txt, model, processor),
                    model_name="SigLIP",
                    alpha=0.5,
                    dataset_type=dataset_type
                )

                print(f"✅ SigLIP: {total} نمونه ارزیابی شد، {skipped} نمونه رد شد.")
                all_results["SigLIP"] = summarize_results(results)
            else:
                print("⚠️ هیچ ویژگی تصویری برای SigLIP استخراج نشد.")

            # آزادسازی حافظه
            del model, processor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ خطا در ارزیابی SigLIP: {e}")
            import traceback  # اضافه کردن این خط به صورت محلی در صورت تداخل
            traceback.print_exc()

    if "lava" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی LLaVA")
        print("=" * 60)
        try:
            model, processor = load_lava_model()
            lava_image_features = build_generic_image_cache(
                image_ids,
                args.image_folder,
                lambda img: get_lava_image_feature(img, model, processor),
                dataset_type=dataset_type
            )
            if lava_image_features:
                l_ids, l_feats = stack_feature_cache(lava_image_features)
                results, total, skipped = evaluate_generic(
                    data,
                    args.image_folder,
                    lava_image_features,
                    l_ids,
                    l_feats,
                    lambda img: get_lava_image_feature(img, model, processor),
                    lambda txt: get_lava_text_feature(txt, model, processor),
                    model_name="LLaVA",
                    alpha=0.5,
                )
                all_results["LLaVA"] = summarize_results(results)
        except Exception as e:
            print(f"❌ خطا در ارزیابی LLaVA: {e}")
            traceback.print_exc()

    print("\n" + "=" * 140)
    print(f"📋 جدول مقایسه‌ای {args.dataset.upper()} (Open-set)")
    print("=" * 140)

    header = f"{'Model/Alpha':<20} {'MRR':<9} {'mAP@5':<9} {'mAP@10':<9} {'mAP@50':<9} {'Prec@1':<9} {'Prec@5':<9} {'Prec@10':<9} {'Prec@50':<9} {'Rec@1':<9} {'Rec@5':<9} {'Rec@10':<9} {'Rec@50':<9}"
    print(header)
    print("-" * len(header))



    for beta in sorted(args.betas):
        for alpha in sorted(args.alphas):
            key = f"CLIP-beta a={alpha:.2f} b={beta:.2f}"
            if key in all_results:
                r = all_results[key]
                print(
                    f"{key:<20} {r['mrr']:<9f} {r['map5']:<9f} {r['map10']:<9f} {r['map50']:<9f} "
                    f"{r['prec1']:<9f} {r['prec5']:<9f} {r['prec10']:<9f} {r['prec50']:<9f} "
                    f"{r['rec1']:<9f} {r['rec5']:<9f} {r['rec10']:<9f} {r['rec10']:<9f} {r['rec50']:<9f}")

    for model_name in ["Searle", "Qwen", "Blip", "LLaVA","SigLIP", "clip_sep"]:
        if model_name in all_results:
            r = all_results[model_name]
            print(
                f"{model_name:<20} {r['mrr']:<10.4f} {r['map5']:<10.4f} {r['map10']:<10.4f} {r['prec1']:<10.4f} {r['prec5']:<10.4f} {r['prec10']:<10.4f} {r['rec1']:<10.4f} {r['rec5']:<10.4f} {r['rec10']:<10.4f}")


if __name__ == "__main__":
    main()




'''
    #ویرایش شده
    for alpha in sorted(args.alphas):
        key = f"CLIP-sep alpha={alpha:.2f}"
        if key in all_results:
            r = all_results[key]
            print(
                f"{key:<20} {r['mrr']:<10.4f} {r['map5']:<10.4f} {r['map10']:<10.4f} {r['prec1']:<10.4f} {r['prec5']:<10.4f} {r['prec10']:<10.4f} {r['rec1']:<10.4f} {r['rec5']:<10.4f} {r['rec10']:<10.4f}")
'''


