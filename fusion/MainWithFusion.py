#1405.02.23
import os
import json
import argparse
import traceback
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
K_VALUES = [1, 5, 10,50]

# ===============================
# Cross-Attention Fusion Module
# ===============================
class CrossModalAttentionFusion(nn.Module):
    """ماژول Attention برای ترکیب هوشمند تصویر و متن"""

    def __init__(self, img_dim=512, txt_dim=512, hidden_dim=512, num_heads=8):
        super().__init__()

        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, img_feat, txt_feat):
        """
        Args:
            img_feat: (batch, img_dim)
            txt_feat: (batch, txt_dim)
        Returns:
            fused_feat: (batch, hidden_dim)
            attn_weights: attention weights
        """
        img = self.img_proj(img_feat).unsqueeze(1)
        txt = self.txt_proj(txt_feat).unsqueeze(1)

        attn_out, attn_weights = self.cross_attn(
            query=img,
            key=txt,
            value=txt
        )

        img = self.norm1(img + attn_out)
        ffn_out = self.ffn(img)
        fused = self.norm2(img + ffn_out)

        return fused.squeeze(1), attn_weights

class CrossModalMLPFusion(nn.Module):
    """f_query = MLP(concat(f_img, f_txt, f_img * f_txt))"""

    def __init__(self, img_dim=512, txt_dim=512, hidden_dim=512):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        # ورودی: concat(img, txt, img*txt) → 3 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, img_feat, txt_feat):
        img = self.img_proj(img_feat)   # (B, hidden_dim)
        txt = self.txt_proj(txt_feat)   # (B, hidden_dim)

        interaction = img * txt         # element-wise product
        fused = self.mlp(torch.cat([img, txt, interaction], dim=-1))
        return F.normalize(fused, dim=-1), None

class CrossModalTransformerFusion(nn.Module):
    """f_query = Transformer([f_img, f_txt]) با bidirectional attention"""

    def __init__(self, img_dim=512, txt_dim=512, hidden_dim=512, num_heads=8):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        # img attends to txt
        self.img2txt_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        # txt attends to img
        self.txt2img_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, img_feat, txt_feat):
        img = self.img_proj(img_feat).unsqueeze(1)  # (B, 1, D)
        txt = self.txt_proj(txt_feat).unsqueeze(1)  # (B, 1, D)

        img_ctx, attn_w = self.img2txt_attn(query=img, key=txt, value=txt)  # img از txt می‌خواند
        txt_ctx, _      = self.txt2img_attn(query=txt, key=img, value=img)  # txt از img می‌خواند

        fused = self.ffn(torch.cat([img_ctx.squeeze(1), txt_ctx.squeeze(1)], dim=-1))
        return F.normalize(fused, dim=-1), attn_w


def zero_shot_fusion(img_feat, txt_feat, alpha=0.5):
    """
    تلفیق Zero-Shot ویژگی‌های تصویر مرجع و متن تغییردهنده.

    Args:
        img_feat (torch.Tensor): بردار ویژگی تصویر
        txt_feat (torch.Tensor): بردار ویژگی متن
        alpha (float): وزن تصویر (بین 0 تا 1). وزن متن برابر 1 - alpha خواهد بود.

    Returns:
        torch.Tensor: بردار تلفیق‌شده و نرمال‌شده نهایی
    """
    # ۱. نرمال‌سازی L2 بردارها قبل از ترکیب (بسیار مهم)
    img_feat_norm = F.normalize(img_feat, p=2, dim=-1)
    txt_feat_norm = F.normalize(txt_feat, p=2, dim=-1)

    # ۲. ترکیب خطی ویژگی‌ها
    # فرمول: alpha * img + (1 - alpha) * txt
    fused_feat = (alpha * img_feat_norm) + ((1.0 - alpha) * txt_feat_norm)

    # ۳. نرمال‌سازی مجدد بردار نهایی برای محاسبه Cosine Similarity
    fused_feat_norm = F.normalize(fused_feat, p=2, dim=-1)

    return fused_feat_norm


# ===============================
# MLP Fusion Module (Trained)
# ===============================
class MLPFusion(nn.Module):
    """
    ماژول MLP برای ترکیب ویژگی‌های تصویر و متن
    این کلاس باید دقیقاً مطابق با مدل آموزش‌دیده باشد
    """

    def __init__(
            self,
            input_dim: int = 512,
            hidden_dim: int = 1024,
            output_dim: int = 512,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.img_proj = nn.Linear(input_dim, hidden_dim)
        self.txt_proj = nn.Linear(input_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, img_feat, text_feat):
        """
        Args:
            img_feat: (batch, input_dim)
            text_feat: (batch, input_dim)
        Returns:
            fused_feat: (batch, output_dim)
        """
        img_proj = self.img_proj(img_feat)
        txt_proj = self.txt_proj(text_feat)

        combined = torch.cat([img_proj, txt_proj], dim=-1)
        fused = self.fusion(combined)

        return F.normalize(fused, dim=-1), None


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

    # if isinstance(data, list):
    # parsed = [parse_cirr_sample(s) for s in data]
    # return "cirr", [x for x in parsed if x is not None]

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
        "map": {5: [], 10: []},
        "mrr": [],
    }


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


def mean_reciprocal_rank(relevant, retrieved):
    for rank, img in enumerate(retrieved, start=1):
        if img in relevant:
            return 1.0 / rank
    return 0.0


def summarize_results(results):
    return {
        "mrr": float(np.mean(results["mrr"])) if results["mrr"] else 0.0,
        "map5": float(np.mean(results["map"][5])) if results["map"][5] else 0.0,
        "map10": float(np.mean(results["map"][10])) if results["map"][10] else 0.0,  # اضافه شدن map10
        "prec1": float(np.mean(results["prec"][1])) if results["prec"][1] else 0.0,
        "prec5": float(np.mean(results["prec"][5])) if results["prec"][5] else 0.0,
        "prec10": float(np.mean(results["prec"][10])) if results["prec"][10] else 0.0,
        "rec1": float(np.mean(results["rec"][1])) if results["rec"][1] else 0.0,
        "rec5": float(np.mean(results["rec"][5])) if results["rec"][5] else 0.0,
        "rec10": float(np.mean(results["rec"][10])) if results["rec"][10] else 0.0,
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


def load_image(image_folder, image_id, dataset_type=None):
    path = find_image_path(image_folder, image_id, dataset_type)
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ===============================
# Model Loaders
# ===============================
def load_clip_model(model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
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

    # اگر مسیر محلی دارید، از آن استفاده کنید
    if model_path is None:
        model_path = "Salesforce/blip-itm-base-coco"  # یا مسیر محلی

    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForImageTextRetrieval.from_pretrained(model_path).to(device)
    model.eval()

    return model, processor

def load_qwen_model(model_path, device):
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
            # device_map="auto",
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


# ===============================
# Load Trained Fusion Model
# ===============================
def load_trained_fusion_model(checkpoint_path: str, device: str = "cuda"):
    """
    بارگذاری مدل آموزش‌دیده از checkpoint

    Args:
        checkpoint_path: مسیر فایل .pth
        device: دستگاه اجرا

    Returns:
        fusion_module: مدل بارگذاری‌شده
    """
    print(f"🔄 بارگذاری مدل Fusion از {checkpoint_path}...")

    # مقداردهی اولیه مدل با همان پارامترهای آموزش
    fusion_module = MLPFusion(
        input_dim=512,
        hidden_dim=1024,
        output_dim=512,
        dropout_rate=0.1
    )

    # بارگذاری checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # استخراج state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # بارگذاری وزن‌ها
    fusion_module.load_state_dict(state_dict)
    fusion_module.to(device)
    fusion_module.eval()

    print(f"✅ مدل Fusion بارگذاری شد (device: {device})")

    # نمایش اطلاعات checkpoint
    if "epoch" in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"   Loss: {checkpoint['loss']:.4f}")

    return fusion_module


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
        # برای ITM، نیاز به یک متن dummy داریم
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

        # گرفتن CLS token از خروجی vision encoder
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]

        # نرمال‌سازی
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        return image_embeds.cpu()

    except Exception as e:
        print(f"Blip image error: {e}")
        import traceback
        traceback.print_exc()
        return None

@torch.no_grad()
def get_blip_text_feature(caption, model, processor):
    """استخراج ویژگی متن با Blip-ITM"""
    try:
        # برای ITM، نیاز به یک تصویر dummy داریم
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

        # گرفتن CLS token از خروجی text encoder
        text_embeds = text_outputs.last_hidden_state[:, 0, :]

        # نرمال‌سازی
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return text_embeds.cpu()

    except Exception as e:
        print(f"Blip text error: {e}")
        import traceback
        traceback.print_exc()
        return None

# @torch.no_grad()
@torch.inference_mode()
def get_qwen_image_feature(image, model, processor, device):
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
def get_qwen_text_feature(text, model, processor, device):
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


# ===============================
# Feature Cache Builders
# ===============================
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

        batch = valid_imgs[i:i + batch_size]

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
    for k in K_VALUES:
        top_k = ranked_ids[:k]
        hits = sum(1 for img in top_k if img in positives)
        results["prec"][k].append(hits / k)
        results["rec"][k].append(hits / len(positives))

    results["map"][5].append(average_precision_at_k(positives, ranked_ids, 5))
    results["map"][10].append(average_precision_at_k(positives, ranked_ids, 10))  # اضافه شدن map@10
    results["mrr"].append(mean_reciprocal_rank(positives, ranked_ids))


# ===============================
# Evaluators
# ===============================
@torch.no_grad()
def evaluate_with_trained_fusion(
        data,
        clip_model,
        fusion_module,
        image_features_cache,
        gallery_ids,
        gallery_feats
):
    """
    ارزیابی با استفاده از مدل Fusion آموزش‌دیده
    """
    results = init_results()
    total = 0
    skipped = 0

    # نرمال‌سازی ویژگی‌های گالری
    gallery_feats_norm = F.normalize(gallery_feats.to(device).float(), dim=-1)

    text_cache = {}

    for item in tqdm(data, desc="CLIP-Fusion (Trained)"):

        reference_id = item["reference_id"]
        target_id = item["target_id"]
        caption = item["caption"]

        positives = {target_id}

        ref_feat = image_features_cache.get(reference_id)

        if ref_feat is None:
            skipped += 1
            continue

        caption_full = "a photo of " + caption

        if caption_full not in text_cache:
            text_cache[caption_full] = get_clip_text_feature(
                caption_full,
                clip_model
            )


@torch.no_grad()
def evaluate_with_fusion(
        data,
        clip_model,
        fusion_module,
        image_features_cache,
        gallery_ids,
        gallery_feats
):
    results = init_results()
    total = 0
    skipped = 0

    # projection گالری به فضای fusion
    projected_gallery = fusion_module.img_proj(
        gallery_feats.to(device)
    )

    projected_gallery = F.normalize(projected_gallery, dim=-1)

    text_cache = {}

    for item in tqdm(data, desc="CLIP-Fusion"):

        reference_id = item["reference_id"]
        target_id = item["target_id"]
        caption = item["caption"]

        positives = {target_id}

        ref_feat = image_features_cache.get(reference_id)

        if ref_feat is None:
            skipped += 1
            continue

        caption_full = "a photo of " + caption

        if caption_full not in text_cache:
            text_cache[caption_full] = get_clip_text_feature(
                caption_full,
                clip_model
            )

        txt_feat = text_cache[caption_full]

        if txt_feat is None:
            skipped += 1
            continue

        try:
            fused_feat, attn_weights = fusion_module(
                ref_feat.to(device).float(),
                txt_feat.to(device).float()
            )

            fused_feat = F.normalize(fused_feat, dim=-1)

            sims = torch.matmul(
                projected_gallery,
                fused_feat.squeeze(0).T
            )

            ranked_idx = torch.argsort(
                sims,
                descending=True
            ).cpu().tolist()

            ranked_ids = [gallery_ids[i] for i in ranked_idx]

            update_metrics(results, ranked_ids, positives)

            total += 1

        except Exception as e:
            print(f"Fusion error: {e}")
            skipped += 1

    return results, total, skipped


@torch.no_grad()
def evaluate_zero_shot(
        data,
        clip_model,
        image_features_cache,
        gallery_ids,
        gallery_feats,
        alpha=0.5  # پارامتر آلفا اضافه شد
):
    results = init_results()
    total = 0
    skipped = 0

    # در حالت زیروشات، نیازی به projection گالری نیست. فقط نرمال‌سازی می‌کنیم.
    normalized_gallery = F.normalize(gallery_feats.to(device).float(), p=2, dim=-1)

    text_cache = {}

    for item in tqdm(data, desc="CLIP Zero-Shot Fusion"):

        reference_id = item["reference_id"]
        target_id = item["target_id"]
        caption = item["caption"]

        positives = {target_id}

        ref_feat = image_features_cache.get(reference_id)

        if ref_feat is None:
            skipped += 1
            continue

        caption_full = "a photo of " + caption

        if caption_full not in text_cache:
            text_cache[caption_full] = get_clip_text_feature(
                caption_full,
                clip_model
            )

        txt_feat = text_cache[caption_full]

        if txt_feat is None:
            skipped += 1
            continue

        try:
            # -----------------------------------------------------
            # استفاده از Zero-shot fusion به جای ماژول پارامتریک
            # -----------------------------------------------------
            ref_feat_tensor = ref_feat.to(device).float()
            txt_feat_tensor = txt_feat.to(device).float()

            # اطمینان از یکسان بودن ابعاد
            if ref_feat_tensor.dim() == 1:
                ref_feat_tensor = ref_feat_tensor.unsqueeze(0)
            if txt_feat_tensor.dim() == 1:
                txt_feat_tensor = txt_feat_tensor.unsqueeze(0)

            query_feat = zero_shot_fusion(ref_feat_tensor, txt_feat_tensor, alpha=alpha)

            # محاسبه شباهت
            sims = torch.matmul(
                normalized_gallery,
                query_feat.squeeze(0).T
            )

            if sims.dim() > 1:
                sims = sims.squeeze(-1)  # اطمینان از 1D بودن sims

            ranked_idx = torch.argsort(
                sims,
                descending=True
            ).cpu().tolist()

            ranked_ids = [gallery_ids[i] for i in ranked_idx]

            update_metrics(results, ranked_ids, positives)

            total += 1

        except Exception as e:
            print(f"Fusion error: {e}")
            skipped += 1

    return results, total, skipped

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
            ref_img = load_image(image_folder, reference_id, dataset_type)
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
    parser.add_argument("--models", type=str, nargs="+",
                        default=["clip", "searle", "qwen", "blip", "lava","clip-fusion"])

    args = parser.parse_args()

    print(f"🔄 بارگذاری دیتاست {args.dataset.upper()}...")
    detected_dataset, data = load_dataset(args.json_path)
    if detected_dataset != args.dataset:
        raise ValueError(f"Dataset mismatch: arg={args.dataset}, detected={detected_dataset}")

    print(f"✅ تعداد {len(data)} نمونه بارگذاری شد.")

    print("🔄 بارگذاری مدل CLIP...")
    clip_model, preprocess = load_clip_model()

    # ===============================
    # Fusion Module
    # ===============================
    '''
    fusion_module = CrossModalAttentionFusion(
        img_dim=512,  # برای CLIP ViT-B/32
        txt_dim=512,
        hidden_dim=512,
        num_heads=8
    ).to(device)
    
    # رویکرد ۱
    #fusion_module = CrossModalMLPFusion(img_dim=512, txt_dim=512, hidden_dim=512).to(device)

    # یا رویکرد ۲
    #fusion_module = CrossModalTransformerFusion(img_dim=512, txt_dim=512, hidden_dim=512, num_heads=8).to(device)
'''
    fusion_module = load_trained_fusion_model(
        checkpoint_path="best_fusion_model.pth",
        device=device
    )

    fusion_module.eval()


    image_ids = set()
    for item in data:
        image_ids.add(item["reference_id"])
        image_ids.add(item["target_id"])
        for member in item.get("members", []):
            image_ids.add(member)

    dataset_type = args.dataset

    print(f"🔄 استخراج ویژگی تصاویر CLIP از {args.image_folder} ...")
    image_features_cache, image_features_cache_raw = build_clip_image_cache(
        image_ids, args.image_folder, clip_model, preprocess, dataset_type
    )
    print(f"✅ تعداد {len(image_features_cache)} تصویر پردازش شد.")

    gallery_ids, gallery_feats = stack_feature_cache(image_features_cache)
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

    if "clip-fusion" in args.models:
        print("\n" + "=" * 60)
        print("🔄 ارزیابی clip-fusion")
        print("=" * 60)

        fusion_results, total_fusion, skipped_fusion = evaluate_with_trained_fusion(
            data=data,  # ✅ اصلاح شد
            clip_model=clip_model,
            fusion_module=fusion_module,  # ✅ اصلاح شد
            image_features_cache=image_features_cache,  # ✅ اصلاح شد
            gallery_ids=gallery_ids,
            gallery_feats=gallery_feats
        )
        print(f"✅ CLIP-Fusion: {total_fusion} نمونه ارزیابی شد، {skipped_fusion} نمونه رد شد.")
        all_results["CLIP-Fusion"] = summarize_results(fusion_results)



        '''
        fusion_results, total_fusion, skipped_fusion = evaluate_with_fusion(
            data=data,
            clip_model=clip_model,
            fusion_module=fusion_module,
            image_features_cache=image_features_cache,
            gallery_ids=gallery_ids,
            gallery_feats=gallery_feats
        )
        '''




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
                    lambda img: get_qwen_image_feature(img, model, processor, device),
                    lambda txt: get_qwen_text_feature(txt, model, processor, device),
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
            model, processor, embed_model = load_blip_model(args.blip_model_path)
            blip_image_features = build_generic_image_cache(
                image_ids,
                args.image_folder,
                lambda img: get_blip_image_feature(img, model, processor, embed_model),
                dataset_type=dataset_type
            )
            if blip_image_features:
                b_ids, b_feats = stack_feature_cache(blip_image_features)
                results, total, skipped = evaluate_generic(
                    data,
                    args.image_folder,
                    blip_image_features,
                    b_ids,
                    b_feats,
                    lambda img: get_blip_image_feature(img, model, processor, embed_model),
                    lambda txt: get_blip_text_feature(txt, embed_model),
                    model_name="Blip",
                    alpha=0.5,
                )
                all_results["Blip"] = summarize_results(results)
        except Exception as e:
            print(f"❌ خطا در ارزیابی Blip: {e}")
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

    header = f"{'Model/Alpha':<20} {'MRR':<10} {'mAP@5':<10} {'mAP@10':<10} {'Prec@1':<10} {'Prec@5':<10} {'Prec@10':<10} {'Rec@1':<10} {'Rec@5':<10} {'Rec@10':<10}"
    print(header)
    print("-" * len(header))

    for alpha in sorted(args.alphas):
        key = f"CLIP alpha={alpha:.2f}"
        if key in all_results:
            r = all_results[key]
            print(
                f"{key:<20} {r['mrr']:<10.4f} {r['map5']:<10.4f} {r['map10']:<10.4f} {r['prec1']:<10.4f} {r['prec5']:<10.4f} {r['prec10']:<10.4f} {r['rec1']:<10.4f} {r['rec5']:<10.4f} {r['rec10']:<10.4f}")

    for model_name in ["Searle", "Qwen", "Blip", "LLaVA","CLIP-Fusion"]:
        if model_name in all_results:
            r = all_results[model_name]
            print(
                f"{model_name:<20} {r['mrr']:<10.4f} {r['map5']:<10.4f} {r['map10']:<10.4f} {r['prec1']:<10.4f} {r['prec5']:<10.4f} {r['prec10']:<10.4f} {r['rec1']:<10.4f} {r['rec5']:<10.4f} {r['rec10']:<10.4f}")


if __name__ == "__main__":
    main()
