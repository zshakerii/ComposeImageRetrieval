import sys
import os
from contextlib import contextmanager

import torch
import yaml


@contextmanager
def _searle_import_context(searle_dir: str):
    """
    موقتاً محیط import را برای بارگذاری SEARLE آماده می‌کند:
    - SEARLE_DIR را ابتدای sys.path می‌گذارد تا src ریپوی SEARLE اولویت بگیرد
    - ماژول‌های src کش‌شده‌ی پروژه‌ی کاربر را کنار می‌گذارد
    - در پایان، وضعیت اولیه‌ی sys.path و sys.modules را بازمی‌گرداند
    """
    # ۱. نگه‌داری وضعیت فعلی sys.path
    original_sys_path = list(sys.path)

    # ۲. کنار گذاشتن ماژول‌های src کاربر (src و همه‌ی src.*)
    stashed_modules = {}
    for name in list(sys.modules.keys()):
        if name == "src" or name.startswith("src."):
            stashed_modules[name] = sys.modules.pop(name)

    # ۳. اولویت دادن به مسیر SEARLE
    sys.path.insert(0, searle_dir)

    try:
        yield
    finally:
        # ۴. بازگرداندن sys.path به حالت اولیه
        sys.path[:] = original_sys_path

        # ۵. حذف هر ماژول src که در حین لود SEARLE ساخته شده
        for name in list(sys.modules.keys()):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]

        # ۶. بازگرداندن ماژول‌های اصلی کاربر
        sys.modules.update(stashed_modules)


def load_searle_config(path: str = "searle.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["searle"]


def load_searle_model(config: dict, device: str = "cuda"):
    """
    مدل SEARLE را از طریق torch.hub بارگذاری می‌کند.

    config کلیدهای زیر را می‌خواند:
        searle_dir: مسیر کلون‌شده‌ی محلی ریپوی SEARLE
        backbone:   'ViT-B/32' یا 'ViT-L/14'
        source:     'local' (پیش‌فرض) یا 'github'
    """
    searle_dir = config["searle_dir"]
    backbone = config.get("backbone", "ViT-L/14")
    source = config.get("source", "local")

    if source == "local" and not os.path.isdir(searle_dir):
        raise FileNotFoundError(
            f"مسیر SEARLE پیدا نشد: {searle_dir}. "
            f"ابتدا ریپو را کلون کن: git clone https://github.com/miccunifi/SEARLE"
        )

    with _searle_import_context(searle_dir):
        phi, clip_model = torch.hub.load(
            repo_or_dir=searle_dir,
            source=source,
            model="searle",
            backbone=backbone,
        )

    phi = phi.to(device).eval()
    clip_model = clip_model.to(device).eval()
    return phi, clip_model


@torch.no_grad()
def get_searle_image_feature(clip_model, images, device: str = "cuda"):
    """
    ویژگی تصویر را استخراج و L2-normalize می‌کند.
    نرمال‌سازی برای عملکرد صحیح SEARLE در فضای شباهت کسینوسی ضروری است.
    """
    images = images.to(device)
    image_features = clip_model.encode_image(images)
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    return image_features


'''
def load_searle_model(clip_model_name="ViT-B/32"):
    searle, encode_with_pseudo_tokens = torch.hub.load(
        repo_or_dir="miccunifi/SEARLE",
        source="github",
        model="searle",
        backbone=clip_model_name,
    )
    searle.to(device).eval()
    return searle, encode_with_pseudo_tokens



@torch.no_grad()
def get_searle_image_feature(image, model, preprocess):
    x = preprocess(image).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    return feat

'''
