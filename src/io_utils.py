import os
import json

from PIL import Image

from .config import IMAGE_EXTS


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
    cap_keys = ("caption", "generated_caption", "text")

    # FIX (باگ ۲): حالت dict که در docstring وعده داده شده بود اما اصلاً
    # پیاده‌سازی نشده بود و همیشه به ValueError می‌رسید، حالا پشتیبانی می‌شود.
    if isinstance(data, dict):
        for raw_id, value in data.items():
            img_id = os.path.splitext(str(raw_id))[0]

            if isinstance(value, dict):
                caption = ""
                for k in cap_keys:
                    if value.get(k):
                        caption = value[k]
                        break
            else:
                caption = value if value else ""

            captions[img_id] = caption
        return captions

    if isinstance(data, list):
        id_keys = ("image_id", "img_id", "reference_img_id", "id", "reference")

        for entry in data:
            if not isinstance(entry, dict):
                continue

            img_id = None
            for k in id_keys:
                if entry.get(k) is not None:
                    img_id = str(entry[k])
                    break
            if img_id is None:
                continue

            # حذف پسوند فایل اگر وجود داشت (هماهنگ با parse_cirr_sample)
            img_id = os.path.splitext(img_id)[0]

            caption = ""
            for k in cap_keys:
                if entry.get(k):
                    caption = entry[k]
                    break

            captions[img_id] = caption
        return captions

    raise ValueError(f"فرمت ناشناخته‌ی generated captions: {type(data)}")
