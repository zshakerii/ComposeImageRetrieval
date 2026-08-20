import json
import os
from pathlib import Path


# =========================
# تنظیمات
# =========================

JSON_PATH = r"cap.rc2.val.json"

IMAGE_FOLDER = r"./nlvr/nlvr2/images/val"

# فایل خروجی تصاویر مفقود
MISSING_OUTPUT = "missing_images.json"


# =========================
# پیدا کردن تصویر
# =========================

def find_image(image_name, image_folder):
    """
    بررسی می‌کند آیا تصویر با نام داده شده در پوشه وجود دارد یا خیر.

    مثال:
        test1-569-2-img0
    بررسی می‌شود برای:
        test1-569-2-img0.jpg
        test1-569-2-img0.jpeg
        test1-569-2-img0.png
    """

    extensions = [".jpg", ".jpeg", ".png", ".webp"]

    image_folder = Path(image_folder)

    for ext in extensions:
        path = image_folder / (image_name + ext)

        if path.exists() and path.is_file():
            return path

    return None


# =========================
# استخراج نام تصاویر
# =========================

def extract_images(data):
    """
    تمام image id های موجود در JSON را استخراج می‌کند.
    """

    images = set()

    if not isinstance(data, list):
        data = [data]

    for item in data:

        # reference
        reference = item.get("reference")

        if reference:
            images.add(reference)

        # target_hard
        target_hard = item.get("target_hard")

        if target_hard:
            images.add(target_hard)

        # target_soft
        target_soft = item.get("target_soft")

        if isinstance(target_soft, dict):
            images.update(target_soft.keys())

        # img_set.members
        img_set = item.get("img_set")

        if isinstance(img_set, dict):

            members = img_set.get("members", [])

            if isinstance(members, list):
                images.update(members)

    return sorted(images)


# =========================
# Main
# =========================

def main():

    print("=" * 70)
    print("Checking dataset images")
    print("=" * 70)

    print(f"JSON   : {JSON_PATH}")
    print(f"Images : {IMAGE_FOLDER}")
    print()

    # -------------------------
    # خواندن JSON
    # -------------------------

    if not os.path.exists(JSON_PATH):
        print(f"[ERROR] JSON file not found: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------
    # استخراج تصاویر
    # -------------------------

    images = extract_images(data)

    print(f"Unique images referenced in JSON: {len(images)}")
    print()

    # -------------------------
    # بررسی فایل‌ها
    # -------------------------

    missing_images = []
    existing_images = []

    for index, image_name in enumerate(images, start=1):

        image_path = find_image(
            image_name,
            IMAGE_FOLDER
        )

        if image_path is None:

            missing_images.append({
                "image_name": image_name,
                "expected_folder": str(IMAGE_FOLDER)
            })

            print(f"[MISSING] {image_name}")

        else:

            existing_images.append({
                "image_name": image_name,
                "path": str(image_path)
            })

    # -------------------------
    # ذخیره تصاویر مفقود
    # -------------------------

    with open(
        MISSING_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            missing_images,
            f,
            ensure_ascii=False,
            indent=2
        )

    # -------------------------
    # گزارش
    # -------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Total unique images : {len(images)}")
    print(f"Existing images     : {len(existing_images)}")
    print(f"Missing images      : {len(missing_images)}")

    print()
    print(f"Missing images saved to:")
    print(MISSING_OUTPUT)


if __name__ == "__main__":
    main()