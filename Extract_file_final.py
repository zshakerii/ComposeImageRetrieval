import os
import json
import pandas as pd


# =========================================================
# تنظیمات
# =========================================================

PARQUET_FILES = [
    "./royokongcirr_val/val-00000-of-00002.parquet",
    "./royokongcirr_val/val-00001-of-00002.parquet",
]

IMAGE_FOLDER = "./cirr-val"

JSON_OUTPUT = "./cirr-val.json"


# =========================================================
# ساخت پوشه تصاویر
# =========================================================

os.makedirs(IMAGE_FOLDER, exist_ok=True)


# =========================================================
# ذخیره تصویر داخل Parquet
# =========================================================

def save_image(image_data):
    """
    image_data ساختاری شبیه:

    {
        "bytes": b"...",
        "path": "dev-1028-1-img1.png"
    }

    خروجی:
        نام فایل ذخیره شده
    """

    if image_data is None:
        return None

    if not isinstance(image_data, dict):
        return None

    image_bytes = image_data.get("bytes")
    image_path = image_data.get("path")

    if image_bytes is None:
        return None

    if not image_path:
        return None

    # فقط نام فایل
    filename = os.path.basename(image_path)

    output_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    # اگر قبلاً ذخیره نشده، ذخیره کن
    if not os.path.exists(output_path):

        with open(output_path, "wb") as f:
            f.write(image_bytes)

    return filename


# =========================================================
# پردازش یک فایل Parquet
# =========================================================

def process_parquet(
    parquet_file,
    json_data,
    saved_images
):
    """
    یک Parquet را پردازش می‌کند
    و رکوردهای JSON را به json_data اضافه می‌کند.
    """

    print()
    print("=" * 70)
    print("Loading:")
    print(parquet_file)
    print("=" * 70)

    # -----------------------------------------------------
    # بررسی وجود فایل
    # -----------------------------------------------------

    if not os.path.exists(parquet_file):

        print(
            f"[ERROR] File not found: {parquet_file}"
        )

        return 0

    # -----------------------------------------------------
    # خواندن Parquet
    # -----------------------------------------------------

    df = pd.read_parquet(
        parquet_file
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {df.columns.tolist()}"
    )

    # -----------------------------------------------------
    # بررسی ستون‌های مورد نیاز
    # -----------------------------------------------------

    required_columns = {
        "candidate_id",
        "caption",
        "group",
        "target_id",
        "target",
        "candidate"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        print(
            f"[ERROR] Missing columns: "
            f"{sorted(missing_columns)}"
        )

        return 0

    # -----------------------------------------------------
    # پردازش رکوردها
    # -----------------------------------------------------

    for index, row in df.iterrows():

        candidate_id = row["candidate_id"]

        caption = row["caption"]

        group = row["group"]

        target_id = row["target_id"]

        target = row["target"]

        candidate = row["candidate"]

        # -------------------------------------------------
        # ذخیره Target
        # -------------------------------------------------

        target_filename = save_image(
            target
        )

        if target_filename:

            saved_images.add(
                target_filename
            )

        # -------------------------------------------------
        # ذخیره Candidate
        # -------------------------------------------------

        candidate_filename = save_image(
            candidate
        )

        if candidate_filename:

            saved_images.add(
                candidate_filename
            )

        # -------------------------------------------------
        # تبدیل numpy array به list
        # -------------------------------------------------

        if hasattr(group, "tolist"):

            group = group.tolist()

        # -------------------------------------------------
        # ساخت JSON record
        # -------------------------------------------------

        record = {
            "candidate_id": candidate_id,

            "caption": caption,

            "group": group,

            "target_id": target_id
        }

        json_data.append(
            record
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (index + 1) % 100 == 0:

            print(
                f"Processed: "
                f"{index + 1:,}/{len(df):,} | "
                f"Total records: {len(json_data):,} | "
                f"Images: {len(saved_images):,}"
            )

    return len(df)


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("CIRR PARQUET -> JSON + IMAGES")
    print("=" * 70)

    # -----------------------------------------------------
    # لیست نهایی JSON
    # -----------------------------------------------------

    json_data = []

    # -----------------------------------------------------
    # تصاویر ذخیره شده در هر دو فایل
    # -----------------------------------------------------

    saved_images = set()

    total_rows = 0

    # -----------------------------------------------------
    # پردازش هر دو Parquet
    # -----------------------------------------------------

    for parquet_file in PARQUET_FILES:

        processed_rows = process_parquet(
            parquet_file=parquet_file,
            json_data=json_data,
            saved_images=saved_images
        )

        total_rows += processed_rows

    # =====================================================
    # ذخیره JSON نهایی
    # =====================================================

    print()
    print("=" * 70)
    print("Saving JSON")
    print("=" * 70)

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # =====================================================
    # نتیجه
    # =====================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Parquet files     : {len(PARQUET_FILES)}"
    )

    print(
        f"Total rows        : {total_rows:,}"
    )

    print(
        f"JSON records      : {len(json_data):,}"
    )

    print(
        f"Unique images     : {len(saved_images):,}"
    )

    print(
        f"Image folder      : "
        f"{os.path.abspath(IMAGE_FOLDER)}"
    )

    print(
        f"JSON file         : "
        f"{os.path.abspath(JSON_OUTPUT)}"
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()