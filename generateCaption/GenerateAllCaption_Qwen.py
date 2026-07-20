import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================
# Settings
# ============================
MODEL_NAME = r"./models_download/Qwen2-VL-2B-Instruct"
IMAGE_FOLDER = "./nlvr/nlvr2/images/test1"
OUTPUT_JSON = "./generateCaption/generated_captions.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# ============================
# Prompt Templates
# ============================
SYSTEM_PROMPT = "You are a helpful assistant that generates detailed image captions."

USER_PROMPT_TEMPLATE = """
Please generate a detailed caption for this image. 
Focus on:
- Main objects and subjects
- Actions and activities
- Scene setting and environment
- Colors and visual details
- Spatial relationships

Caption:
"""


# ============================
# Helper Functions
# ============================
def list_images(image_folder):
    """
    لیست تمام تصاویر داخل پوشه به همراه مسیر و image_id (نام فایل بدون پسوند)
    """
    images = []
    for fname in sorted(os.listdir(image_folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_id = os.path.splitext(fname)[0]
            image_path = os.path.join(image_folder, fname)
            images.append((image_id, image_path))
    return images


def clean_caption(text):
    """
    پاکسازی و نرمال‌سازی کپشن تولید شده
    """
    text = text.strip()

    prefixes = ["Caption:", "caption:", "CAPTION:", "Image caption:", "Description:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text


def load_existing_results(output_path):
    """
    بارگذاری نتایج قبلی (در صورت وجود) برای ادامه از همان‌جا
    خروجی: لیست نتایج + مجموعه image_id هایی که caption معتبر دارند
    """
    if not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Warning: could not read existing output file, starting fresh.")
        return [], set()

    done_ids = set()
    for item in results:
        ref = item.get("reference")
        caption = item.get("caption", "")
        # فقط آن‌هایی که caption غیرخالی دارند را «انجام‌شده» در نظر می‌گیریم
        if ref and caption:
            done_ids.add(ref)

    return results, done_ids


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# ============================
# Load Model and Processor
# ============================
print(f"Loading model from: {MODEL_NAME}")
print(f"Device: {DEVICE}")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True,
    min_pixels=224 * 224,
    max_pixels=512 * 512
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    local_files_only=True
)

if DEVICE == "cpu":
    model = model.to("cpu")

model.eval()
print("Model loaded successfully!")

# ============================
# Collect images + previous results
# ============================
images = list_images(IMAGE_FOLDER)
print(f"\nTotal images found in folder: {len(images)}")

results, done_ids = load_existing_results(OUTPUT_JSON)
print(f"Already captioned (will skip): {len(done_ids)}")

# ============================
# Generate Captions
# ============================
processed_count = 0
skipped_count = 0
error_count = 0

print("\nStarting caption generation...")

for idx, (image_id, image_path) in enumerate(images):
    # رد کردن تصاویری که از قبل caption دارند
    if image_id in done_ids:
        skipped_count += 1
        print(f"[{idx + 1}/{len(images)}] ⏭️  Skipped (already done): {image_id}")
        continue

    try:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": USER_PROMPT_TEMPLATE}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        if DEVICE == "cuda":
            inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        caption = clean_caption(output_text)

        results.append({
            "reference": image_id,
            "caption": caption
        })
        done_ids.add(image_id)
        save_results(results, OUTPUT_JSON)  # ← ذخیره فوری

        processed_count += 1
        print(f"[{idx + 1}/{len(images)}] ✅ Processed: {image_id}")

    except Exception as e:
        print(f"[{idx + 1}/{len(images)}] ❌ Error processing {image_id}: {str(e)}")
        error_count += 1
        results.append({
            "reference": image_id,
            "caption": "",
            "error": str(e)
        })
        save_results(results, OUTPUT_JSON)

# ============================
# Final Save
# ============================
save_results(results, OUTPUT_JSON)

print("\n✅ Caption generation completed!")
print(f"Processed: {processed_count}")
print(f"Skipped (already done): {skipped_count}")
print(f"Errors: {error_count}")
print(f"Saved to: {OUTPUT_JSON}")
