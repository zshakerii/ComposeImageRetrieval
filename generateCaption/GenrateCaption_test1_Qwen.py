import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================
# Settings
# ============================
MODEL_NAME = r"./Models/Qwen2-VL-2B-Instruct"
IMAGE_FOLDER = "./nlvr/nlvr2/images/test1"
INPUT_JSON = "./cap.rc2.test1.json"
OUTPUT_JSON = "./generated_captions.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 1

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
def find_image_path(image_id, image_folder):
    """جستجوی مسیر تصویر بر اساس image_id"""
    possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for ext in possible_extensions:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None


def clean_caption(text):
    """پاکسازی و نرمال‌سازی کپشن تولید شده"""
    text = text.strip()
    prefixes = ["Caption:", "caption:", "CAPTION:", "Image caption:", "Description:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def collect_unique_image_ids(data):
    """
    جمع‌آوری همه image_idهای یکتا از reference و img_set.members
    ترتیب را حفظ می‌کند تا خروجی قابل پیش‌بینی باشد.
    """
    seen = set()
    ordered_ids = []

    for item in data:
        candidates = []

        ref = item.get("reference")
        if ref:
            candidates.append(ref)

        img_set = item.get("img_set", {})
        members = img_set.get("members", []) if isinstance(img_set, dict) else []
        candidates.extend(members)

        for image_id in candidates:
            if image_id and image_id not in seen:
                seen.add(image_id)
                ordered_ids.append(image_id)

    return ordered_ids


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def generate_caption(image_path, model, processor):
    """تولید کپشن برای یک تصویر"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT_TEMPLATE},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if DEVICE == "cuda":
        inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return clean_caption(output_text)


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
    max_pixels=512 * 512,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    local_files_only=True,
)

if DEVICE == "cpu":
    model = model.to("cpu")

model.eval()
print("Model loaded successfully!")

# ============================
# Load Input Data
# ============================
print(f"\nLoading input data from: {INPUT_JSON}")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# اگر فایل یک آبجکت تکی باشد، تبدیل به لیست
if isinstance(data, dict):
    data = [data]

unique_image_ids = collect_unique_image_ids(data)
print(f"Total samples: {len(data)}")
print(f"Total unique images to process: {len(unique_image_ids)}")

# ============================
# Resume support: load existing results if any
# ============================
results = {}
if os.path.exists(OUTPUT_JSON):
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            results = existing
        print(f"Resuming: {len(results)} captions already exist.")
    except Exception:
        results = {}

# ============================
# Generate Captions
# ============================
processed_count = 0
skipped_count = 0
error_count = 0

print("\nStarting caption generation...")

for idx, image_id in enumerate(unique_image_ids):
    # رد کردن تصاویری که قبلاً با موفقیت پردازش شده‌اند
    if image_id in results and results[image_id].get("caption"):
        skipped_count += 1
        continue

    image_path = find_image_path(image_id, IMAGE_FOLDER)

    if not image_path:
        print(f"[{idx + 1}/{len(unique_image_ids)}] Error: Image not found for ID {image_id}")
        error_count += 1
        results[image_id] = {"caption": "", "error": "Image file not found"}
        continue

    try:
        caption = generate_caption(image_path, model, processor)
        results[image_id] = {"caption": caption}
        save_results(results, OUTPUT_JSON)  # ذخیره فوری

        processed_count += 1
        print(f"[{idx + 1}/{len(unique_image_ids)}] ✅ Processed: {image_id}")

    except Exception as e:
        print(f"[{idx + 1}/{len(unique_image_ids)}] ❌ Error processing {image_id}: {str(e)}")
        error_count += 1
        results[image_id] = {"caption": "", "error": str(e)}
        save_results(results, OUTPUT_JSON)

# ============================
# Save Results
# ============================
print("\nSaving results...")
save_results(results, OUTPUT_JSON)

print("\n✅ Caption generation completed!")
print(f"Processed (new): {processed_count}")
print(f"Skipped (already done): {skipped_count}")
print(f"Errors: {error_count}")
print(f"Saved to: {OUTPUT_JSON}")
