import os
import json
import atexit
from typing import List, Tuple, Dict

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================
# Settings
# ============================
MODEL_NAME = r"./models_download/Qwen2-VL-2B-Instruct"
IMAGE_FOLDER = "./nlvr/nlvr2/images/test1"
OUTPUT_JSON = "./generateCaption/generated_captions.json"

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
MAX_NEW_TOKENS = 64
SAVE_EVERY_N_BATCHES = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 if DEVICE == "cuda" else 1

if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
elif DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

SYSTEM_PROMPT = (
    "You are a precise visual captioning assistant. For each image, output "
    "exactly one dense, information-rich caption sentence (two only if "
    "strictly necessary). Never use introductions, labels, prefixes, or "
    "bullet points."
)

USER_PROMPT = (
    "Describe this image in one dense sentence (two maximum). Include the "
    "main subjects, their actions, the setting, and important spatial "
    "relationships. Do not add any introduction, label, or explanation."
)


# ============================
# Helper functions
# ============================
def list_images(image_folder):
    images = []
    for fname in sorted(os.listdir(image_folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_id = os.path.splitext(fname)[0]
            image_path = os.path.join(image_folder, fname)
            images.append((image_id, image_path))
    return images


def clean_caption(text):
    text = text.strip()
    prefixes = ["Caption:", "caption:", "CAPTION:", "Image caption:", "Description:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def load_existing_results(output_path) -> Dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Warning: could not read existing output file, starting fresh.")
        return {}
    results = {}
    for item in raw:
        ref = item.get("reference")
        if ref:
            results[ref] = item
    return results


def save_results_atomic(results: Dict[str, dict], output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, output_path)


def build_messages(image_path):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]


# ============================
# Model loading
# ============================
def load_model_and_processor():
    print(f"Loading model from: {MODEL_NAME}")
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
        min_pixels=224 * 224,
        max_pixels=512 * 512,
    )

    # CRITICAL for correct batched generation on a causal decoder: padding
    # must be on the LEFT so every row's real last token lands at the same
    # relative position. Right-padding (a common tokenizer default) silently
    # corrupts batched generation for models like Qwen2-VL.
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    PAD_TOKEN_ID = processor.tokenizer.pad_token_id
    EOS_TOKEN_ID = processor.tokenizer.eos_token_id

    attn_impl = "eager"
    if DEVICE == "cuda":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

    model_kwargs = dict(
        torch_dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    if DEVICE == "cuda":
        model_kwargs["device_map"] = "auto"

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception as e:
        if attn_impl != "eager":
            print(f"⚠️ {attn_impl} unavailable ({e}); falling back to eager attention.")
            model_kwargs["attn_implementation"] = "eager"
            model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
        else:
            raise

    if DEVICE == "cpu":
        model = model.to("cpu")

    model.eval()
    print(f"Model loaded successfully! (attention: {model_kwargs['attn_implementation']})")
    return model, processor, PAD_TOKEN_ID, EOS_TOKEN_ID


# ============================
# Batched generation
# ============================
@torch.inference_mode()
def generate_batch_captions(
    model, processor, batch_items: List[Tuple[str, str]], pad_token_id: int, eos_token_id: int
) -> List[str]:
    """batch_items: list of (image_id, image_path). Returns cleaned captions
    in the same order as batch_items."""
    all_messages = [build_messages(path) for _, path in batch_items]

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]

    # process_vision_info runs per-conversation; images must be concatenated
    # in the SAME order the conversations were built, matching the text list
    # 1:1 — a misordered concatenation here is the classic cause of
    # corrupted captions under batching in Qwen2-VL.
    image_inputs = []
    for m in all_messages:
        imgs, _ = process_vision_info(m)
        image_inputs.extend(imgs or [])

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )

    if DEVICE == "cuda":
        inputs = inputs.to(DEVICE, non_blocking=True)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=pad_token_id,   # <-- explicit: the actual fix for
        eos_token_id=eos_token_id,   #     empty/garbled batched output
    )

    # Row-by-row trimming: each row's own input_ids length is used to slice
    # off the prompt. Under correct left-padding every row has the same
    # padded length, so this is equivalent to a fixed-index slice — but
    # unifies the batch-of-1 and batch-of-N code paths and removes the
    # "same slice index for every row" assumption from the code itself.
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return [clean_caption(t) for t in decoded]


def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# ============================
# Main loop
# ============================
def main():
    model, processor, pad_token_id, eos_token_id = load_model_and_processor()

    images = list_images(IMAGE_FOLDER)
    print(f"\nTotal images found in folder: {len(images)}")

    results = load_existing_results(OUTPUT_JSON)
    done_ids = {img_id for img_id, item in results.items() if item.get("caption")}
    print(f"Already captioned (will skip): {len(done_ids)}")

    pending = [(img_id, path) for img_id, path in images if img_id not in done_ids]
    print(f"Remaining to process: {len(pending)}")

    processed_count = 0
    error_count = 0
    batches_since_save = 0

    def flush():
        save_results_atomic(results, OUTPUT_JSON)

    atexit.register(flush)

    try:
        i = 0
        while i < len(pending):
            batch = pending[i:i + BATCH_SIZE]
            try:
                captions = generate_batch_captions(
                    model, processor, batch, pad_token_id, eos_token_id
                )
                for (img_id, _), caption in zip(batch, captions):
                    results[img_id] = {"reference": img_id, "caption": caption}
                processed_count += len(batch)
                print(f"[{i + len(batch)}/{len(pending)}] ✅ Processed batch of {len(batch)}")

            except Exception as e:
                # Falls back to per-item processing for ANY batch failure
                # (OOM, shape mismatch, or otherwise) — not just OOM, per
                # your robustness requirement.
                reason = "OOM" if _is_oom(e) else f"error ({e})"
                print(f"⚠️ Batch failed ({reason}) — retrying items one-by-one.")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                for img_id, path in batch:
                    try:
                        caption = generate_batch_captions(
                            model, processor, [(img_id, path)], pad_token_id, eos_token_id
                        )[0]
                        results[img_id] = {"reference": img_id, "caption": caption}
                        processed_count += 1
                    except Exception as e2:
                        print(f"[{img_id}] ❌ Error even alone: {e2}")
                        results[img_id] = {"reference": img_id, "caption": "", "error": str(e2)}
                        error_count += 1
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

            i += len(batch)
            batches_since_save += 1
            if batches_since_save >= SAVE_EVERY_N_BATCHES:
                flush()
                batches_since_save = 0

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user — saving progress before exit.")
    finally:
        flush()

    print("\n✅ Caption generation completed!")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()