from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_PATH = Path("./models_download/Qwen3-VL-Embedding").resolve()

print("Model:", MODEL_PATH)

model = SentenceTransformer(
    str(MODEL_PATH),
    local_files_only=True,
    trust_remote_code=True,
    device="cpu",
)

print("Model loaded successfully")

image_path = next(
    p for p in Path("./cirr-val").rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
)

image = Image.open(image_path).convert("RGB")

inputs = [
    "A person standing outdoors.",
    image,
    {
        "image": image,
        "text": "A person standing outdoors."
    },
]

embeddings = model.encode(
    inputs,
    convert_to_tensor=True,
)

print("Embedding shape:", embeddings.shape)
print("Embedding dtype:", embeddings.dtype)

print("Test passed.")