from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_PATH = Path(
    r"./models_download/Qwen3-VL-Embedding-2B"
).resolve()

print("Model:", MODEL_PATH)

model = SentenceTransformer(
    str(MODEL_PATH),
    device="cpu",
    local_files_only=True,
)

print("MODEL LOADED")

image_path = next(
    p for p in Path("./cirr-val").rglob("*")
    if p.is_file()
    and p.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }
)

print("Test image:", image_path)

image = Image.open(image_path).convert("RGB")

inputs = [
    {
        "image": image,
        "text": "A person outdoors."
    }
]

embeddings = model.encode(
    inputs,
    convert_to_tensor=True,
    show_progress_bar=True,
)

print("Embedding shape:", embeddings.shape)
print("Embedding dtype:", embeddings.dtype)
print("TEST SUCCESS")