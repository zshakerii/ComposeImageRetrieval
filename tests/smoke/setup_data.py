"""Generate minimal images for smoke testing."""

from pathlib import Path

from PIL import Image

IMAGE_DIR = Path(__file__).parent / "images"
COLORS = {
    "1": (220, 50, 50),
    "2": (50, 50, 220),
    "3": (50, 200, 50),
}


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for image_id, color in COLORS.items():
        img = Image.new("RGB", (224, 224), color)
        path = IMAGE_DIR / f"{int(image_id):012d}.jpg"
        img.save(path, quality=90)
        print(f"Created {path}")


if __name__ == "__main__":
    main()
