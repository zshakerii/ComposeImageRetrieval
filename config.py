import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

# k های مورد استفاده برای precision/recall
K_VALUES = [1, 5, 10, 50]

# k های مورد استفاده برای mAP
MAP_K_VALUES = [5, 10, 50]

DEFAULT_MODELS=["clip", "searle", "qwen", "blip", "lava","siglip","clip_sep","clip_beta", "open_clip"]

