"""Global configuration and constants."""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
K_VALUES = [1, 5, 10, 50]
MAP_K_VALUES = [5, 10, 50]

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

CLIP_CAPTION_PREFIX = "a photo of "
SEARLE_CAPTION_PREFIX = "a photo of $"

DEFAULT_MODELS = [
    "clip", "searle", "qwen", "blip", "lava", "siglip", "clip_sep", "clip_beta","open_clip"]

DEFAULT_OPEN_CLIP_MODEL = "ViT-H-14"
DEFAULT_OPEN_CLIP_PRETRAINED = "./models_download/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

OPEN_CLIP_MODELS = ("open_clip", "open_clip_sep", "open_clip_beta")
