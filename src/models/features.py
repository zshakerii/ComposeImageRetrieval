import torch
import torch.nn.functional as F
import clip
from PIL import Image

from ..config import device


def extract_tensor(out):
    """خروجی مدل را به تنسور تبدیل می‌کند (سازگار با SigLIP2)."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    raise TypeError(f"خروجی غیرمنتظره: {type(out)}")


