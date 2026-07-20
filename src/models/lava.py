import os
import torch

from ..config import device

def load_lava_model(hf_token=None):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_path = "llava-hf/llava-1.5-7b-hf"
    hf_token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    processor_kwargs = {}
    if hf_token:
        processor_kwargs["token"] = hf_token

    model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    model.eval()
    return model, processor




@torch.no_grad()
def get_lava_image_feature(image, model, processor):
    try:
        prompt = "USER: <image>\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        feat = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(feat.float(), dim=-1).cpu()
    except Exception:
        return None


@torch.no_grad()
def get_lava_text_feature(caption, model, processor):
    try:
        dummy_image = Image.new("RGB", (336, 336), color="white")
        prompt = f"USER: {caption}\nASSISTANT:"
        inputs = processor(text=prompt, images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        feat = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(feat.float(), dim=-1).cpu()
    except Exception:
        return None

