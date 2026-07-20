import os
import torch

from ..config import device

def load_qwen_model(model_path, target_device=None):
    from transformers import AutoProcessor, AutoModel

    target_device = target_device or device
    print(f"🔄 بارگذاری مدل Qwen از {model_path}...")

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if target_device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(target_device)
        model.eval()
        print(f"✅ مدل Qwen بارگذاری شد (device: {target_device})")
        return model, processor
    except Exception as e:
        import traceback
        print(f"❌ خطا در بارگذاری مدل Qwen: {e}")
        traceback.print_exc()
        return None, None





@torch.inference_mode()
def get_qwen_image_feature(image, model, processor, device):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": " "},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(device)

        pixel_values = inputs["pixel_values"].to(model.dtype)
        kwargs = {}
        if "image_grid_thw" in inputs:
            kwargs["grid_thw"] = inputs["image_grid_thw"]

        vision_outputs = model.visual(pixel_values, **kwargs)

        if hasattr(vision_outputs, "hidden_states"):
            image_features = vision_outputs.hidden_states[-1]
        elif isinstance(vision_outputs, tuple):
            image_features = vision_outputs[0]
        else:
            image_features = vision_outputs

        if image_features.dim() == 3:
            image_features = image_features.mean(dim=1)
        elif image_features.dim() == 2:
            image_features = image_features.mean(dim=0, keepdim=True)

        return F.normalize(image_features, p=2, dim=-1).cpu()
    except Exception as e:
        import traceback
        print(f"Qwen image error: {e}")
        traceback.print_exc()
        return None


@torch.no_grad()
def get_qwen_text_feature(text, model, processor, device):
    if processor is None or model is None:
        return None
    try:
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        outputs = model.language_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        embedding = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(embedding, p=2, dim=-1).cpu()
    except Exception as e:
        import traceback
        print(f"Qwen text error: {e}")
        traceback.print_exc()
        return None
