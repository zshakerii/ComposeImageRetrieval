import os
import torch

from ..config import device

def load_blip_model(model_path=None):
    from transformers import BlipProcessor, BlipForImageTextRetrieval

    if model_path is None:
        model_path = "Salesforce/blip-itm-base-coco"

    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForImageTextRetrieval.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor



@torch.no_grad()
def get_blip_image_feature(image, model, processor):
    try:
        inputs = processor(
            images=image, text="an image", return_tensors="pt", padding=True
        ).to(device)
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"], return_dict=True
        )
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]
        return F.normalize(image_embeds, p=2, dim=-1).cpu()
    except Exception as e:
        import traceback
        print(f"❌ Blip image error: {e}")
        traceback.print_exc()
        return None


@torch.no_grad()
def get_blip_text_feature(caption, model, processor):
    try:
        dummy_image = Image.new("RGB", (384, 384), color="white")
        inputs = processor(
            images=dummy_image, text=caption, return_tensors="pt", padding=True
        ).to(device)
        text_outputs = model.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        return F.normalize(text_embeds, p=2, dim=-1).cpu()
    except Exception as e:
        import traceback
        print(f"❌ Blip text error: {e}")
        traceback.print_exc()
        return None

