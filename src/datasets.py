import os
import json


def parse_cirr_sample(sample):
    reference_id = os.path.splitext(sample.get("reference", ""))[0]
    caption = sample.get("caption", "").strip()
    members = [os.path.splitext(m)[0] for m in sample.get("img_set", {}).get("members", [])]
    ref_index = sample.get("img_set", {}).get("reference_rank", 0)

    if not reference_id or not caption or not members or ref_index >= len(members):
        return None

    target_id = members[ref_index]
    return {
        "reference_id": reference_id,
        "caption": caption,
        "target_id": target_id,
        "members": members,
    }


def parse_circo_sample(sample):
    reference_id = sample.get("reference_img_id")
    caption = sample.get("relative_caption", "").strip()
    target_id = sample.get("target_img_id")
    gt_ids = sample.get("gt_img_ids", [])

    if reference_id is not None:
        reference_id = str(reference_id)
    if target_id is not None:
        target_id = str(target_id)
    gt_ids = [str(x) for x in gt_ids]

    if not reference_id or not caption or not target_id:
        return None

    return {
        "reference_id": reference_id,
        "caption": caption,
        "target_id": target_id,
        "members": gt_ids,
    }


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        parsed = [parse_circo_sample(s) for s in data["annotations"]]
        return "circo", [x for x in parsed if x is not None]

    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]

        # CIRCO: دارای reference_img_id و target_img_id و gt_img_ids
        if "reference_img_id" in first_item and "target_img_id" in first_item:
            parsed = [parse_circo_sample(s) for s in data]
            return "circo", [x for x in parsed if x is not None]

        # CIRR: دارای reference و img_set و target_hard
        if "reference" in first_item and "img_set" in first_item:
            parsed = [parse_cirr_sample(s) for s in data]
            return "cirr", [x for x in parsed if x is not None]

    raise ValueError("Unknown dataset format")
