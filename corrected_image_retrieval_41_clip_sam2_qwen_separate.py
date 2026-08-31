#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIRR Image Retrieval: CLIP / CLIP+Qwen / CLIP+SAM2 / CLIP+Qwen+SAM2

Standalone Qwen modes:
  - Qwen2-VL-2B-Instruct: generate local image captions
  - Qwen3-VL-Embedding-2B: generate local multimodal embeddings

Experimental table implemented exactly as:

    Method                  Reference   CIRR Caption   Qwen Caption   SAM
    ----------------------------------------------------------------------
    CLIP baseline              YES           YES           NO           NO
    CLIP + Qwen                YES           YES           YES          NO
    CLIP + SAM                 YES           YES           NO           YES
    CLIP + Qwen + SAM         YES           YES           YES          YES

Important design choices
------------------------
1. The original CIRR relative caption is always preserved and used.
2. Qwen captions are OPTIONAL auxiliary visual context loaded from a local JSON.
3. SAM 2.1 is used only on unique reference images, never on the entire gallery.
4. CLIP remains the common retrieval space for text, image and SAM regions.
5. All expensive embeddings are persistently cached.
6. The four methods are evaluated against the same gallery and same GT.
7. For region selection, CLIP chooses the SAM regions most relevant to the
   current composed query. With Qwen enabled, Qwen context also participates
   in the region-selection seed.
8. No BLIP and no alpha sweep are used.

Expected local stack
--------------------
- OpenAI CLIP package/repository
- SAM 2.1 package installed locally
- local SAM 2.1 checkpoint (.pt)
- local SAM 2.1 config available in the installed SAM2 repository
- Qwen-generated captions already stored in a local JSON file

Example:
python corrected_image_retrieval_39_clip_sam2_qwen.py \
  --dataset cirr \
  --image_folder ./cirr-val \
  --json_path cirr-val.json \
  --qwen_captions_path ./qwen_captions.json \
  --sam2_checkpoint ./models_download/SAM2/sam2.1_hiera_large.pt \
  --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
  --clip_model ViT-B/32 \
  --cirr_subset

Run only selected ablations with --methods, e.g.:
--methods baseline qwen sam full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import clip
except ImportError:
    clip = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
K_VALUES = (1, 5, 10, 50)
MAP_K_VALUES = (5, 10, 50)


# ============================================================
# Reproducibility / general helpers
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return os.path.splitext(str(value).strip())[0]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
    )


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_cache(path: Path, expected_meta: dict) -> Optional[dict]:
    if not path.is_file():
        return None

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        for key, expected in expected_meta.items():
            if payload.get(key) != expected:
                return None
        return payload
    except Exception as exc:
        print(f"[WARN] Cache load failed: {path}: {type(exc).__name__}: {exc}")
        return None


def get_query_key(item: dict) -> str:
    if str(item.get("query_id", "")).strip():
        return str(item["query_id"])
    if item.get("pairid") is not None:
        return str(item["pairid"])

    ref = normalize_id(item.get("reference_id"))
    target = normalize_id(item.get("target_id"))
    if ref and target:
        return f"{ref}__{target}"
    return ref or "query_unknown"


# ============================================================
# Dataset
# ============================================================

def parse_cirr_sample(sample: dict, index: int) -> dict:
    if not isinstance(sample, dict):
        raise ValueError(f"CIRR sample #{index} is not an object.")

    # New schema: candidate_id, caption, group, target_id
    if "candidate_id" in sample or "group" in sample or "target_id" in sample:
        required = ("candidate_id", "caption", "group", "target_id")
        missing = [k for k in required if k not in sample]
        if missing:
            raise ValueError(f"CIRR sample #{index} missing: {missing}")

        ref = normalize_id(sample.get("candidate_id"))
        target = normalize_id(sample.get("target_id"))
        caption = normalize_text(sample.get("caption"))
        group_raw = sample.get("group")

        if not isinstance(group_raw, list):
            raise ValueError(f"CIRR sample #{index}: group must be a list.")

        members = list(dict.fromkeys(
            normalize_id(x) for x in group_raw if normalize_id(x)
        ))

        if not ref or not target or not members:
            raise ValueError(f"CIRR sample #{index}: invalid reference/target/group")

        query_id = f"{ref}__{target}__{index}"

        return {
            "query_id": query_id,
            "pairid": query_id,
            "candidate_id": ref,
            "reference_id": ref,
            "target_id": target,
            "target_hard": target,
            "target_soft": {},
            "positives": [target],
            "caption": caption,
            "members": members,
            "group": members,
        }

    # Old CIRR schema
    if "reference" not in sample or "caption" not in sample:
        raise ValueError(f"CIRR sample #{index} has unsupported schema.")

    img_set = sample.get("img_set")
    if not isinstance(img_set, dict):
        raise ValueError(f"CIRR sample #{index}: img_set must be an object.")

    ref = normalize_id(sample.get("reference"))
    caption = normalize_text(sample.get("caption"))
    members = list(dict.fromkeys(
        normalize_id(x)
        for x in img_set.get("members", [])
        if normalize_id(x)
    ))

    target = normalize_id(sample.get("target_hard"))
    positives = [target] if target else []

    soft = sample.get("target_soft")
    if isinstance(soft, dict):
        positives.extend(normalize_id(x) for x in soft if normalize_id(x))
    elif isinstance(soft, list):
        positives.extend(normalize_id(x) for x in soft if normalize_id(x))

    positives = list(dict.fromkeys(x for x in positives if x))
    pairid = sample.get("pairid")
    query_id = str(pairid) if pairid is not None else f"{ref}__{index}"

    return {
        "query_id": query_id,
        "pairid": pairid if pairid is not None else query_id,
        "candidate_id": ref,
        "reference_id": ref,
        "target_id": target or None,
        "target_hard": target or None,
        "target_soft": soft if isinstance(soft, dict) else {},
        "positives": positives,
        "caption": caption,
        "members": members,
        "group": members,
    }


def load_cirr(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("CIRR JSON must be a non-empty list.")

    return [parse_cirr_sample(x, i) for i, x in enumerate(data)]


def get_relevant_ids(item: dict) -> Set[str]:
    target = normalize_id(item.get("target_id") or item.get("target_hard"))
    if target:
        return {target}

    relevant: Set[str] = set()
    soft = item.get("target_soft")

    if isinstance(soft, dict):
        for key, score in soft.items():
            key = normalize_id(key)
            if not key:
                continue
            try:
                if float(score) > 0:
                    relevant.add(key)
            except (TypeError, ValueError):
                relevant.add(key)
    elif isinstance(soft, list):
        relevant.update(normalize_id(x) for x in soft if normalize_id(x))

    if not relevant:
        relevant.update(
            normalize_id(x) for x in item.get("positives") or [] if normalize_id(x)
        )

    return relevant


def validate_cirr(data: List[dict], require_gt: bool) -> None:
    no_group = sum(1 for x in data if not x.get("members"))
    no_ref = sum(1 for x in data if not x.get("reference_id"))
    no_caption = sum(1 for x in data if not normalize_text(x.get("caption")))
    no_gt = sum(1 for x in data if not get_relevant_ids(x))

    print("\n=== CIRR validation ===")
    print(f"Queries                  : {len(data)}")
    print(f"Without reference        : {no_ref}")
    print(f"Without group            : {no_group}")
    print(f"Without CIRR caption     : {no_caption}")
    print(f"Without ground truth     : {no_gt}")

    if no_ref or no_group:
        raise RuntimeError("Invalid CIRR annotations: reference/group missing.")

    if require_gt and no_gt:
        raise RuntimeError("Ground truth is required but missing for some queries.")


# ============================================================
# Image indexing
# ============================================================

_IMAGE_PATH_INDEX: Dict[str, str] = {}


def build_image_index(folder: str) -> Dict[str, str]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")

    index: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.stem, str(p))

    if not index:
        raise RuntimeError(f"No images found in {root}")
    return index


def scan_gallery_ids(folder: str) -> List[str]:
    global _IMAGE_PATH_INDEX
    _IMAGE_PATH_INDEX = build_image_index(folder)
    return sorted(_IMAGE_PATH_INDEX)


def find_image_path(folder: str, image_id: str) -> Optional[str]:
    global _IMAGE_PATH_INDEX
    image_id = normalize_id(image_id)

    candidates = [image_id]
    if image_id.isdigit():
        candidates.append(image_id.zfill(12))

    for candidate in dict.fromkeys(candidates):
        p = _IMAGE_PATH_INDEX.get(candidate)
        if p and Path(p).is_file():
            return p

    if not _IMAGE_PATH_INDEX:
        _IMAGE_PATH_INDEX = build_image_index(folder)
        for candidate in dict.fromkeys(candidates):
            p = _IMAGE_PATH_INDEX.get(candidate)
            if p and Path(p).is_file():
                return p

    return None


def load_image(folder: str, image_id: str) -> Optional[Image.Image]:
    path = find_image_path(folder, image_id)
    if path is None:
        return None

    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")
    except Exception as exc:
        print(f"[WARN] Could not read {image_id}: {exc}")
        return None


# ============================================================
# Qwen caption loading
# ============================================================

QWEN_ID_KEYS = (
    "image_id",
    "img_id",
    "reference_img_id",
    "reference_id",
    "reference",
    "candidate_id",
    "id",
)
QWEN_TEXT_KEYS = (
    "caption",
    "generated_caption",
    "qwen_caption",
    "description",
    "text",
    "image_caption",
)


def load_qwen_captions(path: str) -> Dict[str, str]:
    """
    Flexible local JSON reader.

    Supported examples:

    1) List:
       [{"image_id": "abc", "caption": "A man ..."}, ...]

    2) Dict:
       {"abc": "A man ...", "def": "A woman ..."}

    3) Dict values as objects:
       {"abc": {"caption": "A man ..."}}
    """
    if not path:
        return {}

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Qwen caption JSON not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, str] = {}

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            image_id = ""
            for key in QWEN_ID_KEYS:
                if key in item and item[key] is not None:
                    image_id = normalize_id(item[key])
                    if image_id:
                        break

            caption = ""
            for key in QWEN_TEXT_KEYS:
                if key in item and item[key]:
                    caption = normalize_text(item[key])
                    if caption:
                        break

            if image_id and caption:
                result[image_id] = caption

    elif isinstance(data, dict):
        for key, value in data.items():
            image_id = normalize_id(key)
            caption = ""

            if isinstance(value, str):
                caption = normalize_text(value)
            elif isinstance(value, dict):
                for text_key in QWEN_TEXT_KEYS:
                    if value.get(text_key):
                        caption = normalize_text(value[text_key])
                        if caption:
                            break

                if not image_id:
                    for id_key in QWEN_ID_KEYS:
                        if value.get(id_key) is not None:
                            image_id = normalize_id(value[id_key])
                            if image_id:
                                break

            if image_id and caption:
                result[image_id] = caption

    else:
        raise ValueError(f"Unsupported Qwen caption JSON type: {type(data).__name__}")

    print(f"Loaded Qwen captions: {len(result)}")
    return result


# ============================================================
# Metrics
# ============================================================

def average_precision_at_k(relevant: Iterable[str], retrieved: List[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0

    hits = 0
    score = 0.0
    for rank, image_id in enumerate(retrieved[:k], start=1):
        if image_id in relevant:
            hits += 1
            score += hits / rank

    return score / min(len(relevant), k)


def reciprocal_rank(relevant: Iterable[str], retrieved: List[str]) -> float:
    relevant = set(relevant)
    for rank, image_id in enumerate(retrieved, start=1):
        if image_id in relevant:
            return 1.0 / rank
    return 0.0


def init_results() -> dict:
    return {
        "prec": {k: [] for k in K_VALUES},
        "rec": {k: [] for k in K_VALUES},
        "map": {k: [] for k in MAP_K_VALUES},
        "mrr": [],
    }


def update_metrics(results: dict, ranked: List[str], relevant: Iterable[str]) -> None:
    relevant = {normalize_id(x) for x in relevant if normalize_id(x)}
    if not relevant:
        return

    for k in K_VALUES:
        top = ranked[:k]
        hits = sum(x in relevant for x in top)
        results["prec"][k].append(hits / k)
        results["rec"][k].append(hits / len(relevant))

    for k in MAP_K_VALUES:
        results["map"][k].append(
            average_precision_at_k(relevant, ranked, k)
        )

    results["mrr"].append(reciprocal_rank(relevant, ranked))


def summarize(results: dict) -> dict:
    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "mrr": mean(results["mrr"]),
        "map5": mean(results["map"][5]),
        "map10": mean(results["map"][10]),
        "map50": mean(results["map"][50]),
        "prec1": mean(results["prec"][1]),
        "prec5": mean(results["prec"][5]),
        "prec10": mean(results["prec"][10]),
        "prec50": mean(results["prec"][50]),
        "rec1": mean(results["rec"][1]),
        "rec5": mean(results["rec"][5]),
        "rec10": mean(results["rec"][10]),
        "rec50": mean(results["rec"][50]),
    }


# ============================================================
# CLIP
# ============================================================

def load_clip_model(model_name: str):
    if clip is None:
        raise ImportError(
            "OpenAI CLIP is not installed. Install the CLIP package/repository."
        )

    model, preprocess = clip.load(model_name, device=str(DEVICE))
    model.eval()
    return model, preprocess


@torch.inference_mode()
def clip_image_batch(
    images: List[Image.Image],
    model,
    preprocess,
) -> torch.Tensor:
    if not images:
        return torch.empty((0, 0), dtype=torch.float32)

    batch = torch.stack([preprocess(x) for x in images], dim=0).to(DEVICE)
    feat = model.encode_image(batch).float()
    return F.normalize(feat, dim=-1).cpu()


@torch.inference_mode()
def clip_text_batch(
    texts: List[str],
    model,
) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)

    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    feat = model.encode_text(tokens).float()
    return F.normalize(feat, dim=-1).cpu()


def build_clip_gallery_cache(
    gallery_ids: List[str],
    image_folder: str,
    model,
    preprocess,
    cache_dir: str,
    dataset: str,
    split: str,
    model_name: str,
    batch_size: int,
    force: bool,
) -> Dict[str, torch.Tensor]:
    root = Path(cache_dir) / dataset / split
    root.mkdir(parents=True, exist_ok=True)

    key = safe_name(model_name)
    meta = {
        "cache_type": "clip_gallery_norm",
        "version": 2,
        "dataset": dataset,
        "split": split,
        "model_name": model_name,
        "gallery_ids": list(gallery_ids),
    }
    cache_path = root / f"clip_{key}_gallery.pt"
    partial_path = root / f"clip_{key}_gallery.partial.pt"

    if not force:
        payload = load_cache(cache_path, meta)
        if payload is not None:
            features = payload.get("features")
            if isinstance(features, dict) and set(features) == set(gallery_ids):
                print(f"Loaded CLIP gallery cache: {cache_path}")
                return features

    features: Dict[str, torch.Tensor] = {}

    if not force and partial_path.is_file():
        try:
            payload = torch.load(partial_path, map_location="cpu", weights_only=False)
            if all(payload.get(k) == v for k, v in meta.items()):
                cached = payload.get("features")
                if isinstance(cached, dict):
                    features.update(cached)
                    print(f"Resuming CLIP gallery: {len(features)}/{len(gallery_ids)}")
        except Exception as exc:
            print(f"[WARN] CLIP partial cache could not be resumed: {exc}")

    remaining = [x for x in gallery_ids if x not in features]

    for start in tqdm(
        range(0, len(remaining), batch_size),
        desc="CLIP gallery embeddings",
    ):
        ids = remaining[start:start + batch_size]
        images = []

        for image_id in ids:
            img = load_image(image_folder, image_id)
            if img is None:
                raise RuntimeError(f"Gallery image cannot be opened: {image_id}")
            images.append(img)

        batch_features = clip_image_batch(images, model, preprocess)

        for image_id, feature in zip(ids, batch_features):
            features[image_id] = feature.unsqueeze(0)

        atomic_torch_save(
            {**meta, "features": features, "num_cached": len(features)},
            partial_path,
        )

    if set(features) != set(gallery_ids):
        missing = sorted(set(gallery_ids) - set(features))
        raise RuntimeError(
            f"CLIP gallery cache incomplete: {len(missing)} missing. {missing[:10]}"
        )

    atomic_torch_save(
        {**meta, "features": features, "num_cached": len(features)},
        cache_path,
    )

    try:
        partial_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Saved CLIP gallery cache: {cache_path}")
    return features


def build_text_cache(
    texts: Iterable[str],
    model,
    model_name: str,
    cache_dir: str,
    dataset: str,
    split: str,
    cache_name: str,
    batch_size: int,
    force: bool,
) -> Dict[str, torch.Tensor]:
    unique_texts = sorted({normalize_text(x) for x in texts if normalize_text(x)})

    root = Path(cache_dir) / dataset / split
    root.mkdir(parents=True, exist_ok=True)

    meta = {
        "cache_type": f"clip_text_{cache_name}",
        "version": 2,
        "dataset": dataset,
        "split": split,
        "model_name": model_name,
        "text_hash": stable_hash("\n".join(unique_texts)),
    }
    cache_path = root / (
        f"clip_{safe_name(model_name)}_{safe_name(cache_name)}_text.pt"
    )

    if not force:
        payload = load_cache(cache_path, meta)
        if payload is not None:
            features = payload.get("features")
            if isinstance(features, dict) and set(features) == set(unique_texts):
                print(f"Loaded {cache_name} text cache: {cache_path}")
                return features

    features: Dict[str, torch.Tensor] = {}

    for start in tqdm(
        range(0, len(unique_texts), batch_size),
        desc=f"CLIP {cache_name} text embeddings",
    ):
        batch = unique_texts[start:start + batch_size]
        batch_features = clip_text_batch(batch, model)
        for text, feature in zip(batch, batch_features):
            features[text] = feature.unsqueeze(0)

    atomic_torch_save(
        {**meta, "features": features},
        cache_path,
    )

    print(f"Saved {cache_name} text cache: {cache_path}")
    return features


def stack_feature_dict(
    features: Dict[str, torch.Tensor],
    ordered_ids: List[str],
) -> torch.Tensor:
    tensor = torch.cat(
        [features[x].reshape(1, -1).float() for x in ordered_ids],
        dim=0,
    )
    return F.normalize(tensor, dim=-1)


# ============================================================
# SAM 2.1
# ============================================================

def load_sam2(
    checkpoint: str,
    config: str,
):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise ImportError(
            "SAM 2 is not installed. Install the official SAM 2 package locally."
        ) from exc

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    # Official SAM2 build_sam2 uses a Hydra config name. Keep the documented
    # config string first; if a local path is supplied, try its normalized name.
    config_candidates = [config]
    config_path = Path(config).expanduser()
    if config_path.is_file():
        config_candidates.extend([
            str(config_path),
            config_path.name,
            f"configs/sam2.1/{config_path.name}",
        ])

    errors = []
    model = None
    resolved_config = None

    for cfg in dict.fromkeys(config_candidates):
        try:
            model = build_sam2(
                cfg,
                str(checkpoint_path),
                device=DEVICE,
                mode="eval",
                apply_postprocessing=False,
            )
            resolved_config = cfg
            break
        except Exception as exc:
            errors.append(f"{cfg}: {type(exc).__name__}: {exc}")

    if model is None:
        raise RuntimeError(
            "Could not load local SAM 2.1 checkpoint/config.\n"
            + "\n".join(errors[-5:])
        )

    model.eval()

    # Conservative settings for CIRR reference-region discovery.
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=24,
        points_per_batch=64,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        output_mode="binary_mask",
    )

    print(f"SAM2 checkpoint : {checkpoint_path}")
    print(f"SAM2 config     : {resolved_config}")
    print(f"SAM2 device     : {DEVICE}")

    return model, generator, resolved_config


def crop_masked_region(
    image: Image.Image,
    mask: np.ndarray,
    bbox_xywh: Optional[Iterable[float]],
    background_mode: str,
    padding_ratio: float = 0.08,
) -> Optional[Image.Image]:
    image_np = np.asarray(image.convert("RGB"))
    h, w = image_np.shape[:2]

    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    if bbox_xywh is None:
        x = float(xs.min())
        y = float(ys.min())
        bw = float(xs.max() - xs.min() + 1)
        bh = float(ys.max() - ys.min() + 1)
    else:
        x, y, bw, bh = [float(v) for v in bbox_xywh]

    padx = max(1.0, bw * padding_ratio)
    pady = max(1.0, bh * padding_ratio)

    x0 = max(0, int(np.floor(x - padx)))
    y0 = max(0, int(np.floor(y - pady)))
    x1 = min(w, int(np.ceil(x + bw + padx)))
    y1 = min(h, int(np.ceil(y + bh + pady)))

    if x1 <= x0 or y1 <= y0:
        return None

    crop = image_np[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    if background_mode == "masked":
        crop[~crop_mask] = 0
    elif background_mode == "white":
        crop[~crop_mask] = 255

    return Image.fromarray(crop)


def filter_masks(
    masks: List[dict],
    width: int,
    height: int,
    min_area_ratio: float,
    max_area_ratio: float,
    max_regions: int,
) -> List[dict]:
    total = float(width * height)
    selected: List[dict] = []

    for ann in masks:
        segmentation = ann.get("segmentation")
        if segmentation is None:
            continue

        area = float(ann.get("area", 0.0))
        ratio = area / total if total > 0 else 0.0

        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue

        piou = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))
        quality = 0.5 * piou + 0.5 * stability

        selected.append({
            "segmentation": np.asarray(segmentation, dtype=np.bool_),
            "bbox": ann.get("bbox"),
            "area": area,
            "area_ratio": ratio,
            "sam_quality": quality,
        })

    selected.sort(key=lambda x: x["sam_quality"], reverse=True)
    return selected[:max_regions]


def build_region_cache(
    reference_ids: List[str],
    image_folder: str,
    clip_model,
    clip_preprocess,
    sam_generator,
    sam_key: str,
    clip_model_name: str,
    cache_dir: str,
    dataset: str,
    split: str,
    max_regions: int,
    min_area_ratio: float,
    max_area_ratio: float,
    background_mode: str,
    force: bool,
) -> Dict[str, List[dict]]:
    root = Path(cache_dir) / dataset / split
    root.mkdir(parents=True, exist_ok=True)

    meta = {
        "cache_type": "sam2_clip_regions",
        "version": 3,
        "dataset": dataset,
        "split": split,
        "clip_model_name": clip_model_name,
        "sam_key": sam_key,
        "reference_ids": list(reference_ids),
        "max_regions": max_regions,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "background_mode": background_mode,
    }

    tag = (
        f"sam2_{sam_key}_clip_{safe_name(clip_model_name)}"
        f"_r{max_regions}_a{min_area_ratio:g}-{max_area_ratio:g}"
        f"_{background_mode}"
    )
    cache_path = root / f"{tag}_regions.pt"
    partial_path = root / f"{tag}_regions.partial.pt"

    if not force:
        payload = load_cache(cache_path, meta)
        if payload is not None:
            regions = payload.get("regions")
            if isinstance(regions, dict) and set(regions) == set(reference_ids):
                print(f"Loaded SAM+CLIP region cache: {cache_path}")
                return regions

    regions: Dict[str, List[dict]] = {}

    if not force and partial_path.is_file():
        try:
            payload = torch.load(partial_path, map_location="cpu", weights_only=False)
            if all(payload.get(k) == v for k, v in meta.items()):
                cached = payload.get("regions")
                if isinstance(cached, dict):
                    regions.update(cached)
                    print(f"Resuming region cache: {len(regions)}/{len(reference_ids)}")
        except Exception as exc:
            print(f"[WARN] Region partial cache could not be resumed: {exc}")

    remaining = [x for x in reference_ids if x not in regions]

    for reference_id in tqdm(
        remaining,
        desc="SAM2 + CLIP reference regions",
    ):
        image = load_image(image_folder, reference_id)
        if image is None:
            raise RuntimeError(f"Reference image cannot be opened: {reference_id}")

        image_np = np.asarray(image.convert("RGB"))

        try:
            masks = sam_generator.generate(image_np)
        except Exception as exc:
            raise RuntimeError(
                f"SAM2 failed for {reference_id}: {type(exc).__name__}: {exc}"
            ) from exc

        candidate_masks = filter_masks(
            masks,
            image_np.shape[1],
            image_np.shape[0],
            min_area_ratio,
            max_area_ratio,
            max_regions,
        )

        records: List[dict] = []

        for region_index, ann in enumerate(candidate_masks):
            region_image = crop_masked_region(
                image,
                ann["segmentation"],
                ann["bbox"],
                background_mode,
            )
            if region_image is None:
                continue

            try:
                feat = clip_image_batch(
                    [region_image],
                    clip_model,
                    clip_preprocess,
                )
                if feat.numel() == 0:
                    continue
            except Exception as exc:
                print(
                    f"[WARN] Region CLIP failed: ref={reference_id}, "
                    f"region={region_index}: {exc}"
                )
                continue

            records.append({
                "index": region_index,
                "feature": feat[0].float(),
                "bbox": ann["bbox"],
                "area": ann["area"],
                "area_ratio": ann["area_ratio"],
                "sam_quality": ann["sam_quality"],
            })

        regions[reference_id] = records

        atomic_torch_save(
            {**meta, "regions": regions, "num_cached": len(regions)},
            partial_path,
        )

    if set(regions) != set(reference_ids):
        missing = sorted(set(reference_ids) - set(regions))
        raise RuntimeError(
            f"SAM region cache incomplete: {len(missing)} missing. {missing[:10]}"
        )

    atomic_torch_save(
        {**meta, "regions": regions, "num_cached": len(regions)},
        cache_path,
    )

    try:
        partial_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Saved SAM+CLIP region cache: {cache_path}")
    return regions


# ============================================================
# Region selection / query composition
# ============================================================

def normalized_mean(features: List[torch.Tensor]) -> torch.Tensor:
    if not features:
        raise ValueError("Cannot average an empty feature list.")

    stacked = torch.cat(
        [x.reshape(1, -1).float() for x in features],
        dim=0,
    )
    return F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)


def make_seed_query(
    reference_feat: torch.Tensor,
    cirr_feat: torch.Tensor,
    qwen_feat: Optional[torch.Tensor],
    use_qwen: bool,
) -> torch.Tensor:
    parts = [reference_feat, cirr_feat]
    if use_qwen and qwen_feat is not None:
        parts.append(qwen_feat)
    return normalized_mean(parts)


def select_region_feature(
    region_records: List[dict],
    seed_query: torch.Tensor,
    top_regions: int,
    temperature: float,
) -> Tuple[Optional[torch.Tensor], List[dict]]:
    if not region_records:
        return None, []

    features = F.normalize(
        torch.cat(
            [r["feature"].reshape(1, -1).float() for r in region_records],
            dim=0,
        ),
        dim=-1,
    )

    seed = F.normalize(seed_query.reshape(1, -1).float(), dim=-1)
    scores = (features @ seed.squeeze(0)).flatten()

    m = min(top_regions, scores.numel())
    values, indices = torch.topk(scores, k=m)

    weights = F.softmax(values / max(temperature, 1e-4), dim=0)
    selected_features = features[indices]
    region_feat = F.normalize(
        (selected_features * weights.unsqueeze(1)).sum(dim=0, keepdim=True),
        dim=-1,
    )

    selected = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        selected.append({
            "region_index": int(region_records[idx]["index"]),
            "score": float(value),
            "area_ratio": float(region_records[idx]["area_ratio"]),
            "sam_quality": float(region_records[idx]["sam_quality"]),
            "bbox": region_records[idx]["bbox"],
        })

    return region_feat, selected


def compose_final_query(
    reference_feat: torch.Tensor,
    cirr_feat: torch.Tensor,
    qwen_feat: Optional[torch.Tensor],
    region_feat: Optional[torch.Tensor],
    use_qwen: bool,
    use_sam: bool,
    reference_weight: float,
    cirr_weight: float,
    qwen_weight: float,
    region_weight: float,
) -> torch.Tensor:
    components: List[Tuple[float, torch.Tensor, str]] = [
        (reference_weight, reference_feat, "reference"),
        (cirr_weight, cirr_feat, "cirr"),
    ]

    if use_qwen and qwen_feat is not None:
        components.append((qwen_weight, qwen_feat, "qwen"))

    if use_sam and region_feat is not None:
        components.append((region_weight, region_feat, "region"))

    valid = [(w, f) for w, f, _ in components if w > 0]
    if not valid:
        raise ValueError("All final-query weights are zero.")

    weighted = []
    for weight, feature in valid:
        weighted.append(weight * F.normalize(feature.reshape(1, -1).float(), dim=-1))

    return F.normalize(torch.cat(weighted, dim=0).sum(dim=0, keepdim=True), dim=-1)


# ============================================================
# Retrieval evaluation
# ============================================================

METHOD_LABELS = {
    "baseline": "CLIP baseline",
    "qwen": "CLIP + Qwen",
    "sam": "CLIP + SAM2",
    "full": "CLIP + Qwen + SAM2",
}


def evaluate_method(
    method: str,
    data: List[dict],
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    clip_gallery_cache: Dict[str, torch.Tensor],
    cirr_text_cache: Dict[str, torch.Tensor],
    qwen_text_cache: Dict[str, torch.Tensor],
    qwen_captions: Dict[str, str],
    region_cache: Dict[str, List[dict]],
    subset_protocol: bool,
    reference_weight: float,
    cirr_weight: float,
    qwen_weight: float,
    region_weight: float,
    top_regions: int,
    region_temperature: float,
) -> Tuple[dict, int, int, Dict[str, List[str]], Dict[str, int], dict]:
    use_qwen = method in {"qwen", "full"}
    use_sam = method in {"sam", "full"}

    results = init_results()
    rankings: Dict[str, List[str]] = {}
    skip_reasons: Dict[str, int] = {}
    total = 0
    skipped = 0

    region_queries = 0
    region_available = 0
    qwen_queries = 0
    qwen_missing = 0
    selected_region_log = {}

    for item in tqdm(
        data,
        desc=METHOD_LABELS[method],
    ):
        key = get_query_key(item)
        ref_id = item["reference_id"]
        cirr_caption = normalize_text(item.get("caption"))

        reference_feat = clip_gallery_cache.get(ref_id)
        cirr_feat = cirr_text_cache.get(cirr_caption)

        if reference_feat is None:
            skipped += 1
            skip_reasons["missing_reference_clip_embedding"] = (
                skip_reasons.get("missing_reference_clip_embedding", 0) + 1
            )
            continue

        if cirr_feat is None:
            skipped += 1
            skip_reasons["missing_cirr_caption_embedding"] = (
                skip_reasons.get("missing_cirr_caption_embedding", 0) + 1
            )
            continue

        qwen_feat = None
        if use_qwen:
            qwen_caption = normalize_text(qwen_captions.get(ref_id, ""))
            if qwen_caption:
                qwen_feat = qwen_text_cache.get(qwen_caption)
                if qwen_feat is not None:
                    qwen_queries += 1
            else:
                qwen_missing += 1

        if use_qwen and qwen_feat is None:
            # Qwen is an optional auxiliary signal. We do not discard the
            # query merely because one local Qwen caption is missing; this keeps
            # the ablation comparable and makes the missing-Qwen count explicit.
            pass

        try:
            seed = make_seed_query(
                reference_feat=reference_feat,
                cirr_feat=cirr_feat,
                qwen_feat=qwen_feat,
                use_qwen=use_qwen,
            )

            region_feat = None
            selected_regions = []

            if use_sam:
                records = region_cache.get(ref_id, [])
                if records:
                    region_available += 1

                region_queries += 1

                region_feat, selected_regions = select_region_feature(
                    region_records=records,
                    seed_query=seed,
                    top_regions=top_regions,
                    temperature=region_temperature,
                )

                if selected_regions:
                    selected_region_log[key] = selected_regions

            query = compose_final_query(
                reference_feat=reference_feat,
                cirr_feat=cirr_feat,
                qwen_feat=qwen_feat,
                region_feat=region_feat,
                use_qwen=use_qwen,
                use_sam=use_sam,
                reference_weight=reference_weight,
                cirr_weight=cirr_weight,
                qwen_weight=qwen_weight,
                region_weight=region_weight,
            )

            sims = gallery_feats @ query.squeeze(0)

            if subset_protocol:
                allowed = {
                    normalize_id(x)
                    for x in item.get("members", [])
                    if normalize_id(x)
                }
            else:
                allowed = None

            ranked = []
            order = torch.argsort(sims, descending=True).cpu().tolist()

            for idx in order:
                image_id = gallery_ids[idx]
                if image_id == ref_id:
                    continue
                if allowed is not None and image_id not in allowed:
                    continue
                ranked.append(image_id)

            rankings[key] = ranked[:50]

            relevant = get_relevant_ids(item)
            if relevant:
                update_metrics(results, ranked, relevant)

            total += 1

        except Exception as exc:
            skipped += 1
            reason = f"{type(exc).__name__}: {exc}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            print(
                f"[WARN] {METHOD_LABELS[method]} failed: "
                f"query={key!r}, reference={ref_id!r}: {reason}"
            )

    diagnostics = {
        "use_qwen": use_qwen,
        "use_sam": use_sam,
        "qwen_queries_with_caption": qwen_queries,
        "qwen_missing_caption": qwen_missing,
        "sam_region_queries": region_queries,
        "sam_queries_with_regions": region_available,
        "selected_regions": selected_region_log,
    }

    return results, total, skipped, rankings, skip_reasons, diagnostics


# ============================================================
# Output
# ============================================================

def save_predictions(
    path: Path,
    method: str,
    data: List[dict],
    rankings: Dict[str, List[str]],
    total: int,
    skipped: int,
    skip_reasons: Dict[str, int],
    qwen_captions: Dict[str, str],
    selected_region_log: Dict[str, list],
    config: dict,
) -> None:
    predictions = []

    for item in data:
        key = get_query_key(item)
        ranking = rankings.get(key, [])
        target = normalize_id(item.get("target_id"))
        target_rank = None
        if target and target in ranking:
            target_rank = ranking.index(target) + 1

        ref_id = item.get("reference_id")

        predictions.append({
            "query_id": key,
            "reference_id": ref_id,
            "target_id": item.get("target_id"),
            "cirr_caption": item.get("caption", ""),
            "qwen_caption": qwen_captions.get(ref_id, ""),
            "members": item.get("members", []),
            "ranking": ranking,
            "target_rank": target_rank,
            "selected_regions": selected_region_log.get(key, []),
        })

    payload = {
        "dataset": "cirr",
        "method": method,
        "method_description": METHOD_LABELS[method],
        "num_queries": len(data),
        "num_predictions": total,
        "num_skipped": skipped,
        "skip_reasons": skip_reasons,
        "config": config,
        "predictions": predictions,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved: {path}")


def save_metrics(path: Path, dataset: str, split: str, subset: bool, metrics: dict) -> None:
    payload = {
        "dataset": dataset,
        "split": split,
        "cirr_subset": subset,
        "metrics": metrics,
        "metric_order": [
            "MRR",
            "mAP@5",
            "mAP@10",
            "mAP@50",
            "P@1",
            "P@5",
            "P@10",
            "P@50",
            "R@1",
            "R@5",
            "R@10",
            "R@50",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved: {path}")


def print_summary_table(all_results: Dict[str, dict]) -> None:
    print("\n" + "=" * 150)
    print("FINAL RESULTS - CIRR")
    print("=" * 150)

    header = (
        f"{'Method':<28}"
        f"{'MRR':>9}"
        f"{'mAP@5':>10}"
        f"{'mAP@10':>10}"
        f"{'mAP@50':>10}"
        f"{'P@1':>9}"
        f"{'P@5':>9}"
        f"{'P@10':>10}"
        f"{'P@50':>10}"
        f"{'R@1':>9}"
        f"{'R@5':>9}"
        f"{'R@10':>10}"
        f"{'R@50':>10}"
    )
    print(header)
    print("-" * len(header))

    for label, result in all_results.items():
        print(
            f"{label:<28}"
            f"{result['mrr']:>9.4f}"
            f"{result['map5']:>10.4f}"
            f"{result['map10']:>10.4f}"
            f"{result['map50']:>10.4f}"
            f"{result['prec1']:>9.4f}"
            f"{result['prec5']:>9.4f}"
            f"{result['prec10']:>10.4f}"
            f"{result['prec50']:>10.4f}"
            f"{result['rec1']:>9.4f}"
            f"{result['rec5']:>9.4f}"
            f"{result['rec10']:>10.4f}"
            f"{result['rec50']:>10.4f}"
        )



# ============================================================
# Standalone Qwen runners
# ============================================================

def _enable_local_offline_mode() -> None:
    """Prevent Hugging Face loaders from trying the network."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _require_local_dir(path: str, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {p}")
    return p


def load_qwen2vl_local(model_path: str):
    """
    Load Qwen2-VL-2B-Instruct from a local directory only.

    Qwen2-VL is a vision-language generation model, so this runner is used
    to generate image captions that can later be consumed by the CLIP+Qwen
    ablation in retrieval mode.
    """
    _enable_local_offline_mode()

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Qwen2-VL requires a recent transformers installation."
        ) from exc

    local_path = _require_local_dir(
        model_path,
        "Qwen2-VL model",
    )

    if not (local_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Qwen2-VL config.json not found: {local_path}"
        )

    processor = AutoProcessor.from_pretrained(
        str(local_path),
        local_files_only=True,
    )

    dtype = (
        torch.float16
        if DEVICE.type == "cuda"
        else torch.float32
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }

    if DEVICE.type == "cuda":
        model_kwargs["device_map"] = "auto"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(local_path),
        **model_kwargs,
    )

    if DEVICE.type != "cuda":
        model = model.to(DEVICE)

    model.eval()

    print(f"Qwen2-VL model : {local_path}")
    print(f"Qwen2-VL device: {DEVICE}")
    print(f"Qwen2-VL dtype : {getattr(model, 'dtype', dtype)}")

    return model, processor


def _qwen2vl_generate_caption(
    image: Image.Image,
    model,
    processor,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Generate one concise visual caption.

    The prompt is deliberately restrictive so the generated text describes
    visible content and does not invent attributes.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError(
            "Qwen2-VL requires qwen-vl-utils. Install it in the same environment."
        ) from exc

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device

    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(model_device)
        else:
            moved[key] = value

    with torch.inference_mode():
        generated_ids = model.generate(
            **moved,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
        )

    input_len = moved["input_ids"].shape[1]
    generated_trimmed = [
        output_ids[input_len:]
        for output_ids in generated_ids
    ]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return normalize_text(output_text)


def run_qwen2vl_caption_generation(args) -> None:
    """
    Standalone Qwen2-VL caption generation.

    Output:
        JSON file:
        {
          "metadata": {...},
          "captions": {
             "image_id": "caption"
          }
        }
    """
    print("\n" + "=" * 100)
    print("STANDALONE QWEN2-VL-2B-INSTRUCT CAPTION GENERATION")
    print("=" * 100)

    model, processor = load_qwen2vl_local(
        args.qwen_model_path
    )

    gallery_ids = scan_gallery_ids(args.image_folder)

    if args.qwen_limit > 0:
        image_ids = gallery_ids[:args.qwen_limit]
    else:
        image_ids = gallery_ids

    print(f"Images available : {len(gallery_ids)}")
    print(f"Images to process: {len(image_ids)}")

    output_path = (
        Path(args.qwen_output)
        if args.qwen_output
        else Path(args.image_folder).resolve().parent
        / "qwen2vl_captions.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, str] = {}

    if output_path.is_file() and not args.qwen_force:
        try:
            old = json.loads(output_path.read_text(encoding="utf-8"))

            if isinstance(old, dict):
                if isinstance(old.get("captions"), dict):
                    existing = {
                        normalize_id(k): normalize_text(v)
                        for k, v in old["captions"].items()
                        if normalize_id(k) and normalize_text(v)
                    }
                else:
                    existing = {
                        normalize_id(k): normalize_text(v)
                        for k, v in old.items()
                        if normalize_id(k) and normalize_text(v)
                    }

            print(f"Resuming Qwen captions: {len(existing)} already available.")
        except Exception as exc:
            print(f"[WARN] Existing Qwen output could not be loaded: {exc}")

    pending_ids = [
        image_id
        for image_id in image_ids
        if image_id not in existing
    ]

    prompt = args.qwen_prompt.strip()

    for image_id in tqdm(
        pending_ids,
        desc="Qwen2-VL captions",
    ):
        image = load_image(
            args.image_folder,
            image_id,
        )

        if image is None:
            print(
                f"[WARN] Qwen2-VL skipped unreadable image: {image_id}"
            )
            continue

        try:
            caption = _qwen2vl_generate_caption(
                image=image,
                model=model,
                processor=processor,
                prompt=prompt,
                max_new_tokens=args.qwen_max_new_tokens,
            )

            if caption:
                existing[image_id] = caption

        except Exception as exc:
            print(
                f"[WARN] Qwen2-VL failed for {image_id}: "
                f"{type(exc).__name__}: {exc}"
            )

        # Periodic checkpointing so interruption does not lose all work.
        if len(existing) % max(1, args.qwen_checkpoint_every) == 0:
            payload = {
                "metadata": {
                    "model": "Qwen2-VL-2B-Instruct",
                    "model_path": str(
                        Path(args.qwen_model_path).resolve()
                    ),
                    "image_folder": str(
                        Path(args.image_folder).resolve()
                    ),
                    "prompt": prompt,
                    "max_new_tokens": args.qwen_max_new_tokens,
                    "num_captions": len(existing),
                },
                "captions": existing,
            }

            tmp = output_path.with_suffix(
                output_path.suffix + ".tmp"
            )
            tmp.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, output_path)

    payload = {
        "metadata": {
            "model": "Qwen2-VL-2B-Instruct",
            "model_path": str(
                Path(args.qwen_model_path).resolve()
            ),
            "image_folder": str(
                Path(args.image_folder).resolve()
            ),
            "prompt": prompt,
            "max_new_tokens": args.qwen_max_new_tokens,
            "num_images_requested": len(image_ids),
            "num_captions": len(existing),
        },
        "captions": dict(sorted(existing.items())),
    }

    tmp = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, output_path)

    print(f"\nQwen2-VL captions saved: {output_path}")
    print(f"Total captions       : {len(existing)}")

    del model, processor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def load_qwen3_embedding_local(
    model_path: str,
    repo_path: str,
):
    """
    Load the official Qwen3-VL-Embedding implementation locally.

    The official Qwen repository exposes Qwen3VLEmbedder through:
        src.models.qwen3_vl_embedding

    The model itself supports text, image and mixed-modal inputs in a shared
    representation space.
    """
    _enable_local_offline_mode()

    repo = _require_local_dir(
        repo_path,
        "Qwen3-VL-Embedding repository",
    )
    model_dir = _require_local_dir(
        model_path,
        "Qwen3-VL-Embedding model",
    )

    import sys
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    except Exception as exc:
        raise ImportError(
            "Could not import Qwen3VLEmbedder from the local Qwen3-VL-Embedding "
            f"repository: {repo}"
        ) from exc

    dtype = (
        torch.float16
        if DEVICE.type == "cuda"
        else torch.float32
    )

    kwargs = {
        "model_name_or_path": str(model_dir),
    }

    # Official example exposes torch_dtype/attention arguments.  Keep them
    # optional for compatibility with repository revisions.
    try:
        kwargs["torch_dtype"] = dtype
        model = Qwen3VLEmbedder(**kwargs)
    except TypeError:
        kwargs.pop("torch_dtype", None)
        model = Qwen3VLEmbedder(**kwargs)

    print(f"Qwen3-VL-Embedding model: {model_dir}")
    print(f"Qwen3 repository      : {repo}")
    print(f"Qwen3 device           : {DEVICE}")

    return model


def _qwen3_process(
    embedder,
    inputs: List[dict],
) -> torch.Tensor:
    """
    Call the official Qwen3-VL embedder and normalize the returned vectors.
    """
    with torch.inference_mode():
        result = embedder.process(
            inputs,
            normalize=True,
        )

    if not torch.is_tensor(result):
        if isinstance(result, tuple) and result:
            result = result[0]

    if not torch.is_tensor(result):
        raise TypeError(
            f"Qwen3-VL-Embedding returned {type(result).__name__}, "
            "expected torch.Tensor."
        )

    return F.normalize(
        result.float(),
        dim=-1,
    ).cpu()


def run_qwen3_embedding_generation(args) -> None:
    """
    Standalone Qwen3-VL-Embedding generation.

    Outputs separate image/text embedding files. These embeddings are NOT
    mixed directly with CLIP vectors because Qwen3-VL-Embedding uses its own
    2048-dimensional (2B) or 4096-dimensional (8B) representation space.
    Cross-model fusion requires score-level fusion or a trained projection.
    """
    print("\n" + "=" * 100)
    print("STANDALONE QWEN3-VL-EMBEDDING")
    print("=" * 100)

    embedder = load_qwen3_embedding_local(
        model_path=args.qwen_model_path,
        repo_path=args.qwen3_repo_path,
    )

    gallery_ids = scan_gallery_ids(args.image_folder)

    if args.qwen_limit > 0:
        image_ids = gallery_ids[:args.qwen_limit]
    else:
        image_ids = gallery_ids

    output_dir = (
        Path(args.qwen_output_dir)
        if args.qwen_output_dir
        else Path(args.embedding_cache_dir)
        / args.dataset
        / "qwen3_vl_embedding"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "image_embeddings.pt"

    image_meta = {
        "cache_type": "qwen3_vl_embedding_image",
        "version": 1,
        "dataset": args.dataset,
        "model_path": str(Path(args.qwen_model_path).resolve()),
        "image_ids": list(image_ids),
        "instruction": args.qwen3_instruction,
    }

    image_features: Dict[str, torch.Tensor] = {}

    if image_path.is_file() and not args.qwen_force:
        payload = torch.load(
            image_path,
            map_location="cpu",
            weights_only=False,
        )

        valid = (
            isinstance(payload, dict)
            and all(payload.get(k) == v for k, v in image_meta.items())
        )

        if valid and isinstance(payload.get("features"), dict):
            image_features = payload["features"]
            print(
                f"Loaded Qwen3 image embeddings: "
                f"{len(image_features)}"
            )

    remaining = [
        x for x in image_ids
        if x not in image_features
    ]

    for start in tqdm(
        range(
            0,
            len(remaining),
            max(1, args.qwen_batch_size),
        ),
        desc="Qwen3 image embeddings",
    ):
        batch_ids = remaining[start:start + args.qwen_batch_size]
        inputs = []

        for image_id in batch_ids:
            path = find_image_path(
                args.image_folder,
                image_id,
            )

            if path is None:
                print(
                    f"[WARN] Qwen3 image missing: {image_id}"
                )
                continue

            inputs.append(
                {
                    "image": path,
                    "instruction": args.qwen3_instruction,
                }
            )

        if not inputs:
            continue

        # Inputs are kept aligned with valid files.
        valid_ids = [
            image_id
            for image_id in batch_ids
            if find_image_path(args.image_folder, image_id) is not None
        ]

        try:
            features = _qwen3_process(
                embedder,
                inputs,
            )

            if features.shape[0] != len(valid_ids):
                raise ValueError(
                    f"Qwen3 returned {features.shape[0]} embeddings for "
                    f"{len(valid_ids)} images."
                )

            for image_id, feature in zip(valid_ids, features):
                image_features[image_id] = feature

        except Exception as exc:
            raise RuntimeError(
                f"Qwen3 image embedding batch failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        atomic_torch_save(
            {
                **image_meta,
                "features": image_features,
                "num_cached": len(image_features),
            },
            image_path,
        )

    atomic_torch_save(
        {
            **image_meta,
            "features": image_features,
            "num_cached": len(image_features),
        },
        image_path,
    )

    print(f"Saved Qwen3 image embeddings: {image_path}")
    print(f"Image embeddings: {len(image_features)}")

    if args.qwen3_text_json:
        text_source = Path(args.qwen3_text_json).expanduser().resolve()
        if not text_source.is_file():
            raise FileNotFoundError(
                f"Qwen3 text JSON not found: {text_source}"
            )

        data = json.loads(
            text_source.read_text(encoding="utf-8")
        )

        texts: List[Tuple[str, str]] = []

        if isinstance(data, dict) and isinstance(data.get("captions"), dict):
            for key, value in data["captions"].items():
                text = normalize_text(value)
                if text:
                    texts.append((normalize_id(key), text))

        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    text = normalize_text(value)
                elif isinstance(value, dict):
                    text = normalize_text(
                        value.get("caption")
                        or value.get("qwen_caption")
                        or value.get("text")
                    )
                else:
                    text = ""

                if text:
                    texts.append((normalize_id(key), text))

        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue

                image_id = normalize_id(
                    item.get("image_id")
                    or item.get("reference_id")
                    or item.get("id")
                )

                text = normalize_text(
                    item.get("caption")
                    or item.get("qwen_caption")
                    or item.get("text")
                )

                if image_id and text:
                    texts.append((image_id, text))

        text_features: Dict[str, torch.Tensor] = {}
        for start in tqdm(
            range(
                0,
                len(texts),
                max(1, args.qwen_batch_size),
            ),
            desc="Qwen3 text embeddings",
        ):
            batch = texts[start:start + args.qwen_batch_size]

            inputs = [
                {
                    "text": text,
                    "instruction": args.qwen3_instruction,
                }
                for _, text in batch
            ]

            features = _qwen3_process(
                embedder,
                inputs,
            )

            for (image_id, _), feature in zip(batch, features):
                text_features[image_id] = feature

        text_meta = {
            "cache_type": "qwen3_vl_embedding_text",
            "version": 1,
            "dataset": args.dataset,
            "model_path": str(
                Path(args.qwen_model_path).resolve()
            ),
            "instruction": args.qwen3_instruction,
        }

        text_path = output_dir / "text_embeddings.pt"

        atomic_torch_save(
            {
                **text_meta,
                "features": text_features,
                "num_cached": len(text_features),
            },
            text_path,
        )

        print(f"Saved Qwen3 text embeddings: {text_path}")
        print(f"Text embeddings : {len(text_features)}")

    del embedder
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def maybe_run_standalone_qwen(args) -> bool:
    """
    Return True when the command was a standalone Qwen execution.
    """
    if args.run_qwen == "none":
        return False

    if args.qwen_model == "qwen2-vl-2b-instruct":
        run_qwen2vl_caption_generation(args)
        return True

    if args.qwen_model == "qwen3-vl-embedding-2b":
        run_qwen3_embedding_generation(args)
        return True

    raise ValueError(
        f"Unsupported --qwen_model: {args.qwen_model}"
    )



# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIRR retrieval ablation: CLIP / Qwen / SAM2 / Qwen+SAM2"
    )

    parser.add_argument("--dataset", choices=["cirr"], required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--qwen_captions_path", default=None)

    # --------------------------------------------------------
    # Standalone Qwen execution
    # --------------------------------------------------------
    parser.add_argument(
        "--run_qwen",
        choices=["none", "standalone"],
        default="none",
        help=(
            "Run Qwen independently and exit. "
            "Use --qwen_model to select Qwen2-VL caption generation "
            "or Qwen3-VL-Embedding generation."
        ),
    )
    parser.add_argument(
        "--qwen_model",
        choices=[
            "qwen2-vl-2b-instruct",
            "qwen3-vl-embedding-2b",
        ],
        default="qwen2-vl-2b-instruct",
        help="Standalone Qwen model to execute.",
    )
    parser.add_argument(
        "--qwen_model_path",
        default=None,
        help="Local path to the selected Qwen model.",
    )
    parser.add_argument(
        "--qwen3_repo_path",
        default="./models_download/Qwen3-VL-Embedding",
        help=(
            "Local path to the Qwen3-VL-Embedding source repository "
            "(used only with qwen3-vl-embedding-2b)."
        ),
    )
    parser.add_argument(
        "--qwen_output",
        default=None,
        help=(
            "Output JSON for Qwen2-VL captions. "
            "Default: ../qwen2vl_captions.json relative to image folder."
        ),
    )
    parser.add_argument(
        "--qwen_output_dir",
        default=None,
        help="Output directory for Qwen3-VL-Embedding .pt caches.",
    )
    parser.add_argument(
        "--qwen_batch_size",
        type=int,
        default=4,
        help="Batch size for Qwen3-VL-Embedding.",
    )
    parser.add_argument(
        "--qwen_limit",
        type=int,
        default=0,
        help="Process only the first N images; 0 means all images.",
    )
    parser.add_argument(
        "--qwen_force",
        action="store_true",
        help="Ignore existing Qwen output/cache and rebuild it.",
    )
    parser.add_argument(
        "--qwen_checkpoint_every",
        type=int,
        default=50,
        help="Checkpoint Qwen2-VL caption JSON every N successful captions.",
    )
    parser.add_argument(
        "--qwen_max_new_tokens",
        type=int,
        default=80,
        help="Maximum generated tokens for Qwen2-VL caption generation.",
    )
    parser.add_argument(
        "--qwen_prompt",
        default=(
            "Describe only the visible content of this image in one concise English sentence. "
            "Preserve visible objects, attributes, colors, counts, spatial relations, and clothing. "
            "Do not infer identity, intent, age, location, or facts that are not visually supported. "
            "Do not mention that you are analyzing an image. Output only the caption."
        ),
        help="Strict visual-caption prompt for Qwen2-VL.",
    )
    parser.add_argument(
        "--qwen3_instruction",
        default="Retrieve images or text relevant to the given input.",
        help="Instruction passed to Qwen3-VL-Embedding.",
    )
    parser.add_argument(
        "--qwen3_text_json",
        default=None,
        help=(
            "Optional local JSON containing captions/text to embed with "
            "Qwen3-VL-Embedding."
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "qwen", "sam", "full"],
        default=["baseline", "qwen", "sam", "full"],
        help="Ablations to run. Default: all four.",
    )

    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument(
        "--sam2_config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )

    parser.add_argument("--embedding_cache_dir", default="./embedding_cache")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--clip_text_batch_size", type=int, default=64)

    parser.add_argument("--cirr_subset", action="store_true")
    parser.add_argument("--require_cirr_gt", action="store_true")
    parser.add_argument("--strict_evaluation", action="store_true")
    parser.add_argument("--force_rebuild_embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Final fusion weights. Same defaults are used for all methods so that the
    # contribution of each signal remains interpretable.
    parser.add_argument("--reference_weight", type=float, default=1.0)
    parser.add_argument("--cirr_weight", type=float, default=1.0)
    parser.add_argument("--qwen_weight", type=float, default=0.60)
    parser.add_argument("--region_weight", type=float, default=0.80)

    parser.add_argument("--top_regions", type=int, default=3)
    parser.add_argument("--region_temperature", type=float, default=0.07)

    parser.add_argument("--sam_max_regions", type=int, default=20)
    parser.add_argument("--sam_min_area_ratio", type=float, default=0.01)
    parser.add_argument("--sam_max_area_ratio", type=float, default=0.80)
    parser.add_argument(
        "--sam_background_mode",
        choices=["crop", "masked", "white"],
        default="crop",
    )

    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--predictions_dir", default=None)

    args = parser.parse_args()

    if args.qwen_batch_size < 1:
        raise ValueError("--qwen_batch_size must be >= 1.")

    if args.qwen_limit < 0:
        raise ValueError("--qwen_limit must be >= 0.")

    if args.qwen_checkpoint_every < 1:
        raise ValueError("--qwen_checkpoint_every must be >= 1.")

    if args.qwen_max_new_tokens < 1:
        raise ValueError("--qwen_max_new_tokens must be >= 1.")

    if args.run_qwen != "none":
        if not args.qwen_model_path:
            raise ValueError(
                "--qwen_model_path is required when --run_qwen standalone is used."
            )

        # Standalone execution intentionally happens before CLIP/SAM loading,
        # so Qwen can be run completely independently.
        if maybe_run_standalone_qwen(args):
            return

    set_seed(args.seed)

    if args.clip_batch_size < 1 or args.clip_text_batch_size < 1:
        raise ValueError("CLIP batch sizes must be >= 1.")
    if args.top_regions < 1 or args.sam_max_regions < 1:
        raise ValueError("Region counts must be >= 1.")
    if args.region_temperature <= 0:
        raise ValueError("--region_temperature must be > 0.")
    if not 0 <= args.sam_min_area_ratio < args.sam_max_area_ratio <= 1:
        raise ValueError("Invalid SAM area-ratio range.")

    for name in (
        "reference_weight",
        "cirr_weight",
        "qwen_weight",
        "region_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be >= 0.")

    split_name = "val"
    text_for_split = f"{args.json_path} {args.image_folder}".lower().replace("\\", "/")
    if "test1" in text_for_split or "/test/" in text_for_split:
        split_name = "test"
    elif "train" in text_for_split:
        split_name = "train"

    print(f"Device       : {DEVICE}")
    print(f"Split        : {split_name}")
    print(f"Selected     : {', '.join(args.methods)}")

    # -------------------------------
    # Data
    # -------------------------------
    data = load_cirr(args.json_path)
    validate_cirr(data, require_gt=args.require_cirr_gt)

    gallery_ids = scan_gallery_ids(args.image_folder)
    print(f"Gallery images: {len(gallery_ids)}")

    gallery_set = set(gallery_ids)
    missing_refs = sorted({
        x["reference_id"]
        for x in data
        if x.get("reference_id") not in gallery_set
    })
    missing_targets = sorted({
        normalize_id(x.get("target_id"))
        for x in data
        if normalize_id(x.get("target_id"))
        and normalize_id(x.get("target_id")) not in gallery_set
    })

    print(f"Missing references: {len(missing_refs)}")
    print(f"Missing targets   : {len(missing_targets)}")

    if missing_refs:
        raise RuntimeError(f"Reference images missing: {missing_refs[:10]}")

    subset_protocol = bool(args.cirr_subset)
    print(
        "Retrieval protocol: "
        + ("CIRR SUBSET" if subset_protocol else "FULL GALLERY")
    )

    qwen_captions = {}
    if any(m in {"qwen", "full"} for m in args.methods):
        if not args.qwen_captions_path:
            raise ValueError(
                "--qwen_captions_path is required when qwen/full is selected."
            )
        qwen_captions = load_qwen_captions(args.qwen_captions_path)

    # -------------------------------
    # CLIP
    # -------------------------------
    print("\n=== Loading CLIP ===")
    clip_model, clip_preprocess = load_clip_model(args.clip_model)

    clip_gallery_cache = build_clip_gallery_cache(
        gallery_ids=gallery_ids,
        image_folder=args.image_folder,
        model=clip_model,
        preprocess=clip_preprocess,
        cache_dir=args.embedding_cache_dir,
        dataset=args.dataset,
        split=split_name,
        model_name=args.clip_model,
        batch_size=args.clip_batch_size,
        force=args.force_rebuild_embeddings,
    )

    gallery_feats = stack_feature_dict(clip_gallery_cache, gallery_ids)
    print(f"CLIP gallery tensor: {tuple(gallery_feats.shape)}")

    cirr_text_cache = build_text_cache(
        texts=[x.get("caption", "") for x in data],
        model=clip_model,
        model_name=args.clip_model,
        cache_dir=args.embedding_cache_dir,
        dataset=args.dataset,
        split=split_name,
        cache_name="cirr",
        batch_size=args.clip_text_batch_size,
        force=args.force_rebuild_embeddings,
    )

    qwen_text_cache: Dict[str, torch.Tensor] = {}
    if qwen_captions:
        qwen_text_cache = build_text_cache(
            texts=qwen_captions.values(),
            model=clip_model,
            model_name=args.clip_model,
            cache_dir=args.embedding_cache_dir,
            dataset=args.dataset,
            split=split_name,
            cache_name="qwen",
            batch_size=args.clip_text_batch_size,
            force=args.force_rebuild_embeddings,
        )

    # -------------------------------
    # SAM 2.1 only if needed
    # -------------------------------
    region_cache: Dict[str, List[dict]] = {}

    if any(m in {"sam", "full"} for m in args.methods):
        if not args.sam2_checkpoint:
            raise ValueError(
                "--sam2_checkpoint is required when sam/full is selected."
            )

        print("\n=== Loading SAM 2.1 ===")
        sam_model, sam_generator, resolved_config = load_sam2(
            checkpoint=args.sam2_checkpoint,
            config=args.sam2_config,
        )

        checkpoint_path = Path(args.sam2_checkpoint).expanduser().resolve()
        stat = checkpoint_path.stat()
        sam_fingerprint = stable_hash(
            f"{checkpoint_path}|{stat.st_size}|{stat.st_mtime_ns}|{resolved_config}"
        )

        unique_refs = sorted({x["reference_id"] for x in data})
        print(f"Unique references for SAM: {len(unique_refs)}")

        region_cache = build_region_cache(
            reference_ids=unique_refs,
            image_folder=args.image_folder,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            sam_generator=sam_generator,
            sam_key=sam_fingerprint,
            clip_model_name=args.clip_model,
            cache_dir=args.embedding_cache_dir,
            dataset=args.dataset,
            split=split_name,
            max_regions=args.sam_max_regions,
            min_area_ratio=args.sam_min_area_ratio,
            max_area_ratio=args.sam_max_area_ratio,
            background_mode=args.sam_background_mode,
            force=args.force_rebuild_embeddings,
        )

        # Release SAM before evaluation; regions are already cached on CPU.
        del sam_model, sam_generator
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------
    # Four ablations
    # -------------------------------
    all_results: Dict[str, dict] = {}
    diagnostics: Dict[str, dict] = {}

    for method in ("baseline", "qwen", "sam", "full"):
        if method not in args.methods:
            continue

        print(f"\n=== {METHOD_LABELS[method]} ===")

        (
            results,
            total,
            skipped,
            rankings,
            skip_reasons,
            method_diag,
        ) = evaluate_method(
            method=method,
            data=data,
            gallery_ids=gallery_ids,
            gallery_feats=gallery_feats,
            clip_gallery_cache=clip_gallery_cache,
            cirr_text_cache=cirr_text_cache,
            qwen_text_cache=qwen_text_cache,
            qwen_captions=qwen_captions,
            region_cache=region_cache,
            subset_protocol=subset_protocol,
            reference_weight=args.reference_weight,
            cirr_weight=args.cirr_weight,
            qwen_weight=args.qwen_weight,
            region_weight=args.region_weight,
            top_regions=args.top_regions,
            region_temperature=args.region_temperature,
        )

        if skipped:
            message = (
                f"{METHOD_LABELS[method]} incomplete: "
                f"processed={total}/{len(data)}, skipped={skipped}"
            )
            if args.strict_evaluation:
                raise RuntimeError(message)
            print(f"[WARN] {message}")

        summary = summarize(results)
        label = METHOD_LABELS[method]
        all_results[label] = summary
        diagnostics[label] = {
            "processed": total,
            "skipped": skipped,
            "skip_reasons": skip_reasons,
            "method_diagnostics": method_diag,
        }

        print(
            f"{label} | "
            f"MRR={summary['mrr']:.4f} | "
            f"mAP@5={summary['map5']:.4f} | "
            f"mAP@10={summary['map10']:.4f} | "
            f"mAP@50={summary['map50']:.4f} | "
            f"R@1={summary['rec1']:.4f} | "
            f"R@5={summary['rec5']:.4f} | "
            f"R@10={summary['rec10']:.4f} | "
            f"R@50={summary['rec50']:.4f} | "
            f"n={total}, skipped={skipped}"
        )

        prediction_dir = (
            Path(args.predictions_dir)
            if args.predictions_dir
            else Path(args.json_path).parent / "predictions"
        )

        method_cfg = {
            "method": method,
            "description": label,
            "reference": True,
            "cirr_caption": True,
            "qwen_caption": method in {"qwen", "full"},
            "sam": method in {"sam", "full"},
            "clip_model": args.clip_model,
            "reference_weight": args.reference_weight,
            "cirr_weight": args.cirr_weight,
            "qwen_weight": args.qwen_weight,
            "region_weight": args.region_weight,
            "top_regions": args.top_regions,
            "region_temperature": args.region_temperature,
            "sam_max_regions": args.sam_max_regions,
            "sam_min_area_ratio": args.sam_min_area_ratio,
            "sam_max_area_ratio": args.sam_max_area_ratio,
            "sam_background_mode": args.sam_background_mode,
            "subset_protocol": subset_protocol,
        }

        selected_regions = method_diag.get("selected_regions", {})

        save_predictions(
            path=prediction_dir / f"cirr_{method}_predictions.json",
            method=method,
            data=data,
            rankings=rankings,
            total=total,
            skipped=skipped,
            skip_reasons=skip_reasons,
            qwen_captions=qwen_captions,
            selected_region_log=selected_regions,
            config=method_cfg,
        )

    # -------------------------------
    # Final outputs
    # -------------------------------
    print_summary_table(all_results)

    metrics_output = (
        Path(args.metrics_output)
        if args.metrics_output
        else Path(args.json_path).with_name(
            Path(args.json_path).stem + "_clip_qwen_sam2_ablation_metrics.json"
        )
    )

    save_metrics(
        path=metrics_output,
        dataset=args.dataset,
        split=split_name,
        subset=subset_protocol,
        metrics={
            "results": all_results,
            "diagnostics": diagnostics,
            "configuration": {
                "clip_model": args.clip_model,
                "qwen_captions_path": args.qwen_captions_path,
                "sam2_checkpoint": args.sam2_checkpoint,
                "sam2_config": args.sam2_config,
                "reference_weight": args.reference_weight,
                "cirr_weight": args.cirr_weight,
                "qwen_weight": args.qwen_weight,
                "region_weight": args.region_weight,
                "top_regions": args.top_regions,
                "region_temperature": args.region_temperature,
                "sam_max_regions": args.sam_max_regions,
                "sam_min_area_ratio": args.sam_min_area_ratio,
                "sam_max_area_ratio": args.sam_max_area_ratio,
                "sam_background_mode": args.sam_background_mode,
            },
        },
    )

    print("\nDone.")
    print(f"Metrics     : {metrics_output}")
    print(f"Cache       : {Path(args.embedding_cache_dir).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise
    except Exception as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
