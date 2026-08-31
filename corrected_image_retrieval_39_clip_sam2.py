#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIRR retrieval with CLIP + SAM 2.1 region-aware fusion.

Pipeline:
    reference image -> CLIP global feature
                     -> SAM2 automatic regions -> CLIP region features
    caption        -> CLIP text feature
    seed = normalize(global + text)
    top-M regions are selected by region/seed similarity
    final_query = normalize(w_text*text + w_global*global + w_region*region)

Persistent caches:
    - CLIP gallery embeddings
    - CLIP text embeddings
    - SAM2 + CLIP region embeddings for unique references

The script intentionally has no BLIP and no alpha parameter.
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
_IMAGE_PATH_INDEX: Dict[str, str] = {}


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


def normalize_prompt(value: Any) -> str:
    return str(value or "").strip()


def safe_name(value: str) -> str:
    return (
        str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")
        .replace(":", "_").replace("|", "_").replace("*", "_")
        .replace("?", "_").replace('"', "_").replace("<", "_")
        .replace(">", "_")
    )


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def get_query_key(item: dict) -> str:
    q = str(item.get("query_id", "")).strip()
    if q:
        return q
    if item.get("pairid") is not None:
        return str(item["pairid"])
    ref = normalize_id(item.get("reference_id"))
    target = normalize_id(item.get("target_id"))
    if ref and target:
        return f"{ref}__{target}"
    return ref or "query_unknown"


def parse_cirr_sample(sample: dict, index: int) -> dict:
    if not isinstance(sample, dict):
        raise ValueError(f"CIRR sample #{index} is not an object")

    if "candidate_id" in sample or "group" in sample or "target_id" in sample:
        required = ("candidate_id", "caption", "group", "target_id")
        missing = [x for x in required if x not in sample]
        if missing:
            raise ValueError(f"CIRR sample #{index} missing fields: {missing}")

        ref = normalize_id(sample.get("candidate_id"))
        target = normalize_id(sample.get("target_id"))
        caption = normalize_prompt(sample.get("caption"))
        group = sample.get("group")
        if not isinstance(group, list):
            raise ValueError(f"CIRR sample #{index}: group must be a list")
        members = list(dict.fromkeys(normalize_id(x) for x in group if normalize_id(x)))
        if not ref or not target or not members:
            raise ValueError(f"Invalid CIRR sample #{index}")
        if ref not in members or target not in members:
            raise ValueError(f"CIRR sample #{index}: reference/target not in group")
        query_id = f"{ref}__{target}__{index}"
        return {
            "query_id": query_id, "pairid": query_id,
            "candidate_id": ref, "reference_id": ref,
            "target_id": target, "target_hard": target,
            "target_soft": {}, "positives": [target],
            "caption": caption, "group": members, "members": members,
            "annotation_format": "candidate_group",
        }

    if "reference" not in sample or "caption" not in sample:
        raise ValueError(f"Unsupported CIRR sample #{index}")
    img_set = sample.get("img_set")
    if not isinstance(img_set, dict):
        raise ValueError(f"CIRR sample #{index}: img_set missing")
    ref = normalize_id(sample.get("reference"))
    caption = normalize_prompt(sample.get("caption"))
    raw_members = img_set.get("members", [])
    if not isinstance(raw_members, list):
        raise ValueError(f"CIRR sample #{index}: members must be a list")
    members = list(dict.fromkeys(normalize_id(x) for x in raw_members if normalize_id(x)))
    target = normalize_id(sample.get("target_hard"))
    positives = [target] if target else []
    soft = sample.get("target_soft")
    if isinstance(soft, dict):
        positives += [normalize_id(x) for x in soft if normalize_id(x)]
    elif isinstance(soft, list):
        positives += [normalize_id(x) for x in soft if normalize_id(x)]
    positives = list(dict.fromkeys(x for x in positives if x))
    pairid = sample.get("pairid")
    query_id = str(pairid) if pairid is not None else f"{ref}__{index}"
    return {
        "query_id": query_id, "pairid": query_id,
        "candidate_id": ref, "reference_id": ref,
        "target_id": target or None, "target_hard": target or None,
        "target_soft": soft if isinstance(soft, dict) else {},
        "positives": positives, "caption": caption,
        "group": members, "members": members, "annotation_format": "cap_rc2",
    }


def load_dataset(path: str) -> Tuple[str, List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset JSON must be a non-empty list")
    first = data[0]
    if isinstance(first, dict) and ("candidate_id" in first or "group" in first or "target_id" in first):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]
    if isinstance(first, dict) and "reference" in first and isinstance(first.get("img_set"), dict):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]
    if isinstance(first, dict) and ("reference_img_id" in first or "gt_img_ids" in first):
        out = []
        for x in data:
            target = normalize_id(x.get("target_img_id"))
            positives = [normalize_id(v) for v in x.get("gt_img_ids", []) if normalize_id(v)]
            if target and target not in positives:
                positives.insert(0, target)
            out.append({
                "query_id": str(x.get("query_id") or x.get("pairid") or x.get("reference_img_id")),
                "pairid": x.get("pairid"), "reference_id": normalize_id(x.get("reference_img_id")),
                "target_id": target or None, "positives": list(dict.fromkeys(positives)),
                "caption": normalize_prompt(x.get("relative_caption")), "members": [], "group": [],
                "annotation_format": "circo",
            })
        return "circo", out
    raise ValueError(f"Unknown dataset format. keys={list(first.keys())}")


def get_relevant_ids(item: dict) -> Set[str]:
    target = normalize_id(item.get("target_id") or item.get("target_hard"))
    if target:
        return {target}
    soft = item.get("target_soft")
    relevant: Set[str] = set()
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
        relevant.update(normalize_id(x) for x in item.get("positives", []) if normalize_id(x))
    return relevant


def validate_cirr(data: List[dict], require_gt: bool) -> None:
    missing_members = [get_query_key(x) for x in data if not x.get("members")]
    missing_refs = [get_query_key(x) for x in data if not x.get("reference_id")]
    no_caption = [get_query_key(x) for x in data if not normalize_prompt(x.get("caption"))]
    no_gt = sum(1 for x in data if not get_relevant_ids(x))
    print("\n=== CIRR annotation check ===")
    print(f"Queries                  : {len(data)}")
    print(f"Queries with GT          : {len(data)-no_gt}")
    print(f"Queries without GT       : {no_gt}")
    print(f"Queries without members  : {len(missing_members)}")
    print(f"Queries without reference: {len(missing_refs)}")
    print(f"Queries without caption  : {len(no_caption)}")
    if missing_members or missing_refs:
        raise RuntimeError("CIRR annotations are structurally invalid")
    if require_gt and no_gt:
        raise RuntimeError("Ground truth is required but some queries have no target")


def build_image_path_index(image_folder: str) -> Dict[str, str]:
    root = Path(image_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    index: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.stem, str(p))
    if not index:
        raise RuntimeError(f"No images found in {root}")
    return index


def scan_gallery_ids(image_folder: str) -> List[str]:
    global _IMAGE_PATH_INDEX
    _IMAGE_PATH_INDEX = build_image_path_index(image_folder)
    return sorted(_IMAGE_PATH_INDEX)


def find_image_path(image_folder: str, image_id: str) -> Optional[str]:
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
        _IMAGE_PATH_INDEX = build_image_path_index(image_folder)
        for candidate in dict.fromkeys(candidates):
            p = _IMAGE_PATH_INDEX.get(candidate)
            if p and Path(p).is_file():
                return p
    return None


def load_image(image_folder: str, image_id: str) -> Optional[Image.Image]:
    path = find_image_path(image_folder, image_id)
    if path is None:
        return None
    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")
    except Exception as exc:
        print(f"[WARN] Could not open image {image_id}: {exc}")
        return None


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_cache(path: Path, meta: dict) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        for k, v in meta.items():
            if payload.get(k) != v:
                return None
        return payload
    except Exception as exc:
        print(f"[WARN] Could not load cache {path}: {exc}")
        return None


# ============================================================
# CLIP
# ============================================================

def load_clip_model(model_name: str):
    if clip is None:
        raise ImportError("OpenAI CLIP is not installed")
    model, preprocess = clip.load(model_name, device=str(DEVICE))
    model.eval()
    return model, preprocess


@torch.inference_mode()
def encode_clip_images(images: List[Image.Image], model, preprocess) -> torch.Tensor:
    if not images:
        return torch.empty((0, 0), dtype=torch.float32)
    batch = torch.stack([preprocess(img) for img in images]).to(DEVICE)
    feat = model.encode_image(batch).float()
    return F.normalize(feat, dim=-1).cpu()


@torch.inference_mode()
def encode_clip_texts(texts: List[str], model) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    feat = model.encode_text(tokens).float()
    return F.normalize(feat, dim=-1).cpu()


def build_clip_gallery_cache(
    gallery_ids: List[str], image_folder: str, model, preprocess,
    cache_dir: str, dataset: str, split: str, model_name: str,
    batch_size: int, force_rebuild: bool,
) -> Dict[str, torch.Tensor]:
    root = Path(cache_dir) / dataset.lower() / split.lower()
    key = safe_name(model_name)
    path = root / f"clip_{key}_gallery.pt"
    partial = root / f"clip_{key}_gallery.partial.pt"
    meta = {
        "cache_type": "clip_gallery", "version": 3, "dataset": dataset,
        "split": split, "model_name": model_name, "gallery_ids": list(gallery_ids),
    }
    if not force_rebuild:
        payload = load_cache(path, meta)
        if payload and isinstance(payload.get("features"), dict) and set(payload["features"]) == set(gallery_ids):
            print(f"Loaded CLIP gallery cache: {path}")
            return payload["features"]

    features: Dict[str, torch.Tensor] = {}
    if not force_rebuild and partial.is_file():
        try:
            payload = torch.load(partial, map_location="cpu", weights_only=False)
            if all(payload.get(k) == v for k, v in meta.items()) and isinstance(payload.get("features"), dict):
                features.update(payload["features"])
                print(f"Resuming CLIP gallery cache: {len(features)}/{len(gallery_ids)}")
        except Exception as exc:
            print(f"[WARN] Partial CLIP cache ignored: {exc}")

    remaining = [x for x in gallery_ids if x not in features]
    for start in tqdm(range(0, len(remaining), batch_size), desc="CLIP gallery embeddings"):
        ids = remaining[start:start + batch_size]
        images = []
        for image_id in ids:
            img = load_image(image_folder, image_id)
            if img is None:
                raise RuntimeError(f"Unreadable gallery image: {image_id}")
            images.append(img)
        feats = encode_clip_images(images, model, preprocess)
        for image_id, feat in zip(ids, feats):
            features[image_id] = feat.unsqueeze(0)
        atomic_torch_save({**meta, "features": features, "num_cached": len(features)}, partial)

    if set(features) != set(gallery_ids):
        missing = sorted(set(gallery_ids) - set(features))
        raise RuntimeError(f"CLIP gallery cache incomplete: {missing[:20]}")
    atomic_torch_save({**meta, "features": features, "num_cached": len(features)}, path)
    try:
        partial.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"Saved CLIP gallery cache: {path}")
    return features


def build_clip_text_cache(
    captions: Iterable[str], model, model_name: str,
    cache_dir: str, dataset: str, split: str, batch_size: int,
    force_rebuild: bool,
) -> Dict[str, torch.Tensor]:
    captions = sorted({normalize_prompt(x) for x in captions if normalize_prompt(x)})
    root = Path(cache_dir) / dataset.lower() / split.lower()
    path = root / f"clip_{safe_name(model_name)}_text.pt"
    meta = {
        "cache_type": "clip_text", "version": 2, "dataset": dataset,
        "split": split, "model_name": model_name,
        "caption_hash": stable_hash("\n".join(captions)),
    }
    if not force_rebuild:
        payload = load_cache(path, meta)
        if payload and isinstance(payload.get("features"), dict) and set(payload["features"]) == set(captions):
            print(f"Loaded CLIP text cache: {path}")
            return payload["features"]

    features: Dict[str, torch.Tensor] = {}
    for start in tqdm(range(0, len(captions), batch_size), desc="CLIP text embeddings"):
        batch = captions[start:start + batch_size]
        feats = encode_clip_texts(batch, model)
        for caption, feat in zip(batch, feats):
            features[caption] = feat.unsqueeze(0)
    atomic_torch_save({**meta, "features": features}, path)
    print(f"Saved CLIP text cache: {path}")
    return features


def stack_feature_dict(features: Dict[str, torch.Tensor], ordered_ids: List[str]) -> torch.Tensor:
    x = torch.cat([features[k].reshape(1, -1).float() for k in ordered_ids], dim=0)
    return F.normalize(x, dim=-1)


# ============================================================
# SAM2 + region CLIP cache
# ============================================================

def load_sam2(checkpoint: str, config: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not installed. Install the official SAM2 repository first."
        ) from exc

    checkpoint = str(Path(checkpoint).expanduser().resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    candidates = [config]
    p = Path(config)
    candidates += [p.name, f"configs/sam2.1/{p.name}"]
    if p.suffix == ".yaml":
        candidates.append(f"configs/sam2.1/{p.stem}.yaml")

    last_error = None
    sam = None
    for cfg in dict.fromkeys(candidates):
        try:
            sam = build_sam2(cfg, checkpoint, device=DEVICE, apply_postprocessing=False)
            print(f"SAM2 config resolved: {cfg}")
            break
        except Exception as exc:
            last_error = exc
    if sam is None:
        raise RuntimeError(f"Could not load SAM2. Last error: {last_error}")

    sam.eval()
    generator = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=24,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    print(f"SAM2 checkpoint: {checkpoint}")
    print(f"SAM2 device    : {DEVICE}")
    return sam, generator


def crop_region(
    image: Image.Image, mask: np.ndarray, bbox: Optional[Iterable[float]],
    padding_ratio: float, background_mode: str,
) -> Optional[Image.Image]:
    arr = np.asarray(image.convert("RGB"), copy=True)
    h, w = arr.shape[:2]
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if bbox is None:
        x, y = float(xs.min()), float(ys.min())
        bw, bh = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
    else:
        x, y, bw, bh = [float(v) for v in bbox]
    px, py = bw * padding_ratio, bh * padding_ratio
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(w, int(x + bw + px)), min(h, int(y + bh + py))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = arr[y0:y1, x0:x1].copy()
    m = mask[y0:y1, x0:x1]
    if background_mode == "masked":
        crop[~m] = 0
    elif background_mode == "white":
        crop[~m] = 255
    return Image.fromarray(crop)


def select_sam_masks(
    masks: List[dict], width: int, height: int,
    min_area_ratio: float, max_area_ratio: float, max_regions: int,
) -> List[dict]:
    total = float(width * height)
    out = []
    for ann in masks:
        seg = ann.get("segmentation")
        area = float(ann.get("area", 0))
        if seg is None or area <= 0:
            continue
        ratio = area / total
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        iou = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))
        quality = 0.45 * iou + 0.45 * stability + 0.10 * min(1.0, ratio * 10.0)
        out.append({
            "segmentation": np.asarray(seg, dtype=np.bool_),
            "bbox": ann.get("bbox"), "area": area, "area_ratio": ratio,
            "sam_quality": quality,
        })
    out.sort(key=lambda x: x["sam_quality"], reverse=True)
    return out[:max_regions]


def build_region_cache(
    reference_ids: List[str], image_folder: str,
    clip_model, clip_preprocess, clip_model_name: str,
    sam_generator, sam_key: str, cache_dir: str, dataset: str, split: str,
    max_regions: int, min_area_ratio: float, max_area_ratio: float,
    background_mode: str, padding_ratio: float, force_rebuild: bool,
) -> Dict[str, List[dict]]:
    root = Path(cache_dir) / dataset.lower() / split.lower()
    key = (
        f"sam2_{sam_key}_clip_{safe_name(clip_model_name)}"
        f"_r{max_regions}_a{min_area_ratio:g}-{max_area_ratio:g}_{background_mode}"
    )
    path = root / f"{key}_regions.pt"
    partial = root / f"{key}_regions.partial.pt"
    meta = {
        "cache_type": "sam2_clip_regions", "version": 3, "dataset": dataset,
        "split": split, "clip_model_name": clip_model_name, "sam_key": sam_key,
        "reference_ids": list(reference_ids), "max_regions": max_regions,
        "min_area_ratio": min_area_ratio, "max_area_ratio": max_area_ratio,
        "background_mode": background_mode, "padding_ratio": padding_ratio,
    }
    if not force_rebuild:
        payload = load_cache(path, meta)
        if payload and isinstance(payload.get("regions"), dict) and set(payload["regions"]) == set(reference_ids):
            print(f"Loaded SAM+CLIP region cache: {path}")
            return payload["regions"]

    regions: Dict[str, List[dict]] = {}
    if not force_rebuild and partial.is_file():
        try:
            payload = torch.load(partial, map_location="cpu", weights_only=False)
            if all(payload.get(k) == v for k, v in meta.items()) and isinstance(payload.get("regions"), dict):
                regions.update(payload["regions"])
                print(f"Resuming SAM region cache: {len(regions)}/{len(reference_ids)}")
        except Exception as exc:
            print(f"[WARN] Partial SAM cache ignored: {exc}")

    remaining = [x for x in reference_ids if x not in regions]
    for ref_id in tqdm(remaining, desc="SAM2 + CLIP reference regions"):
        image = load_image(image_folder, ref_id)
        if image is None:
            raise RuntimeError(f"Reference image could not be opened: {ref_id}")
        arr = np.asarray(image.convert("RGB"), copy=True)
        try:
            with torch.inference_mode():
                masks = sam_generator.generate(arr)
        except Exception as exc:
            raise RuntimeError(f"SAM2 failed for {ref_id}: {exc}") from exc

        selected = select_sam_masks(
            masks, arr.shape[1], arr.shape[0],
            min_area_ratio, max_area_ratio, max_regions,
        )
        records: List[dict] = []
        for idx, reg in enumerate(selected):
            crop = crop_region(
                image, reg["segmentation"], reg["bbox"], padding_ratio, background_mode
            )
            if crop is None:
                continue
            try:
                feat = encode_clip_images([crop], clip_model, clip_preprocess)[0]
            except Exception as exc:
                print(f"[WARN] Region CLIP failed: ref={ref_id}, region={idx}: {exc}")
                continue
            records.append({
                "index": idx, "feature": feat,
                "bbox": reg["bbox"], "area": reg["area"],
                "area_ratio": reg["area_ratio"], "sam_quality": reg["sam_quality"],
            })
        regions[ref_id] = records
        atomic_torch_save({**meta, "regions": regions, "num_cached": len(regions)}, partial)

    atomic_torch_save({**meta, "regions": regions, "num_cached": len(regions)}, path)
    try:
        partial.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"Saved SAM+CLIP region cache: {path}")
    return regions


def aggregate_regions(
    records: List[dict], seed: torch.Tensor, top_m: int, temperature: float,
) -> Optional[torch.Tensor]:
    if not records:
        return None
    feats = torch.cat([x["feature"].reshape(1, -1).float() for x in records], dim=0)
    feats = F.normalize(feats, dim=-1)
    seed = F.normalize(seed.reshape(1, -1).float(), dim=-1)
    scores = feats @ seed.squeeze(0)
    m = min(top_m, scores.numel())
    vals, idx = torch.topk(scores, k=m)
    selected = feats[idx]
    if m == 1:
        return selected[:1]
    weights = F.softmax(vals / max(temperature, 1e-4), dim=0)
    agg = (selected * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    return F.normalize(agg, dim=-1)


def make_query(
    global_ref: torch.Tensor, text: torch.Tensor, regions: List[dict],
    text_weight: float, global_weight: float, region_weight: float,
    top_regions: int, temperature: float,
) -> Tuple[torch.Tensor, int]:
    global_ref = F.normalize(global_ref.reshape(1, -1).float(), dim=-1)
    text = F.normalize(text.reshape(1, -1).float(), dim=-1)
    seed = F.normalize(global_ref + text, dim=-1)
    region = aggregate_regions(regions, seed, top_regions, temperature)
    if region is None:
        q = text_weight * text + global_weight * global_ref
        return F.normalize(q, dim=-1), 0
    q = text_weight * text + global_weight * global_ref + region_weight * region
    return F.normalize(q, dim=-1), len(regions)


# ============================================================
# Ranking / metrics
# ============================================================

def rank_from_sims(
    sims: torch.Tensor, gallery_ids: List[str],
    exclude_ids: Iterable[str] = (), restrict_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    sims = sims.detach().float().flatten().cpu()
    if sims.numel() != len(gallery_ids):
        raise ValueError(f"Similarity length {sims.numel()} != gallery {len(gallery_ids)}")
    excluded = {normalize_id(x) for x in exclude_ids if normalize_id(x)}
    allowed = None if restrict_ids is None else {normalize_id(x) for x in restrict_ids if normalize_id(x)}
    valid = torch.tensor([
        x not in excluded and (allowed is None or x in allowed)
        for x in gallery_ids
    ], dtype=torch.bool)
    sims[~valid] = -float("inf")
    order = torch.argsort(sims, descending=True).tolist()
    return [gallery_ids[i] for i in order if torch.isfinite(sims[i])]


def average_precision_at_k(relevant: Iterable[str], retrieved: List[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    for rank, image_id in enumerate(retrieved[:k], start=1):
        if image_id in rel:
            hits += 1
            score += hits / rank
    return score / min(len(rel), k)


def reciprocal_rank(relevant: Iterable[str], retrieved: List[str]) -> float:
    rel = set(relevant)
    for rank, image_id in enumerate(retrieved, start=1):
        if image_id in rel:
            return 1.0 / rank
    return 0.0


def init_results() -> dict:
    return {"prec": {k: [] for k in K_VALUES}, "rec": {k: [] for k in K_VALUES}, "map": {k: [] for k in MAP_K_VALUES}, "mrr": []}


def update_metrics(results: dict, ranked: List[str], positives: Iterable[str]) -> None:
    positives = {normalize_id(x) for x in positives if normalize_id(x)}
    if not positives:
        return
    for k in K_VALUES:
        hits = sum(1 for x in ranked[:k] if x in positives)
        results["prec"][k].append(hits / k)
        results["rec"][k].append(hits / len(positives))
    for k in MAP_K_VALUES:
        results["map"][k].append(average_precision_at_k(positives, ranked, k))
    results["mrr"].append(reciprocal_rank(positives, ranked))


def summarize(results: dict) -> dict:
    mean = lambda x: float(np.mean(x)) if x else 0.0
    return {
        "mrr": mean(results["mrr"]), "map5": mean(results["map"][5]),
        "map10": mean(results["map"][10]), "map50": mean(results["map"][50]),
        "prec1": mean(results["prec"][1]), "prec5": mean(results["prec"][5]),
        "prec10": mean(results["prec"][10]), "prec50": mean(results["prec"][50]),
        "rec1": mean(results["rec"][1]), "rec5": mean(results["rec"][5]),
        "rec10": mean(results["rec"][10]), "rec50": mean(results["rec"][50]),
    }


def print_summary(label: str, s: dict, total: int, skipped: int) -> None:
    print(
        f"{label} | MRR={s['mrr']:.4f} | mAP@5={s['map5']:.4f} | "
        f"mAP@10={s['map10']:.4f} | mAP@50={s['map50']:.4f} | "
        f"P@1={s['prec1']:.4f} | P@5={s['prec5']:.4f} | "
        f"P@10={s['prec10']:.4f} | P@50={s['prec50']:.4f} | "
        f"R@1={s['rec1']:.4f} | R@5={s['rec5']:.4f} | "
        f"R@10={s['rec10']:.4f} | R@50={s['rec50']:.4f} | "
        f"n={total}, skipped={skipped}"
    )


def evaluate(
    data: List[dict], gallery_ids: List[str], gallery_feats: torch.Tensor,
    clip_gallery: Dict[str, torch.Tensor], text_cache: Dict[str, torch.Tensor],
    region_cache: Dict[str, List[dict]], subset: bool,
    text_weight: float, global_weight: float, region_weight: float,
    top_regions: int, temperature: float,
):
    results = init_results()
    rankings: Dict[str, List[str]] = {}
    skipped = 0
    total = 0
    reasons: Dict[str, int] = {}
    region_counts = []

    for item in tqdm(
        data,
        desc=(f"CLIP+SAM t={text_weight:.2f} g={global_weight:.2f} r={region_weight:.2f}"),
    ):
        qkey = get_query_key(item)
        ref_id = item["reference_id"]
        caption = normalize_prompt(item.get("caption"))
        global_feat = clip_gallery.get(ref_id)
        text_feat = text_cache.get(caption)
        if not caption:
            skipped += 1; reasons["empty_caption"] = reasons.get("empty_caption", 0) + 1; continue
        if global_feat is None:
            skipped += 1; reasons["missing_reference_clip"] = reasons.get("missing_reference_clip", 0) + 1; continue
        if text_feat is None:
            skipped += 1; reasons["missing_text_clip"] = reasons.get("missing_text_clip", 0) + 1; continue
        try:
            query, n_regions = make_query(
                global_feat, text_feat, region_cache.get(ref_id, []),
                text_weight, global_weight, region_weight,
                top_regions, temperature,
            )
            region_counts.append(n_regions)
            sims = gallery_feats @ query.squeeze(0)
            ranked = rank_from_sims(
                sims, gallery_ids, exclude_ids=[ref_id],
                restrict_ids=item.get("members", []) if subset else None,
            )
            rankings[qkey] = ranked[:50]
            positives = get_relevant_ids(item)
            if positives:
                update_metrics(results, ranked, positives)
            total += 1
        except Exception as exc:
            skipped += 1
            reason = f"{type(exc).__name__}: {exc}"
            reasons[reason] = reasons.get(reason, 0) + 1
            print(f"[WARN] Query failed: {qkey!r}, ref={ref_id!r}: {reason}")

    stats = {
        "avg_available_regions": float(np.mean(region_counts)) if region_counts else 0.0,
        "queries_with_regions": sum(1 for x in region_counts if x > 0),
        "queries_without_regions": sum(1 for x in region_counts if x == 0),
    }
    return results, total, skipped, rankings, reasons, stats


def save_predictions(path: Path, data: List[dict], rankings: Dict[str, List[str]], total: int, skipped: int, reasons: dict, config: dict) -> None:
    preds = []
    for item in data:
        key = get_query_key(item)
        ranking = rankings.get(key, [])
        target = normalize_id(item.get("target_id"))
        target_rank = ranking.index(target) + 1 if target and target in ranking else None
        preds.append({
            "query_id": key, "candidate_id": item.get("candidate_id") or item.get("reference_id"),
            "reference": item.get("reference_id"), "target_id": item.get("target_id"),
            "caption": item.get("caption", ""), "members": item.get("members", []),
            "ranking": ranking, "target_rank": target_rank,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "cirr", "model": "CLIP+SAM2", "num_queries": len(data),
            "num_predictions": total, "num_skipped": skipped, "skip_reasons": reasons,
            "config": config, "predictions": preds,
        }, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to: {path}")


def save_metrics(path: Path, dataset: str, split: str, subset: bool, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "split": split, "cirr_subset": subset, "metrics": result}, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {path}")


def infer_split(json_path: str, image_folder: str) -> str:
    text = f"{json_path} {image_folder}".lower().replace("\\", "/")
    if "test1" in text or "/test/" in text or text.endswith("/test"):
        return "test"
    if "val" in text or "dev" in text:
        return "val"
    if "train" in text:
        return "train"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="CIRR/CIRCO retrieval with CLIP + SAM2")
    parser.add_argument("--dataset", choices=["cirr", "circo"], required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--split", choices=["auto", "train", "val", "test", "unknown"], default="auto")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--embedding_cache_dir", default="./embedding_cache")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--clip_text_batch_size", type=int, default=64)
    parser.add_argument("--sam_max_regions", type=int, default=20)
    parser.add_argument("--sam_min_area_ratio", type=float, default=0.01)
    parser.add_argument("--sam_max_area_ratio", type=float, default=0.80)
    parser.add_argument("--sam_background_mode", choices=["crop", "masked", "white"], default="crop")
    parser.add_argument("--sam_padding_ratio", type=float, default=0.08)
    parser.add_argument("--top_regions", type=int, default=3)
    parser.add_argument("--region_temperature", type=float, default=0.07)
    parser.add_argument("--text_weight", type=float, default=0.35)
    parser.add_argument("--global_weight", type=float, default=0.30)
    parser.add_argument("--region_weight", type=float, default=0.35)
    parser.add_argument("--cirr_subset", action="store_true")
    parser.add_argument("--require_cirr_gt", action="store_true")
    parser.add_argument("--force_rebuild_embeddings", action="store_true")
    parser.add_argument("--strict_evaluation", action="store_true")
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--predictions_output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.clip_batch_size < 1 or args.clip_text_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    if args.sam_max_regions < 1 or args.top_regions < 1:
        raise ValueError("Region counts must be >= 1")
    if not (0.0 <= args.sam_min_area_ratio < args.sam_max_area_ratio <= 1.0):
        raise ValueError("Invalid SAM area ratios")
    if args.region_temperature <= 0:
        raise ValueError("--region_temperature must be > 0")
    if min(args.text_weight, args.global_weight, args.region_weight) < 0:
        raise ValueError("Fusion weights cannot be negative")
    if args.text_weight + args.global_weight + args.region_weight <= 0:
        raise ValueError("At least one fusion weight must be > 0")

    set_seed(args.seed)
    split = args.split if args.split != "auto" else infer_split(args.json_path, args.image_folder)
    print(f"Split : {split}")
    print(f"Device: {DEVICE}")

    detected, data = load_dataset(args.json_path)
    if detected != args.dataset:
        raise ValueError(f"Dataset mismatch: detected={detected}, requested={args.dataset}")
    subset = bool(args.cirr_subset) if args.dataset == "cirr" else False
    if args.dataset == "cirr":
        validate_cirr(data, args.require_cirr_gt)

    gallery_ids = scan_gallery_ids(args.image_folder)
    print(f"Gallery images: {len(gallery_ids)}")

    if args.dataset == "cirr":
        gallery_set = set(gallery_ids)
        missing_refs = sorted({x["reference_id"] for x in data if x.get("reference_id") not in gallery_set})
        missing_targets = sorted({normalize_id(x.get("target_id")) for x in data if normalize_id(x.get("target_id")) and normalize_id(x.get("target_id")) not in gallery_set})
        if missing_refs:
            raise RuntimeError(f"Missing reference images: {missing_refs[:20]}")
        if missing_targets:
            print(f"[WARN] Missing target images in gallery: {missing_targets[:20]}")

    print("Retrieval protocol: " + ("CIRR SUBSET" if subset else "FULL GALLERY"))

    print("\n=== CLIP ===")
    clip_model, clip_preprocess = load_clip_model(args.clip_model)
    clip_gallery = build_clip_gallery_cache(
        gallery_ids, args.image_folder, clip_model, clip_preprocess,
        args.embedding_cache_dir, args.dataset, split, args.clip_model,
        args.clip_batch_size, args.force_rebuild_embeddings,
    )
    gallery_feats = stack_feature_dict(clip_gallery, gallery_ids)
    print(f"CLIP gallery tensor: {tuple(gallery_feats.shape)}")

    text_cache = build_clip_text_cache(
        [x.get("caption", "") for x in data], clip_model, args.clip_model,
        args.embedding_cache_dir, args.dataset, split, args.clip_text_batch_size,
        args.force_rebuild_embeddings,
    )

    print("\n=== SAM 2.1 ===")
    _, sam_generator = load_sam2(args.sam2_checkpoint, args.sam2_config)
    unique_refs = sorted({x["reference_id"] for x in data if x.get("reference_id")})
    print(f"Unique reference images: {len(unique_refs)}")
    cp = Path(args.sam2_checkpoint).expanduser().resolve()
    stat = cp.stat()
    sam_key = stable_hash(f"{cp}|{stat.st_size}|{stat.st_mtime_ns}")

    region_cache = build_region_cache(
        unique_refs, args.image_folder, clip_model, clip_preprocess,
        args.clip_model, sam_generator, sam_key,
        args.embedding_cache_dir, args.dataset, split,
        args.sam_max_regions, args.sam_min_area_ratio, args.sam_max_area_ratio,
        args.sam_background_mode, args.sam_padding_ratio,
        args.force_rebuild_embeddings,
    )

    print("\n=== CLIP + SAM2 retrieval ===")
    results, total, skipped, rankings, reasons, region_stats = evaluate(
        data, gallery_ids, gallery_feats, clip_gallery, text_cache, region_cache,
        subset, args.text_weight, args.global_weight, args.region_weight,
        args.top_regions, args.region_temperature,
    )

    if skipped:
        msg = f"Incomplete evaluation: processed={total}/{len(data)}, skipped={skipped}"
        if args.strict_evaluation:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
        print(f"[WARN] Skip reasons: {reasons}")

    summary = summarize(results)
    tag = f"CLIP+SAM2 t={args.text_weight:.2f} g={args.global_weight:.2f} r={args.region_weight:.2f}"
    if any(get_relevant_ids(x) for x in data):
        print_summary(tag, summary, total, skipped)
    else:
        print("[INFO] No ground truth available; rankings were generated but metrics are not meaningful.")

    print("\n=== Region statistics ===")
    print(f"Queries with regions    : {region_stats['queries_with_regions']}")
    print(f"Queries without regions : {region_stats['queries_without_regions']}")
    print(f"Average available       : {region_stats['avg_available_regions']:.2f}")

    config = {
        "model": "CLIP+SAM2", "clip_model": args.clip_model,
        "sam2_checkpoint": str(cp), "sam2_config": args.sam2_config,
        "text_weight": args.text_weight, "global_weight": args.global_weight,
        "region_weight": args.region_weight, "top_regions": args.top_regions,
        "region_temperature": args.region_temperature,
        "sam_max_regions": args.sam_max_regions,
        "sam_min_area_ratio": args.sam_min_area_ratio,
        "sam_max_area_ratio": args.sam_max_area_ratio,
        "sam_background_mode": args.sam_background_mode,
        "sam_padding_ratio": args.sam_padding_ratio,
        "cirr_subset": subset,
    }

    pred_path = Path(args.predictions_output) if args.predictions_output else Path(args.json_path).with_name(Path(args.json_path).stem + "_clip_sam2_predictions.json")
    metric_path = Path(args.metrics_output) if args.metrics_output else Path(args.json_path).with_name(Path(args.json_path).stem + "_clip_sam2_metrics.json")
    save_predictions(pred_path, data, rankings, total, skipped, reasons, config)
    save_metrics(metric_path, args.dataset, split, subset, {tag: summary})

    print("\n" + "=" * 150)
    print("FINAL RESULTS - CLIP + SAM2")
    print("=" * 150)
    print(f"{tag:<45}{summary['mrr']:>9.4f}{summary['map5']:>10.4f}{summary['map10']:>10.4f}{summary['map50']:>10.4f}{summary['rec1']:>10.4f}{summary['rec5']:>10.4f}{summary['rec10']:>10.4f}{summary['rec50']:>10.4f}")
    print(f"Predictions: {pred_path}")
    print(f"Metrics    : {metric_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        raise
    except Exception:
        traceback.print_exc()
        raise
