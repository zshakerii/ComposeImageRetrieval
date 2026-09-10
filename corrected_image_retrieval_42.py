#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-Embedding-2B evaluator for CIRR/CIRCO.

Designed for local/offline evaluation and compatibility with the user's
existing image-retrieval evaluation protocol:
  - recursive image gallery discovery
  - CIRR new/old JSON schemas
  - optional CIRR subset protocol
  - reference-image + relative-caption composed queries
  - full-gallery image embeddings cached on disk
  - resumable query embedding cache
  - Recall@1/5/10/50, Precision@1/5/10/50,
    mAP@5/10/50 and MRR
  - query-level prediction JSON
  - metrics JSON

Recommended model for this task:
  Qwen/Qwen3-VL-Embedding-2B

This is intentionally NOT a Qwen2-VL-2B-Instruct caption-generation evaluator.
Qwen3-VL-Embedding is an embedding model designed for multimodal retrieval.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
K_VALUES = (1, 5, 10, 50)
MAP_K_VALUES = (5, 10, 50)
DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."

IMAGE_PATH_INDEX: Dict[str, str] = {}


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
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def query_key(item: dict, index: int = 0) -> str:
    for key in ("query_id", "pairid"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    ref = normalize_id(item.get("reference_id"))
    target = normalize_id(item.get("target_id") or item.get("target_hard"))
    if ref and target:
        return f"{ref}__{target}__{index}"
    return ref or f"query_{index}"


# ---------------------------------------------------------------------------
# Dataset parsing: compatible with the user's current script
# ---------------------------------------------------------------------------

def parse_cirr_sample(sample: dict, index: int = 0) -> dict:
    if not isinstance(sample, dict):
        raise ValueError(f"CIRR sample #{index} must be a JSON object")

    # New CIRR schema: candidate_id, group, target_id, caption
    if any(k in sample for k in ("candidate_id", "group", "target_id")):
        required = ("candidate_id", "caption", "group", "target_id")
        missing = [k for k in required if k not in sample]
        if missing:
            raise ValueError(f"CIRR sample #{index} missing fields: {missing}")

        ref = normalize_id(sample.get("candidate_id"))
        caption = normalize_text(sample.get("caption"))
        group_raw = sample.get("group")
        if not isinstance(group_raw, list) or not group_raw:
            raise ValueError(f"CIRR sample #{index}: 'group' must be non-empty list")
        members = list(dict.fromkeys(normalize_id(x) for x in group_raw if normalize_id(x)))
        target = normalize_id(sample.get("target_id"))

        if not ref or not target:
            raise ValueError(f"CIRR sample #{index}: empty reference/target")
        if ref not in members:
            raise ValueError(f"CIRR sample #{index}: candidate_id not in group")
        if target not in members:
            raise ValueError(f"CIRR sample #{index}: target_id not in group")

        qid = f"{ref}__{target}__{index}"
        return {
            "query_id": qid,
            "pairid": qid,
            "reference_id": ref,
            "target_id": target,
            "target_hard": target,
            "target_soft": {},
            "positives": [target],
            "members": members,
            "group": members,
            "caption": caption,
        }

    # Old cap.rc2 schema
    if "reference" not in sample or "caption" not in sample:
        raise ValueError(f"CIRR sample #{index}: unsupported schema")
    img_set = sample.get("img_set")
    if not isinstance(img_set, dict):
        raise ValueError(f"CIRR sample #{index}: img_set must be dict")

    ref = normalize_id(sample.get("reference"))
    members = list(dict.fromkeys(
        normalize_id(x) for x in img_set.get("members", []) if normalize_id(x)
    ))
    if not ref or ref not in members:
        raise ValueError(f"CIRR sample #{index}: invalid reference/members")

    target_hard = normalize_id(sample.get("target_hard"))
    positives: List[str] = [target_hard] if target_hard else []
    soft = sample.get("target_soft")
    if isinstance(soft, dict):
        positives.extend(normalize_id(x) for x in soft.keys() if normalize_id(x))
    elif isinstance(soft, list):
        positives.extend(normalize_id(x) for x in soft if normalize_id(x))
    positives = list(dict.fromkeys(x for x in positives if x))

    pairid = sample.get("pairid")
    qid = str(pairid) if pairid is not None else f"{ref}__{index}"
    return {
        "query_id": qid,
        "pairid": pairid if pairid is not None else qid,
        "reference_id": ref,
        "target_id": target_hard or None,
        "target_hard": target_hard or None,
        "target_soft": soft if isinstance(soft, (dict, list)) else {},
        "positives": positives,
        "members": members,
        "caption": normalize_text(sample.get("caption")),
    }


def parse_circo_sample(sample: dict, index: int = 0) -> dict:
    gt_ids = [normalize_id(x) for x in sample.get("gt_img_ids", []) if normalize_id(x)]
    target = normalize_id(sample.get("target_img_id"))
    if target and target not in gt_ids:
        gt_ids.insert(0, target)
    return {
        "query_id": str(sample.get("query_id") or sample.get("pairid") or index),
        "pairid": sample.get("pairid") or index,
        "reference_id": normalize_id(sample.get("reference_img_id")),
        "caption": normalize_text(sample.get("relative_caption")),
        "target_id": target or None,
        "positives": list(dict.fromkeys(gt_ids)),
        "members": [],
    }


def load_dataset(path: str) -> Tuple[str, List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset JSON must be a non-empty list")

    first = data[0]
    if isinstance(first, dict) and all(k in first for k in ("candidate_id", "caption", "group", "target_id")):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]
    if isinstance(first, dict) and "reference" in first and isinstance(first.get("img_set"), dict):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]
    if isinstance(first, dict) and ("reference_img_id" in first or "gt_img_ids" in first):
        return "circo", [parse_circo_sample(x, i) for i, x in enumerate(data)]
    raise ValueError(f"Unknown dataset JSON schema. Keys: {list(first.keys()) if isinstance(first, dict) else 'N/A'}")


def get_relevant_ids(item: dict) -> Set[str]:
    target = normalize_id(item.get("target_id") or item.get("target_hard"))
    if target:
        return {target}

    positives: Set[str] = set()
    soft = item.get("target_soft")
    if isinstance(soft, dict):
        for k, v in soft.items():
            k = normalize_id(k)
            if not k:
                continue
            try:
                if float(v) > 0:
                    positives.add(k)
            except (TypeError, ValueError):
                positives.add(k)
    elif isinstance(soft, list):
        positives.update(normalize_id(x) for x in soft if normalize_id(x))

    positives.update(normalize_id(x) for x in item.get("positives", []) if normalize_id(x))
    return positives


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def build_image_path_index(image_folder: str) -> Dict[str, str]:
    root = Path(image_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.setdefault(p.stem, str(p))
    if not out:
        raise RuntimeError(f"No images found in {root}")
    return out


def find_image_path(image_id: str) -> Optional[str]:
    image_id = normalize_id(image_id)
    candidates = [image_id]
    if image_id.isdigit():
        candidates.append(image_id.zfill(12))
    for key in dict.fromkeys(candidates):
        p = IMAGE_PATH_INDEX.get(key)
        if p and Path(p).is_file():
            return p
    return None


def load_image(image_id: str) -> Image.Image:
    p = find_image_path(image_id)
    if not p:
        raise FileNotFoundError(f"Image not found: {image_id}")
    with Image.open(p) as img:
        img.load()
        return img.convert("RGB")


# ---------------------------------------------------------------------------
# Qwen3-VL-Embedding local loader
# ---------------------------------------------------------------------------

def _load_official_embedder_class(model_path: Path):
    """Load Qwen3VLEmbedder from the model's local scripts if present."""
    candidates = [
        model_path / "scripts" / "qwen3_vl_embedding.py",
        model_path / "qwen3_vl_embedding.py",
        model_path.parent / "Qwen3-VL-Embedding" / "scripts" / "qwen3_vl_embedding.py",
    ]
    for script in candidates:
        if not script.is_file():
            continue
        spec = importlib.util.spec_from_file_location("local_qwen3_vl_embedding", str(script))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = getattr(module, "Qwen3VLEmbedder", None)
        if cls is not None:
            print(f"Qwen3 embedder helper: {script}")
            return cls
    return None


class Qwen3Local:
    def __init__(self, model_path: str, dtype: str = "auto"):
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Qwen3 model directory does not exist: {model_path}")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

        self.path = model_path
        self.dtype = torch_dtype
        self.backend = None
        self.model = None

        # Preferred: official Qwen3 helper shipped with the model repository.
        embedder_cls = _load_official_embedder_class(model_path)
        if embedder_cls is not None:
            kwargs: Dict[str, Any] = {}
            if DEVICE.type == "cuda":
                kwargs["torch_dtype"] = torch_dtype
                # FlashAttention is optional; do not force it because many Windows installs lack it.
            self.model = embedder_cls(str(model_path), **kwargs)
            self.backend = "official-Qwen3VLEmbedder"
            return

        # Fallback: current HF model implementation exposes encode().
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch_dtype,
            )
            self.model = self.model.to(DEVICE).eval()
            self.backend = "transformers-AutoModel"
        except Exception as exc:
            raise RuntimeError(
                "Could not load Qwen3-VL-Embedding locally. "
                "Make sure the model folder is a complete local snapshot including the official "
                "scripts/qwen3_vl_embedding.py, or install a recent Transformers version. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        print(f"Qwen3 backend: {self.backend}")
        print(f"Qwen3 device: {DEVICE}")
        print(f"Qwen3 dtype: {self.dtype}")

    @torch.inference_mode()
    def encode(self, inputs: List[Dict[str, Any]], instruction: Optional[str] = None) -> torch.Tensor:
        """Return L2-normalized [N,D] embeddings for text/image/mixed inputs."""
        if not inputs:
            return torch.empty((0, 0), dtype=torch.float32)

        prepared = []
        for x in inputs:
            item = dict(x)
            if instruction is not None and "instruction" not in item:
                item["instruction"] = instruction
            prepared.append(item)

        # Official helper uses process(list_of_dict).
        if self.backend == "official-Qwen3VLEmbedder":
            out = self.model.process(prepared, normalize=True)
        else:
            # Current official HF model supports encode(inputs).
            try:
                out = self.model.encode(prepared)
            except TypeError:
                out = self.model.encode(prepared, normalize=True)

        if not torch.is_tensor(out):
            out = torch.as_tensor(out)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        out = out.float().cpu()
        out = F.normalize(out, p=2, dim=-1)
        return out


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in s).replace(" ", "_")


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_image_cache(path: Path, gallery_ids: List[str], model_key: str) -> Optional[torch.Tensor]:
    if not path.is_file():
        return None
    try:
        p = torch.load(path, map_location="cpu", weights_only=False)
        if p.get("version") != 2 or p.get("model_key") != model_key:
            return None
        if list(p.get("gallery_ids", [])) != list(gallery_ids):
            return None
        feat = p.get("features")
        if not torch.is_tensor(feat) or feat.ndim != 2 or feat.shape[0] != len(gallery_ids):
            return None
        return F.normalize(feat.float(), dim=-1).cpu()
    except Exception as exc:
        print(f"[WARN] Invalid Qwen3 image cache {path}: {type(exc).__name__}: {exc}")
        return None


def build_image_cache(
    embedder: Qwen3Local,
    gallery_ids: List[str],
    image_folder: str,
    cache_path: Path,
    batch_size: int,
    instruction: str,
    force_rebuild: bool,
) -> torch.Tensor:
    if not force_rebuild:
        cached = load_image_cache(cache_path, gallery_ids, "Qwen3-VL-Embedding-2B")
        if cached is not None:
            print(f"Loaded Qwen3 gallery cache: {cache_path}")
            print(f"Cached gallery images: {cached.shape[0]} x {cached.shape[1]}")
            return cached

    partial = cache_path.with_suffix(cache_path.suffix + ".partial")
    features: Dict[str, torch.Tensor] = {}
    if partial.is_file() and not force_rebuild:
        try:
            p = torch.load(partial, map_location="cpu", weights_only=False)
            if (
                p.get("version") == 2
                and p.get("model_key") == "Qwen3-VL-Embedding-2B"
                and list(p.get("gallery_ids", [])) == list(gallery_ids)
                and isinstance(p.get("features"), dict)
            ):
                features = {k: v for k, v in p["features"].items() if torch.is_tensor(v)}
                print(f"Resuming Qwen3 image cache: {len(features)}/{len(gallery_ids)}")
        except Exception as exc:
            print(f"[WARN] Could not resume partial Qwen3 cache: {exc}")

    def save_partial() -> None:
        atomic_torch_save(
            {
                "version": 2,
                "model_key": "Qwen3-VL-Embedding-2B",
                "gallery_ids": list(gallery_ids),
                "features": features,
            },
            partial,
        )

    remaining = [x for x in gallery_ids if x not in features]
    for start in tqdm(range(0, len(remaining), batch_size), desc="Qwen3 gallery image embeddings"):
        batch_ids = remaining[start:start + batch_size]
        inputs = []
        valid_ids = []
        for image_id in batch_ids:
            p = find_image_path(image_id)
            if not p:
                raise RuntimeError(f"Gallery image missing: {image_id}")
            inputs.append({"image": p})
            valid_ids.append(image_id)

        try:
            batch = embedder.encode(inputs, instruction=None)
            if batch.ndim != 2 or batch.shape[0] != len(valid_ids):
                raise RuntimeError(f"Unexpected embedding shape {tuple(batch.shape)}")
            for image_id, feat in zip(valid_ids, batch):
                features[image_id] = feat.cpu()
        except Exception as exc:
            save_partial()
            raise RuntimeError(
                f"Qwen3 gallery batch failed at {valid_ids[:3]}: {type(exc).__name__}: {exc}"
            ) from exc
        save_partial()
        print(f"[CACHE] Qwen3: {len(features)}/{len(gallery_ids)} gallery images")

    missing = [x for x in gallery_ids if x not in features]
    if missing:
        save_partial()
        raise RuntimeError(f"Qwen3 gallery cache incomplete: {len(missing)} missing")

    matrix = torch.stack([features[x] for x in gallery_ids], dim=0)
    matrix = F.normalize(matrix.float(), dim=-1).cpu()
    atomic_torch_save(
        {
            "version": 2,
            "model_key": "Qwen3-VL-Embedding-2B",
            "gallery_ids": list(gallery_ids),
            "features": matrix,
        },
        cache_path,
    )
    try:
        partial.unlink()
    except OSError:
        pass
    print(f"Saved Qwen3 gallery cache: {cache_path}")
    return matrix


def load_query_cache(path: Path) -> Optional[Dict[str, torch.Tensor]]:
    if not path.is_file():
        return None
    try:
        p = torch.load(path, map_location="cpu", weights_only=False)
        if p.get("version") != 1 or not isinstance(p.get("features"), dict):
            return None
        return {str(k): v.float() for k, v in p["features"].items() if torch.is_tensor(v)}
    except Exception:
        return None


def save_query_cache(path: Path, features: Dict[str, torch.Tensor]) -> None:
    atomic_torch_save({"version": 1, "features": features}, path)


# ---------------------------------------------------------------------------
# Metrics and ranking
# ---------------------------------------------------------------------------

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
    return {
        "prec": {k: [] for k in K_VALUES},
        "rec": {k: [] for k in K_VALUES},
        "map": {k: [] for k in MAP_K_VALUES},
        "mrr": [],
    }


def update_metrics(results: dict, ranked_ids: List[str], positives: Iterable[str]) -> None:
    positives = {normalize_id(x) for x in positives if normalize_id(x)}
    if not positives:
        return
    for k in K_VALUES:
        top_k = ranked_ids[:k]
        hits = sum(1 for x in top_k if x in positives)
        results["prec"][k].append(hits / k)
        results["rec"][k].append(hits / len(positives))
    for k in MAP_K_VALUES:
        results["map"][k].append(average_precision_at_k(positives, ranked_ids, k))
    results["mrr"].append(reciprocal_rank(positives, ranked_ids))


def summarize(results: dict) -> dict:
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
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


def rank_ids(scores: torch.Tensor, gallery_ids: List[str], exclude_id: str, restrict_ids: Optional[Iterable[str]]) -> List[str]:
    scores = scores.float().flatten().clone()
    if scores.numel() != len(gallery_ids):
        raise ValueError(f"Score length {scores.numel()} != gallery size {len(gallery_ids)}")

    allowed = None if restrict_ids is None else {normalize_id(x) for x in restrict_ids if normalize_id(x)}
    mask = torch.ones(len(gallery_ids), dtype=torch.bool)
    for i, image_id in enumerate(gallery_ids):
        if image_id == normalize_id(exclude_id):
            mask[i] = False
        if allowed is not None and image_id not in allowed:
            mask[i] = False
    scores[~mask] = -float("inf")
    order = torch.argsort(scores, descending=True).tolist()
    return [gallery_ids[i] for i in order if torch.isfinite(scores[i])]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    data: List[dict],
    embedder: Qwen3Local,
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    subset_protocol: bool,
    instruction: str,
    query_cache_path: Path,
    predictions_path: Path,
    query_batch_size: int,
) -> Tuple[dict, Dict[str, List[str]], int, int, Dict[str, int]]:
    cache = load_query_cache(query_cache_path) or {}
    results = init_results()
    rankings: Dict[str, List[str]] = {}
    skip_reasons: Dict[str, int] = {}
    processed = 0
    skipped = 0

    id_to_row = {image_id: i for i, image_id in enumerate(gallery_ids)}

    missing_cached: List[Tuple[str, dict, str]] = []
    query_embeddings: Dict[str, torch.Tensor] = {}
    for idx, item in enumerate(data):
        qid = query_key(item, idx)
        if qid in cache and torch.is_tensor(cache[qid]):
            query_embeddings[qid] = cache[qid]
            continue
        caption = normalize_text(item.get("caption"))
        ref = normalize_id(item.get("reference_id"))
        if not caption:
            skipped += 1
            reason = "empty_caption"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        ref_path = find_image_path(ref)
        if not ref_path:
            skipped += 1
            reason = "missing_reference_image"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        # IMPORTANT: composed CIRR query is encoded as ONE multimodal item:
        # reference image + relative caption. No alpha fusion is used.
        missing_cached.append((qid, item, ref_path))

    for start in tqdm(range(0, len(missing_cached), query_batch_size), desc="Qwen3 composed queries"):
        batch = missing_cached[start:start + query_batch_size]
        inputs = [
            {
                "image": ref_path,
                "text": normalize_text(item["caption"]),
                "instruction": instruction,
            }
            for _, item, ref_path in batch
        ]
        try:
            emb = embedder.encode(inputs)
            if emb.ndim != 2 or emb.shape[0] != len(batch):
                raise RuntimeError(f"Unexpected query embedding shape {tuple(emb.shape)}")
            for (qid, _item, _ref), vec in zip(batch, emb):
                query_embeddings[qid] = vec.cpu()
                cache[qid] = vec.cpu()
            save_query_cache(query_cache_path, cache)
        except Exception as exc:
            for qid, _item, _ref in batch:
                skipped += 1
                reason = f"embedding:{type(exc).__name__}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            print(f"[WARN] Qwen3 query batch failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    predictions = []
    for idx, item in enumerate(data):
        qid = query_key(item, idx)
        qfeat = query_embeddings.get(qid)
        if qfeat is None:
            continue
        try:
            scores = gallery_feats @ qfeat.reshape(-1, 1)
            scores = scores.flatten()
            allowed = item.get("members", []) if subset_protocol else None
            ranked = rank_ids(scores, gallery_ids, item.get("reference_id", ""), allowed)
            rankings[qid] = ranked[:50]
            positives = get_relevant_ids(item)
            if positives:
                update_metrics(results, ranked, positives)
            processed += 1
            target_id = normalize_id(item.get("target_id"))
            target_rank = ranked.index(target_id) + 1 if target_id in ranked else None
            predictions.append({
                "query_id": qid,
                "reference": item.get("reference_id"),
                "target_id": item.get("target_id"),
                "caption": item.get("caption", ""),
                "members": item.get("members", []),
                "ranking": ranked[:50],
                "target_rank": target_rank,
            })
        except Exception as exc:
            skipped += 1
            reason = f"ranking:{type(exc).__name__}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            print(f"[WARN] Ranking failed for {qid}: {type(exc).__name__}: {exc}")

    payload = {
        "dataset": "cirr",
        "model": "Qwen3-VL-Embedding-2B",
        "composition": "single multimodal embedding = reference image + relative caption",
        "instruction": instruction,
        "subset_protocol": subset_protocol,
        "num_queries": len(data),
        "num_predictions": processed,
        "num_skipped": skipped,
        "skip_reasons": skip_reasons,
        "predictions": predictions,
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved: {predictions_path}")

    return results, rankings, processed, skipped, skip_reasons


def print_summary(name: str, summary: dict, processed: int, skipped: int) -> None:
    print("\n" + "=" * 150)
    print(f"{name} | processed={processed} skipped={skipped}")
    print("=" * 150)
    print(
        f"MRR={summary['mrr']:.4f}  "
        f"mAP@5={summary['map5']:.4f}  mAP@10={summary['map10']:.4f}  mAP@50={summary['map50']:.4f}\n"
        f"P@1={summary['prec1']:.4f} P@5={summary['prec5']:.4f} P@10={summary['prec10']:.4f} P@50={summary['prec50']:.4f}\n"
        f"R@1={summary['rec1']:.4f} R@5={summary['rec5']:.4f} R@10={summary['rec10']:.4f} R@50={summary['rec50']:.4f}"
    )


def validate(data: List[dict], require_gt: bool) -> None:
    no_ref = sum(1 for x in data if not normalize_id(x.get("reference_id")))
    no_caption = sum(1 for x in data if not normalize_text(x.get("caption")))
    no_members = sum(1 for x in data if not x.get("members"))
    no_gt = sum(1 for x in data if not get_relevant_ids(x))
    print("\n=== Annotation check ===")
    print(f"Queries                 : {len(data)}")
    print(f"Without reference       : {no_ref}")
    print(f"Without caption         : {no_caption}")
    print(f"Without candidate group : {no_members}")
    print(f"Without ground truth    : {no_gt}")
    if no_ref:
        raise RuntimeError("Some queries have no reference image ID")
    if require_gt and no_gt:
        raise RuntimeError("Ground truth required, but missing for some queries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL-Embedding-2B CIRR evaluator")
    parser.add_argument("--dataset", choices=["cirr", "circo"], default="cirr")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--qwen3_path", required=True)
    parser.add_argument("--split", choices=["auto", "train", "val", "test", "unknown"], default="auto")
    parser.add_argument("--cirr_subset", action="store_true", help="Restrict CIRR ranking to group/members.")
    parser.add_argument("--require_gt", action="store_true")
    parser.add_argument("--embedding_cache_dir", default="./embedding_cache")
    parser.add_argument("--image_batch_size", type=int, default=4)
    parser.add_argument("--query_batch_size", type=int, default=4)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--force_rebuild_embeddings", action="store_true")
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--predictions_output", default=None)
    parser.add_argument("--strict_evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    args = parser.parse_args()

    if args.image_batch_size < 1 or args.query_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")

    set_seed(args.seed)
    print(f"Device: {DEVICE}")
    print(f"Dataset JSON: {args.json_path}")
    print(f"Image folder: {args.image_folder}")
    print(f"Qwen3 path: {args.qwen3_path}")
    print("NOTE: Qwen3 uses a single composed multimodal embedding; alpha sweep is not used.")

    detected, data = load_dataset(args.json_path)
    if detected != args.dataset:
        raise ValueError(f"Dataset mismatch: argument={args.dataset}, detected={detected}")
    if args.dataset != "cirr":
        raise NotImplementedError("This evaluator is intentionally focused on CIRR; add CIRCO separately if needed.")

    validate(data, args.require_gt)

    global IMAGE_PATH_INDEX
    IMAGE_PATH_INDEX = build_image_path_index(args.image_folder)
    gallery_ids = sorted(IMAGE_PATH_INDEX.keys())
    print(f"Gallery images: {len(gallery_ids)}")

    # Check reference/group coverage before expensive model execution.
    missing_ref: Set[str] = set()
    missing_members: Set[str] = set()
    missing_target: Set[str] = set()
    for item in data:
        ref = normalize_id(item.get("reference_id"))
        if ref and ref not in IMAGE_PATH_INDEX:
            missing_ref.add(ref)
        target = normalize_id(item.get("target_id"))
        if target and target not in IMAGE_PATH_INDEX:
            missing_target.add(target)
        for member in item.get("members", []) or []:
            member = normalize_id(member)
            if member and member not in IMAGE_PATH_INDEX:
                missing_members.add(member)
    print("\n=== Gallery coverage ===")
    print(f"Missing references : {len(missing_ref)}")
    print(f"Missing targets    : {len(missing_target)}")
    print(f"Missing group IDs  : {len(missing_members)}")
    if args.require_gt and (missing_ref or missing_target or missing_members):
        raise RuntimeError("Gallery coverage validation failed")

    split = args.split
    if split == "auto":
        text = f"{args.json_path} {args.image_folder}".lower().replace("\\", "/")
        if "test1" in text or "/test/" in text or text.endswith("/test"):
            split = "test"
        elif "val" in text or "dev" in text:
            split = "val"
        elif "train" in text:
            split = "train"
        else:
            split = "unknown"
    print(f"Split: {split}")

    embedder = Qwen3Local(args.qwen3_path, dtype=args.dtype)

    cache_root = Path(args.embedding_cache_dir) / args.dataset / split
    cache_root.mkdir(parents=True, exist_ok=True)
    model_tag = safe_name("Qwen3-VL-Embedding-2B")
    image_cache_path = cache_root / f"{model_tag}_image_embeddings.pt"
    query_cache_path = cache_root / f"{model_tag}_query_embeddings.pt"

    gallery_feats = build_image_cache(
        embedder=embedder,
        gallery_ids=gallery_ids,
        image_folder=args.image_folder,
        cache_path=image_cache_path,
        batch_size=args.image_batch_size,
        instruction=args.instruction,
        force_rebuild=args.force_rebuild_embeddings,
    )

    predictions_path = Path(args.predictions_output) if args.predictions_output else Path(args.json_path).with_name(
        Path(args.json_path).stem + "_qwen3_predictions.json"
    )
    metrics_path = Path(args.metrics_output) if args.metrics_output else Path(args.json_path).with_name(
        Path(args.json_path).stem + "_qwen3_metrics_12.json"
    )

    results, rankings, processed, skipped, skip_reasons = evaluate(
        data=data,
        embedder=embedder,
        gallery_ids=gallery_ids,
        gallery_feats=gallery_feats,
        subset_protocol=args.cirr_subset,
        instruction=args.instruction,
        query_cache_path=query_cache_path,
        predictions_path=predictions_path,
        query_batch_size=args.query_batch_size,
    )

    if skipped and args.strict_evaluation:
        raise RuntimeError(
            f"Qwen3 evaluation incomplete: processed={processed}/{len(data)}, skipped={skipped}. "
            f"Reasons={skip_reasons}"
        )

    has_gt = any(bool(get_relevant_ids(x)) for x in data)
    summary = summarize(results) if has_gt and processed else None
    if summary is not None:
        print_summary("Qwen3-VL-Embedding-2B", summary, processed, skipped)
        payload = {
            "dataset": args.dataset,
            "split": split,
            "model": "Qwen3-VL-Embedding-2B",
            "device": str(DEVICE),
            "subset_protocol": args.cirr_subset,
            "composition": "single multimodal embedding = reference image + relative caption",
            "instruction": args.instruction,
            "processed": processed,
            "skipped": skipped,
            "skip_reasons": skip_reasons,
            "metrics": summary,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved: {metrics_path}")
    else:
        print("\nNo complete local ground-truth metrics available. Predictions were still generated.")


if __name__ == "__main__":
    main()
