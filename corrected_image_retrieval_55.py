
import os
import json
import argparse
import traceback
import hashlib
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

# Allow Pillow to decode JPEG files that are truncated by a small number of bytes.
# This is needed for the corrupted NLVR gallery image encountered during retrieval.
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

try:
    import clip
except ImportError:
    clip = None

try:
    import open_clip
except ImportError:
    open_clip = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
K_VALUES = (1, 5, 10, 50)
MAP_K_VALUES = (5, 10, 50)

FULL_GALLERY = "full_gallery"
CIRR_SUBSET = "cirr_subset"


def set_deterministic_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible evaluation."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset parsing
# ============================================================

def normalize_id(value) -> str:
    """Normalize dataset image IDs while preserving leading zeros."""
    if value is None:
        return ""
    return os.path.splitext(str(value).strip())[0]



def get_query_key(item: dict) -> str:
    """Return a unique/stable key for storing a query ranking."""
    query_id = str(item.get("query_id", "")).strip()
    if query_id:
        return query_id

    pairid = item.get("pairid")
    if pairid is not None:
        return str(pairid)

    reference_id = normalize_id(item.get("reference_id"))
    target_id = normalize_id(item.get("target_id"))
    if reference_id and target_id:
        return f"{reference_id}__{target_id}"
    if reference_id:
        return reference_id
    return "query_unknown"



def parse_cirr_sample(sample: dict, index: int = 0) -> dict:
    """*
    Parse both the new CIRR JSON schema and the old cap.rc2 schema.
    New schema:
        candidate_id -> reference image
        group        -> candidate subset
        target_id    -> hard ground-truth image
        caption      -> relative caption
    *"""
    if not isinstance(sample, dict):
        raise ValueError(f"CIRR sample #{index} must be a JSON object.")

    # --------------------------------------------------------
    # NEW SCHEMA
    # --------------------------------------------------------
    if "candidate_id" in sample or "group" in sample or "target_id" in sample:
        required = ("candidate_id", "caption", "group", "target_id")
        missing = [k for k in required if k not in sample]
        if missing:
            raise ValueError(
                f"CIRR sample #{index} is missing required fields: {missing}"
            )

        reference_id = normalize_id(sample.get("candidate_id"))
        caption = str(sample.get("caption", "")).strip()

        group_raw = sample.get("group")
        if not isinstance(group_raw, list):
            raise ValueError(f"CIRR sample #{index}: 'group' must be a list.")

        members = []
        for value in group_raw:
            image_id = normalize_id(value)
            if image_id:
                members.append(image_id)
        members = list(dict.fromkeys(members))

        target_id = normalize_id(sample.get("target_id"))

        if not reference_id:
            raise ValueError(f"CIRR sample #{index}: empty candidate_id.")
        if not members:
            raise ValueError(f"CIRR sample #{index}: empty group.")
        if reference_id not in members:
            raise ValueError(
                f"CIRR sample #{index}: candidate_id '{reference_id}' is not present in group."
            )
        if not target_id:
            raise ValueError(f"CIRR sample #{index}: empty target_id.")
        if target_id not in members:
            raise ValueError(
                f"CIRR sample #{index}: target_id '{target_id}' is not present in group."
            )

        query_id = f"{reference_id}__{target_id}__{index}"

        return {
            "query_id": query_id,
            "pairid": query_id,
            "annotation_format": "candidate_group",
            "candidate_id": reference_id,
            "target_id": target_id,
            "caption": caption,
            "group": members,
            "reference_id": reference_id,
            "target_hard": target_id,
            "target_soft": {},
            "positives": [target_id],
            "members": members,
            "reference_rank": None,
            "target_rank": None,
            "img_set_id": None,
        }

    # --------------------------------------------------------
    # OLD cap.rc2.* SCHEMA
    # --------------------------------------------------------
    if "reference" not in sample:
        raise ValueError("CIRR sample is missing 'reference'.")
    if "caption" not in sample:
        raise ValueError("CIRR sample is missing 'caption'.")

    img_set = sample.get("img_set")
    if not isinstance(img_set, dict):
        raise ValueError(
            f"CIRR sample pairid={sample.get('pairid')!r} must contain 'img_set'."
        )

    reference_id = normalize_id(sample.get("reference"))
    caption = str(sample.get("caption", "")).strip()

    members_raw = img_set.get("members", [])
    if not isinstance(members_raw, list):
        raise ValueError("CIRR img_set.members must be a list.")

    members = [normalize_id(x) for x in members_raw if normalize_id(x)]
    members = list(dict.fromkeys(members))

    if not reference_id:
        raise ValueError(f"CIRR sample pairid={sample.get('pairid')!r} has empty reference.")
    if reference_id not in members:
        raise ValueError(
            f"CIRR sample pairid={sample.get('pairid')!r}: reference '{reference_id}' is not in members."
        )

    target_hard = normalize_id(sample.get("target_hard"))
    positives = [target_hard] if target_hard else []

    target_soft = sample.get("target_soft")
    if isinstance(target_soft, dict):
        positives.extend(
            normalize_id(x) for x in target_soft.keys() if normalize_id(x)
        )
    elif isinstance(target_soft, list):
        positives.extend(
            normalize_id(x) for x in target_soft if normalize_id(x)
        )
    positives = list(dict.fromkeys(x for x in positives if x))

    pairid = sample.get("pairid")
    query_id = str(pairid) if pairid is not None else f"{reference_id}__{index}"

    return {
        "query_id": query_id,
        "pairid": pairid if pairid is not None else query_id,
        "annotation_format": "cap_rc2",
        "candidate_id": reference_id,
        "target_id": target_hard or None,
        "caption": caption,
        "group": members,
        "reference_id": reference_id,
        "target_hard": target_hard or None,
        "target_soft": target_soft if isinstance(target_soft, dict) else {},
        "positives": positives,
        "members": members,
        "reference_rank": img_set.get("reference_rank"),
        "target_rank": img_set.get("target_rank"),
        "img_set_id": img_set.get("id"),
    }




def parse_circo_sample(sample: dict) -> dict:
    gt_ids = [normalize_id(x) for x in sample.get("gt_img_ids", [])]
    target_id = normalize_id(sample.get("target_img_id"))

    if target_id and target_id not in gt_ids:
        gt_ids.insert(0, target_id)

    return {
        "reference_id": normalize_id(sample.get("reference_img_id")),
        "caption": str(sample.get("relative_caption", "")).strip(),
        "target_id": target_id,
        "positives": gt_ids,
        # CIRCO does NOT define gallery as GT images.
        "members": [],
    }


def load_dataset(json_path: str) -> Tuple[str, List[dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON list.")

    first = data[0]

    if (
        isinstance(first, dict)
        and "candidate_id" in first
        and "caption" in first
        and "group" in first
        and "target_id" in first
    ):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]

    if (
        isinstance(first, dict)
        and "reference" in first
        and "caption" in first
        and isinstance(first.get("img_set"), dict)
        and "members" in first["img_set"]
    ):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]

    if isinstance(first, dict) and (
        "reference_img_id" in first or "gt_img_ids" in first
    ):
        return "circo", [parse_circo_sample(x) for x in data]

    raise ValueError(
        "Unknown CIRR/CIRCO JSON format. "
        f"First keys: {list(first.keys()) if isinstance(first, dict) else 'N/A'}"
    )




def summarize_cirr_ground_truth(data: List[dict]) -> Tuple[int, int]:
    """Return (queries_with_gt, queries_without_gt)."""
    with_gt = sum(1 for item in data if get_relevant_ids(item))
    return with_gt, len(data) - with_gt




def validate_cirr_annotations(
    data: List[dict],
    require_ground_truth: bool = False,
) -> None:
    """*
    Validate CIRR annotations.
    For test1, img_set.members is mandatory because it defines the candidate
    subset. Ground truth is optional.
    *"""
    with_gt, without_gt = summarize_cirr_ground_truth(data)

    print("\n=== CIRR annotation check ===")
    print(f"Queries                  : {len(data)}")
    print(f"Queries with GT          : {with_gt}")
    print(f"Queries without GT       : {without_gt}")

    missing_members = []
    missing_reference = []
    missing_caption = []
    invalid_reference_rank = []

    for item in data:
        pairid = item.get("pairid")

        if not item.get("members"):
            missing_members.append(pairid)

        if not item.get("reference_id"):
            missing_reference.append(pairid)

        if not normalize_prompt(item.get("caption", "")):
            missing_caption.append(pairid)

        reference_rank = item.get("reference_rank")
        if reference_rank is not None:
            try:
                rank = int(reference_rank)
                if rank < 0:
                    invalid_reference_rank.append(pairid)
            except (TypeError, ValueError):
                invalid_reference_rank.append(pairid)

    print(f"Queries without members  : {len(missing_members)}")
    print(f"Queries without reference: {len(missing_reference)}")
    print(f"Queries without caption   : {len(missing_caption)}")
    print(f"Invalid reference_rank    : {len(invalid_reference_rank)}")

    if missing_members:
        raise RuntimeError(
            f"{len(missing_members)} CIRR queries have empty img_set.members. "
            "CIRR test1 requires a candidate subset for each query."
        )

    if missing_reference:
        raise RuntimeError(
            f"{len(missing_reference)} CIRR queries have no reference image ID."
        )

    # Empty captions are not fatal for CIRR test1.
    # The affected query is skipped during model inference because models
    # such as CLIP/SEARLE require a textual query. We keep the query in the
    # dataset so the final prediction file can report it explicitly.
    if missing_caption:
        print("\n[WARN] Queries with empty annotation captions require fallback resolution before inference:")
        for pairid in missing_caption[:30]:
            item = next((x for x in data if x.get("pairid") == pairid), None)
            if item is None:
                print(f"  pairid={pairid!r}")
            else:
                print(
                    f"  pairid={pairid!r}, "
                    f"reference={item.get('reference_id')!r}, "
                    f"members={len(item.get('members') or [])}"
                )

        if len(missing_caption) > 30:
            print(f"  ... and {len(missing_caption) - 30} more")

    if invalid_reference_rank:
        raise RuntimeError(
            f"{len(invalid_reference_rank)} CIRR queries have invalid reference_rank."
        )

    if require_ground_truth and without_gt > 0:
        raise RuntimeError(
            "Ground truth is required, but the supplied CIRR JSON does not "
            "contain target_hard/target_soft for every query."
        )



def load_optional_ground_truth(path: Optional[str]) -> Optional[List[dict]]:
    """Load an optional ground-truth JSON and normalize common CIRR formats."""
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    if not isinstance(gt_data, list):
        raise ValueError("--ground_truth_path must contain a JSON list.")

    normalized = []
    for sample in gt_data:
        if not isinstance(sample, dict):
            continue

        pairid = sample.get("pairid")
        reference_id = normalize_id(
            sample.get("reference")
            or sample.get("reference_img_id")
            or sample.get("reference_id")
        )

        target_id = normalize_id(
            sample.get("target_hard")
            or sample.get("target_img_id")
            or sample.get("target_id")
        )

        positives = []
        soft = sample.get("target_soft")
        if isinstance(soft, dict):
            positives = [normalize_id(k) for k in soft.keys() if normalize_id(k)]

        for key in ("gt_img_ids", "positives", "positive_ids"):
            value = sample.get(key)
            if isinstance(value, list):
                positives.extend(normalize_id(x) for x in value if normalize_id(x))

        if target_id:
            positives.insert(0, target_id)

        normalized.append({
            "pairid": pairid,
            "reference_id": reference_id,
            "target_id": target_id or None,
            "positives": list(dict.fromkeys(x for x in positives if x)),
        })

    return normalized


def attach_ground_truth(data: List[dict], gt_data: Optional[List[dict]]) -> Tuple[List[dict], int]:
    """Merge optional GT annotations into query data by pairid, then reference ID."""
    if gt_data is None:
        return data, 0

    by_pairid = {str(x["pairid"]): x for x in gt_data if x.get("pairid") is not None}
    by_reference = {}
    for x in gt_data:
        ref = x.get("reference_id") or ""
        if ref:
            by_reference.setdefault(ref, []).append(x)

    matched = 0
    for item in data:
        gt = None
        if item.get("pairid") is not None:
            gt = by_pairid.get(str(item["pairid"]))

        # A reference image may occur in multiple queries.  Never attach an
        # arbitrary GT row based only on the reference ID.
        if gt is None:
            candidates = by_reference.get(item.get("reference_id", ""), [])
            if len(candidates) == 1:
                gt = candidates[0]

        if gt is None:
            continue

        item["target_id"] = gt.get("target_id") or None
        item["positives"] = list(gt.get("positives") or [])
        matched += 1

    return data, matched


# ============================================================
# Image handling
# ============================================================

# Built once from the actual image folder. This is important because CIRR/NLVR
# folders can contain images in nested directories.
_IMAGE_PATH_INDEX: Dict[str, str] = {}

def build_image_path_index(image_folder: str) -> Dict[str, str]:
    root = Path(image_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

    index: Dict[str, str] = {}

    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = p.stem
        # Keep first deterministic occurrence if duplicate stems exist.
        index.setdefault(stem, str(p))

    return index


def find_image_path(image_folder: str, image_id: str) -> Optional[str]:
    global _IMAGE_PATH_INDEX

    image_id = normalize_id(image_id)
    candidates = [image_id]

    if image_id.isdigit():
        candidates.append(image_id.zfill(12))

    # Fast path: use the recursively-built index.
    for candidate in dict.fromkeys(candidates):
        path = _IMAGE_PATH_INDEX.get(candidate)
        if path and Path(path).is_file():
            return path

    # Fallback for callers that use find_image_path before scan_gallery_ids().
    if not _IMAGE_PATH_INDEX:
        _IMAGE_PATH_INDEX = build_image_path_index(image_folder)
        for candidate in dict.fromkeys(candidates):
            path = _IMAGE_PATH_INDEX.get(candidate)
            if path and Path(path).is_file():
                return path

    return None


def load_image(image_folder: str, image_id: str) -> Optional[Image.Image]:
    """Load an image, including JPEGs that are truncated by a few bytes."""
    path = find_image_path(image_folder, image_id)
    if path is None:
        return None

    try:
        with Image.open(path) as img:
            # Force decoding while LOAD_TRUNCATED_IMAGES is enabled.
            img.load()
            return img.convert("RGB")
    except Exception as exc:
        print(f"[WARN] Could not open image {image_id} at {path}: {exc}")
        return None


def scan_gallery_ids(image_folder: str) -> List[str]:
    """*
    Build the actual retrieval gallery and a recursive image-path index.
    *"""
    global _IMAGE_PATH_INDEX

    _IMAGE_PATH_INDEX = build_image_path_index(image_folder)

    if not _IMAGE_PATH_INDEX:
        raise RuntimeError(f"No images found in {image_folder}")

    return sorted(_IMAGE_PATH_INDEX.keys())


# ============================================================
# Generated captions
# ============================================================

def load_generated_captions(json_path: str) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_keys = (
        "image_id", "img_id", "reference_img_id", "reference",
        "candidate_id", "reference_id", "id", "query_id", "pairid"
    )
    cap_keys = ("caption", "relative_caption", "generated_caption", "text")

    result: Dict[str, str] = {}

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            image_id = next(
                (normalize_id(item[k]) for k in id_keys
                 if k in item and item[k] is not None),
                None,
            )
            if not image_id:
                continue

            caption = next(
                (str(item[k]).strip() for k in cap_keys if item.get(k)),
                "",
            )
            result[image_id] = caption

        return result

    if isinstance(data, dict):
        for key, value in data.items():
            image_id = normalize_id(key)

            if isinstance(value, dict):
                caption = next(
                    (str(value[k]).strip() for k in cap_keys if value.get(k)),
                    "",
                )
            elif isinstance(value, str):
                caption = value.strip()
            else:
                caption = ""

            result[image_id] = caption

        return result

    raise ValueError(f"Unsupported generated captions format: {type(data)}")


# ============================================================
# Caption preparation / fallback
# ============================================================

def apply_caption_fallback(
    data: List[dict],
    generated_captions: Optional[Dict[str, str]],
    mode: str = "auto",
) -> Tuple[List[dict], Dict[str, int]]:
    """*
    Resolve the effective caption used by retrieval models.
    mode:
      **auto      : use annotation caption; if empty, use generated caption*
                  **for the same reference when available.*
      **generated : for empty annotation captions only, require/use generated*
                  **caption when available.*
      **skip      : keep empty captions; model will skip those queries.*
      **error     : fail immediately on an empty annotation caption.*

    The original annotation caption is preserved in ``original_caption``.
    The selected caption source is stored in ``caption_source``.
    IMPORTANT:
    We never replace a valid CIRR annotation caption with a generated caption.
    The fallback is used ONLY when the annotation caption is empty.
    *"""
    if mode not in {"auto", "generated", "skip", "error"}:
        raise ValueError(f"Unknown caption fallback mode: {mode}")

    generated_captions = generated_captions or {}
    stats = {
        "annotation": 0,
        "generated_fallback": 0,
        "empty_unresolved": 0,
    }

    for item in data:
        original = normalize_prompt(item.get("caption", ""))
        item["original_caption"] = original

        if original:
            item["caption"] = original
            item["caption_source"] = "annotation"
            stats["annotation"] += 1
            continue

        if mode == "error":
            raise RuntimeError(
                "Empty CIRR caption encountered: "
                f"pairid={item.get('pairid')!r}, "
                f"reference={item.get('reference_id')!r}"
            )

        generated = normalize_prompt(
            generated_captions.get(item.get("reference_id", ""), "")
        )

        if mode in {"auto", "generated"} and generated:
            item["caption"] = generated
            item["caption_source"] = "generated_fallback"
            stats["generated_fallback"] += 1
        else:
            item["caption"] = ""
            item["caption_source"] = "empty"
            stats["empty_unresolved"] += 1

    print("\n=== Caption resolution ===")
    print(f"Annotation captions       : {stats['annotation']}")
    print(f"Generated fallbacks       : {stats['generated_fallback']}")
    print(f"Unresolved empty captions : {stats['empty_unresolved']}")

    if stats["generated_fallback"]:
        for item in data:
            if item.get("caption_source") == "generated_fallback":
                print(
                    f"[WARN] Generated-caption fallback used: "
                    f"pairid={item.get('pairid')!r}, "
                    f"reference={item.get('reference_id')!r}"
                )

    if stats["empty_unresolved"]:
        for item in data:
            if item.get("caption_source") == "empty":
                print(
                    f"[WARN] Caption unresolved: "
                    f"pairid={item.get('pairid')!r}, "
                    f"reference={item.get('reference_id')!r}"
                )

    return data, stats


# ============================================================
# Metrics
# ============================================================

def average_precision_at_k(relevant: Iterable[str],
                           retrieved: List[str],
                           k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = 0
    score = 0.0

    for rank, image_id in enumerate(retrieved[:k], start=1):
        if image_id in relevant_set:
            hits += 1
            score += hits / rank

    return score / min(len(relevant_set), k)


def reciprocal_rank(relevant: Iterable[str], retrieved: List[str]) -> float:
    relevant_set = set(relevant)
    for rank, image_id in enumerate(retrieved, start=1):
        if image_id in relevant_set:
            return 1.0 / rank
    return 0.0


def init_results() -> dict:
    return {
        "prec": {k: [] for k in K_VALUES},
        "rec": {k: [] for k in K_VALUES},
        "map": {k: [] for k in MAP_K_VALUES},
        "mrr": [],
    }



def get_relevant_ids(item: dict) -> Set[str]:
    """*
    Return Ground Truth image IDs.
    For the new CIRR schema, target_id is the hard target. It is intentionally
    treated as the primary positive and is never inferred from group order.
    *"""
    target_id = normalize_id(item.get("target_id") or item.get("target_hard"))
    if target_id:
        return {target_id}

    relevant: Set[str] = set()
    soft = item.get("target_soft")
    if isinstance(soft, dict):
        for image_id, score in soft.items():
            image_id = normalize_id(image_id)
            if not image_id:
                continue
            try:
                if float(score) > 0:
                    relevant.add(image_id)
            except (TypeError, ValueError):
                relevant.add(image_id)
    elif isinstance(soft, list):
        relevant.update(
            normalize_id(x) for x in soft if normalize_id(x)
        )

    if not relevant:
        relevant.update(
            normalize_id(x) for x in item.get("positives") or [] if normalize_id(x)
        )
    return relevant



def update_metrics(
    results: dict,
    ranked_ids: List[str],
    positives: Iterable[str],
) -> None:
    positives = {normalize_id(x) for x in positives if normalize_id(x)}
    if not positives:
        return

    for k in K_VALUES:
        top_k = ranked_ids[:k]
        hits = sum(1 for x in top_k if x in positives)

        # Precision@K = relevant results in top-K / K
        results["prec"][k].append(hits / k)

        # Recall@K = relevant results in top-K / number of relevant images
        results["rec"][k].append(hits / len(positives))

    for k in MAP_K_VALUES:
        results["map"][k].append(
            average_precision_at_k(positives, ranked_ids, k)
        )

    results["mrr"].append(
        reciprocal_rank(positives, ranked_ids)
    )


def summarize_results(results: dict) -> dict:
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
# Ranking
# ============================================================

def build_id_index(gallery_ids: List[str]) -> Dict[str, int]:
    return {str(x): i for i, x in enumerate(gallery_ids)}


def rank_from_sims(
    sims: torch.Tensor,
    gallery_ids: List[str],
    id2idx: Dict[str, int],
    exclude_ids: Iterable[str] = (),
    restrict_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """Rank only the candidates that can actually participate.

    Full-gallery mode sorts the complete gallery.
    CIRR subset mode sorts only ``members`` instead of masking/sorting the
    entire gallery. This preserves exact ranking while reducing work.
    """
    scores = sims.detach().float().flatten()
    if scores.numel() != len(gallery_ids):
        raise ValueError(
            f"Similarity length ({scores.numel()}) != gallery size ({len(gallery_ids)})"
        )

    excluded = {
        normalize_id(x) for x in exclude_ids if normalize_id(x)
    }

    if restrict_ids is None:
        candidate_indices = [
            i for i, image_id in enumerate(gallery_ids)
            if image_id not in excluded
        ]
    else:
        candidate_indices = []
        seen = set()
        for image_id in restrict_ids:
            image_id = normalize_id(image_id)
            if not image_id or image_id in seen or image_id in excluded:
                continue
            idx = id2idx.get(image_id)
            if idx is not None:
                candidate_indices.append(idx)
                seen.add(image_id)

    if not candidate_indices:
        return []

    index_tensor = torch.tensor(candidate_indices, dtype=torch.long)
    candidate_scores = scores[index_tensor]
    order = torch.argsort(candidate_scores, descending=True).cpu().tolist()
    return [gallery_ids[candidate_indices[pos]] for pos in order]


# ============================================================
# CLIP embedding cache
# ============================================================

def _safe_cache_model_name(model_name: str) -> str:
    return (
        model_name
        .replace("/", "_")
        .replace("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def clip_embedding_cache_path(
    cache_dir: str,
    model_name: str,
    dataset: str,
    split: str,
) -> Path:
    """Return a split-aware cache path.*

    Example:
        embedding_cache/cirr/val/clip_ViT-B_32_image_embeddings.pt
        embedding_cache/cirr/test/clip_ViT-B_32_image_embeddings.pt
    *"""
    cache_root = Path(cache_dir) / dataset.lower() / split.lower()
    cache_root.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_cache_model_name(model_name)
    return cache_root / f"clip_{safe_name}_image_embeddings.pt"


def save_clip_image_cache(
    cache_path: Path,
    gallery_ids: List[str],
    normalized: Dict[str, torch.Tensor],
    raw: Dict[str, torch.Tensor],
    model_name: str,
    dataset: str,
    split: str,
) -> None:
    payload = {
        "version": 4,
        "dataset": dataset,
        "split": split,
        "model_name": model_name,
        "gallery_ids": list(gallery_ids),
        "normalized": normalized,
        "raw": raw,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def load_clip_image_cache(
    cache_path: Path,
    gallery_ids: List[str],
    model_name: str,
    dataset: str,
    split: str,
) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    if not cache_path.is_file():
        return None

    try:
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(payload, dict):
            return None

        if payload.get("version") != 4:
            return None

        if payload.get("dataset") != dataset:
            return None

        if payload.get("split") != split:
            return None

        if payload.get("model_name") != model_name:
            return None

        cached_ids = list(payload.get("gallery_ids", []))
        if cached_ids != list(gallery_ids):
            return None

        normalized = payload.get("normalized")
        raw = payload.get("raw")
        if not isinstance(normalized, dict) or not isinstance(raw, dict):
            return None

        if set(normalized.keys()) != set(gallery_ids):
            return None

        if set(raw.keys()) != set(gallery_ids):
            return None

        if gallery_ids:
            probe_n = normalized[gallery_ids[0]]
            probe_r = raw[gallery_ids[0]]
            if not torch.is_tensor(probe_n) or not torch.is_tensor(probe_r):
                return None
            if probe_n.numel() == 0 or probe_r.numel() == 0:
                return None

        return normalized, raw

    except Exception as exc:
        print(f"[WARN] Could not load embedding cache {cache_path}: {exc}")
        return None


# ============================================================
# CLIP
# ============================================================

def load_clip_model(model_name: str = "ViT-B/32"):
    if clip is None:
        raise ImportError(
            "OpenAI CLIP is not installed. Install the CLIP package/repository "
            "before using --models clip, searle or clip_beta."
        )
    model, preprocess = clip.load(model_name, device=str(DEVICE))
    model.eval()
    return model, preprocess


@torch.inference_mode()
def clip_image_embedding(image: Image.Image, model, preprocess) -> torch.Tensor:
    x = preprocess(image).unsqueeze(0).to(DEVICE)
    feat = model.encode_image(x)
    return F.normalize(feat.float(), dim=-1).cpu()


@torch.inference_mode()
def clip_text_embedding(text: str, model) -> torch.Tensor:
    tokens = clip.tokenize([text], truncate=True).to(DEVICE)
    feat = model.encode_text(tokens)
    return F.normalize(feat.float(), dim=-1).cpu()


def build_clip_image_cache(
    image_ids: Iterable[str],
    image_folder: str,
    model,
    preprocess,
    partial_cache_path: Optional[Path] = None,
    model_name: str = "ViT-B/32",
    checkpoint_every: int = 250,
    cache_dataset: str = "cirr",
    cache_split: str = "val",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """*
    Build CLIP image embeddings with resumable checkpoints.
    A partial cache is written periodically, so an interruption does not
    force another full pass over all 8082 images.
    *"""
    normalized: Dict[str, torch.Tensor] = {}
    raw: Dict[str, torch.Tensor] = {}
    image_ids = list(image_ids)

    # Resume an interrupted build when possible.
    if partial_cache_path is not None and partial_cache_path.is_file():
        try:
            payload = torch.load(
                partial_cache_path,
                map_location="cpu",
                weights_only=False,
            )
            if (
                payload.get("version") == 4
                and payload.get("dataset") == cache_dataset
                and payload.get("split") == cache_split
                and payload.get("model_name") == model_name
                and list(payload.get("gallery_ids", [])) == image_ids
            ):
                n = payload.get("normalized")
                r = payload.get("raw")
                if isinstance(n, dict) and isinstance(r, dict):
                    normalized.update(n)
                    raw.update(r)
                    print(
                        f"Resuming partial CLIP cache: "
                        f"{len(normalized)}/{len(image_ids)} images"
                    )
        except Exception as exc:
            print(f"[WARN] Could not resume partial embedding cache: {exc}")

    def save_partial() -> None:
        if partial_cache_path is None:
            return
        partial_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 4,
            "dataset": cache_dataset,
            "split": cache_split,
            "model_name": model_name,
            "gallery_ids": image_ids,
            "normalized": normalized,
            "raw": raw,
        }
        tmp = partial_cache_path.with_suffix(partial_cache_path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, partial_cache_path)

    remaining = [x for x in image_ids if x not in normalized or x not in raw]

    for processed, image_id in enumerate(
        tqdm(remaining, desc="CLIP image embeddings"),
        start=1,
    ):
        image = load_image(image_folder, image_id)
        if image is None:
            print(
                f"[WARN] Skipping unreadable image during embedding: "
                f"{image_id} -> {find_image_path(image_folder, image_id)!r}"
            )
            continue

        try:
            x = preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                feat = model.encode_image(x).float()

            raw[image_id] = feat.cpu()
            normalized[image_id] = F.normalize(feat, dim=-1).cpu()

        except Exception as exc:
            print(f"[WARN] CLIP image {image_id}: {type(exc).__name__}: {exc}")

        if processed % checkpoint_every == 0:
            save_partial()
            print(
                f"[CACHE] Partial CLIP cache saved: "
                f"{len(normalized)}/{len(image_ids)}"
            )

    # Always save the latest state, including a failed/incomplete build.
    save_partial()

    return normalized, raw


# ============================================================
# Persistent image-embedding caches
# ============================================================

def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def model_image_cache_path(
    cache_dir: str,
    dataset: str,
    split: str,
    model_key: str,
) -> Path:
    root = Path(cache_dir) / dataset.lower() / split.lower()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_safe_cache_model_name(model_key)}_image_embeddings.pt"


def load_generic_image_cache(
    cache_path: Path,
    gallery_ids: List[str],
    dataset: str,
    split: str,
    model_key: str,
) -> Optional[Dict[str, torch.Tensor]]:
    if not cache_path.is_file():
        return None

    try:
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            return None

        if payload.get("cache_version") != 1:
            return None
        if payload.get("dataset") != dataset:
            return None
        if payload.get("split") != split:
            return None
        if payload.get("model_key") != model_key:
            return None
        if list(payload.get("gallery_ids", [])) != list(gallery_ids):
            return None

        features = payload.get("features")
        if not isinstance(features, dict):
            return None
        if set(features.keys()) != set(gallery_ids):
            return None

        return features

    except Exception as exc:
        print(f"[WARN] Could not load cache {cache_path}: {type(exc).__name__}: {exc}")
        return None


def build_generic_image_cache(
    image_ids: List[str],
    image_folder: str,
    encode_batch,
    cache_path: Path,
    model_key: str,
    dataset: str,
    split: str,
    batch_size: int = 32,
    force_rebuild: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build a persistent full-gallery image cache with resumable checkpoints."""
    image_ids = list(image_ids)
    features: Dict[str, torch.Tensor] = {}
    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")

    if force_rebuild:
        for path in (cache_path, partial_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                print(f"[WARN] Could not remove cache {path}: {exc}")

    # Resume partial cache.
    if partial_path.is_file() and not force_rebuild:
        try:
            payload = torch.load(
                partial_path,
                map_location="cpu",
                weights_only=False,
            )
            if (
                isinstance(payload, dict)
                and payload.get("cache_version") == 1
                and payload.get("dataset") == dataset
                and payload.get("split") == split
                and payload.get("model_key") == model_key
                and list(payload.get("gallery_ids", [])) == image_ids
                and isinstance(payload.get("features"), dict)
            ):
                features.update(payload["features"])
                print(
                    f"Resuming {model_key} image cache: "
                    f"{len(features)}/{len(image_ids)} images"
                )
        except Exception as exc:
            print(f"[WARN] Could not resume partial cache: {type(exc).__name__}: {exc}")

    def save_partial() -> None:
        _atomic_torch_save(
            {
                "cache_version": 1,
                "dataset": dataset,
                "split": split,
                "model_key": model_key,
                "gallery_ids": image_ids,
                "features": features,
                "num_cached": len(features),
            },
            partial_path,
        )

    remaining = [image_id for image_id in image_ids if image_id not in features]

    for start in tqdm(
        range(0, len(remaining), batch_size),
        desc=f"{model_key} image embeddings",
    ):
        batch_ids = remaining[start:start + batch_size]
        images = []
        valid_ids = []

        for image_id in batch_ids:
            image = load_image(image_folder, image_id)
            if image is None:
                raise RuntimeError(
                    f"Cannot build FULL gallery cache: image could not be opened: {image_id}"
                )
            images.append(image)
            valid_ids.append(image_id)

        try:
            batch_features = encode_batch(images)
            if not torch.is_tensor(batch_features):
                raise TypeError(
                    f"Encoder returned {type(batch_features).__name__}, expected torch.Tensor."
                )
            if batch_features.ndim != 2 or batch_features.shape[0] != len(valid_ids):
                raise ValueError(
                    f"Invalid embedding shape {tuple(batch_features.shape)} for "
                    f"batch size {len(valid_ids)}"
                )

            batch_features = F.normalize(batch_features.float(), dim=-1).cpu()

            for image_id, feat in zip(valid_ids, batch_features):
                features[image_id] = feat.unsqueeze(0)

        except Exception as exc:
            save_partial()
            raise RuntimeError(
                f"{model_key} batch failed for images "
                f"{valid_ids[:3]}{'...' if len(valid_ids) > 3 else ''}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        save_partial()
        print(
            f"[CACHE] {model_key}: {len(features)}/{len(image_ids)} images cached"
        )

    if set(features.keys()) != set(image_ids):
        missing = sorted(set(image_ids) - set(features))
        save_partial()
        raise RuntimeError(
            f"{model_key} cache is incomplete: "
            f"{len(missing)} gallery images are missing. First missing: {missing[:20]}"
        )

    payload = {
        "cache_version": 1,
        "dataset": dataset,
        "split": split,
        "model_key": model_key,
        "gallery_ids": image_ids,
        "features": features,
        "num_cached": len(features),
    }
    _atomic_torch_save(payload, cache_path)

    try:
        if partial_path.exists():
            partial_path.unlink()
    except OSError:
        pass

    print(f"Saved FULL {model_key} image cache: {cache_path}")
    return features


# ============================================================
# OpenCLIP
# ============================================================


# ============================================================
# BLIP-ITM (Salesforce/blip-itm-base-coco)
# ============================================================

BLIP_TEXT_MAX_LEN = 40


def load_blip_itm_model(model_path: str):
    """Load BLIP-ITM locally/offline for image-text retrieval."""
    if not model_path:
        raise ValueError("--blip_path is required when using --models blip")

    from transformers import AutoProcessor, BlipForImageTextRetrieval

    local_path = Path(model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"BLIP model directory does not exist: {local_path}")
    if not (local_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in BLIP directory: {local_path}")

    # Keep Transformers completely local for reproducible/offline evaluation.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        str(local_path),
        local_files_only=True,
    )

    try:
        model = BlipForImageTextRetrieval.from_pretrained(
            str(local_path),
            dtype=dtype,
            local_files_only=True,
        )
    except TypeError:
        model = BlipForImageTextRetrieval.from_pretrained(
            str(local_path),
            torch_dtype=dtype,
            local_files_only=True,
        )

    model = model.to(DEVICE).eval()
    model.requires_grad_(False)

    print(f"BLIP checkpoint : {local_path}")
    print(f"BLIP model      : {model.__class__.__name__}")
    print(f"BLIP device     : {DEVICE}")
    print(f"BLIP dtype      : {model.dtype}")
    return model, processor


def _move_blip_inputs(inputs: dict) -> dict:
    return {
        key: value.to(DEVICE) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


@torch.inference_mode()
def blip_image_features(
        images: List[Image.Image],
        model,
        processor,
) -> torch.Tensor:
    """Return normalized BLIP image-text retrieval features."""
    if not images:
        return torch.empty((0, 0), dtype=torch.float32)

    inputs = processor(images=images, return_tensors="pt")
    inputs = _move_blip_inputs(inputs)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    outputs = model.vision_model(pixel_values=inputs["pixel_values"])

    # Handle both dict-like and tuple outputs
    if isinstance(outputs, tuple):
        hidden = outputs[0]
    else:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]

    # BLIP-ITM's image retrieval branch uses the CLS token followed by vision_proj.
    pooled = hidden[:, 0, :]
    projected = model.vision_proj(pooled)
    return F.normalize(projected.float(), dim=-1).cpu()


@torch.inference_mode()
def blip_text_features(
        texts: List[str],
        model,
        processor,
) -> torch.Tensor:
    """Return normalized BLIP text retrieval features."""
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)

    clean = [normalize_prompt(x) for x in texts]
    inputs = processor(
        text=clean,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=BLIP_TEXT_MAX_LEN,
    )
    inputs = _move_blip_inputs(inputs)

    outputs = model.text_encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        return_dict=True,
    )

    # Handle both dict-like and tuple outputs
    if isinstance(outputs, tuple):
        hidden = outputs[0]
    else:
        hidden = outputs.last_hidden_state

    pooled = hidden[:, 0, :]
    projected = model.text_proj(pooled)
    return F.normalize(projected.float(), dim=-1).cpu()


#------------------------------------------------------------------------------
@torch.inference_mode()
def blip_itm_score_batch(
    caption: str,
    images: List[Image.Image],
    model,
    processor,
) -> torch.Tensor:
    """Return positive-class BLIP ITM probabilities for image/text pairs."""
    if not images:
        return torch.empty(0, dtype=torch.float32)

    inputs = processor(
        images=images,
        text=[caption] * len(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=BLIP_TEXT_MAX_LEN,
    )
    inputs = _move_blip_inputs(inputs)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    outputs = model(inputs, use_itm_head=True)

    logits = getattr(outputs, "itm_score", None)
    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if not torch.is_tensor(logits):
        raise RuntimeError(
            f"Unsupported BLIP ITM output type: {type(logits).__name__}"
        )

    if logits.ndim == 2 and logits.shape[-1] == 2:
        return logits.float().softmax(dim=-1)[:, 1].cpu()
    if logits.ndim == 1:
        return logits.float().sigmoid().cpu()
    raise RuntimeError(f"Unexpected BLIP ITM score shape: {tuple(logits.shape)}")


def build_blip_image_cache(
    image_ids: Iterable[str],
    image_folder: str,
    model,
    processor,
    cache_dir: str,
    dataset: str,
    split: str,
    model_key: str,
    batch_size: int,
    force_rebuild: bool = False,
) -> Dict[str, torch.Tensor]:
    image_ids = list(image_ids)
    cache_path = model_image_cache_path(
        cache_dir,
        dataset,
        split,
        model_key,
    )

    if not force_rebuild:
        cached = load_generic_image_cache(
            cache_path,
            image_ids,
            dataset,
            split,
            model_key,
        )
        if cached is not None:
            print(f"Loaded BLIP image embeddings from cache: {cache_path}")
            print(f"Cached images: {len(cached)}")
            return cached

    def encode_batch(images):
        return blip_image_features(images, model, processor)

    return build_generic_image_cache(
        image_ids=image_ids,
        image_folder=image_folder,
        encode_batch=encode_batch,
        cache_path=cache_path,
        model_key=model_key,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )


def blip_itm_rerank(
    caption: str,
    candidate_ids: List[str],
    image_folder: str,
    model,
    processor,
    top_k: int,
    batch_size: int,
) -> List[str]:
    """Rerank only the first top_k candidates with BLIP ITM."""
    if top_k <= 0 or not candidate_ids or not caption:
        return candidate_ids

    candidates = list(candidate_ids[:top_k])
    scored: List[Tuple[str, float]] = []

    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        images: List[Image.Image] = []
        valid_ids: List[str] = []

        for image_id in chunk:
            image = load_image(image_folder, image_id)
            if image is None:
                continue
            images.append(image)
            valid_ids.append(image_id)

        if not images:
            continue

        scores = blip_itm_score_batch(
            caption,
            images,
            model,
            processor,
        )
        scored.extend(
            (image_id, float(score))
            for image_id, score in zip(valid_ids, scores.tolist())
        )

    score_map = dict(scored)
    candidate_set = set(candidates)
    reranked_top = sorted(
        candidates,
        key=lambda image_id: score_map.get(image_id, -float("inf")),
        reverse=True,
    )
    suffix = [image_id for image_id in candidate_ids if image_id not in candidate_set]
    return reranked_top + suffix


def evaluate_blip(
    data: List[dict],
    image_folder: str,
    reference_image_folder: str,
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    model,
    processor,
    protocols: Tuple[str, ...],
    text_batch_size: int,
    itm_topk: int,
    itm_batch_size: int,
):
    """Evaluate BLIP once and report independent protocol results."""
    id2idx = build_id_index(gallery_ids)

    if not torch.is_tensor(gallery_feats) or gallery_feats.ndim != 2:
        raise TypeError("gallery_feats must be a 2-D torch.Tensor")
    if gallery_feats.shape[0] != len(gallery_ids):
        raise ValueError(
            f"Invalid BLIP gallery feature shape {tuple(gallery_feats.shape)}; "
            f"expected ({len(gallery_ids)}, feature_dim)."
        )

    unique_captions = sorted({
        normalize_prompt(item.get("caption", ""))
        for item in data
        if normalize_prompt(item.get("caption", ""))
    })
    text_cache: Dict[str, torch.Tensor] = {}
    print(f"BLIP unique captions: {len(unique_captions)}")

    for start in tqdm(
        range(0, len(unique_captions), max(1, text_batch_size)),
        desc="BLIP text embeddings",
    ):
        batch = unique_captions[start:start + max(1, text_batch_size)]
        features = blip_text_features(batch, model, processor)
        if features.ndim != 2 or features.shape[0] != len(batch):
            raise ValueError(
                f"Invalid BLIP text feature shape {tuple(features.shape)} for {len(batch)} texts."
            )
        for i, caption in enumerate(batch):
            text_cache[caption] = features[i:i + 1].float().cpu()

    unique_refs = sorted({
        item.get("reference_id", "") for item in data if item.get("reference_id")
    })
    ref_cache: Dict[str, Optional[torch.Tensor]] = {}
    print(f"BLIP unique references: {len(unique_refs)}")

    for ref_id in tqdm(unique_refs, desc="BLIP reference embeddings"):
        image = load_image(reference_image_folder, ref_id)
        if image is None:
            ref_cache[ref_id] = None
            continue
        try:
            feat = blip_image_features([image], model, processor)
            ref_cache[ref_id] = feat.float().cpu()
        except Exception as exc:
            print(f"[WARN] BLIP reference embedding failed: {ref_id}: {type(exc).__name__}: {exc}")
            ref_cache[ref_id] = None

    outputs = {
        protocol: {
            "results": init_results(),
            "rankings": {},
            "total": 0,
            "skipped": 0,
            "skip_reasons": {},
        }
        for protocol in protocols
    }

    for item in tqdm(data, desc="BLIP retrieval"):
        ref_id = item["reference_id"]
        caption = normalize_prompt(item.get("caption", ""))
        ref_feat = ref_cache.get(ref_id)
        text_feat = text_cache.get(caption)

        if ref_feat is None or text_feat is None:
            reason = (
                "missing_reference_embedding" if ref_feat is None
                else "empty_or_missing_text_embedding"
            )
            for state in outputs.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            continue

        try:
            query = F.normalize(
                F.normalize(ref_feat.float(), dim=-1)
                + F.normalize(text_feat.float(), dim=-1),
                dim=-1,
            ).squeeze(0)
            sims = gallery_feats.float() @ query

            protocol_rankings = {}
            for protocol in protocols:
                ranked = rank_from_sims(
                    sims,
                    gallery_ids,
                    id2idx,
                    exclude_ids=[ref_id],
                    restrict_ids=protocol_restrict_ids(item, protocol),
                )
                if itm_topk > 0:
                    ranked = blip_itm_rerank(
                        caption=caption,
                        candidate_ids=ranked,
                        image_folder=image_folder,
                        model=model,
                        processor=processor,
                        top_k=min(itm_topk, len(ranked)),
                        batch_size=itm_batch_size,
                    )
                protocol_rankings[protocol] = ranked

            positives = get_relevant_ids(item)
            query_key = get_query_key(item)
            for protocol in protocols:
                state = outputs[protocol]
                ranked = protocol_rankings[protocol]
                state["rankings"][query_key] = ranked[:50]
                if positives:
                    update_metrics(state["results"], ranked, positives)
                state["total"] += 1

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for state in outputs.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            print(f"[WARN] BLIP query failed: {get_query_key(item)!r}: {reason}")

    return {
        protocol: (
            state["results"],
            state["total"],
            state["skipped"],
            state["rankings"],
            state["skip_reasons"],
        )
        for protocol, state in outputs.items()
    }

def load_searle_model(
    clip_model_name: str = "ViT-B/32",
    searle_path: str = r"C:\Users\user\Desktop\Python\ImageRetrieval\models_download\SEARLE",
):
    """Load SEARLE from a local repository without GitHub/torch.hub cache."""
    import importlib
    import sys

    requested_root = Path(searle_path).expanduser().resolve()

    if not requested_root.exists():
        raise FileNotFoundError(
            f"Local SEARLE directory does not exist:\n{requested_root}"
        )
    if not requested_root.is_dir():
        raise NotADirectoryError(
            f"Local SEARLE path is not a directory:\n{requested_root}"
        )

    def find_repo_root(root: Path) -> Optional[Path]:
        candidates = [root, root / "SEARLE", root / "main", root / "miccunifi_SEARLE_main"]
        try:
            candidates.extend(x for x in root.iterdir() if x.is_dir())
        except OSError:
            pass

        seen = set()
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
            except OSError:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "hubconf.py").is_file() and (candidate / "src" / "encode_with_pseudo_tokens.py").is_file():
                return candidate

        try:
            for encoder in root.rglob("encode_with_pseudo_tokens.py"):
                candidate = encoder.parent.parent
                if (candidate / "hubconf.py").is_file():
                    return candidate.resolve()
        except OSError:
            pass
        return None

    repo_dir = find_repo_root(requested_root)
    if repo_dir is None:
        discovered = []
        try:
            for path in requested_root.rglob("*"):
                if path.is_file():
                    discovered.append(str(path.relative_to(requested_root)))
        except OSError:
            pass
        message = (
            "Invalid local SEARLE repository.\n"
            f"Requested path: {requested_root}\n\n"
            "Required files:\n"
            "  hubconf.py\n"
            "  src/encode_with_pseudo_tokens.py"
        )
        if discovered:
            message += "\n\nFiles found under the requested path:\n  " + "\n  ".join(discovered[:100])
        raise RuntimeError(message)

    hubconf_file = repo_dir / "hubconf.py"
    src_dir = repo_dir / "src"
    encoder_file = src_dir / "encode_with_pseudo_tokens.py"

    print(f"SEARLE local path: {requested_root}")
    print(f"SEARLE repository root: {repo_dir}")
    print(f"SEARLE hubconf: {hubconf_file}")
    print(f"SEARLE encoder: {encoder_file}")

    repo_str = str(repo_dir)
    src_str = str(src_dir.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    importlib.invalidate_caches()

    # IMPORTANT: do not rely on normal `import src...` resolution.
    # On Windows, another installed package named `src` can shadow this
    # repository, and some SEARLE snapshots do not contain src/__init__.py.
    # We explicitly install a package object for this repository's src folder
    # and then load the encoder module from its exact file path.
    try:
        sys.modules.pop("src.encode_with_pseudo_tokens", None)
        sys.modules.pop("src", None)

        src_init = src_dir / "__init__.py"

        if src_init.is_file():
            src_spec = importlib.util.spec_from_file_location(
                "src",
                str(src_init),
                submodule_search_locations=[src_str],
            )
            if src_spec is None or src_spec.loader is None:
                raise ImportError(f"Could not create import spec for {src_init}")

            src_module = importlib.util.module_from_spec(src_spec)
            sys.modules["src"] = src_module
            src_spec.loader.exec_module(src_module)
        else:
            # Namespace-package fallback when src/__init__.py does not exist.
            src_spec = importlib.machinery.ModuleSpec(
                name="src",
                loader=None,
                is_package=True,
            )
            src_module = importlib.util.module_from_spec(src_spec)
            src_module.__path__ = [src_str]
            src_module.__package__ = "src"
            sys.modules["src"] = src_module

        encoder_spec = importlib.util.spec_from_file_location(
            "src.encode_with_pseudo_tokens",
            str(encoder_file),
        )
        if encoder_spec is None or encoder_spec.loader is None:
            raise ImportError(
                f"Could not create import spec for {encoder_file}"
            )

        encode_module = importlib.util.module_from_spec(encoder_spec)
        encode_module.__package__ = "src"
        sys.modules["src.encode_with_pseudo_tokens"] = encode_module
        encoder_spec.loader.exec_module(encode_module)

        encode_with_pseudo_tokens = getattr(
            encode_module,
            "encode_with_pseudo_tokens",
            None,
        )
        if encode_with_pseudo_tokens is None:
            raise AttributeError(
                "encode_with_pseudo_tokens function was not found in "
                f"{encoder_file}"
            )

        print("SEARLE pseudo-token encoder import: OK")

    except Exception as exc:
        raise RuntimeError(
            "The local SEARLE encoder file exists but could not be loaded directly.\n\n"
            f"Repository: {repo_dir}\n"
            f"Expected file: {encoder_file}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        print("Loading SEARLE from local hubconf.py ...")
        searle, hub_encoder = torch.hub.load(
            repo_or_dir=str(repo_dir),
            source="local",
            model="searle",
            backbone=clip_model_name,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SEARLE from the local repository.\n\n"
            f"Repository: {repo_dir}\n"
            f"hubconf.py: {hubconf_file}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if hub_encoder is not None:
        encode_with_pseudo_tokens = hub_encoder

    if searle is None:
        raise RuntimeError("SEARLE model returned None from hubconf.py")

    searle = searle.to(DEVICE).eval()
    print("SEARLE loaded successfully")
    print(f"SEARLE device: {DEVICE}")
    print(f"SEARLE backbone: {clip_model_name}")
    return searle, encode_with_pseudo_tokens


@torch.inference_mode()
def build_searle_query(
    reference_raw_feature: torch.Tensor,
    caption: str,
    searle,
    encode_with_pseudo_tokens,
    clip_model,
    prompt_templates: Optional[List[str]] = None,
) -> torch.Tensor:
    """*
    Build the composed SEARLE query.
    IMPORTANT: `$` is NOT a normal character here. It is the placeholder
    replaced by SEARLE pseudo tokens. The official repository explicitly
    requires the prompt to contain `$`.
    *"""
    caption = normalize_prompt(caption)
    if not caption:
        raise ValueError("SEARLE requires a non-empty relative caption.")

    raw = reference_raw_feature.to(DEVICE).float()
    pseudo_tokens = searle(raw)

    if not prompt_templates:
        prompt_templates = ["a photo of $ {caption}"]

    embeddings = []
    for template in prompt_templates:
        prompt = template.format(caption=caption)
        if "$" not in prompt:
            raise ValueError(
                f"Invalid SEARLE prompt: {prompt!r}. It must contain '$'."
            )
        tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
        feat = encode_with_pseudo_tokens(
            clip_model,
            tokens,
            pseudo_tokens,
        )
        embeddings.append(F.normalize(feat.float(), dim=-1))

    # Prompt ensembling is done in embedding space, then normalized again.
    query = torch.stack(embeddings, dim=0).mean(dim=0)
    return F.normalize(query, dim=-1).cpu()


def evaluate_searle(
    data: List[dict],
    searle,
    encode_with_pseudo_tokens,
    clip_model,
    clip_preprocess,
    image_features_cache: Dict[str, torch.Tensor],
    gallery_ids: List[str],
    reference_image_folder: str,
    gallery_feats: torch.Tensor,
    protocols: Tuple[str, ...] = (FULL_GALLERY, CIRR_SUBSET),
    prompt_templates: Optional[List[str]] = None,
    hybrid_weights: Optional[List[float]] = None,
    clip_text_cache: Optional[Dict[str, torch.Tensor]] = None,
):
    """Evaluate SEARLE once per query/weight and emit both protocols."""
    id2idx = build_id_index(gallery_ids)
    weights = hybrid_weights or [1.0]
    outputs = {}
    reference_feature_cache: Dict[str, Optional[torch.Tensor]] = {}

    for weight in weights:
        protocol_state = {
            protocol: {
                "results": init_results(),
                "predictions": [],
                "total": 0,
                "skipped": 0,
                "skip_reasons": {},
            }
            for protocol in protocols
        }

        for item in tqdm(data, desc=f"SEARLE w={weight:.2f}"):
            ref_id = item["reference_id"]
            caption = normalize_prompt(item["caption"])
            raw_ref = image_features_cache.get(ref_id)

            if raw_ref is None and ref_id in reference_feature_cache:
                raw_ref = reference_feature_cache[ref_id]

            if raw_ref is None:
                ref_image = load_image(reference_image_folder, ref_id)
                if ref_image is not None:
                    try:
                        raw_ref = clip_model.encode_image(
                            clip_preprocess(ref_image).unsqueeze(0).to(DEVICE)
                        ).float().cpu()
                        reference_feature_cache[ref_id] = raw_ref
                    except Exception:
                        raw_ref = None
                else:
                    raw_ref = None
                if ref_id not in reference_feature_cache:
                    reference_feature_cache[ref_id] = raw_ref

            if raw_ref is None or not caption:
                reason = "missing_reference_embedding" if raw_ref is None else "empty_caption"
                for state in protocol_state.values():
                    state["skipped"] += 1
                    state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
                continue

            try:
                q_searle = build_searle_query(
                    raw_ref,
                    caption,
                    searle,
                    encode_with_pseudo_tokens,
                    clip_model,
                    prompt_templates=prompt_templates,
                )
                sims = gallery_feats @ q_searle.squeeze(0)

                if weight < 0.999999:
                    if clip_text_cache is None:
                        raise RuntimeError("clip_text_cache is required for hybrid SEARLE.")
                    t = clip_text_cache.get(caption)
                    if t is None:
                        raise RuntimeError(f"Missing CLIP text feature for: {caption}")
                    sims_clip = gallery_feats @ t.squeeze(0)
                    sims = weight * sims + (1.0 - weight) * sims_clip

                query_key = get_query_key(item)
                positives = get_relevant_ids(item)
                for protocol in protocols:
                    ranked = rank_from_sims(
                        sims,
                        gallery_ids,
                        id2idx,
                        exclude_ids=[ref_id],
                        restrict_ids=protocol_restrict_ids(item, protocol),
                    )
                    state = protocol_state[protocol]
                    if positives:
                        update_metrics(state["results"], ranked, positives)
                    state["predictions"].append({
                        "query_id": query_key,
                        "candidate_id": item.get("candidate_id") or ref_id,
                        "pairid": item.get("pairid"),
                        "reference": ref_id,
                        "target_id": item.get("target_id"),
                        "caption": caption,
                        "img_set_id": item.get("img_set_id"),
                        "reference_rank": item.get("reference_rank"),
                        "members": item.get("members", []),
                        "ranking": ranked[:50],
                    })
                    state["total"] += 1
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for state in protocol_state.values():
                    state["skipped"] += 1
                    state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
                print(f"[WARN] SEARLE failed for {ref_id}: {reason}")

        outputs[weight] = {
            protocol: (
                state["results"],
                state["predictions"],
                state["total"],
                state["skipped"],
                state["skip_reasons"],
            )
            for protocol, state in protocol_state.items()
        }

    return outputs

def load_open_clip_model(
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2b_s32b_b79k",
):
    if open_clip is None:
        raise ImportError(
            "open_clip_torch is not installed. Install it before using --models open_clip."
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=str(DEVICE),
    )
    model.eval()
    return model, preprocess


@torch.inference_mode()
def open_clip_image_embedding(image, model, preprocess):
    x = preprocess(image).unsqueeze(0).to(DEVICE)
    feat = model.encode_image(x, normalize=True)
    return F.normalize(feat.float(), dim=-1).cpu()


@torch.inference_mode()
def open_clip_text_embedding(text, model, tokenizer):
    tokens = tokenizer([text]).to(DEVICE)
    feat = model.encode_text(tokens, normalize=True)
    return F.normalize(feat.float(), dim=-1).cpu()


def build_open_clip_image_cache(
    image_ids,
    image_folder,
    model,
    preprocess,
    cache_dir: str,
    dataset: str,
    split: str,
    model_key: str,
    batch_size: int = 32,
    force_rebuild: bool = False,
):
    cache_path = model_image_cache_path(
        cache_dir,
        dataset,
        split,
        model_key,
    )

    if not force_rebuild:
        cached = load_generic_image_cache(
            cache_path,
            list(image_ids),
            dataset,
            split,
            model_key,
        )
        if cached is not None:
            print(f"Loaded OpenCLIP image embeddings from cache: {cache_path}")
            print(f"Cached images: {len(cached)}")
            return cached

    def encode_batch(images):
        xs = torch.stack([preprocess(img) for img in images], dim=0).to(DEVICE)
        with torch.inference_mode():
            feats = model.encode_image(xs, normalize=True)
        return feats

    return build_generic_image_cache(
        list(image_ids),
        image_folder,
        encode_batch,
        cache_path,
        model_key,
        dataset,
        split,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )


# ============================================================
# SigLIP / SigLIP2
# ============================================================

def load_siglip_model(model_path: str):
    if not model_path:
        raise ValueError("--siglip_path is required for siglip")

    from transformers import AutoModel, AutoProcessor

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

    try:
        model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
        ).to(DEVICE)
    except TypeError:
        # Backward compatibility with older transformers.
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    print(f"SigLIP model: {model.__class__.__name__}")
    print(f"SigLIP device: {DEVICE}")
    return model, processor


@torch.inference_mode()
def _siglip_forward_image_features(inputs, model) -> torch.Tensor:
    """Return normalized-compatible SigLIP image embeddings.*

    The user's checkpoint is loaded as ``SiglipModel``. In the
    Hugging Face implementation, ``SiglipVisionModel.head`` performs
    the multi-head attention pooling and its result is returned as
    ``pooler_output``. Do NOT feed ``pooler_output`` back through
    ``vision_model.head`` or another projection.
    *"""

    # Preferred path: full model forward exposes image_embeds.
    try:
        outputs = model(inputs)
        image_embeds = getattr(outputs, "image_embeds", None)
        if torch.is_tensor(image_embeds):
            return image_embeds.float()
    except Exception:
        pass

    # SiglipModel.get_image_features() returns
    # BaseModelOutputWithPooling in current Transformers.
    output = model.get_image_features(inputs)

    if torch.is_tensor(output):
        return output.float()

    image_embeds = getattr(output, "image_embeds", None)
    if torch.is_tensor(image_embeds):
        return image_embeds.float()

    pooler = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooler):
        # IMPORTANT: for SiglipModel this is already the final pooled
        # image representation produced by SiglipMultiheadAttentionPoolingHead.
        return pooler.float()

    raise TypeError(
        "Unsupported SigLIP image output: "
        f"{type(output).__name__}. No tensor/image_embeds/pooler_output found."
    )


@torch.inference_mode()
def _siglip_forward_text_features(inputs, model) -> torch.Tensor:
    """Return SigLIP text embeddings.*

    For ``SiglipModel``, ``text_model.head`` already produces the final
    projected text representation stored in ``pooler_output``.
    *"""

    try:
        outputs = model(inputs)
        text_embeds = getattr(outputs, "text_embeds", None)
        if torch.is_tensor(text_embeds):
            return text_embeds.float()
    except Exception:
        pass

    output = model.get_text_features(inputs)

    if torch.is_tensor(output):
        return output.float()

    text_embeds = getattr(output, "text_embeds", None)
    if torch.is_tensor(text_embeds):
        return text_embeds.float()

    pooler = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooler):
        return pooler.float()

    raise TypeError(
        "Unsupported SigLIP text output: "
        f"{type(output).__name__}. No tensor/text_embeds/pooler_output found."
    )


@torch.inference_mode()
def siglip_image_embedding(image, model, processor) -> torch.Tensor:
    """Encode one image with SigLIP/SigLIP2 and return L2-normalized features."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    feat = _siglip_forward_image_features(inputs, model)
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
    return F.normalize(feat.float(), dim=-1).cpu()


@torch.inference_mode()
def siglip_text_embedding(text: str, model, processor) -> torch.Tensor:
    """Encode one text string with SigLIP/SigLIP2."""
    text = normalize_prompt(text)
    if not text:
        raise ValueError("SigLIP text input cannot be empty.")

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    feat = _siglip_forward_text_features(inputs, model)
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
    return F.normalize(feat.float(), dim=-1).cpu()


def build_siglip_image_cache(
    image_ids,
    image_folder,
    model,
    processor,
    cache_dir: str,
    dataset: str,
    split: str,
    model_key: str,
    batch_size: int = 16,
    force_rebuild: bool = False,
):
    cache_path = model_image_cache_path(
        cache_dir,
        dataset,
        split,
        model_key,
    )

    if not force_rebuild:
        cached = load_generic_image_cache(
            cache_path,
            list(image_ids),
            dataset,
            split,
            model_key,
        )
        if cached is not None:
            print(f"Loaded SigLIP image embeddings from cache: {cache_path}")
            print(f"Cached images: {len(cached)}")
            return cached

    def encode_batch(images):
        inputs = processor(
            images=images,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(DEVICE) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        return _siglip_forward_image_features(inputs, model)

    return build_generic_image_cache(
        list(image_ids),
        image_folder,
        encode_batch,
        cache_path,
        model_key,
        dataset,
        split,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )



# ============================================================
# ALIGN (kakaobrain/align-base)
# ============================================================


def _extract_align_feature(output, feature_name: str) -> torch.Tensor:
    """Extract a tensor from ALIGN get_*_features output across Transformers versions."""
    if torch.is_tensor(output):
        return output.float()

    if feature_name == "image":
        candidate_names = ("image_embeds", "pooler_output", "last_hidden_state")
    else:
        candidate_names = ("text_embeds", "pooler_output", "last_hidden_state")

    for name in candidate_names:
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            if name == "last_hidden_state" and value.ndim == 3:
                return value[:, 0, :].float()
            return value.float()

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                if value.ndim == 3:
                    return value[:, 0, :].float()
                return value.float()

    raise TypeError(
        f"Unsupported ALIGN {feature_name} output: {type(output).__name__}"
    )


def load_align_model(model_path: str):
    """Load ALIGN base locally/offline using Transformers."""
    if not model_path:
        raise ValueError("--align_path is required when using --models align.")

    from transformers import AlignModel, AlignProcessor

    local_path = Path(model_path).expanduser().resolve()

    if not local_path.is_dir():
        raise FileNotFoundError(
            f"ALIGN model directory does not exist: {local_path}\n"
            "Expected a local copy of kakaobrain/align-base."
        )

    if not (local_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing config.json in ALIGN directory: {local_path}"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        processor = AlignProcessor.from_pretrained(
            str(local_path),
            local_files_only=True,
        )
    except TypeError:
        processor = AlignProcessor.from_pretrained(str(local_path))

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

    try:
        model = AlignModel.from_pretrained(
            str(local_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
    except TypeError:
        try:
            model = AlignModel.from_pretrained(
                str(local_path),
                dtype=dtype,
                local_files_only=True,
            )
        except TypeError:
            model = AlignModel.from_pretrained(
                str(local_path),
                local_files_only=True,
            )

    model = model.to(DEVICE).eval()
    model.requires_grad_(False)

    print(f"ALIGN checkpoint : {local_path}")
    print(f"ALIGN model      : {model.__class__.__name__}")
    print(f"ALIGN device     : {DEVICE}")
    print(f"ALIGN dtype      : {getattr(model, 'dtype', dtype)}")

    return model, processor


@torch.inference_mode()
def align_image_embedding(
    image: Image.Image,
    model,
    processor,
) -> torch.Tensor:
    """Return one L2-normalized ALIGN image embedding."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    output = model.get_image_features(**inputs)
    feat = _extract_align_feature(output, "image")
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)

    return F.normalize(feat.float(), dim=-1).cpu()


@torch.inference_mode()
def align_text_embedding(
    text: str,
    model,
    processor,
) -> torch.Tensor:
    """Return one L2-normalized ALIGN text embedding."""
    text = normalize_prompt(text)
    if not text:
        raise ValueError("ALIGN text input cannot be empty.")

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    output = model.get_text_features(**inputs)
    feat = _extract_align_feature(output, "text")
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)

    return F.normalize(feat.float(), dim=-1).cpu()


def build_align_image_cache(
    image_ids,
    image_folder,
    model,
    processor,
    cache_dir: str,
    dataset: str,
    split: str,
    model_key: str,
    batch_size: int = 16,
    force_rebuild: bool = False,
):
    """Build/resume a persistent full-gallery ALIGN image embedding cache."""
    cache_path = model_image_cache_path(
        cache_dir,
        dataset,
        split,
        model_key,
    )

    if not force_rebuild:
        cached = load_generic_image_cache(
            cache_path,
            list(image_ids),
            dataset,
            split,
            model_key,
        )
        if cached is not None:
            print(f"Loaded ALIGN image embeddings from cache: {cache_path}")
            print(f"Cached images: {len(cached)}")
            return cached

    def encode_batch(images):
        inputs = processor(
            images=images,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(DEVICE) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        output = model.get_image_features(**inputs)
        return _extract_align_feature(output, "image")

    return build_generic_image_cache(
        list(image_ids),
        image_folder,
        encode_batch,
        cache_path,
        model_key,
        dataset,
        split,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )


# ============================================================
# Cache utilities
# ============================================================

def stack_feature_cache(cache: Dict[str, torch.Tensor]):
    if not cache:
        raise RuntimeError("Feature cache is empty.")

    ids = list(cache.keys())
    feats = torch.cat(
        [cache[x].reshape(1, -1) for x in ids],
        dim=0,
    ).float()

    feats = F.normalize(feats, dim=-1)
    return ids, feats


def normalize_prompt(caption: str) -> str:
    caption = caption.strip()
    return caption if caption else ""


# ============================================================
# Generic cross-modal evaluation
# ============================================================



def evaluate_cross_modal(
    data: List[dict],
    image_folder: str,
    reference_image_folder: str,
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    get_image_feature,
    get_text_feature,
    model_name: str,
    alpha: float = 0.5,
    protocols: Tuple[str, ...] = (FULL_GALLERY, CIRR_SUBSET),
):
    """Evaluate a shared image/text embedding model under both protocols."""
    id2idx = build_id_index(gallery_ids)
    ref_cache: Dict[str, Optional[torch.Tensor]] = {}
    text_cache: Dict[str, Optional[torch.Tensor]] = {}

    for item in tqdm(data, desc=f"{model_name}: cache queries"):
        ref_id = item["reference_id"]
        caption = normalize_prompt(item["caption"])
        if ref_id not in ref_cache:
            image = load_image(reference_image_folder, ref_id)
            ref_cache[ref_id] = None if image is None else get_image_feature(image)
        if caption not in text_cache:
            text_cache[caption] = get_text_feature(caption) if caption else None

    states = {
        protocol: {
            "results": init_results(),
            "rankings": {},
            "total": 0,
            "skipped": 0,
            "skip_reasons": {},
        }
        for protocol in protocols
    }

    for item in tqdm(data, desc=f"{model_name}: ranking"):
        ref_id = item["reference_id"]
        query_key = get_query_key(item)
        caption = normalize_prompt(item["caption"])
        ref_feat = ref_cache.get(ref_id)
        text_feat = text_cache.get(caption)

        if ref_feat is None or text_feat is None:
            reason = "missing_reference_embedding" if ref_feat is None else "empty_or_missing_text_embedding"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            continue

        try:
            ref_feat = F.normalize(ref_feat.float(), dim=-1)
            text_feat = F.normalize(text_feat.float(), dim=-1)
            if ref_feat.shape[-1] != gallery_feats.shape[-1] or text_feat.shape[-1] != gallery_feats.shape[-1]:
                raise ValueError(
                    f"{model_name}: query/gallery feature dimensions do not match: "
                    f"ref={ref_feat.shape[-1]}, text={text_feat.shape[-1]}, gallery={gallery_feats.shape[-1]}"
                )

            sims_text = gallery_feats @ text_feat.squeeze(0)
            sims_ref = gallery_feats @ ref_feat.squeeze(0)
            sims = alpha * sims_text + (1.0 - alpha) * sims_ref
            positives = get_relevant_ids(item)

            for protocol in protocols:
                ranked_ids = rank_from_sims(
                    sims,
                    gallery_ids,
                    id2idx,
                    exclude_ids=[ref_id],
                    restrict_ids=protocol_restrict_ids(item, protocol),
                )
                state = states[protocol]
                state["rankings"][query_key] = ranked_ids[:50]
                if positives:
                    update_metrics(state["results"], ranked_ids, positives)
                state["total"] += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            print(f"[WARN] {model_name} failed for query={query_key!r}: {reason}")

    return {
        protocol: (
            state["results"],
            state["total"],
            state["skipped"],
            state["rankings"],
            state["skip_reasons"],
        )
        for protocol, state in states.items()
    }

def evaluate_clip_beta(
    data,
    image_folder,
    reference_image_folder,
    gallery_ids,
    gallery_feats,
    clip_model,
    clip_model_preprocess,
    image_features_cache,
    generated_captions,
    alphas,
    betas,
    protocols=(FULL_GALLERY, CIRR_SUBSET),
):
    """Evaluate CLIP-beta and generate both protocols in one query pass."""
    id2idx = build_id_index(gallery_ids)
    rel_cache = {}
    gen_cache = {}

    for item in tqdm(data, desc="CLIP-beta: text cache"):
        rel = normalize_prompt(item["caption"])
        if rel not in rel_cache:
            rel_cache[rel] = clip_text_embedding(rel, clip_model) if rel else None
        ref_id = item["reference_id"]
        gen = generated_captions.get(ref_id, "").strip()
        if gen and ref_id not in gen_cache:
            gen_cache[ref_id] = clip_text_embedding(gen, clip_model)

    outputs = {}
    ref_feature_cache: Dict[str, Optional[torch.Tensor]] = {}

    for beta in betas:
        for alpha in alphas:
            states = {
                protocol: {
                    "results": init_results(),
                    "rankings": {},
                    "total": 0,
                    "skipped": 0,
                    "skip_reasons": {},
                }
                for protocol in protocols
            }

            for item in tqdm(data, desc=f"CLIP-beta a={alpha:.2f} b={beta:.2f}"):
                ref_id = item["reference_id"]
                query_key = get_query_key(item)
                rel = normalize_prompt(item["caption"])
                gen = generated_captions.get(ref_id, "").strip()
                t_rel = rel_cache.get(rel)
                r = image_features_cache.get(ref_id)

                if r is None:
                    if ref_id not in ref_feature_cache:
                        image = load_image(reference_image_folder, ref_id)
                        ref_feature_cache[ref_id] = (
                            clip_image_embedding(image, clip_model, clip_model_preprocess)
                            if image is not None else None
                        )
                    r = ref_feature_cache.get(ref_id)

                t_gen = gen_cache.get(ref_id)
                if t_rel is None or r is None:
                    reason = "missing_relative_text_embedding" if t_rel is None else "missing_reference_embedding"
                    for state in states.values():
                        state["skipped"] += 1
                        state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
                    continue

                try:
                    if t_gen is None:
                        q1, q2 = t_rel, r
                    else:
                        q1 = beta * t_rel + (1.0 - beta) * t_gen
                        q2 = beta * r + (1.0 - beta) * t_gen
                    q1 = F.normalize(q1.float(), dim=-1)
                    q2 = F.normalize(q2.float(), dim=-1)
                    sims1 = gallery_feats @ q1.squeeze(0)
                    sims2 = gallery_feats @ q2.squeeze(0)
                    sims = alpha * sims1 + (1.0 - alpha) * sims2
                    positives = get_relevant_ids(item)

                    for protocol in protocols:
                        ranked_ids = rank_from_sims(
                            sims,
                            gallery_ids,
                            id2idx,
                            exclude_ids=[ref_id],
                            restrict_ids=protocol_restrict_ids(item, protocol),
                        )
                        state = states[protocol]
                        state["rankings"][query_key] = ranked_ids[:50]
                        if positives:
                            update_metrics(state["results"], ranked_ids, positives)
                        state["total"] += 1
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    for state in states.values():
                        state["skipped"] += 1
                        state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
                    print(f"[WARN] CLIP-beta failed for query={query_key!r}: {reason}")

            outputs[(alpha, beta)] = {
                protocol: (
                    state["results"],
                    state["total"],
                    state["skipped"],
                    state["rankings"],
                    state["skip_reasons"],
                )
                for protocol, state in states.items()
            }

    return outputs

def save_cirr_test_predictions(
    path: Path,
    data: List[dict],
    model_name: str,
    split: str,
    rankings: Dict[str, List[str]],
    total: int,
    skipped: int,
    skip_reasons: Optional[Dict[str, int]] = None,
    extra: Optional[dict] = None,
) -> None:
    """Save query-level CIRR predictions including candidate/target IDs."""
    predictions = []

    for item in data:
        query_key = get_query_key(item)
        ranking = rankings.get(query_key, [])
        target_id = normalize_id(item.get("target_id"))

        target_rank = None
        if target_id and target_id in ranking:
            target_rank = ranking.index(target_id) + 1

        predictions.append({
            "query_id": query_key,
            "candidate_id": item.get("candidate_id") or item.get("reference_id"),
            "reference": item.get("reference_id"),
            "target_id": item.get("target_id"),
            "caption": item.get("caption", ""),
            "caption_source": item.get("caption_source", "annotation"),
            "members": item.get("members", []),
            "ranking": ranking,
            "target_rank": target_rank,
        })

    payload = {
        "dataset": "cirr",
        "split": split,
        "model": model_name,
        "num_queries": len(data),
        "num_predictions": total,
        "num_skipped": skipped,
        "skip_reasons": skip_reasons or {},
        "ground_truth_type": "target_id",
        "metrics_available": any(bool(get_relevant_ids(x)) for x in data),
        "predictions": predictions,
    }

    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to: {path}")



def enforce_complete_evaluation(
    model_name: str,
    total_queries: int,
    total_processed: int,
    skipped: int,
    strict: bool,
) -> None:
    if skipped == 0:
        return

    message = (
        f"{model_name}: incomplete evaluation: "
        f"processed={total_processed}/{total_queries}, skipped={skipped}."
    )

    if strict:
        raise RuntimeError(message)

    print(f"[WARN] {message}")


def print_metric_summary(label: str, summary: dict, total: int, skipped: int) -> None:
    """Print the exact same 12 metrics for every model/configuration."""
    print(
        f"{label} | "
        f"MRR={summary['mrr']:.4f} | "
        f"mAP@5={summary['map5']:.4f} | "
        f"mAP@10={summary['map10']:.4f} | "
        f"mAP@50={summary['map50']:.4f} | "
        f"P@1={summary['prec1']:.4f} | "
        f"P@5={summary['prec5']:.4f} | "
        f"P@10={summary['prec10']:.4f} | "
        f"P@50={summary['prec50']:.4f} | "
        f"R@1={summary['rec1']:.4f} | "
        f"R@5={summary['rec5']:.4f} | "
        f"R@10={summary['rec10']:.4f} | "
        f"R@50={summary['rec50']:.4f} | "
        f"n={total}, skipped={skipped}"
    )


def save_metrics_json(
    path: Path,
    dataset: str,
    split: str,
    protocol_results: Dict[str, Dict[str, dict]],
) -> None:
    """Save metrics separated by evaluation protocol."""
    payload = {
        "dataset": dataset,
        "split": split,
        "protocols": protocol_results,
        "metric_order": [
            "MRR",
            "mAP@5", "mAP@10", "mAP@50",
            "P@1", "P@5", "P@10", "P@50",
            "R@1", "R@5", "R@10", "R@50",
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Metrics JSON saved to: {path}")

def validate_reference_images(
    data: List[dict],
    image_folder: str,
    fail_on_missing: bool = False,
) -> Tuple[int, int]:
    """Check that every unique CIRR reference image can be resolved."""
    references = sorted({
        item.get("reference_id", "")
        for item in data
        if item.get("reference_id")
    })

    found = 0
    missing_ids: List[str] = []

    for ref_id in references:
        if find_image_path(image_folder, ref_id) is None:
            missing_ids.append(ref_id)
        else:
            found += 1

    print("\n=== CIRR reference image validation ===")
    print(f"Unique references : {len(references)}")
    print(f"Found             : {found}")
    print(f"Missing           : {len(missing_ids)}")

    if missing_ids:
        print("First missing reference IDs:")
        for ref_id in missing_ids[:30]:
            print(f"  {ref_id}")

    if fail_on_missing and missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)}/{len(references)} unique reference images "
            f"cannot be resolved from: {image_folder}"
        )

    return found, len(missing_ids)



def validate_cirr_gallery_coverage(
    data: List[dict],
    gallery_ids: List[str],
    require_ground_truth: bool = False,
) -> None:
    """Validate reference, target_id and group members against the image gallery."""
    gallery_set = set(gallery_ids)

    missing_refs = sorted({
        item.get("reference_id")
        for item in data
        if item.get("reference_id") and item["reference_id"] not in gallery_set
    })

    missing_targets = sorted({
        target_id
        for item in data
        for target_id in [normalize_id(item.get("target_id") or item.get("target_hard"))]
        if target_id and target_id not in gallery_set
    })

    missing_members = sorted({
        normalize_id(member)
        for item in data
        for member in item.get("members") or []
        if normalize_id(member) and normalize_id(member) not in gallery_set
    })

    print("\n=== CIRR gallery coverage check ===")
    print(f"Gallery IDs               : {len(gallery_ids)}")
    print(f"Missing reference IDs     : {len(missing_refs)}")
    print(f"Missing target IDs        : {len(missing_targets)}")
    print(f"Missing group member IDs  : {len(missing_members)}")

    if missing_refs:
        print("First missing references:")
        for value in missing_refs[:20]:
            print(f"  {value}")

    if missing_targets:
        print("First missing targets:")
        for value in missing_targets[:20]:
            print(f"  {value}")

    if missing_members:
        print("First missing group members:")
        for value in missing_members[:20]:
            print(f"  {value}")

    if require_ground_truth and (missing_refs or missing_targets or missing_members):
        raise RuntimeError(
            "CIRR gallery validation failed. Check candidate_id/reference_id, "
            "target_id and group against the actual image folder."
        )



# ============================================================
# Qwen3-VL-Embedding-2B
# ============================================================

QWEN3_DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve images relevant to the user's composed image and text query."
)
QWEN3_DEFAULT_DOCUMENT_INSTRUCTION = "Represent the user's input."


def _load_qwen3_embedder_class(model_path: str, code_path: Optional[str] = None):
    """Load Qwen3VLEmbedder from an explicitly local implementation.*

    The official Qwen3-VL-Embedding project has used both:
      - scripts/qwen3_vl_embedding.py (model repo / HF snapshot)
      - src/models/qwen3_vl_embedding.py (GitHub source tree)
    We support both layouts and never download code automatically.
    *"""
    import importlib
    import importlib.util
    import sys

    model_dir = Path(model_path).resolve()
    errors = []

    candidates = []
    roots = []
    if code_path:
        roots.append(Path(code_path).expanduser().resolve())
    roots.extend([
        model_dir,
        model_dir.parent,
        Path.cwd().resolve(),
    ])

    seen = set()
    for root in roots:
        for rel in (
            Path("scripts") / "qwen3_vl_embedding.py",
            Path("src") / "models" / "qwen3_vl_embedding.py",
            Path("qwen3_vl_embedding.py"),
        ):
            fp = root / rel
            if fp not in seen:
                seen.add(fp)
                candidates.append(fp)

    # Also accept a direct path to the implementation file.
    if code_path:
        cp = Path(code_path).expanduser().resolve()
        if cp.is_file() and cp not in seen:
            candidates.insert(0, cp)

    for file_path in candidates:
        if not file_path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                "qwen3_vl_embedding_local", str(file_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {file_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cls = getattr(module, "Qwen3VLEmbedder", None)
            if cls is None:
                raise AttributeError(f"Qwen3VLEmbedder not found in {file_path}")
            print(f"Qwen3 embedding implementation: {file_path}")
            return cls
        except Exception as exc:
            errors.append(f"{file_path}: {type(exc).__name__}: {exc}")

    # Last local-package fallback. We add candidate roots to sys.path so that
    # namespace packages such as `scripts` work even without __init__.py.
    for root in roots:
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))

    for module_name in (
        "src.models.qwen3_vl_embedding",
        "scripts.qwen3_vl_embedding",
        "qwen3_vl_embedding",
    ):
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, "Qwen3VLEmbedder", None)
            if cls is not None:
                print(f"Qwen3 embedding implementation: module={module_name}")
                return cls
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "Could not import Qwen3VLEmbedder.\n"
        "Provide the official/local implementation using --qwen3_code_path, "
        "or place qwen3_vl_embedding.py under ./scripts/ or ./src/models/.\n\n"
        + "\n".join(errors)
    )


def load_qwen3_model(model_path: str, query_instruction: str, max_length: int = 8192,
                     qwen3_dtype: str = "auto", use_flash_attention: bool = False,
                     code_path: Optional[str] = None):
    """Load local Qwen3-VL-Embedding.*

    This loader deliberately prefers the Sentence-Transformers multimodal backend
    documented by the supplied local model card. The official Qwen3VLEmbedder
    wrapper is used only as a fallback when available.
    *"""
    if not model_path:
        raise ValueError("--qwen3_path is required when using --models qwen3.")

    local_path = Path(model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"Qwen3 model directory does not exist: {local_path}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # ------------------------------------------------------------------
    # Backend 1: Sentence-Transformers multimodal embedding.
    # The supplied model card explicitly documents:
    # SentenceTransformer(...).encode(text/image/text+image).
    # ------------------------------------------------------------------
    st_error = None
    try:
        from sentence_transformers import SentenceTransformer

        st_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }

        if qwen3_dtype == "float16":
            st_kwargs["torch_dtype"] = torch.float16
        elif qwen3_dtype == "bfloat16":
            st_kwargs["torch_dtype"] = torch.bfloat16
        elif qwen3_dtype == "float32":
            st_kwargs["torch_dtype"] = torch.float32
        elif qwen3_dtype != "auto":
            raise ValueError(
                "--qwen3_dtype must be auto, float16, bfloat16, or float32"
            )

        model = SentenceTransformer(
            str(local_path),
            device=str(DEVICE),
            **st_kwargs,
        )
        model._qwen3_backend = "sentence_transformers"
        print("Qwen3 backend     : SentenceTransformer multimodal")
        print(f"Qwen3 checkpoint  : {local_path}")
        print(f"Qwen3 device      : {DEVICE}")
        print(f"Qwen3 query inst. : {query_instruction}")
        return model
    except Exception as exc:
        st_error = exc
        print(
            "[WARN] Local SentenceTransformer backend could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------
    # Backend 2: official/local Qwen3VLEmbedder implementation.
    # ------------------------------------------------------------------
    try:
        Embedder = _load_qwen3_embedder_class(str(local_path), code_path=code_path)

        kwargs = {}
        if qwen3_dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif qwen3_dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif qwen3_dtype == "float32":
            kwargs["torch_dtype"] = torch.float32
        elif qwen3_dtype != "auto":
            raise ValueError(
                "--qwen3_dtype must be auto, float16, bfloat16, or float32"
            )
        if use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        model = Embedder(
            model_name_or_path=str(local_path),
            max_length=max_length,
            **kwargs,
        )
        model._qwen3_backend = "official"
        print("Qwen3 backend     : official Qwen3VLEmbedder")
        print(f"Qwen3 checkpoint  : {local_path}")
        print(f"Qwen3 query inst. : {query_instruction}")
        return model
    except Exception as official_exc:
        raise RuntimeError(
            "Could not load the local Qwen3-VL-Embedding model.\n\n"
            f"Model path: {local_path}\n"
            f"SentenceTransformer error: {type(st_error).__name__}: {st_error}\n"
            f"Official Qwen3VLEmbedder error: {type(official_exc).__name__}: {official_exc}\n\n"
            "The supplied model card documents the SentenceTransformer backend. "
            "Verify that sentence-transformers and the local multimodal modules "
            "are installed and present in the model directory."
        ) from official_exc


class _Qwen3SentenceTransformerAdapter:
    """Adapter exposing the same process() API as Qwen3VLEmbedder."""

    def __init__(self, model, default_instruction: str):
        self.model = model
        self.default_instruction = default_instruction
        self._qwen3_backend = "sentence_transformers"

    def process(self, items, normalize=True):
        if not items:
            return torch.empty((0, 0), dtype=torch.float32)

        # The model card supports multimodal records such as:
        # {"text": ..., "image": ...}
        # and SentenceTransformer.encode accepts these records directly.
        # All records in a batch use the same task instruction.
        # Sentence-Transformers multimodal dictionaries accept only modality keys
        # such as image/text/video/audio. The task instruction must be supplied
        # through the separate `prompt=` argument, not inside each item.
        clean_items = []
        for item in items:
            if not isinstance(item, dict):
                clean_items.append(item)
                continue
            clean_items.append({
                key: value
                for key, value in item.items()
                if key in {"image", "text", "video", "audio"}
            })

        embeddings = self.model.encode(
            clean_items,
            prompt=self.default_instruction,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(embeddings)
        return embeddings.float().cpu()


def _make_qwen3_process_adapter(model, instruction: str):
    """Return a model object exposing process(items, normalize=True)."""
    if getattr(model, "_qwen3_backend", None) == "sentence_transformers":
        return _Qwen3SentenceTransformerAdapter(model, instruction)
    return model

def _qwen3_model_process(model, items, normalize=True):
    """Call Qwen3 backend and return a normalized CPU tensor."""
    if not items:
        return torch.empty((0, 0), dtype=torch.float32)

    embeddings = model.process(items, normalize=normalize)
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    embeddings = embeddings.float().cpu()
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
        raise ValueError(
            f"Unexpected Qwen3 embedding shape {tuple(embeddings.shape)} "
            f"for {len(items)} inputs."
        )

    # SentenceTransformer may already normalize; official Qwen3 may or may not,
    # so normalizing once here guarantees cosine/dot-product equivalence.
    return F.normalize(embeddings, p=2, dim=-1)


def qwen3_image_embedding(image_path: str, model) -> torch.Tensor:
    """Encode a gallery image only."""
    return _qwen3_model_process(
        model,
        [{"image": image_path, "instruction": QWEN3_DEFAULT_DOCUMENT_INSTRUCTION}],
        normalize=True,
    )[0:1]


def qwen3_composed_query_embedding(reference_image_path: str, caption: str, model,
                                   instruction: str) -> torch.Tensor:
    """Encode the CIRR composed query as one joint image+text embedding."""
    caption = normalize_prompt(caption)
    if not caption:
        raise ValueError("Qwen3 composed query requires a non-empty caption.")
    if not reference_image_path or not Path(reference_image_path).is_file():
        raise FileNotFoundError(
            f"Reference image does not exist: {reference_image_path}"
        )

    item = {
        "image": reference_image_path,
        "text": caption,
        "instruction": instruction,
    }
    return _qwen3_model_process(model, [item], normalize=True)


def build_qwen3_image_cache(
    image_ids: List[str],
    image_folder: str,
    model,
    cache_dir: str,
    dataset: str,
    split: str,
    model_key: str,
    batch_size: int = 1,
    force_rebuild: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build a resumable full-gallery cache for Qwen3 image embeddings."""
    image_ids = list(image_ids)
    cache_path = model_image_cache_path(
        cache_dir, dataset, split, model_key
    )

    if not force_rebuild:
        cached = load_generic_image_cache(
            cache_path, image_ids, dataset, split, model_key
        )
        if cached is not None:
            print(f"Loaded Qwen3 image embeddings from cache: {cache_path}")
            print(f"Cached images: {len(cached)}")
            return cached

    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")
    features: Dict[str, torch.Tensor] = {}

    if not force_rebuild and partial_path.is_file():
        try:
            payload = torch.load(
                partial_path, map_location="cpu", weights_only=False
            )
            if (
                isinstance(payload, dict)
                and payload.get("cache_version") == 1
                and payload.get("dataset") == dataset
                and payload.get("split") == split
                and payload.get("model_key") == model_key
                and list(payload.get("gallery_ids", [])) == image_ids
                and isinstance(payload.get("features"), dict)
            ):
                features.update(payload["features"])
                print(
                    f"Resuming Qwen3 image cache: "
                    f"{len(features)}/{len(image_ids)}"
                )
        except Exception as exc:
            print(f"[WARN] Could not resume Qwen3 cache: {type(exc).__name__}: {exc}")

    def save_partial():
        _atomic_torch_save(
            {
                "cache_version": 1,
                "dataset": dataset,
                "split": split,
                "model_key": model_key,
                "gallery_ids": image_ids,
                "features": features,
                "num_cached": len(features),
            },
            partial_path,
        )

    remaining = [x for x in image_ids if x not in features]
    for start in tqdm(range(0, len(remaining), max(1, batch_size)), desc="Qwen3 gallery embeddings"):
        batch_ids = remaining[start:start + max(1, batch_size)]
        items = []
        valid_ids = []
        for image_id in batch_ids:
            path = find_image_path(image_folder, image_id)
            if path is None:
                raise RuntimeError(f"Cannot resolve gallery image: {image_id}")
            items.append({
                "image": path,
                "instruction": QWEN3_DEFAULT_DOCUMENT_INSTRUCTION,
            })
            valid_ids.append(image_id)

        try:
            batch_features = _qwen3_model_process(model, items, normalize=True)
            for image_id, feat in zip(valid_ids, batch_features):
                features[image_id] = feat.unsqueeze(0).cpu()
        except Exception as exc:
            save_partial()
            raise RuntimeError(
                f"Qwen3 gallery batch failed for {valid_ids[:3]}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        save_partial()
        print(f"[CACHE] Qwen3: {len(features)}/{len(image_ids)} images cached")

    missing = sorted(set(image_ids) - set(features))
    if missing:
        save_partial()
        raise RuntimeError(
            f"Qwen3 cache is incomplete: {len(missing)} images missing. "
            f"First missing: {missing[:20]}"
        )

    _atomic_torch_save(
        {
            "cache_version": 1,
            "dataset": dataset,
            "split": split,
            "model_key": model_key,
            "gallery_ids": image_ids,
            "features": features,
            "num_cached": len(features),
        },
        cache_path,
    )
    try:
        if partial_path.exists():
            partial_path.unlink()
    except OSError:
        pass

    print(f"Saved FULL Qwen3 image cache: {cache_path}")
    return features


def evaluate_qwen3(
    data: List[dict],
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    reference_image_folder: str,
    qwen3_model,
    protocols: Tuple[str, ...],
    query_instruction: str,
):
    """Evaluate Qwen3 joint image+text queries in all requested protocols."""
    id2idx = build_id_index(gallery_ids)
    if not torch.is_tensor(gallery_feats) or gallery_feats.ndim != 2:
        raise ValueError("Qwen3 gallery features must be a 2-D tensor.")
    if gallery_feats.shape[0] != len(gallery_ids):
        raise ValueError(
            f"Qwen3 gallery mismatch: {gallery_feats.shape[0]} features vs {len(gallery_ids)} IDs."
        )

    states = {
        protocol: {
            "results": init_results(),
            "rankings": {},
            "total": 0,
            "skipped": 0,
            "skip_reasons": {},
        }
        for protocol in protocols
    }

    ref_path_cache: Dict[str, Optional[str]] = {}
    query_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    for item in tqdm(data, desc="Qwen3 composed retrieval"):
        query_key = get_query_key(item)
        ref_id = normalize_id(item.get("reference_id"))
        caption = normalize_prompt(item.get("caption", ""))
        if not ref_id or not caption:
            reason = "missing_reference_id" if not ref_id else "empty_caption"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            continue

        if ref_id not in ref_path_cache:
            ref_path_cache[ref_id] = find_image_path(reference_image_folder, ref_id)
        ref_path = ref_path_cache[ref_id]
        if ref_path is None:
            reason = "missing_reference_image"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            continue

        try:
            cache_key = (ref_id, caption)
            q = query_cache.get(cache_key)
            if q is None:
                q = qwen3_composed_query_embedding(
                    ref_path, caption, qwen3_model, query_instruction
                )
                query_cache[cache_key] = q.cpu()

            sims = gallery_feats @ q.squeeze(0)
            positives = get_relevant_ids(item)
            for protocol in protocols:
                ranked_ids = rank_from_sims(
                    sims,
                    gallery_ids,
                    id2idx,
                    exclude_ids=[ref_id],
                    restrict_ids=protocol_restrict_ids(item, protocol),
                )
                state = states[protocol]
                state["rankings"][query_key] = ranked_ids[:50]
                if positives:
                    update_metrics(state["results"], ranked_ids, positives)
                state["total"] += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            print(f"[WARN] Qwen3 failed for query={query_key!r}, reference={ref_id!r}: {reason}")

    return {
        protocol: (
            state["results"],
            state["total"],
            state["skipped"],
            state["rankings"],
            state["skip_reasons"],
        )
        for protocol, state in states.items()
    }


# ============================================================
# Adaptive Multi-Signal RRF: CLIP + Qwen caption + SAM2
# ============================================================

ADAPTIVE_RRF_METHOD = "adaptive_rrf"
ADAPTIVE_RRF_LABEL = "CLIP + Qwen + SAM2 + Adaptive-RRF"


def _load_json_with_diagnostics(path: Path) -> Any:
    """Load JSON strictly, with a precise error context for malformed files.

    A single safe normalization is attempted after a strict parse failure:
    removal of a UTF-8 BOM and trailing commas immediately before ] or }.
    No arbitrary text/comment stripping is performed, because that could silently
    change experimental data.
    """
    raw = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        lines = raw.splitlines()
        start = max(1, exc.lineno - 3)
        end = min(len(lines), exc.lineno + 3)
        context = "\n".join(
            f"{i:6d}: {lines[i-1]}" for i in range(start, end + 1)
        )

        # Common accidental JSON formatting error: trailing comma before a
        # closing object/array. This preserves all semantic content.
        repaired = re.sub(r",\s*([}\]])", r"\1", raw)
        if repaired != raw:
            try:
                data = json.loads(repaired)
                print(
                    f"[WARN] Recovered malformed JSON by removing trailing commas: {path}"
                )
                print(
                    f"[WARN] Original JSON error: line={exc.lineno}, "
                    f"column={exc.colno}, char={exc.pos}"
                )
                return data
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Malformed Qwen captions JSON: {path}\n"
            f"JSON error: {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}, character {exc.pos}\n"
            f"Context around the error:\n{context}\n\n"
            "Open the JSON at that line and fix the syntax. "
            "Do not continue with evaluation until the JSON is valid, "
            "because silently dropping a caption would change the experiment."
        ) from exc


def load_qwen_caption_map(path: str) -> Dict[str, str]:
    """Load Qwen-generated visual captions from common local JSON formats."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Qwen captions JSON not found: {source}")

    data = _load_json_with_diagnostics(source)

    captions: Dict[str, str] = {}
    id_keys = (
        "image_id", "img_id", "reference_img_id", "reference",
        "candidate_id", "reference_id", "id", "query_id", "pairid"
    )
    cap_keys = (
        "caption",
        "caption_fa",
        "caption_fa_per",
        "persian_caption",
        "translated_caption",
        "translation",
        "qwen_caption",
        "generated_caption",
        "relative_caption",
        "text",
    )

    def add(image_id: Any, caption: Any) -> None:
        image_id = normalize_id(image_id)
        caption = str(caption or "").strip()
        if image_id and caption:
            captions[image_id] = caption

    # 41-style output: {"metadata": {...}, "captions": {...}}
    if isinstance(data, dict) and isinstance(data.get("captions"), dict):
        for image_id, value in data["captions"].items():
            if isinstance(value, str):
                add(image_id, value)
            elif isinstance(value, dict):
                text = next(
                    (value.get(k) for k in cap_keys if value.get(k)),
                    "",
                )
                add(image_id, text)
        print(f"Qwen captions extracted from nested \"captions\": {len(captions)}")
        return captions

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "metadata":
                continue
            if isinstance(value, str):
                add(key, value)
            elif isinstance(value, dict):
                text = next(
                    (value.get(k) for k in cap_keys if value.get(k)),
                    "",
                )
                add(key, text)
        print(f"Qwen captions extracted from object mapping: {len(captions)}")
        return captions

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            image_id = next(
                (item.get(k) for k in id_keys if item.get(k) is not None),
                None,
            )
            text = next(
                (item.get(k) for k in cap_keys if item.get(k)),
                "",
            )
            add(image_id, text)
        print(f"Qwen captions extracted from list: {len(captions)}")
        return captions

    raise ValueError(f"Unsupported Qwen caption JSON format: {type(data)}")


@torch.inference_mode()
def clip_text_embedding_batch_51(
    texts: List[str],
    model,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Encode unique text strings efficiently with the shared CLIP text encoder."""
    result: Dict[str, torch.Tensor] = {}
    unique = list(dict.fromkeys(x.strip() for x in texts if str(x).strip()))
    if not unique:
        return result

    for start in tqdm(
        range(0, len(unique), max(1, batch_size)),
        desc="CLIP text embeddings (adaptive-RRF)",
    ):
        batch = unique[start:start + max(1, batch_size)]
        tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
        features = model.encode_text(tokens)
        features = F.normalize(features.float(), dim=-1).cpu()
        for text, feature in zip(batch, features):
            result[text] = feature.reshape(1, -1)
    return result


def _prepare_sam2_import(sam2_repo_path: Optional[str]) -> List[str]:
    """Add a local SAM2 repository to sys.path without modifying site-packages.

    The official repository normally has the structure:
        <repo>/sam2/build_sam.py
        <repo>/configs/sam2.1/...

    We first honor --sam2_repo_path, then try common local project locations.
    """
    candidates: List[Path] = []

    if sam2_repo_path:
        candidates.append(Path(sam2_repo_path).expanduser())

    here = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    common = [
        here / "sam2",
        here / "SAM2",
        here / "segment-anything-2",
        here / "models_download" / "sam2",
        here / "models_download" / "SAM2",
        script_dir / "sam2",
        script_dir / "SAM2",
        script_dir / "segment-anything-2",
        script_dir / "models_download" / "sam2",
        script_dir / "models_download" / "SAM2",
    ]
    candidates.extend(common)

    added: List[str] = []
    seen = set()

    for candidate in candidates:
        try:
            root = candidate.resolve()
        except Exception:
            continue
        if str(root) in seen or not root.is_dir():
            continue

        # Repository root: <root>/sam2/build_sam.py
        if (root / "sam2" / "build_sam.py").is_file():
            path_to_add = root
        # Python package folder itself: <root>/build_sam.py
        elif (root / "build_sam.py").is_file() and root.name.lower() == "sam2":
            path_to_add = root.parent
        else:
            continue

        seen.add(str(path_to_add))
        path_str = str(path_to_add)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added.append(path_str)

    return added


def load_sam2_51(checkpoint: str, config: str, sam2_repo_path: Optional[str] = None):
    """Load SAM 2.1 from an installed package or a local official repository."""
    added_paths = _prepare_sam2_import(sam2_repo_path)

    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        searched = []
        if sam2_repo_path:
            searched.append(str(Path(sam2_repo_path).expanduser().resolve()))
        searched.extend(added_paths)
        search_text = "; ".join(dict.fromkeys(searched)) or "none"
        raise ImportError(
            "SAM 2 Python package could not be imported. "
            "Either install SAM 2 with `python -m pip install -e .` from the official "
            "SAM2 repository, or provide --sam2_repo_path pointing to that repository. "
            f"\nSearched local paths: {search_text}"
        ) from exc

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

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
    return model, generator


def filter_masks_51(
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


def crop_masked_region_51(
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


def build_adaptive_region_cache_51(
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
    padding_ratio: float,
    checkpoint_every: int,
    force: bool,
) -> Dict[str, List[dict]]:
    """Resumable SAM2+CLIP region cache for unique reference images."""
    root = Path(cache_dir) / dataset / split
    root.mkdir(parents=True, exist_ok=True)

    meta = {
        "cache_type": "adaptive_rrf_sam2_clip_regions",
        "version": 1,
        "dataset": dataset,
        "split": split,
        "clip_model_name": clip_model_name,
        "sam_key": sam_key,
        "reference_ids": list(reference_ids),
        "max_regions": max_regions,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "background_mode": background_mode,
        "padding_ratio": padding_ratio,
    }

    tag = (
        f"adaptive_rrf_sam2_{sam_key}_clip_{_safe_cache_model_name(clip_model_name)}"
        f"_r{max_regions}_a{min_area_ratio:g}-{max_area_ratio:g}"
        f"_{background_mode}_p{padding_ratio:g}"
    )
    cache_path = root / f"{tag}_regions.pt"
    partial_path = root / f"{tag}_regions.partial.pt"

    if not force:
        payload = load_cache(cache_path, meta)
        if payload is not None:
            regions = payload.get("regions")
            if isinstance(regions, dict) and set(regions) == set(reference_ids):
                print(f"Loaded Adaptive-RRF SAM2 region cache: {cache_path}")
                return regions

    regions: Dict[str, List[dict]] = {}
    if not force and partial_path.is_file():
        try:
            payload = torch.load(
                partial_path,
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(payload, dict) and all(payload.get(k) == v for k, v in meta.items()):
                cached = payload.get("regions")
                if isinstance(cached, dict):
                    regions.update(cached)
                    print(
                        f"Resuming Adaptive-RRF region cache: "
                        f"{len(regions)}/{len(reference_ids)}"
                    )
        except Exception as exc:
            print(f"[WARN] Region partial cache could not be resumed: {exc}")

    remaining = [x for x in reference_ids if x not in regions]
    for processed, reference_id in enumerate(
        tqdm(remaining, desc="SAM2 + CLIP adaptive-RRF regions"),
        start=1,
    ):
        image = load_image(image_folder, reference_id)
        if image is None:
            raise RuntimeError(
                f"Reference image cannot be opened: {reference_id}"
            )

        image_np = np.asarray(image.convert("RGB"))
        try:
            masks = sam_generator.generate(image_np)
        except Exception as exc:
            raise RuntimeError(
                f"SAM2 failed for {reference_id}: {type(exc).__name__}: {exc}"
            ) from exc

        candidate_masks = filter_masks_51(
            masks,
            image_np.shape[1],
            image_np.shape[0],
            min_area_ratio,
            max_area_ratio,
            max_regions,
        )

        records: List[dict] = []
        for region_index, ann in enumerate(candidate_masks):
            region_image = crop_masked_region_51(
                image,
                ann["segmentation"],
                ann["bbox"],
                background_mode,
                padding_ratio,
            )
            if region_image is None:
                continue

            try:
                feat = clip_image_embedding(
                    region_image,
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

        if processed % max(1, checkpoint_every) == 0:
            _atomic_torch_save(
                {**meta, "regions": regions, "num_cached": len(regions)},
                partial_path,
            )

    if set(regions) != set(reference_ids):
        missing = sorted(set(reference_ids) - set(regions))
        raise RuntimeError(
            f"Adaptive-RRF region cache incomplete: {len(missing)} missing. "
            f"First missing: {missing[:10]}"
        )

    _atomic_torch_save(
        {**meta, "regions": regions, "num_cached": len(regions)},
        cache_path,
    )
    try:
        partial_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Saved Adaptive-RRF SAM2 region cache: {cache_path}")
    return regions


def select_adaptive_region_feature_51(
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

    m = min(max(1, top_regions), scores.numel())
    values, indices = torch.topk(scores, k=m)
    weights = F.softmax(values / max(temperature, 1e-4), dim=0)

    region_feat = F.normalize(
        (features[indices] * weights.unsqueeze(1)).sum(dim=0, keepdim=True),
        dim=-1,
    )

    selected = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        record = region_records[idx]
        selected.append({
            "region_index": int(record["index"]),
            "score": float(value),
            "area_ratio": float(record["area_ratio"]),
            "sam_quality": float(record["sam_quality"]),
            "bbox": record["bbox"],
        })
    return region_feat, selected


def _candidate_ids_for_protocol_51(
    item: dict,
    gallery_ids: List[str],
    id2idx: Dict[str, int],
    protocol: str,
    reference_id: str,
) -> List[int]:
    excluded = {reference_id}
    if protocol == CIRR_SUBSET:
        candidates = []
        seen = set()
        for value in item.get("members") or []:
            image_id = normalize_id(value)
            if not image_id or image_id in seen or image_id in excluded:
                continue
            idx = id2idx.get(image_id)
            if idx is not None:
                candidates.append(idx)
                seen.add(image_id)
        return candidates

    return [
        idx for idx, image_id in enumerate(gallery_ids)
        if image_id not in excluded
    ]


def _top_ranked_51(
    sims: torch.Tensor,
    gallery_ids: List[str],
    candidate_indices: List[int],
    topk: int,
) -> Tuple[List[str], np.ndarray]:
    if not candidate_indices:
        return [], np.empty((0,), dtype=np.float32)

    index_tensor = torch.tensor(candidate_indices, dtype=torch.long)
    candidate_scores = sims.detach().float()[index_tensor]
    k = min(max(1, topk), candidate_scores.numel())
    values, order = torch.topk(candidate_scores, k=k)
    ids = [gallery_ids[candidate_indices[int(i)]] for i in order.cpu().tolist()]
    return ids, values.cpu().numpy()


def _channel_confidence_51(scores: np.ndarray) -> float:
    """Convert top-score separation into a bounded query-specific confidence."""
    if scores.size < 2:
        return 0.5
    top = float(scores[0])
    tail = scores[1:]
    scale = float(np.std(tail)) + 1e-6
    margin = (top - float(np.mean(tail))) / scale
    margin = float(np.clip(margin, -4.0, 4.0))
    return float(1.0 / (1.0 + np.exp(-0.75 * margin)))


def weighted_adaptive_rrf_51(
    rankings: Dict[str, List[str]],
    confidences: Dict[str, float],
    base_weights: Dict[str, float],
    rrf_k: int,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Adaptive weighted Reciprocal Rank Fusion.

    For channel c:
        adaptive_weight_c =
            base_weight_c * (0.5 + 0.5 * confidence_c)

    The final score is:
        sum_c adaptive_weight_c / (rrf_k + rank_c)

    This is rank-level fusion, so components do not need the same raw-score
    calibration beyond their individual rankings.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")

    effective_weights = {}
    for channel, weight in base_weights.items():
        if channel not in rankings or not rankings[channel] or weight <= 0:
            continue
        confidence = float(np.clip(confidences.get(channel, 0.5), 0.0, 1.0))
        effective_weights[channel] = float(weight * (0.5 + 0.5 * confidence))

    if not effective_weights:
        raise RuntimeError("Adaptive-RRF has no valid ranking channels.")

    fused: Dict[str, float] = {}
    for channel, weight in effective_weights.items():
        for rank, image_id in enumerate(rankings[channel], start=1):
            fused[image_id] = fused.get(image_id, 0.0) + weight / (
                float(rrf_k) + float(rank)
            )

    ranking = [
        image_id
        for image_id, _ in sorted(
            fused.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return ranking, effective_weights


def evaluate_adaptive_rrf_51(
    data: List[dict],
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    clip_gallery_cache: Dict[str, torch.Tensor],
    clip_text_cache: Dict[str, torch.Tensor],
    qwen_caption_map: Dict[str, str],
    qwen_text_cache: Dict[str, torch.Tensor],
    region_cache: Dict[str, List[dict]],
    protocols: Tuple[str, ...],
    rrf_k: int,
    component_topk: int,
    output_topk: int,
    top_regions: int,
    region_temperature: float,
    confidence_topk: int,
    ref_weight: float,
    text_weight: float,
    base_weight: float,
    qwen_weight: float,
    region_weight: float,
) -> Dict[str, Tuple[dict, int, int, Dict[str, List[str]], Dict[str, int], dict]]:
    """Evaluate the new adaptive rank-fusion method under all protocols."""
    if gallery_feats.ndim != 2 or gallery_feats.shape[0] != len(gallery_ids):
        raise ValueError("Invalid gallery feature tensor for adaptive-RRF.")

    id2idx = build_id_index(gallery_ids)

    states = {
        protocol: {
            "results": init_results(),
            "rankings": {},
            "total": 0,
            "skipped": 0,
            "skip_reasons": {},
            "selected_regions": {},
            "effective_weights": {},
            "channel_confidence": {},
        }
        for protocol in protocols
    }

    for item in tqdm(data, desc=ADAPTIVE_RRF_LABEL):
        query_key = get_query_key(item)
        ref_id = normalize_id(item.get("reference_id"))
        caption = normalize_prompt(item.get("caption", ""))

        reference_feat = clip_gallery_cache.get(ref_id)
        cirr_feat = clip_text_cache.get(caption)

        if reference_feat is None or cirr_feat is None:
            reason = (
                "missing_reference_embedding"
                if reference_feat is None
                else "empty_or_missing_text_embedding"
            )
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = (
                    state["skip_reasons"].get(reason, 0) + 1
                )
            continue

        qwen_caption = normalize_prompt(qwen_caption_map.get(ref_id, ""))
        qwen_feat = qwen_text_cache.get(qwen_caption) if qwen_caption else None

        try:
            reference_feat = F.normalize(reference_feat.float(), dim=-1)
            cirr_feat = F.normalize(cirr_feat.float(), dim=-1)

            if reference_feat.shape[-1] != gallery_feats.shape[-1]:
                raise ValueError("Reference/gallery embedding dimension mismatch.")
            if cirr_feat.shape[-1] != gallery_feats.shape[-1]:
                raise ValueError("Text/gallery embedding dimension mismatch.")

            base_query = F.normalize(
                reference_feat + cirr_feat,
                dim=-1,
            )

            region_feat = None
            selected_regions = []
            records = region_cache.get(ref_id, [])
            if records:
                region_seed_parts = [reference_feat, cirr_feat]
                if qwen_feat is not None:
                    region_seed_parts.append(
                        F.normalize(qwen_feat.float(), dim=-1)
                    )
                region_seed = F.normalize(
                    torch.cat(
                        [x.reshape(1, -1) for x in region_seed_parts],
                        dim=0,
                    ).mean(dim=0, keepdim=True),
                    dim=-1,
                )
                region_feat, selected_regions = select_adaptive_region_feature_51(
                    records,
                    region_seed,
                    top_regions=top_regions,
                    temperature=region_temperature,
                )

            channel_features: Dict[str, torch.Tensor] = {
                "reference": reference_feat,
                "text": cirr_feat,
                "base": base_query,
            }
            if qwen_feat is not None:
                channel_features["qwen"] = F.normalize(
                    qwen_feat.float(),
                    dim=-1,
                )
            if region_feat is not None:
                channel_features["region"] = F.normalize(
                    region_feat.float(),
                    dim=-1,
                )

            base_weights = {
                "reference": ref_weight,
                "text": text_weight,
                "base": base_weight,
                "qwen": qwen_weight,
                "region": region_weight,
            }

            for protocol in protocols:
                candidate_indices = _candidate_ids_for_protocol_51(
                    item,
                    gallery_ids,
                    id2idx,
                    protocol,
                    ref_id,
                )

                rankings_by_channel: Dict[str, List[str]] = {}
                confidences: Dict[str, float] = {}

                for channel, feature in channel_features.items():
                    sims = gallery_feats @ feature.squeeze(0)
                    top_ids, top_values = _top_ranked_51(
                        sims,
                        gallery_ids,
                        candidate_indices,
                        topk=max(component_topk, confidence_topk),
                    )
                    rankings_by_channel[channel] = top_ids[:component_topk]
                    confidences[channel] = _channel_confidence_51(
                        top_values[:max(2, min(confidence_topk, len(top_values)))]
                    )

                fused_ranking, effective_weights = weighted_adaptive_rrf_51(
                    rankings=rankings_by_channel,
                    confidences=confidences,
                    base_weights=base_weights,
                    rrf_k=rrf_k,
                )

                ranked = fused_ranking[:output_topk]
                state = states[protocol]
                state["rankings"][query_key] = ranked

                if selected_regions:
                    state["selected_regions"][query_key] = selected_regions
                state["effective_weights"][query_key] = effective_weights
                state["channel_confidence"][query_key] = confidences

                positives = get_relevant_ids(item)
                if positives:
                    update_metrics(state["results"], ranked, positives)
                state["total"] += 1

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = (
                    state["skip_reasons"].get(reason, 0) + 1
                )
            print(
                f"[WARN] {ADAPTIVE_RRF_LABEL} failed for "
                f"query={query_key!r}, reference={ref_id!r}: {reason}"
            )

    return {
        protocol: (
            state["results"],
            state["total"],
            state["skipped"],
            state["rankings"],
            state["skip_reasons"],
            {
                "selected_regions": state["selected_regions"],
                "effective_weights": state["effective_weights"],
                "channel_confidence": state["channel_confidence"],
            },
        )
        for protocol, state in states.items()
    }


# ============================================================
# Evaluation protocols
# ============================================================

FULL_GALLERY = "full_gallery"
CIRR_SUBSET = "cirr_subset"


def evaluate_cross_modal_multi_alpha(
    data: List[dict],
    reference_image_folder: str,
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    get_image_feature,
    get_text_feature,
    model_name: str,
    alphas: Iterable[float],
    protocols: Tuple[str, ...],
):
    """Evaluate all alpha values and both protocols with one query-feature pass."""
    id2idx = build_id_index(gallery_ids)
    alphas = list(alphas)

    ref_cache: Dict[str, Optional[torch.Tensor]] = {}
    text_cache: Dict[str, Optional[torch.Tensor]] = {}
    for item in tqdm(data, desc=f"{model_name}: cache queries"):
        ref_id = item["reference_id"]
        caption = normalize_prompt(item["caption"])
        if ref_id not in ref_cache:
            image = load_image(reference_image_folder, ref_id)
            ref_cache[ref_id] = None if image is None else get_image_feature(image)
        if caption not in text_cache:
            text_cache[caption] = get_text_feature(caption) if caption else None

    states = {
        (alpha, protocol): {
            "results": init_results(), "rankings": {},
            "total": 0, "skipped": 0, "skip_reasons": {},
        }
        for alpha in alphas for protocol in protocols
    }

    for item in tqdm(data, desc=f"{model_name}: similarity/ranking"):
        ref_id = item["reference_id"]
        query_key = get_query_key(item)
        caption = normalize_prompt(item["caption"])
        ref_feat = ref_cache.get(ref_id)
        text_feat = text_cache.get(caption)
        if ref_feat is None or text_feat is None:
            reason = "missing_reference_embedding" if ref_feat is None else "empty_or_missing_text_embedding"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            continue
        try:
            ref_feat = F.normalize(ref_feat.float(), dim=-1)
            text_feat = F.normalize(text_feat.float(), dim=-1)
            if ref_feat.shape[-1] != gallery_feats.shape[-1] or text_feat.shape[-1] != gallery_feats.shape[-1]:
                raise ValueError(
                    f"{model_name}: query/gallery dimensions mismatch: "
                    f"ref={ref_feat.shape[-1]}, text={text_feat.shape[-1]}, gallery={gallery_feats.shape[-1]}"
                )
            sims_text = gallery_feats @ text_feat.squeeze(0)
            sims_ref = gallery_feats @ ref_feat.squeeze(0)
            positives = get_relevant_ids(item)
            for alpha in alphas:
                sims = alpha * sims_text + (1.0 - alpha) * sims_ref
                for protocol in protocols:
                    ranked_ids = rank_from_sims(
                        sims, gallery_ids, id2idx,
                        exclude_ids=[ref_id],
                        restrict_ids=protocol_restrict_ids(item, protocol),
                    )
                    state = states[(alpha, protocol)]
                    state["rankings"][query_key] = ranked_ids[:50]
                    if positives:
                        update_metrics(state["results"], ranked_ids, positives)
                    state["total"] += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for state in states.values():
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
            print(f"[WARN] {model_name} failed for query={query_key!r}: {reason}")

    return {
        alpha: {
            protocol: (
                states[(alpha, protocol)]["results"],
                states[(alpha, protocol)]["total"],
                states[(alpha, protocol)]["skipped"],
                states[(alpha, protocol)]["rankings"],
                states[(alpha, protocol)]["skip_reasons"],
            ) for protocol in protocols
        } for alpha in alphas
    }

def get_evaluation_protocols(dataset: str) -> Tuple[str, ...]:
    """CIRR is always evaluated under both official ranking views.

    CIRCO has only a full-gallery protocol.
    The caller must never select one protocol through a CLI switch.
    """
    if dataset.lower() == "cirr":
        return (FULL_GALLERY, CIRR_SUBSET)
    return (FULL_GALLERY,)


def protocol_title(protocol: str) -> str:
    if protocol == CIRR_SUBSET:
        return "CIRR SUBSET"
    return "FULL GALLERY"


def protocol_restrict_ids(item: dict, protocol: str) -> Optional[List[str]]:
    if protocol == CIRR_SUBSET:
        members = item.get("members") or []
        return list(members)
    return None

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optimized CIRR/CIRCO image retrieval evaluation. "
            "CIRR is ALWAYS evaluated as both full-gallery and subset; "
            "CIRCO is evaluated as full-gallery."
        )
    )

    parser.add_argument("--dataset", choices=["cirr", "circo"], required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument(
        "--split", choices=["auto", "train", "val", "test", "unknown"], default="auto"
    )
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--strict_evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference_image_folder", default=None)
    parser.add_argument("--ground_truth_path", default=None)
    parser.add_argument(
        "--models", nargs="+", default=["clip"],
        choices=["clip", "searle", "open_clip", "siglip", "clip_beta", "blip", "align", "qwen3", "adaptive_rrf"],
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--generated_captions_path", default=None)
    parser.add_argument("--caption_fallback", choices=["auto", "generated", "skip", "error"], default="auto")
    parser.add_argument("--siglip_path", default=None)
    parser.add_argument("--qwen3_path", default=None)
    parser.add_argument("--qwen3_query_instruction", default=QWEN3_DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--qwen3_image_batch_size", type=int, default=1)
    parser.add_argument("--qwen3_max_length", type=int, default=8192)
    parser.add_argument("--qwen3_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--qwen3_flash_attention", action="store_true")
    parser.add_argument("--qwen3_code_path", default=None)

    # Adaptive Multi-Signal RRF (new research method)
    parser.add_argument("--qwen_captions_path", default=None)
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--sam2_repo_path", default=None, help="Local SAM2 repository root; optional if SAM2 is installed in the current Python environment.")
    parser.add_argument(
        "--sam2_config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--adaptive_rrf_rrf_k", type=int, default=60)
    parser.add_argument("--adaptive_rrf_component_topk", type=int, default=200)
    parser.add_argument("--adaptive_rrf_output_topk", type=int, default=50)
    parser.add_argument("--adaptive_rrf_confidence_topk", type=int, default=50)
    parser.add_argument("--adaptive_rrf_top_regions", type=int, default=3)
    parser.add_argument("--adaptive_rrf_region_temperature", type=float, default=0.07)
    parser.add_argument("--adaptive_rrf_ref_weight", type=float, default=0.50)
    parser.add_argument("--adaptive_rrf_text_weight", type=float, default=0.70)
    parser.add_argument("--adaptive_rrf_base_weight", type=float, default=1.00)
    parser.add_argument("--adaptive_rrf_qwen_weight", type=float, default=0.50)
    parser.add_argument("--adaptive_rrf_region_weight", type=float, default=0.80)
    parser.add_argument("--adaptive_rrf_sam_max_regions", type=int, default=12)
    parser.add_argument("--adaptive_rrf_sam_min_area_ratio", type=float, default=0.01)
    parser.add_argument("--adaptive_rrf_sam_max_area_ratio", type=float, default=0.70)
    parser.add_argument(
        "--adaptive_rrf_sam_background_mode",
        choices=["original", "masked", "white"],
        default="masked",
    )
    parser.add_argument("--adaptive_rrf_sam_padding_ratio", type=float, default=0.08)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument(
        "--searle_path",
        default=r"C:\Users\user\Desktop\Python\ImageRetrieval\models_download\SEARLE",
    )
    parser.add_argument("--embedding_cache_dir", default="./embedding_cache")
    parser.add_argument("--blip_path", default=None)
    parser.add_argument("--blip_batch_size", type=int, default=16)
    parser.add_argument("--blip_text_batch_size", type=int, default=32)
    # Shared CLIP text batch size used by Adaptive-RRF. Kept separate for backward compatibility.
    parser.add_argument("--clip_text_batch_size", type=int, default=64)
    parser.add_argument("--blip_itm_topk", type=int, default=0)
    parser.add_argument("--blip_itm_batch_size", type=int, default=8)
    parser.add_argument("--open_clip_batch_size", type=int, default=32)
    parser.add_argument("--siglip_batch_size", type=int, default=16)
    parser.add_argument("--align_path", default="./models_download/align-base")
    parser.add_argument("--align_batch_size", type=int, default=8)
    parser.add_argument("--embedding_checkpoint_every", type=int, default=250)
    parser.add_argument("--force_rebuild_embeddings", action="store_true")
    parser.add_argument("--open_clip_model", default="ViT-H-14")
    parser.add_argument("--open_clip_pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--require_cirr_gt", action="store_true")
    parser.add_argument("--searle_prompt_ensemble", action="store_true")
    parser.add_argument("--searle_hybrid_weights", nargs="+", type=float, default=[1.0])
    # Backward-compatible, intentionally ignored. Results are ALWAYS both protocols.
    parser.add_argument("--cirr_subset", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    set_deterministic_seed(args.seed)

    for value in (*args.alphas, *args.betas, *args.searle_hybrid_weights):
        if not 0.0 <= value <= 1.0:
            raise ValueError("alpha/beta/SEARLE weights must be in [0,1]")
    if any(x < 1 for x in (
        args.open_clip_batch_size,
        args.siglip_batch_size,
        args.align_batch_size,
        args.qwen3_image_batch_size,
        args.blip_batch_size,
        args.blip_text_batch_size,
        args.blip_itm_batch_size,
    )):
        raise ValueError("Batch sizes must be >= 1")
    if args.qwen3_max_length < 128:
        raise ValueError("--qwen3_max_length must be >= 128")
    if args.blip_itm_topk < 0:
        raise ValueError("--blip_itm_topk must be >= 0")
    if "align" in args.models and not args.align_path:
        raise ValueError("--align_path is required when align is selected")
    if "qwen3" in args.models and not args.qwen3_path:
        raise ValueError("--qwen3_path is required when qwen3 is selected")
    if "adaptive_rrf" in args.models:
        if args.dataset != "cirr":
            raise ValueError("adaptive_rrf is currently defined for CIRR only.")
        if not args.qwen_captions_path:
            raise ValueError("--qwen_captions_path is required for adaptive_rrf.")
        if not args.sam2_checkpoint:
            raise ValueError("--sam2_checkpoint is required for adaptive_rrf.")
        if args.adaptive_rrf_rrf_k <= 0:
            raise ValueError("--adaptive_rrf_rrf_k must be > 0")
        if args.adaptive_rrf_component_topk < 1:
            raise ValueError("--adaptive_rrf_component_topk must be >= 1")
        if args.adaptive_rrf_output_topk < 1:
            raise ValueError("--adaptive_rrf_output_topk must be >= 1")
        if args.adaptive_rrf_confidence_topk < 2:
            raise ValueError("--adaptive_rrf_confidence_topk must be >= 2")
        if args.adaptive_rrf_top_regions < 1:
            raise ValueError("--adaptive_rrf_top_regions must be >= 1")
        if args.adaptive_rrf_region_temperature <= 0:
            raise ValueError("--adaptive_rrf_region_temperature must be > 0")
        adaptive_weights = (
            args.adaptive_rrf_ref_weight,
            args.adaptive_rrf_text_weight,
            args.adaptive_rrf_base_weight,
            args.adaptive_rrf_qwen_weight,
            args.adaptive_rrf_region_weight,
        )
        if any(w < 0 for w in adaptive_weights) or sum(adaptive_weights) <= 0:
            raise ValueError("Adaptive-RRF weights must be non-negative and not all zero")
        if not (0.0 <= args.adaptive_rrf_sam_min_area_ratio < args.adaptive_rrf_sam_max_area_ratio <= 1.0):
            raise ValueError("Invalid Adaptive-RRF SAM area ratios")
        if args.adaptive_rrf_sam_padding_ratio < 0:
            raise ValueError("--adaptive_rrf_sam_padding_ratio must be >= 0")
    if "blip" in args.models and not args.blip_path:
        raise ValueError("--blip_path is required when blip is selected")
    if "clip_beta" in args.models and not args.generated_captions_path:
        raise ValueError("--generated_captions_path is required for clip_beta")
    if args.dataset == "cirr" and args.cirr_subset:
        print("[INFO] --cirr_subset is deprecated and ignored; both protocols are always evaluated.")

    def infer_split() -> str:
        if args.split != "auto":
            return args.split
        text_value = f"{args.json_path} {args.image_folder}".lower().replace("\\", "/")
        if "test1" in text_value or "/test/" in text_value or text_value.endswith("/test"):
            return "test"
        if "val" in text_value or "dev" in text_value:
            return "val"
        if "train" in text_value:
            return "train"
        return "unknown"

    split_name = infer_split()
    protocols = get_evaluation_protocols(args.dataset)
    print(f"Split: {split_name}")
    print(f"Device: {DEVICE}")
    print("Selected models:", ", ".join(args.models))
    print("Evaluation protocols:", ", ".join(protocol_title(p) for p in protocols))
    print(f"Loading {args.dataset.upper()}...")

    detected, data = load_dataset(args.json_path)
    if detected != args.dataset:
        raise ValueError(f"Dataset mismatch: argument={args.dataset}, detected={detected}")

    is_new_cirr_schema = (
        args.dataset == "cirr" and any(item.get("annotation_format") == "candidate_group" for item in data)
    )
    if args.dataset == "cirr":
        require_gt = args.require_cirr_gt or is_new_cirr_schema
        validate_cirr_annotations(data, require_ground_truth=require_gt)

    if args.ground_truth_path:
        gt_data = load_optional_ground_truth(args.ground_truth_path)
        data, matched_gt = attach_ground_truth(data, gt_data)
        print(f"Ground truth: loaded {len(gt_data or [])} entries; matched {matched_gt}/{len(data)} queries")

    generated_captions = {}
    if args.generated_captions_path:
        generated_captions = load_generated_captions(args.generated_captions_path)
        print(f"Generated captions loaded: {len(generated_captions)}")
    data, _caption_stats = apply_caption_fallback(
        data, generated_captions=generated_captions, mode=args.caption_fallback
    )

    print(f"Queries: {len(data)}")
    reference_image_folder = args.reference_image_folder or args.image_folder
    if args.dataset == "cirr":
        validate_reference_images(data, reference_image_folder, fail_on_missing=args.strict_evaluation)

    gallery_ids = scan_gallery_ids(args.image_folder)
    print(f"Gallery images: {len(gallery_ids)}")
    if args.dataset == "cirr":
        validate_cirr_gallery_coverage(data, gallery_ids, require_ground_truth=False)

    protocol_results: Dict[str, Dict[str, dict]] = {protocol: {} for protocol in protocols}

    def record_result(protocol, label, results, total, skipped, rankings=None):
        if any(bool(get_relevant_ids(x)) for x in data):
            summary = summarize_results(results)
            protocol_results[protocol][label] = summary
            print_metric_summary(f"{protocol_title(protocol)} | {label}", summary, total, skipped)
        enforce_complete_evaluation(
            f"{protocol_title(protocol)} | {label}", len(data), total, skipped, args.strict_evaluation
        )
        return summary if any(bool(get_relevant_ids(x)) for x in data) else None

    # -------------------------------------------------------- CLIP shared backbone
    need_clip = any(x in args.models for x in ("clip", "searle", "clip_beta", "adaptive_rrf"))
    clip_model = clip_preprocess = clip_cache = clip_cache_raw = None
    clip_gallery_ids = clip_gallery_feats = None

    if need_clip:
        print("\n=== Loading CLIP ===")
        clip_model, clip_preprocess = load_clip_model(args.clip_model)
        cache_path = clip_embedding_cache_path(
            args.embedding_cache_dir, args.clip_model, args.dataset, split_name
        )
        partial_cache_path = cache_path.with_suffix(cache_path.suffix + ".partial")
        if args.force_rebuild_embeddings and partial_cache_path.exists():
            partial_cache_path.unlink()

        cached = None if args.force_rebuild_embeddings else load_clip_image_cache(
            cache_path, gallery_ids, args.clip_model, args.dataset, split_name
        )
        if cached is not None:
            clip_cache, clip_cache_raw = cached
            print(f"Loaded CLIP image embeddings from cache: {cache_path}")
        else:
            clip_cache, clip_cache_raw = build_clip_image_cache(
                gallery_ids, args.image_folder, clip_model, clip_preprocess,
                partial_cache_path=partial_cache_path,
                model_name=args.clip_model,
                checkpoint_every=max(1, args.embedding_checkpoint_every),
                cache_dataset=args.dataset, cache_split=split_name,
            )
            save_clip_image_cache(
                cache_path, gallery_ids, clip_cache, clip_cache_raw,
                args.clip_model, args.dataset, split_name
            )
            if partial_cache_path.exists():
                partial_cache_path.unlink()
        clip_gallery_ids, clip_gallery_feats = stack_feature_cache(clip_cache)
        print(f"CLIP gallery embeddings: {len(clip_gallery_ids)}")

    if "clip" in args.models:
        print("\n=== CLIP ===")
        outputs_by_alpha = evaluate_cross_modal_multi_alpha(
            data=data, reference_image_folder=reference_image_folder,
            gallery_ids=clip_gallery_ids, gallery_feats=clip_gallery_feats,
            get_image_feature=lambda img: clip_image_embedding(img, clip_model, clip_preprocess),
            get_text_feature=lambda txt: clip_text_embedding(txt, clip_model),
            model_name="CLIP", alphas=args.alphas, protocols=protocols,
        )
        for alpha, by_protocol in outputs_by_alpha.items():
            label = f"CLIP alpha={alpha:.2f}"
            for protocol, payload in by_protocol.items():
                results, total, skipped, rankings, skip_reasons = payload
                record_result(protocol, label, results, total, skipped, rankings)

    # -------------------------------------------------------- BLIP
    if "blip" in args.models:
        print("\n=== BLIP-ITM ===")
        blip_model, blip_processor = load_blip_itm_model(args.blip_path)
        resolved = str(Path(args.blip_path).expanduser().resolve())
        try:
            st = (Path(resolved) / "config.json").stat()
            fingerprint = f"{resolved}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            fingerprint = resolved
        blip_hash = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
        blip_key = f"blip_itm_base_coco_{blip_hash}"
        blip_cache = build_blip_image_cache(
            gallery_ids, args.image_folder, blip_model, blip_processor,
            cache_dir=args.embedding_cache_dir, dataset=args.dataset, split=split_name,
            model_key=blip_key, batch_size=args.blip_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )
        blip_ids, blip_feats = stack_feature_cache(blip_cache)
        blip_outputs = evaluate_blip(
            data=data,
            image_folder=args.image_folder,
            reference_image_folder=reference_image_folder,
            gallery_ids=blip_ids,
            gallery_feats=blip_feats,
            model=blip_model,
            processor=blip_processor,
            protocols=protocols,
            text_batch_size=args.blip_text_batch_size,
            itm_topk=args.blip_itm_topk,
            itm_batch_size=args.blip_itm_batch_size,
        )
        label = "BLIP" + (f" + ITM@{args.blip_itm_topk}" if args.blip_itm_topk > 0 else "")
        for protocol, payload in blip_outputs.items():
            results, total, skipped, rankings, skip_reasons = payload
            record_result(protocol, label, results, total, skipped, rankings)
        del blip_model, blip_processor, blip_cache, blip_feats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------- SEARLE
    if "searle" in args.models:
        print("\n=== SEARLE ===")
        if clip_model is None or clip_cache_raw is None:
            raise RuntimeError("SEARLE requires CLIP and its raw image cache")
        prompt_templates = ["a photo of $ {caption}"]
        if args.searle_prompt_ensemble:
            prompt_templates = [
                "a photo of $ {caption}",
                "an image of $ {caption}",
                "$ {caption}",
            ]
        searle, encode_with_pseudo_tokens = load_searle_model(
            clip_model_name=args.clip_model, searle_path=args.searle_path
        )
        clip_text_cache = None
        if any(w < 0.999999 for w in args.searle_hybrid_weights):
            clip_text_cache = {}
            for item in tqdm(data, desc="SEARLE: CLIP text cache"):
                caption = normalize_prompt(item["caption"])
                if caption and caption not in clip_text_cache:
                    clip_text_cache[caption] = clip_text_embedding(caption, clip_model)
        outputs = evaluate_searle(
            data=data,
            searle=searle,
            encode_with_pseudo_tokens=encode_with_pseudo_tokens,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            image_features_cache=clip_cache_raw,
            gallery_ids=clip_gallery_ids,
            reference_image_folder=reference_image_folder,
            gallery_feats=clip_gallery_feats,
            protocols=protocols,
            prompt_templates=prompt_templates,
            hybrid_weights=args.searle_hybrid_weights,
            clip_text_cache=clip_text_cache,
        )
        for weight, by_protocol in outputs.items():
            label = f"SEARLE w={weight:.2f}"
            for protocol, payload in by_protocol.items():
                results, predictions, total, skipped, skip_reasons = payload
                record_result(protocol, label, results, total, skipped)
                if args.dataset == "cirr":
                    rankings = {
                        str(p.get("query_id")): p.get("ranking", []) for p in predictions
                    }
                    output_path = Path(args.json_path).with_name(
                        Path(args.json_path).stem
                        + f"_searle_w{weight:.2f}_{protocol}_predictions.json"
                    )
                    save_cirr_test_predictions(
                        path=output_path,
                        data=data,
                        model_name=label,
                        split=split_name,
                        rankings=rankings,
                        total=total,
                        skipped=skipped,
                        skip_reasons=skip_reasons,
                        extra={"weight": weight, "protocol": protocol_title(protocol)},
                    )
        del searle, encode_with_pseudo_tokens
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------- CLIP-beta
    if "clip_beta" in args.models:
        print("\n=== CLIP-beta ===")
        beta_outputs = evaluate_clip_beta(
            data=data,
            image_folder=args.image_folder,
            reference_image_folder=reference_image_folder,
            gallery_ids=clip_gallery_ids,
            gallery_feats=clip_gallery_feats,
            clip_model=clip_model,
            clip_model_preprocess=clip_preprocess,
            image_features_cache=clip_cache,
            generated_captions=generated_captions,
            alphas=args.alphas,
            betas=args.betas,
            protocols=protocols,
        )
        for (alpha, beta), by_protocol in beta_outputs.items():
            label = f"CLIP-beta a={alpha:.2f} b={beta:.2f}"
            for protocol, payload in by_protocol.items():
                results, total, skipped, rankings, skip_reasons = payload
                record_result(protocol, label, results, total, skipped, rankings)

    # -------------------------------------------------------- OpenCLIP
    if "open_clip" in args.models:
        print("\n=== OpenCLIP ===")
        oc_model, oc_preprocess = load_open_clip_model(args.open_clip_model, args.open_clip_pretrained)
        tokenizer = open_clip.get_tokenizer(args.open_clip_model)
        oc_key = f"openclip_{args.open_clip_model}_{args.open_clip_pretrained}"
        oc_cache = build_open_clip_image_cache(
            gallery_ids, args.image_folder, oc_model, oc_preprocess,
            cache_dir=args.embedding_cache_dir, dataset=args.dataset, split=split_name,
            model_key=oc_key, batch_size=args.open_clip_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )
        oc_ids, oc_feats = stack_feature_cache(oc_cache)
        outputs_by_alpha = evaluate_cross_modal_multi_alpha(
            data=data, reference_image_folder=reference_image_folder,
            gallery_ids=oc_ids, gallery_feats=oc_feats,
            get_image_feature=lambda img: open_clip_image_embedding(img, oc_model, oc_preprocess),
            get_text_feature=lambda txt: open_clip_text_embedding(txt, oc_model, tokenizer),
            model_name="OpenCLIP", alphas=args.alphas, protocols=protocols,
        )
        for alpha, by_protocol in outputs_by_alpha.items():
            label = f"OpenCLIP alpha={alpha:.2f}"
            for protocol, payload in by_protocol.items():
                results, total, skipped, rankings, skip_reasons = payload
                record_result(protocol, label, results, total, skipped, rankings)
        del oc_model, oc_preprocess, oc_cache, oc_feats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------- SigLIP
    if "siglip" in args.models:
        print("\n=== SigLIP ===")
        sig_model, sig_processor = load_siglip_model(args.siglip_path)
        sig_key = f"siglip_{Path(args.siglip_path).resolve()}"
        sig_cache = build_siglip_image_cache(
            gallery_ids, args.image_folder, sig_model, sig_processor,
            cache_dir=args.embedding_cache_dir, dataset=args.dataset, split=split_name,
            model_key=sig_key, batch_size=args.siglip_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )
        sig_ids, sig_feats = stack_feature_cache(sig_cache)
        outputs_by_alpha = evaluate_cross_modal_multi_alpha(
            data=data, reference_image_folder=reference_image_folder,
            gallery_ids=sig_ids, gallery_feats=sig_feats,
            get_image_feature=lambda img: siglip_image_embedding(img, sig_model, sig_processor),
            get_text_feature=lambda txt: siglip_text_embedding(txt, sig_model, sig_processor),
            model_name="SigLIP", alphas=args.alphas, protocols=protocols,
        )
        for alpha, by_protocol in outputs_by_alpha.items():
            label = f"SigLIP alpha={alpha:.2f}"
            for protocol, payload in by_protocol.items():
                results, total, skipped, rankings, skip_reasons = payload
                record_result(protocol, label, results, total, skipped, rankings)
        del sig_model, sig_processor, sig_cache, sig_feats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------- ALIGN
    if "align" in args.models:
        print("\n=== ALIGN ===")
        align_model, align_processor = load_align_model(args.align_path)
        resolved = str(Path(args.align_path).expanduser().resolve())
        try:
            st = (Path(resolved) / "config.json").stat()
            fingerprint = f"{resolved}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            fingerprint = resolved
        align_hash = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
        align_key = f"align_base_{align_hash}"
        align_cache = build_align_image_cache(
            gallery_ids, args.image_folder, align_model, align_processor,
            cache_dir=args.embedding_cache_dir, dataset=args.dataset, split=split_name,
            model_key=align_key, batch_size=args.align_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )
        align_ids, align_feats = stack_feature_cache(align_cache)
        outputs_by_alpha = evaluate_cross_modal_multi_alpha(
            data=data, reference_image_folder=reference_image_folder,
            gallery_ids=align_ids, gallery_feats=align_feats,
            get_image_feature=lambda img: align_image_embedding(img, align_model, align_processor),
            get_text_feature=lambda txt: align_text_embedding(txt, align_model, align_processor),
            model_name="ALIGN", alphas=args.alphas, protocols=protocols,
        )
        for alpha, by_protocol in outputs_by_alpha.items():
            label = f"ALIGN alpha={alpha:.2f}"
            for protocol, payload in by_protocol.items():
                results, total, skipped, rankings, skip_reasons = payload
                record_result(protocol, label, results, total, skipped, rankings)
        del align_model, align_processor, align_cache, align_feats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------- Qwen3
    if "qwen3" in args.models:
        print("\n=== Qwen3-VL-Embedding-2B ===")
        qwen3_model = load_qwen3_model(
            args.qwen3_path,
            query_instruction=args.qwen3_query_instruction,
            max_length=args.qwen3_max_length,
            qwen3_dtype=args.qwen3_dtype,
            use_flash_attention=args.qwen3_flash_attention,
            code_path=args.qwen3_code_path,
        )
        qwen3_model = _make_qwen3_process_adapter(qwen3_model, args.qwen3_query_instruction)
        seed = str(Path(args.qwen3_path).resolve()) + "|qwen3"
        qhash = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        qkey = f"qwen3_vl_embedding_{qhash}"
        qcache = build_qwen3_image_cache(
            gallery_ids, args.image_folder, qwen3_model,
            cache_dir=args.embedding_cache_dir, dataset=args.dataset, split=split_name,
            model_key=qkey, batch_size=args.qwen3_image_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )
        qids, qfeats = stack_feature_cache(qcache)
        outputs = evaluate_qwen3(
            data=data,
            gallery_ids=qids,
            gallery_feats=qfeats,
            reference_image_folder=reference_image_folder,
            qwen3_model=qwen3_model,
            protocols=protocols,
            query_instruction=args.qwen3_query_instruction,
        )
        label = "Qwen3-VL-Embedding-2B"
        for protocol, payload in outputs.items():
            results, total, skipped, rankings, skip_reasons = payload
            record_result(protocol, label, results, total, skipped, rankings)
            if args.dataset == "cirr":
                qwen_output = Path(args.json_path).with_name(
                    Path(args.json_path).stem + f"_qwen3_{protocol}_predictions.json"
                )
                save_cirr_test_predictions(
                    path=qwen_output,
                    data=data,
                    model_name=label,
                    split=split_name,
                    rankings=rankings,
                    total=total,
                    skipped=skipped,
                    skip_reasons=skip_reasons,
                    extra={
                        "query_type": "joint_image_text",
                        "query_instruction": args.qwen3_query_instruction,
                        "protocol": protocol_title(protocol),
                    },
                )
        del qwen3_model, qcache, qfeats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------- Adaptive Multi-Signal RRF
    if "adaptive_rrf" in args.models:
        print("\n=== ADAPTIVE MULTI-SIGNAL RRF: CLIP + Qwen + SAM2 ===")
        qwen_caption_map = load_qwen_caption_map(args.qwen_captions_path)
        print(f"Qwen captions loaded: {len(qwen_caption_map)}")

        unique_qwen_captions = sorted({
            normalize_prompt(qwen_caption_map.get(item["reference_id"], ""))
            for item in data
            if normalize_prompt(qwen_caption_map.get(item["reference_id"], ""))
        })
        qwen_text_cache = clip_text_embedding_batch_51(
            unique_qwen_captions,
            clip_model,
            batch_size=args.clip_text_batch_size,
        )

        unique_refs = sorted({
            item["reference_id"] for item in data if item.get("reference_id")
        })
        print(f"Unique reference images for Adaptive-RRF: {len(unique_refs)}")

        sam_model_51, sam_generator_51 = load_sam2_51(
            args.sam2_checkpoint,
            args.sam2_config,
            args.sam2_repo_path,
        )
        sam_checkpoint_path = Path(args.sam2_checkpoint).expanduser().resolve()
        try:
            sam_stat = sam_checkpoint_path.stat()
            sam_fingerprint = (
                f"{sam_checkpoint_path}|{sam_stat.st_size}|{sam_stat.st_mtime_ns}"
            )
        except OSError:
            sam_fingerprint = str(sam_checkpoint_path)
        sam_hash_51 = hashlib.sha1(
            sam_fingerprint.encode("utf-8")
        ).hexdigest()[:12]

        adaptive_region_cache = build_adaptive_region_cache_51(
            reference_ids=unique_refs,
            image_folder=args.image_folder,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            sam_generator=sam_generator_51,
            sam_key=sam_hash_51,
            clip_model_name=args.clip_model,
            cache_dir=args.embedding_cache_dir,
            dataset=args.dataset,
            split=split_name,
            max_regions=args.adaptive_rrf_sam_max_regions,
            min_area_ratio=args.adaptive_rrf_sam_min_area_ratio,
            max_area_ratio=args.adaptive_rrf_sam_max_area_ratio,
            background_mode=args.adaptive_rrf_sam_background_mode,
            padding_ratio=args.adaptive_rrf_sam_padding_ratio,
            checkpoint_every=args.embedding_checkpoint_every,
            force=args.force_rebuild_embeddings,
        )

        del sam_model_51, sam_generator_51
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        adaptive_cirr_captions = sorted({
            normalize_prompt(x.get("caption", ""))
            for x in data
            if normalize_prompt(x.get("caption", ""))
        })
        adaptive_cirr_text_cache = clip_text_embedding_batch_51(
            adaptive_cirr_captions,
            clip_model,
            batch_size=args.clip_text_batch_size,
        )

        adaptive_outputs = evaluate_adaptive_rrf_51(
            data=data,
            gallery_ids=clip_gallery_ids,
            gallery_feats=clip_gallery_feats,
            clip_gallery_cache=clip_cache,
            clip_text_cache=adaptive_cirr_text_cache,
            qwen_caption_map=qwen_caption_map,
            qwen_text_cache=qwen_text_cache,
            region_cache=adaptive_region_cache,
            protocols=protocols,
            rrf_k=args.adaptive_rrf_rrf_k,
            component_topk=args.adaptive_rrf_component_topk,
            output_topk=args.adaptive_rrf_output_topk,
            top_regions=args.adaptive_rrf_top_regions,
            region_temperature=args.adaptive_rrf_region_temperature,
            confidence_topk=args.adaptive_rrf_confidence_topk,
            ref_weight=args.adaptive_rrf_ref_weight,
            text_weight=args.adaptive_rrf_text_weight,
            base_weight=args.adaptive_rrf_base_weight,
            qwen_weight=args.adaptive_rrf_qwen_weight,
            region_weight=args.adaptive_rrf_region_weight,
        )

        for protocol, payload in adaptive_outputs.items():
            (
                results,
                total,
                skipped,
                rankings,
                skip_reasons,
                diagnostics,
            ) = payload

            label = ADAPTIVE_RRF_LABEL
            record_result(
                protocol,
                label,
                results,
                total,
                skipped,
                rankings,
            )

            if args.dataset == "cirr":
                output_path = Path(args.json_path).with_name(
                    Path(args.json_path).stem
                    + f"_adaptive_rrf_{protocol}_predictions.json"
                )
                save_cirr_test_predictions(
                    path=output_path,
                    data=data,
                    model_name=label,
                    split=split_name,
                    rankings=rankings,
                    total=total,
                    skipped=skipped,
                    skip_reasons=skip_reasons,
                    extra={
                        "protocol": protocol_title(protocol),
                        "method_family": "adaptive_multisignal_rrf",
                        "rrf_k": args.adaptive_rrf_rrf_k,
                        "component_topk": args.adaptive_rrf_component_topk,
                        "output_topk": args.adaptive_rrf_output_topk,
                        "confidence_topk": args.adaptive_rrf_confidence_topk,
                        "top_regions": args.adaptive_rrf_top_regions,
                        "region_temperature": args.adaptive_rrf_region_temperature,
                        "base_weights": {
                            "reference": args.adaptive_rrf_ref_weight,
                            "text": args.adaptive_rrf_text_weight,
                            "base": args.adaptive_rrf_base_weight,
                            "qwen": args.adaptive_rrf_qwen_weight,
                            "region": args.adaptive_rrf_region_weight,
                        },
                        "diagnostics": diagnostics,
                    },
                )

    # -------------------------------------------------------- Final tables
    has_gt = any(bool(get_relevant_ids(x)) for x in data)
    print("\n" + "=" * 170)
    print(f"FINAL RESULTS - {args.dataset.upper()}")
    print("=" * 170)

    header = (
        f"{'Model':<34}{'MRR':>9}{'mAP@5':>10}{'mAP@10':>10}{'mAP@50':>10}"
        f"{'P@1':>9}{'P@5':>9}{'P@10':>10}{'P@50':>10}"
        f"{'R@1':>9}{'R@5':>9}{'R@10':>10}{'R@50':>10}"
    )

    for protocol in protocols:
        print("\n" + "-" * 170)
        print(f"PROTOCOL: {protocol_title(protocol)}")
        print("-" * 170)
        print(header)
        print("-" * len(header))
        for name, result in protocol_results[protocol].items():
            print(
                f"{name:<34}"
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

    if not has_gt:
        print(
            "\nNOTE: No ground-truth positives/target_id are available. "
            "Rankings were generated but retrieval metrics cannot be computed locally."
        )

    metrics_output = (
        Path(args.metrics_output)
        if args.metrics_output
        else Path(args.json_path).with_name(Path(args.json_path).stem + "_metrics_12_both_protocols.json")
    )
    if has_gt:
        save_metrics_json(metrics_output, args.dataset, split_name, protocol_results)


if __name__ == "__main__":
    main()
