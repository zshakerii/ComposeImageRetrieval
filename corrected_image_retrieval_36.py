
import os
import json
import argparse
import traceback
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

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
    """
    *Parse both the new CIRR JSON schema and the old cap.rc2 schema.*

    *New schema:*
        *candidate_id -> reference image*
        *group        -> candidate subset*
        *target_id    -> hard ground-truth image*
        *caption      -> relative caption*
    """
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
    """
    *Validate CIRR annotations.

    *For test1, img_set.members is mandatory because it defines the candidate
    *subset. Ground truth is optional.
    """
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
    """
    *Build the actual retrieval gallery and a recursive image-path index.
    """
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

    id_keys = ("image_id", "img_id", "reference_img_id", "reference", "id")
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
    """
    *Resolve the effective caption used by retrieval models.

    *mode:
      *auto      : use annotation caption; if empty, use generated caption
                  *for the same reference when available.
      *generated : for empty annotation captions only, require/use generated
                  *caption when available.
      *skip      : keep empty captions; model will skip those queries.
      *error     : fail immediately on an empty annotation caption.

    *The original annotation caption is preserved in ``original_caption``.
    *The selected caption source is stored in ``caption_source``.

    *IMPORTANT:
    *We never replace a valid CIRR annotation caption with a generated caption.
    *The fallback is used ONLY when the annotation caption is empty.
    """
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
    """
    *Return Ground Truth image IDs.*

    *For the new CIRR schema, target_id is the hard target. It is intentionally*
    *treated as the primary positive and is never inferred from group order.*
    """
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
    """Rank gallery IDs by similarity.

    When ``restrict_ids`` is supplied, only those gallery IDs participate.
    Excluded IDs are always removed.
    """
    sims = sims.detach().float().flatten().clone()

    if sims.numel() != len(gallery_ids):
        raise ValueError(
            f"Similarity length ({sims.numel()}) != gallery size ({len(gallery_ids)})"
        )

    excluded = {
        normalize_id(x)
        for x in exclude_ids
        if normalize_id(x)
    }

    if restrict_ids is None:
        valid_mask = torch.ones(len(gallery_ids), dtype=torch.bool)
    else:
        allowed = {
            normalize_id(x)
            for x in restrict_ids
            if normalize_id(x) in id2idx
        }
        valid_mask = torch.tensor(
            [image_id in allowed for image_id in gallery_ids],
            dtype=torch.bool,
        )

    if excluded:
        valid_mask &= torch.tensor(
            [image_id not in excluded for image_id in gallery_ids],
            dtype=torch.bool,
        )

    if not torch.any(valid_mask):
        return []

    sims[~valid_mask] = -float("inf")
    order = torch.argsort(sims, descending=True).cpu().tolist()
    return [gallery_ids[i] for i in order if torch.isfinite(sims[i])]

# ============================================================
# CLIP embedding cache
# ============================================================

def _safe_cache_model_name(model_name: str) -> str:
    return (
        model_name
        .replace("/", "_")
        .replace("\\\\\\\\", "_")
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

    *Example:*
        *embedding_cache/cirr/val/clip_ViT-B_32_image_embeddings.pt*
        *embedding_cache/cirr/test/clip_ViT-B_32_image_embeddings.pt*
    """
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
    """
    *Build CLIP image embeddings with resumable checkpoints.

    *A partial cache is written periodically, so an interruption does not
    *force another full pass over all 8082 images.
    """
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
    hidden = outputs.last_hidden_state
    pooled = hidden[:, 0, :]
    projected = model.text_proj(pooled)
    return F.normalize(projected.float(), dim=-1).cpu()


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

    outputs = model(**inputs, use_itm_head=True)

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
    alphas: List[float],
    subset_protocol: bool,
    text_batch_size: int,
    itm_topk: int,
    itm_batch_size: int,
):
    """Evaluate BLIP retrieval with reusable text/reference embeddings and optional ITM reranking."""
    id2idx = build_id_index(gallery_ids)

    unique_captions = sorted({
        normalize_prompt(item.get("caption", ""))
        for item in data
        if normalize_prompt(item.get("caption", ""))
    })
    text_cache: Dict[str, torch.Tensor] = {}
    print(f"BLIP unique captions: {len(unique_captions)}")

    for start in tqdm(
        range(0, len(unique_captions), text_batch_size),
        desc="BLIP text embeddings",
    ):
        batch = unique_captions[start:start + text_batch_size]
        features = blip_text_features(batch, model, processor)
        for caption, feature in zip(batch, features):
            text_cache[caption] = feature.unsqueeze(0)

    unique_refs = sorted({
        item.get("reference_id", "")
        for item in data
        if item.get("reference_id")
    })
    ref_cache: Dict[str, Optional[torch.Tensor]] = {}
    print(f"BLIP unique references: {len(unique_refs)}")
    for ref_id in tqdm(unique_refs, desc="BLIP reference embeddings"):
        image = load_image(reference_image_folder, ref_id)
        ref_cache[ref_id] = (
            None
            if image is None
            else blip_image_features([image], model, processor)
        )

    outputs = {}
    for alpha in alphas:
        results = init_results()
        rankings: Dict[str, List[str]] = {}
        total = 0
        skipped = 0
        skip_reasons: Dict[str, int] = {}

        for item in tqdm(data, desc=f"BLIP alpha={alpha:.2f}"):
            ref_id = item["reference_id"]
            caption = normalize_prompt(item.get("caption", ""))
            ref_feat = ref_cache.get(ref_id)
            text_feat = text_cache.get(caption)

            if ref_feat is None:
                skipped += 1
                skip_reasons["missing_reference_embedding"] = (
                    skip_reasons.get("missing_reference_embedding", 0) + 1
                )
                continue

            if text_feat is None:
                skipped += 1
                skip_reasons["empty_or_missing_text_embedding"] = (
                    skip_reasons.get("empty_or_missing_text_embedding", 0) + 1
                )
                continue

            try:
                # Shared BLIP retrieval space.
                # alpha=1.0 -> text only; alpha=0.0 -> reference only.
                query = F.normalize(
                    alpha * text_feat.float()
                    + (1.0 - alpha) * ref_feat.float(),
                    dim=-1,
                )
                sims = gallery_feats @ query.squeeze(0)

                restrict_ids = item.get("members", []) if subset_protocol else None
                ranked = rank_from_sims(
                    sims,
                    gallery_ids,
                    id2idx,
                    exclude_ids=[ref_id],
                    restrict_ids=restrict_ids,
                )

                # BLIP ITM is a cross-attention second stage. It is deliberately
                # optional because CIRR captions are relative/editing instructions,
                # not necessarily complete descriptions of the target image.
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

                query_key = get_query_key(item)
                rankings[query_key] = ranked[:50]
                positives = get_relevant_ids(item)
                if positives:
                    update_metrics(results, ranked, positives)
                total += 1

            except Exception as exc:
                skipped += 1
                reason = f"{type(exc).__name__}: {exc}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                print(
                    f"[WARN] BLIP query failed: "
                    f"query={get_query_key(item)!r}, reference={ref_id!r}: {reason}"
                )

        outputs[alpha] = (
            results,
            total,
            skipped,
            rankings,
            skip_reasons,
        )

    return outputs


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
    """
    *Build the composed SEARLE query.

    *IMPORTANT: `$` is NOT a normal character here. It is the placeholder
    *replaced by SEARLE pseudo tokens. The official repository explicitly
    requires the prompt to contain `$`.
    """
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
    image_features_cache_raw: Dict[str, torch.Tensor],
    gallery_ids: List[str],
    reference_image_folder: str,
    gallery_feats: torch.Tensor,
    subset_protocol: bool = False,
    prompt_templates: Optional[List[str]] = None,
    hybrid_weights: Optional[List[float]] = None,
    clip_text_cache: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    *Evaluate/rank SEARLE.

    *If positives are present, metrics are returned. For cap.rc2.test1.json,
    *positives are absent, so predictions are generated without fake metrics.

    *hybrid score:
        *w * SEARLE_similarity + (1-w) * CLIP_text_similarity

    *This is deliberately optional; w=1.0 is pure SEARLE.
    """
    id2idx = build_id_index(gallery_ids)
    weights = hybrid_weights or [1.0]
    outputs = {}
    reference_feature_cache: Dict[str, Optional[torch.Tensor]] = {}

    for weight in weights:
        results = init_results()
        predictions = []
        total = skipped = 0
        skip_reasons = {}

        for item in tqdm(data, desc=f"SEARLE w={weight:.2f}"):
            ref_id = item["reference_id"]
            caption = normalize_prompt(item["caption"])
            raw_ref = image_features_cache_raw.get(ref_id)

            if raw_ref is None and ref_id in reference_feature_cache:
                raw_ref = reference_feature_cache[ref_id]

            if raw_ref is None:
                ref_image = load_image(
                    reference_image_folder,
                    ref_id,
                )
                if ref_image is not None:
                    try:
                        raw_ref = clip_model.encode_image(
                            clip_preprocess(ref_image).unsqueeze(0).to(DEVICE)
                        ).float().cpu()
                        reference_feature_cache[ref_id] = raw_ref
                    except Exception as exc:
                        print(
                            f"[WARN] Reference embedding failed for {ref_id}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        reference_feature_cache[ref_id] = None
                else:
                    reference_feature_cache[ref_id] = None

            if raw_ref is None or not caption:
                skipped += 1
                reason = "missing_reference_embedding" if raw_ref is None else "empty_caption"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                print(
                    f"[WARN] SEARLE query skipped: pairid={item.get('pairid')!r}, "
                    f"reference={ref_id!r}, reason={reason}, caption={caption!r}"
                )
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

                sims_searle = gallery_feats @ q_searle.squeeze(0)
                sims = sims_searle.clone()

                # Optional CLIP text residual. The same CLIP backbone/gallery
                # is used, so the two scores are on the same cosine scale.
                if weight < 0.999999:
                    if clip_text_cache is None:
                        raise RuntimeError("clip_text_cache is required for hybrid SEARLE.")
                    t = clip_text_cache.get(caption)
                    if t is None:
                        raise RuntimeError(f"Missing CLIP text feature for: {caption}")
                    sims_clip = gallery_feats @ t.squeeze(0)
                    sims = weight * sims_searle + (1.0 - weight) * sims_clip

                restrict_ids = item.get("members", []) if subset_protocol else None
                ranked_ids = rank_from_sims(
                    sims,
                    gallery_ids,
                    id2idx,
                    exclude_ids=[ref_id],
                    restrict_ids=restrict_ids,
                )

                positives = get_relevant_ids(item)

                if positives:
                    update_metrics(results, ranked_ids, positives)

                predictions.append({
                    "query_id": get_query_key(item),
                    "candidate_id": item.get("candidate_id") or ref_id,
                    "pairid": item.get("pairid"),
                    "reference": ref_id,
                    "target_id": item.get("target_id"),
                    "caption": caption,
                    "img_set_id": item.get("img_set_id"),
                    "reference_rank": item.get("reference_rank"),
                    "members": item.get("members", []),
                    "ranking": ranked_ids[:50],
                })
                total += 1

            except Exception as exc:
                skipped += 1
                reason = f"{type(exc).__name__}: {exc}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                print(
                    f"[WARN] SEARLE failed for {ref_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        outputs[weight] = (results, predictions, total, skipped, skip_reasons)

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

    *The user's checkpoint is loaded as ``SiglipModel``. In the*
    *Hugging Face implementation, ``SiglipVisionModel.head`` performs*
    *the multi-head attention pooling and its result is returned as*
    *``pooler_output``. Do NOT feed ``pooler_output`` back through*
    *``vision_model.head`` or another projection.*
    """

    # Preferred path: full model forward exposes image_embeds.
    try:
        outputs = model(**inputs)
        image_embeds = getattr(outputs, "image_embeds", None)
        if torch.is_tensor(image_embeds):
            return image_embeds.float()
    except Exception:
        pass

    # SiglipModel.get_image_features() returns
    # BaseModelOutputWithPooling in current Transformers.
    output = model.get_image_features(**inputs)

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

    *For ``SiglipModel``, ``text_model.head`` already produces the final*
    *projected text representation stored in ``pooler_output``.*
    """

    try:
        outputs = model(**inputs)
        text_embeds = getattr(outputs, "text_embeds", None)
        if torch.is_tensor(text_embeds):
            return text_embeds.float()
    except Exception:
        pass

    output = model.get_text_features(**inputs)

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
    subset_protocol: bool = False,
):
    """Evaluate image-text retrieval in a shared embedding space."""
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

    results = init_results()
    rankings: Dict[str, List[str]] = {}
    total = 0
    skipped = 0
    skip_reasons: Dict[str, int] = {}

    for item in tqdm(data, desc=f"{model_name}: ranking"):
        ref_id = item["reference_id"]
        query_key = get_query_key(item)
        caption = normalize_prompt(item["caption"])

        ref_feat = ref_cache.get(ref_id)
        text_feat = text_cache.get(caption)

        if ref_feat is None:
            skipped += 1
            skip_reasons["missing_reference_embedding"] = skip_reasons.get(
                "missing_reference_embedding", 0
            ) + 1
            continue

        if text_feat is None:
            skipped += 1
            skip_reasons["empty_or_missing_text_embedding"] = skip_reasons.get(
                "empty_or_missing_text_embedding", 0
            ) + 1
            continue

        try:
            ref_feat = F.normalize(ref_feat.float(), dim=-1)
            text_feat = F.normalize(text_feat.float(), dim=-1)

            if gallery_feats.ndim != 2:
                raise ValueError(
                    f"{model_name}: gallery_feats must be [N,D], "
                    f"got {tuple(gallery_feats.shape)}"
                )
            if ref_feat.shape[-1] != gallery_feats.shape[-1]:
                raise ValueError(
                    f"{model_name}: reference dimension {ref_feat.shape[-1]} "
                    f"!= gallery dimension {gallery_feats.shape[-1]}"
                )
            if text_feat.shape[-1] != gallery_feats.shape[-1]:
                raise ValueError(
                    f"{model_name}: text dimension {text_feat.shape[-1]} "
                    f"!= gallery dimension {gallery_feats.shape[-1]}"
                )

            sims_text = gallery_feats @ text_feat.squeeze(0)
            sims_ref = gallery_feats @ ref_feat.squeeze(0)
            sims = alpha * sims_text + (1.0 - alpha) * sims_ref

            restrict_ids = item.get("members", []) if subset_protocol else None
            ranked_ids = rank_from_sims(
                sims,
                gallery_ids,
                id2idx,
                exclude_ids=[ref_id],
                restrict_ids=restrict_ids,
            )

            rankings[query_key] = ranked_ids[:50]
            update_metrics(results, ranked_ids, get_relevant_ids(item))
            total += 1

        except Exception as exc:
            skipped += 1
            reason = f"{type(exc).__name__}: {exc}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            print(
                f"[WARN] {model_name} failed for query={query_key!r}, "
                f"reference={ref_id!r}: {reason}"
            )

    return results, total, skipped, rankings, skip_reasons






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
    subset_protocol=False,
):
    """Evaluate CLIP-beta with stable per-query ranking keys."""
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
            results = init_results()
            rankings: Dict[str, List[str]] = {}
            total = 0
            skipped = 0
            skip_reasons: Dict[str, int] = {}

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
                        if image is None:
                            ref_feature_cache[ref_id] = None
                        else:
                            try:
                                ref_feature_cache[ref_id] = clip_image_embedding(
                                    image, clip_model, clip_model_preprocess
                                )
                            except Exception as exc:
                                print(
                                    f"[WARN] Reference embedding failed for {ref_id}: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                ref_feature_cache[ref_id] = None
                    r = ref_feature_cache.get(ref_id)

                t_gen = gen_cache.get(ref_id)

                if t_rel is None or r is None:
                    skipped += 1
                    reason = (
                        "missing_relative_text_embedding"
                        if t_rel is None
                        else "missing_reference_embedding"
                    )
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue

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

                restrict_ids = item.get("members", []) if subset_protocol else None

                try:
                    ranked_ids = rank_from_sims(
                        sims,
                        gallery_ids,
                        id2idx,
                        exclude_ids=[ref_id],
                        restrict_ids=restrict_ids,
                    )
                except Exception as exc:
                    skipped += 1
                    reason = f"{type(exc).__name__}: {exc}"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    print(
                        f"[WARN] CLIP-beta failed for query={query_key!r}: {reason}"
                    )
                    continue

                rankings[query_key] = ranked_ids[:50]
                update_metrics(results, ranked_ids, get_relevant_ids(item))
                total += 1

            outputs[(alpha, beta)] = (
                results, total, skipped, rankings, skip_reasons
            )

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
    subset_protocol: bool,
    results: dict,
) -> None:
    payload = {
        "dataset": dataset,
        "split": split,
        "cirr_subset": subset_protocol,
        "metrics": results,
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Corrected image retrieval evaluation for CIRR/CIRCO"
    )

    parser.add_argument(
        "--dataset",
        choices=["cirr", "circo"],
        required=True,
    )
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument(
        "--split",
        choices=["auto", "train", "val", "test", "unknown"],
        default="auto",
        help=(
            "Dataset split. With auto, inferred from JSON/image paths: "
            "dev/val -> val, train -> train, test1/test -> test."
        ),
    )
    parser.add_argument(
        "--metrics_output",
        default=None,
        help="Optional path for the final 12-metric JSON output.",
    )
    parser.add_argument(
        "--strict_evaluation",
        action="store_true",
        help="Fail if any query is skipped by a selected model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--reference_image_folder",
        default=None,
        help=(
            "Optional folder used to resolve reference images. "
            "Defaults to --image_folder."
        ),
    )
    parser.add_argument(
        "--ground_truth_path",
        default=None,
        help=(
            "Optional JSON containing target_hard/target_img_id/positives for metric evaluation. "
            "Useful for CIRR train/val or a private test GT; public CIRR test GT is not released."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["clip"],
        choices=["clip", "searle", "open_clip", "siglip", "clip_beta", "blip"],
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    parser.add_argument(
        "--generated_captions_path",
        default=None,
        help="Optional generated-caption JSON. Used only as fallback for empty annotation captions.",
    )
    parser.add_argument(
        "--caption_fallback",
        choices=["auto", "generated", "skip", "error"],
        default="auto",
        help=(
            "How to handle empty annotation captions. "
            "auto=use generated caption only when annotation is empty; "
            "generated=same behavior; skip=leave empty; error=fail."
        ),
    )
    parser.add_argument(
        "--siglip_path",
        default=None,
    )

    parser.add_argument(
        "--clip_model",
        default="ViT-B/32",
    )

    parser.add_argument(
        "--searle_path",
        default=r"C:\Users\user\Desktop\Python\ImageRetrieval\models_download\SEARLE",
        help="Local SEARLE repository path.",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        default="./embedding_cache",
        help="Directory used to store persistent image-embedding caches.",
    )
    parser.add_argument(
        "--blip_path",
        default=None,
        help=(
            "Local Salesforce/blip-itm-base-coco directory. Required when "
            "--models includes blip."
        ),
    )
    parser.add_argument(
        "--blip_batch_size",
        type=int,
        default=16,
        help="Batch size for BLIP gallery image embeddings.",
    )
    parser.add_argument(
        "--blip_text_batch_size",
        type=int,
        default=32,
        help="Batch size for BLIP text embeddings.",
    )
    parser.add_argument(
        "--blip_itm_topk",
        type=int,
        default=0,
        help=(
            "Optional BLIP ITM second-stage reranking. 0 disables ITM; "
            "for CIRR relative captions this is recommended as the default."
        ),
    )
    parser.add_argument(
        "--blip_itm_batch_size",
        type=int,
        default=8,
        help="Batch size for BLIP ITM reranking.",
    )

    parser.add_argument(
        "--open_clip_batch_size",
        type=int,
        default=32,
        help="Batch size for OpenCLIP full-gallery image embeddings.",
    )
    parser.add_argument(
        "--siglip_batch_size",
        type=int,
        default=16,
        help="Batch size for SigLIP/SigLIP2 full-gallery image embeddings.",
    )

    parser.add_argument(
        "--embedding_checkpoint_every",
        type=int,
        default=250,
        help="Save a resumable partial embedding cache every N processed images.",
    )

    parser.add_argument(
        "--force_rebuild_embeddings",
        action="store_true",
        help="Force rebuilding CLIP gallery image embeddings instead of using the cache.",
    )
    parser.add_argument(
        "--open_clip_model",
        default="ViT-H-14",
    )
    parser.add_argument(
        "--open_clip_pretrained",
        default="laion2b_s32b_b79k",
    )

    parser.add_argument(
        "--cirr_subset",
        action="store_true",
        help="Use CIRR subset protocol. Do not enable for CIRCO.",
    )
    parser.add_argument(
        "--require_cirr_gt",
        action="store_true",
        help="Require CIRR target_hard/target_soft. Use for train/validation evaluation.",
    )
    parser.add_argument(
        "--searle_prompt_ensemble",
        action="store_true",
        help="Average several valid SEARLE prompts instead of using only the official prompt.",
    )
    parser.add_argument(
        "--searle_hybrid_weights",
        nargs="+",
        type=float,
        default=[1.0],
        help="SEARLE weight(s). 1.0=pure SEARLE; e.g. 0.85 0.90 0.95 1.0 tests CLIP residual fusion.",
    )

    args = parser.parse_args()
    set_deterministic_seed(args.seed)

    for value in args.alphas:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--alphas values must be in [0,1], got {value}")
    for value in args.betas:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--betas values must be in [0,1], got {value}")
    for value in args.searle_hybrid_weights:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--searle_hybrid_weights values must be in [0,1], got {value}"
            )
    if args.open_clip_batch_size < 1 or args.siglip_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1.")
    if args.blip_batch_size < 1 or args.blip_text_batch_size < 1 or args.blip_itm_batch_size < 1:
        raise ValueError("BLIP batch sizes must be >= 1.")
    if args.blip_itm_topk < 0:
        raise ValueError("--blip_itm_topk must be >= 0.")

    if "blip" in args.models and not args.blip_path:
        raise ValueError("--blip_path is required when --models includes blip.")

    def infer_split() -> str:
        if args.split != "auto":
            return args.split

        text = f"{args.json_path} {args.image_folder}".lower().replace("\\\\\\\\", "/")

        if "test1" in text or "/test/" in text or text.endswith("/test"):
            return "test"
        if "val" in text or "dev" in text:
            return "val"
        if "train" in text:
            return "train"
        return "unknown"


    split_name = infer_split()
    print(f"Split: {split_name}")

    if args.dataset == "circo" and args.cirr_subset:
        raise ValueError("--cirr_subset is only valid for CIRR.")

    if "clip_beta" in args.models and not args.generated_captions_path:
        raise ValueError(
            "--generated_captions_path is required for clip_beta."
        )

    print(f"Device: {DEVICE}")
    print("Selected models:", ", ".join(args.models))
    print(f"Loading {args.dataset.upper()}...")

    detected, data = load_dataset(args.json_path)
    if detected != args.dataset:
        raise ValueError(
            f"Dataset mismatch: argument={args.dataset}, detected={detected}"
        )

    is_new_cirr_schema = (
        args.dataset == "cirr"
        and any(item.get("annotation_format") == "candidate_group" for item in data)
    )

    if args.dataset == "cirr":
        require_gt = args.require_cirr_gt or is_new_cirr_schema

        if args.require_cirr_gt and split_name == "test" and not is_new_cirr_schema:
            raise RuntimeError(
                "--require_cirr_gt cannot be used with public CIRR test annotations "
                "unless target_id is explicitly present in the supplied JSON."
            )

        validate_cirr_annotations(
            data,
            require_ground_truth=require_gt,
        )

    if args.ground_truth_path:
        gt_data = load_optional_ground_truth(args.ground_truth_path)
        data, matched_gt = attach_ground_truth(data, gt_data)
        print(
            f"Ground truth: loaded {len(gt_data or [])} entries; "
            f"matched {matched_gt}/{len(data)} queries"
        )
    else:
        matched_gt = 0

    generated_captions = {}
    if args.generated_captions_path:
        generated_captions = load_generated_captions(args.generated_captions_path)
        print(f"Generated captions loaded: {len(generated_captions)}")

    data, caption_stats = apply_caption_fallback(
        data,
        generated_captions=generated_captions,
        mode=args.caption_fallback,
    )

    print(f"Queries: {len(data)}")

    reference_image_folder = args.reference_image_folder or args.image_folder

    if args.dataset == "cirr":
        validate_reference_images(
            data,
            reference_image_folder,
            fail_on_missing=args.strict_evaluation,
        )

    # IMPORTANT: use the actual image directory as gallery.
    gallery_ids = scan_gallery_ids(args.image_folder)
    print(f"Gallery images: {len(gallery_ids)}")

    if args.dataset == "cirr":
        validate_cirr_gallery_coverage(
            data,
            gallery_ids,
            require_ground_truth=False,
        )
        # Missing group members are not retrieval-fatal because ranking uses the full gallery.

    # CIRR supports two evaluation protocols:
    #   - full gallery: rank against every gallery image
    #   - subset: rank only inside the query's candidate group
    # CIRCO is always full-gallery.
    subset_protocol = bool(args.cirr_subset) if args.dataset == "cirr" else False
    print(
        "Retrieval protocol: "
        + ("CIRR SUBSET (members only)" if subset_protocol else "FULL GALLERY")
    )
    if args.dataset == "cirr" and args.cirr_subset:
        print("[INFO] CIRR group members are used as the ranking candidate set.")

    all_results = {}

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------
    need_clip = any(
        x in args.models
        for x in ("clip", "searle", "clip_beta")
    )

    clip_model = None
    clip_preprocess = None
    clip_cache = None
    clip_cache_raw = None
    clip_gallery_ids = None
    clip_gallery_feats = None

    if need_clip:
        print("\n=== Loading CLIP ===")
        clip_model, clip_preprocess = load_clip_model(
            args.clip_model
        )

        cache_path = clip_embedding_cache_path(
            args.embedding_cache_dir,
            args.clip_model,
            args.dataset,
            split_name,
        )
        partial_cache_path = cache_path.with_suffix(cache_path.suffix + ".partial")

        if args.force_rebuild_embeddings and partial_cache_path.exists():
            try:
                partial_cache_path.unlink()
            except OSError as exc:
                print(f"[WARN] Could not remove old partial cache: {exc}")

        clip_cache = None
        clip_cache_raw = None

        if not args.force_rebuild_embeddings:
            cached = load_clip_image_cache(
                cache_path,
                gallery_ids,
                args.clip_model,
                args.dataset,
                split_name,
            )
            if cached is not None:
                clip_cache, clip_cache_raw = cached
                print(f"Loaded CLIP image embeddings from cache: {cache_path}")
                print(f"Cached images: {len(clip_cache)}")

        if clip_cache is None or clip_cache_raw is None:
            print("No valid CLIP embedding cache found. Building image embeddings...")
            clip_cache, clip_cache_raw = build_clip_image_cache(
                gallery_ids,
                args.image_folder,
                clip_model,
                clip_preprocess,
                partial_cache_path=partial_cache_path,
                model_name=args.clip_model,
                checkpoint_every=max(1, args.embedding_checkpoint_every),
                cache_dataset=args.dataset,
                cache_split=split_name,
            )

            missing = [x for x in gallery_ids if x not in clip_cache or x not in clip_cache_raw]
            if missing:
                diagnostics = []
                for image_id in missing[:10]:
                    diagnostics.append(
                        f"{image_id} -> {find_image_path(args.image_folder, image_id)!r}"
                    )

                raise RuntimeError(
                    f"Failed to compute embeddings for {len(missing)} gallery images.\n"
                    f"First missing IDs / resolved paths:\n"
                    + "\n".join(diagnostics)
                )

            save_clip_image_cache(
                cache_path,
                gallery_ids,
                clip_cache,
                clip_cache_raw,
                args.clip_model,
                args.dataset,
                split_name,
            )
            partial_cache_path = cache_path.with_suffix(cache_path.suffix + ".partial")
            try:
                if partial_cache_path.exists():
                    partial_cache_path.unlink()
            except OSError as exc:
                print(f"[WARN] Could not remove partial cache: {exc}")
            print(f"Saved CLIP image embedding cache: {cache_path}")

        clip_gallery_ids, clip_gallery_feats = stack_feature_cache(
            clip_cache
        )

        print(
            f"CLIP gallery embeddings: "
            f"{len(clip_gallery_ids)}"
        )

    if "clip" in args.models:
        print("\n=== CLIP ===")

        results_by_alpha = {}

        for alpha in args.alphas:
            results, total, skipped, rankings, skip_reasons = evaluate_cross_modal(
                data=data,
                image_folder=args.image_folder,
                reference_image_folder=reference_image_folder,
                gallery_ids=clip_gallery_ids,
                gallery_feats=clip_gallery_feats,
                get_image_feature=lambda img: clip_image_embedding(
                    img, clip_model, clip_preprocess
                ),
                get_text_feature=lambda txt: clip_text_embedding(
                    txt, clip_model
                ),
                model_name="CLIP",
                alpha=alpha,
                subset_protocol=subset_protocol,
            )

            enforce_complete_evaluation(
                f"CLIP alpha={alpha:.2f}",
                len(data),
                total,
                skipped,
                args.strict_evaluation,
            )

            has_gt = any(bool(get_relevant_ids(x)) for x in data)

            if has_gt:
                summary = summarize_results(results)
                results_by_alpha[alpha] = summary

                print_metric_summary(
                    f"CLIP alpha={alpha:.2f}", summary, total, skipped
                )

                all_results[f"CLIP alpha={alpha:.2f}"] = summary

            if split_name == "test":
                output_path = Path(args.json_path).with_name(
                    Path(args.json_path).stem
                    + f"_clip_a{alpha:.2f}_predictions.json"
                )
                save_cirr_test_predictions(
                    path=output_path,
                    data=data,
                    model_name=f"CLIP alpha={alpha:.2f}",
                    split=split_name,
                    rankings=rankings,
                    total=total,
                    skipped=skipped,
                    skip_reasons=skip_reasons,
                    extra={"alpha": alpha},
                )

    # --------------------------------------------------------
    # BLIP
    # --------------------------------------------------------
    if "blip" in args.models:
        print("\n=== BLIP-ITM ===")
        print("Checkpoint: Salesforce/blip-itm-base-coco")
        print(
            "BLIP ITM reranking: "
            + ("OFF" if args.blip_itm_topk == 0 else f"TOP-{args.blip_itm_topk}")
        )

        blip_model, blip_processor = load_blip_itm_model(args.blip_path)

        resolved_blip_path = str(Path(args.blip_path).expanduser().resolve())
        try:
            config_stat = (Path(resolved_blip_path) / "config.json").stat()
            blip_fingerprint_source = (
                f"{resolved_blip_path}|{config_stat.st_size}|{config_stat.st_mtime_ns}"
            )
        except OSError:
            blip_fingerprint_source = resolved_blip_path

        blip_hash = hashlib.sha1(
            blip_fingerprint_source.encode("utf-8")
        ).hexdigest()[:12]
        blip_model_key = f"blip_itm_base_coco_{blip_hash}"

        blip_cache = build_blip_image_cache(
            gallery_ids,
            args.image_folder,
            blip_model,
            blip_processor,
            cache_dir=args.embedding_cache_dir,
            dataset=args.dataset,
            split=split_name,
            model_key=blip_model_key,
            batch_size=args.blip_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )
        blip_gallery_ids = list(gallery_ids)
        blip_gallery_feats = stack_feature_cache(blip_cache)

        blip_outputs = evaluate_blip(
            data=data,
            image_folder=args.image_folder,
            reference_image_folder=reference_image_folder,
            gallery_ids=blip_gallery_ids,
            gallery_feats=blip_gallery_feats,
            model=blip_model,
            processor=blip_processor,
            alphas=args.alphas,
            subset_protocol=subset_protocol,
            text_batch_size=args.blip_text_batch_size,
            itm_topk=args.blip_itm_topk,
            itm_batch_size=args.blip_itm_batch_size,
        )

        for alpha, (
            results,
            total,
            skipped,
            rankings,
            skip_reasons,
        ) in blip_outputs.items():
            tag = (
                f"BLIP alpha={alpha:.2f}"
                + (
                    f" + ITM@{args.blip_itm_topk}"
                    if args.blip_itm_topk > 0
                    else ""
                )
            )

            enforce_complete_evaluation(
                tag,
                len(data),
                total,
                skipped,
                args.strict_evaluation,
            )

            if any(bool(get_relevant_ids(x)) for x in data):
                summary = summarize_results(results)
                all_results[tag] = summary
                print_metric_summary(
                    tag,
                    summary,
                    total,
                    skipped,
                )

            if split_name == "test":
                output_path = Path(args.json_path).with_name(
                    Path(args.json_path).stem
                    + f"_blip_a{alpha:.2f}"
                    + (
                        f"_itm{args.blip_itm_topk}"
                        if args.blip_itm_topk > 0
                        else ""
                    )
                    + "_predictions.json"
                )
                save_cirr_test_predictions(
                    path=output_path,
                    data=data,
                    model_name=tag,
                    split=split_name,
                    rankings=rankings,
                    total=total,
                    skipped=skipped,
                    skip_reasons=skip_reasons,
                    extra={
                        "alpha": alpha,
                        "itm_topk": args.blip_itm_topk,
                        "model_checkpoint": resolved_blip_path,
                    },
                )

        del blip_model, blip_processor, blip_cache, blip_gallery_feats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # SEARLE
    # --------------------------------------------------------
    if "searle" in args.models:
        print("\n=== SEARLE ===")

        if clip_model is None or clip_cache_raw is None:
            raise RuntimeError(
                "SEARLE requires the CLIP model and raw CLIP image cache."
            )

        try:
            # The official prompt contains `$`; SEARLE replaces it with the
            # pseudo-word generated from the reference image.
            prompt_templates = ["a photo of $ {caption}"]
            if args.searle_prompt_ensemble:
                prompt_templates = [
                    "a photo of $ {caption}",
                    "an image of $ {caption}",
                    "$ {caption}",
               ]

            searle, encode_with_pseudo_tokens = load_searle_model(
                clip_model_name=args.clip_model,
                searle_path=args.searle_path,
            )

            # Text cache is only needed for hybrid weights < 1.
            clip_text_cache = None
            if any(w < 0.999999 for w in args.searle_hybrid_weights):
                clip_text_cache = {}
                for item in tqdm(data, desc="SEARLE: CLIP text cache"):
                    caption = normalize_prompt(item["caption"])
                    if caption and caption not in clip_text_cache:
                        clip_text_cache[caption] = clip_text_embedding(
                            caption, clip_model
                        )

            outputs = evaluate_searle(
                data=data,
                searle=searle,
                encode_with_pseudo_tokens=encode_with_pseudo_tokens,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                image_features_cache_raw=clip_cache_raw,
                gallery_ids=clip_gallery_ids,
                gallery_feats=clip_gallery_feats,
                reference_image_folder=reference_image_folder,
                subset_protocol=subset_protocol,
                prompt_templates=prompt_templates,
                hybrid_weights=args.searle_hybrid_weights,
                clip_text_cache=clip_text_cache,
            )

            for weight, (results, predictions, total, skipped, skip_reasons) in outputs.items():
                tag = f"Searle w={weight:.2f}"
                positives_count = sum(
                    1 for x in data if get_relevant_ids(x)
                )

                enforce_complete_evaluation(
                    tag,
                    len(data),
                    total,
                    skipped,
                    args.strict_evaluation,
                )

                if positives_count:
                    summary = summarize_results(results)
                    all_results[tag] = summary
                    print_metric_summary(
                        tag, summary, total, skipped
                    )

                output_path = Path(args.json_path).with_name(
                    Path(args.json_path).stem
                    + f"_searle_w{weight:.2f}_predictions.json"
                )
                # Convert SEARLE prediction objects to the same transparent format
                # used by CLIP/OpenCLIP/SigLIP.
                searle_rankings = {
                    str(p.get("query_id")): p.get("ranking", [])
                    for p in predictions
                }
                if args.dataset == "cirr":
                    save_cirr_test_predictions(
                        path=output_path,
                        data=data,
                        model_name=f"SEARLE w={weight:.2f}",
                        split=split_name,
                        rankings=searle_rankings,
                        total=total,
                        skipped=skipped,
                        skip_reasons=skip_reasons,
                        extra={"weight": weight},
                    )
                else:
                    prediction_payload = {
                        "dataset": args.dataset,
                        "model": "SEARLE",
                        "weight": weight,
                        "num_queries": len(data),
                        "num_predictions": total,
                        "num_skipped": skipped,
                        "skip_reasons": skip_reasons,
                        "metrics_available": any(
                            bool(get_relevant_ids(x)) for x in data
                        ),
                        "predictions": predictions,
                    }
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(prediction_payload, f, ensure_ascii=False, indent=2)
                print(
                    f"{tag}: queries={total}, skipped={skipped}; "
                    f"predictions -> {output_path}"
                )
                if skip_reasons:
                    print(f"{tag}: skip reasons -> {skip_reasons}")

            del searle, encode_with_pseudo_tokens
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"[ERROR] SEARLE evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            if args.strict_evaluation:
                raise

    # --------------------------------------------------------
    # CLIP beta
    # --------------------------------------------------------
    if "clip_beta" in args.models:
        print("\n=== CLIP-beta ===")

        generated = generated_captions

        beta_results = evaluate_clip_beta(
            data=data,
            image_folder=args.image_folder,
            reference_image_folder=reference_image_folder,
            gallery_ids=clip_gallery_ids,
            gallery_feats=clip_gallery_feats,
            clip_model=clip_model,
            clip_model_preprocess=clip_preprocess,
            image_features_cache=clip_cache,
            generated_captions=generated,
            alphas=args.alphas,
            betas=args.betas,
            subset_protocol=subset_protocol,
        )

        for (alpha, beta), (results, total, skipped, rankings, skip_reasons) in beta_results.items():
            enforce_complete_evaluation(
                f"CLIP-beta a={alpha:.2f} b={beta:.2f}",
                len(data),
                total,
                skipped,
                args.strict_evaluation,
            )
            has_gt = any(bool(get_relevant_ids(x)) for x in data)

            if has_gt:
                summary = summarize_results(results)

                all_results[
                    f"CLIP-beta a={alpha:.2f} b={beta:.2f}"
                ] = summary

                print_metric_summary(
                    f"CLIP-beta a={alpha:.2f} b={beta:.2f}",
                    summary, total, skipped
                )

            if split_name == "test":
                output_path = Path(args.json_path).with_name(
                    Path(args.json_path).stem
                    + f"_clip_beta_a{alpha:.2f}_b{beta:.2f}_predictions.json"
                )
                save_cirr_test_predictions(
                    path=output_path,
                    data=data,
                    model_name=f"CLIP-beta a={alpha:.2f} b={beta:.2f}",
                    split=split_name,
                    rankings=rankings,
                    total=total,
                    skipped=skipped,
                    skip_reasons=skip_reasons,
                    extra={"alpha": alpha, "beta": beta},
                )

    # --------------------------------------------------------
    # OpenCLIP
    # --------------------------------------------------------
    if "open_clip" in args.models:
        print("\n=== OpenCLIP ===")

        oc_model, oc_preprocess = load_open_clip_model(
            args.open_clip_model,
            args.open_clip_pretrained,
        )
        tokenizer = open_clip.get_tokenizer(args.open_clip_model)

        oc_model_key = (
            f"openclip_{args.open_clip_model}_{args.open_clip_pretrained}"
        )
        oc_cache = build_open_clip_image_cache(
            gallery_ids,
            args.image_folder,
            oc_model,
            oc_preprocess,
            cache_dir=args.embedding_cache_dir,
            dataset=args.dataset,
            split=split_name,
            model_key=oc_model_key,
            batch_size=args.open_clip_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )

        oc_ids, oc_feats = stack_feature_cache(oc_cache)

        for alpha in args.alphas:
            results, total, skipped, rankings, skip_reasons = evaluate_cross_modal(
                data=data,
                image_folder=args.image_folder,
                reference_image_folder=reference_image_folder,
                gallery_ids=oc_ids,
                gallery_feats=oc_feats,
                get_image_feature=lambda img: open_clip_image_embedding(
                    img, oc_model, oc_preprocess
                ),
                get_text_feature=lambda txt: open_clip_text_embedding(
                    txt, oc_model, tokenizer
                ),
                model_name="OpenCLIP",
                alpha=alpha,
                subset_protocol=subset_protocol,
            )

            enforce_complete_evaluation(
                f"OpenCLIP alpha={alpha:.2f}",
                len(data),
                total,
                skipped,
                args.strict_evaluation,
            )
            has_gt = any(bool(get_relevant_ids(x)) for x in data)

            if has_gt:
                summary = summarize_results(results)
                all_results[f"OpenCLIP alpha={alpha:.2f}"] = summary

                print_metric_summary(
                    f"OpenCLIP alpha={alpha:.2f}", summary, total, skipped
                )

            if split_name == "test":
                output_path = Path(args.json_path).with_name(
                    Path(args.json_path).stem
                    + f"_open_clip_a{alpha:.2f}_predictions.json"
                )
                save_cirr_test_predictions(
                    path=output_path,
                    data=data,
                    model_name=f"OpenCLIP alpha={alpha:.2f}",
                    split=split_name,
                    rankings=rankings,
                    total=total,
                    skipped=skipped,
                    skip_reasons=skip_reasons,
                    extra={"alpha": alpha},
                )

        del oc_model, oc_preprocess, oc_cache, oc_feats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # SigLIP
    # --------------------------------------------------------
    if "siglip" in args.models:
        print("\n=== SigLIP ===")

        sig_model, sig_processor = load_siglip_model(
            args.siglip_path
        )

        sig_model_key = f"siglip_{Path(args.siglip_path).resolve()}"
        sig_cache = build_siglip_image_cache(
            gallery_ids,
            args.image_folder,
            sig_model,
            sig_processor,
            cache_dir=args.embedding_cache_dir,
            dataset=args.dataset,
            split=split_name,
            model_key=sig_model_key,
            batch_size=args.siglip_batch_size,
            force_rebuild=args.force_rebuild_embeddings,
        )

        sig_ids, sig_feats = stack_feature_cache(sig_cache)

        for alpha in args.alphas:
            results, total, skipped, rankings, skip_reasons = evaluate_cross_modal(
                data=data,
                image_folder=args.image_folder,
                reference_image_folder=reference_image_folder,
                gallery_ids=sig_ids,
                gallery_feats=sig_feats,
                get_image_feature=lambda img: siglip_image_embedding(
                    img, sig_model, sig_processor
                ),
                get_text_feature=lambda txt: siglip_text_embedding(
                    txt, sig_model, sig_processor
                ),
                model_name="SigLIP",
                alpha=alpha,
                subset_protocol=subset_protocol,
            )

            enforce_complete_evaluation(
                f"SigLIP alpha={alpha:.2f}",
                len(data),
                total,
                skipped,
                args.strict_evaluation,
            )
            has_gt = any(bool(get_relevant_ids(x)) for x in data)

            if has_gt:
                summary = summarize_results(results)
                all_results[f"SigLIP alpha={alpha:.2f}"] = summary

                print_metric_summary(
                    f"SigLIP alpha={alpha:.2f}", summary, total, skipped
                )

            if split_name == "test":
                output_path = Path(args.json_path).with_name(
                    Path(args.json_path).stem
                    + f"_siglip_a{alpha:.2f}_predictions.json"
                )
                save_cirr_test_predictions(
                    path=output_path,
                    data=data,
                    model_name=f"SigLIP alpha={alpha:.2f}",
                    split=split_name,
                    rankings=rankings,
                    total=total,
                    skipped=skipped,
                    skip_reasons=skip_reasons,
                    extra={"alpha": alpha},
                )

        del sig_model, sig_processor, sig_cache, sig_feats
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------
    if not all_results:
        positives_count = sum(
            1 for x in data if get_relevant_ids(x)
        )
        if positives_count == 0:
            if args.dataset == "cirr" and not args.ground_truth_path:
                print(
                    "\nNOTE: No ground-truth positives/target_id are present in the provided "
                    "CIRR JSON. For the public CIRR test split, ground truth is not released; "
                    "use the official test server for leaderboard metrics. Rankings/predictions "
                    "were generated, but MRR/mAP/Recall cannot be computed locally."
                )
            else:
                print(
                    "\nNOTE: No ground-truth positives/target_id are available for the evaluated queries. "
                    "Rankings/predictions were generated, but MRR/mAP/Recall cannot be computed."
                )

    print("\n" + "=" * 150)
    print(f"FINAL RESULTS - {args.dataset.upper()}")
    print("=" * 150)

    header = (
        f"{'Model':<30}"
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

    for name, result in all_results.items():
        print(
            f"{name:<30}"
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

    metrics_output = (
        Path(args.metrics_output)
        if args.metrics_output
        else Path(args.json_path).with_name(
            Path(args.json_path).stem + "_metrics_12.json"
        )
    )
    if all_results:
        save_metrics_json(
            metrics_output,
            args.dataset,
            split_name,
            subset_protocol,
            all_results,
        )



if __name__ == "__main__":
    main()