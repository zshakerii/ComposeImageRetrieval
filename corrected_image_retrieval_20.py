
import os
import json
import argparse
import traceback
import shutil
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

import clip
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
K_VALUES = (1, 5, 10, 50)
MAP_K_VALUES = (5, 10, 50)


# ============================================================
# Dataset parsing
# ============================================================

def normalize_id(value) -> str:
    """Normalize dataset image IDs while preserving leading zeros."""
    if value is None:
        return ""
    return os.path.splitext(str(value).strip())[0]


def parse_cirr_sample(sample: dict) -> dict:
    """
    *Parse CIRR train/validation/test annotations.*

    *CIRR validation/train contain target_hard and target_soft.*
    *Public test annotations normally do not contain ground truth.*
    """
    if "reference" not in sample:
        raise ValueError("CIRR sample is missing 'reference'.")

    img_set = sample.get("img_set") or {}
    reference_id = normalize_id(sample["reference"])
    caption = str(sample.get("caption", "")).strip()

    members = [
        normalize_id(x)
        for x in img_set.get("members", [])
        if normalize_id(x)
    ]

    target_hard = normalize_id(sample.get("target_hard"))

    positives = []
    if target_hard:
        positives.append(target_hard)

    target_soft = sample.get("target_soft")
    if isinstance(target_soft, dict):
        positives.extend(
            normalize_id(x)
            for x in target_soft.keys()
            if normalize_id(x)
        )
    elif isinstance(target_soft, list):
        positives.extend(
            normalize_id(x)
            for x in target_soft
            if normalize_id(x)
        )

    positives = list(dict.fromkeys(x for x in positives if x))

    return {
        "pairid": sample.get("pairid"),
        "reference_id": reference_id,
        "caption": caption,
        "target_id": target_hard or None,
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

    # CIRR cap.rc2.* format
    if (
        isinstance(first, dict)
        and "reference" in first
        and "caption" in first
        and isinstance(first.get("img_set"), dict)
        and "members" in first["img_set"]
    ):
        return "cirr", [parse_cirr_sample(x) for x in data]

    # CIRCO list format
    if isinstance(first, dict) and (
        "reference_img_id" in first or "gt_img_ids" in first
    ):
        return "circo", [parse_circo_sample(x) for x in data]

    raise ValueError(
        "Unknown CIRR/CIRCO JSON format. "
        f"First keys: {list(first.keys()) if isinstance(first, dict) else 'N/A'}"
    )


def summarize_cirr_ground_truth(data: List[dict]) -> Tuple[int, int]:
    """Return (queries_with_gt, queries_without_gt) for CIRR."""
    with_gt = sum(
        1 for item in data
        if item.get("target_hard") or item.get("positives")
    )
    return with_gt, len(data) - with_gt


def validate_cirr_annotations(data: List[dict], require_ground_truth: bool = False) -> None:
    """Validate CIRR train/val/test annotations before model evaluation."""
    with_gt, without_gt = summarize_cirr_ground_truth(data)
    print("\n=== CIRR annotation check ===")
    print(f"Queries                 : {len(data)}")
    print(f"Queries with target_hard: {sum(1 for x in data if x.get('target_hard'))}")
    print(f"Queries with positives  : {with_gt}")
    print(f"Queries without GT      : {without_gt}")

    bad_members = sum(1 for x in data if not x.get("members"))
    if bad_members:
        print(f"[WARN] {bad_members} queries have no img_set.members.")

    if require_ground_truth and without_gt > 0:
        raise RuntimeError(
            f"CIRR local metric evaluation requires ground truth for every query, "
            f"but {without_gt}/{len(data)} queries have no target_hard/target_soft. "
            "Use the CIRR validation/train JSON for local metrics."
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
            by_reference.setdefault(ref, x)

    matched = 0
    for item in data:
        gt = None
        if item.get("pairid") is not None:
            gt = by_pairid.get(str(item["pairid"]))
        if gt is None:
            gt = by_reference.get(item.get("reference_id", ""))

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
    **Build the actual retrieval gallery and a recursive image-path index.**
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
    """Return the relevant image IDs for a CIRR query."""
    relevant: Set[str] = set()

    hard = normalize_id(item.get("target_hard") or item.get("target_id"))
    if hard:
        relevant.add(hard)

    soft = item.get("target_soft")
    if isinstance(soft, dict):
        for image_id, score in soft.items():
            image_id = normalize_id(image_id)
            if not image_id:
                continue
            try:
                if float(score) > 0.0:
                    relevant.add(image_id)
            except (TypeError, ValueError):
                relevant.add(image_id)
    elif isinstance(soft, list):
        for image_id in soft:
            image_id = normalize_id(image_id)
            if image_id:
                relevant.add(image_id)

    for image_id in item.get("positives") or []:
        image_id = normalize_id(image_id)
        if image_id:
            relevant.add(image_id)

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
    """
    **Correct ranking implementation.**

    **- never mutates the caller's similarity tensor**
    **- supports CIRR subset protocol**
    **- supports reference exclusion**
    **- returns IDs, not tensor indices**
    """
    sims = sims.detach().float().flatten().clone()

    if sims.numel() != len(gallery_ids):
        raise ValueError(
            f"Similarity length ({sims.numel()}) != gallery size "
            f"({len(gallery_ids)})"
        )

    if restrict_ids is not None:
        allowed = torch.zeros_like(sims, dtype=torch.bool)
        for image_id in restrict_ids:
            idx = id2idx.get(str(image_id))
            if idx is not None:
                allowed[idx] = True
        sims[~allowed] = -float("inf")

    for image_id in exclude_ids:
        idx = id2idx.get(str(image_id))
        if idx is not None:
            sims[idx] = -float("inf")

    order = torch.argsort(sims, descending=True).cpu().tolist()
    return [
        gallery_ids[i]
        for i in order
        if torch.isfinite(sims[i])
   ]


# ============================================================
# CLIP embedding cache
# ============================================================

def _safe_cache_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\\\\\\\", "_").replace(" ", "_")


def clip_embedding_cache_path(cache_dir: str, model_name: str) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"clip_{_safe_cache_model_name(model_name)}_image_embeddings.pt"


def save_clip_image_cache(
    cache_path: Path,
    gallery_ids: List[str],
    normalized: Dict[str, torch.Tensor],
    raw: Dict[str, torch.Tensor],
    model_name: str,
) -> None:
    payload = {
        "version": 3,
        "model_name": model_name,
        "gallery_ids": list(gallery_ids),
        "normalized": normalized,
        "raw": raw,
    }
    torch.save(payload, cache_path)


def load_clip_image_cache(
    cache_path: Path,
    gallery_ids: List[str],
    model_name: str,
) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    if not cache_path.is_file():
        return None

    try:
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        if payload.get("version") != 3:
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

        # Validate that every current gallery image has both embeddings.
        if set(normalized.keys()) != set(gallery_ids):
            return None
        if set(raw.keys()) != set(gallery_ids):
            return None

        return normalized, raw

    except Exception as exc:
        print(f"[WARN] Could not load embedding cache {cache_path}: {exc}")
        return None


# ============================================================
# CLIP
# ============================================================

def load_clip_model(model_name: str = "ViT-B/32"):
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
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    **Build CLIP image embeddings with resumable checkpoints.**

    **A partial cache is written periodically, so an interruption does not**
    **force another full pass over all 8082 images.**
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
                payload.get("version") == 3
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
            "version": 3,
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
# OpenCLIP
# ============================================================


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
    **Build the composed SEARLE query.**

    *IMPORTANT: `$` is NOT a normal character here. It is the placeholder**
    **replaced by SEARLE pseudo tokens. The official repository explicitly**
    *requires the prompt to contain `$`.**
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
    image_features_cache_raw: Dict[str, torch.Tensor],
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    subset_protocol: bool = False,
    prompt_templates: Optional[List[str]] = None,
    hybrid_weights: Optional[List[float]] = None,
    clip_text_cache: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    **Evaluate/rank SEARLE.**

    **If positives are present, metrics are returned. For cap.rc2.test1.json,**
    **positives are absent, so predictions are generated without fake metrics.**

    **hybrid score:**
        **w * SEARLE_similarity + (1-w) * CLIP_text_similarity**

    **This is deliberately optional; w=1.0 is pure SEARLE.**
    """
    id2idx = build_id_index(gallery_ids)
    weights = hybrid_weights or [1.0]
    outputs = {}

    for weight in weights:
        results = init_results()
        predictions = []
        total = skipped = 0
        skip_reasons = {}

        for item in tqdm(data, desc=f"SEARLE w={weight:.2f}"):
            ref_id = item["reference_id"]
            caption = normalize_prompt(item["caption"])
            raw_ref = image_features_cache_raw.get(ref_id)

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
                    "pairid": item.get("pairid"),
                    "reference": ref_id,
                    "caption": caption,
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
):
    cache = {}

    for image_id in tqdm(list(image_ids), desc="OpenCLIP image embeddings"):
        image = load_image(image_folder, image_id)
        if image is None:
            continue

        try:
            cache[image_id] = open_clip_image_embedding(
                image, model, preprocess
            )
        except Exception as exc:
            print(f"[WARN] OpenCLIP image {image_id}: {exc}")

    return cache


# ============================================================
# SigLIP
# ============================================================

def load_siglip_model(model_path: str):
    if not model_path:
        raise ValueError("--siglip_path is required for siglip")

    from transformers import AutoModel, AutoProcessor

    dtype = (
        torch.float16
        if DEVICE.type == "cuda"
        else torch.float32
    )

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    return model, processor


@torch.inference_mode()
def siglip_image_embedding(image, model, processor):
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    feat = model.get_image_features(**inputs)
    return F.normalize(feat.float(), dim=-1).cpu()


@torch.inference_mode()
def siglip_text_embedding(text, model, processor):
    inputs = processor(
        text=[text],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    feat = model.get_text_features(**inputs)
    return F.normalize(feat.float(), dim=-1).cpu()


def build_siglip_image_cache(
    image_ids,
    image_folder,
    model,
    processor,
    batch_size=32,
):
    cache = {}
    image_ids = list(image_ids)

    for start in tqdm(
        range(0, len(image_ids), batch_size),
        desc="SigLIP image embeddings",
    ):
        ids = image_ids[start:start + batch_size]
        images = []
        valid_ids = []

        for image_id in ids:
            image = load_image(image_folder, image_id)
            if image is not None:
                images.append(image)
                valid_ids.append(image_id)

        if not images:
            continue

        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {
                k: v.to(DEVICE) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            feats = model.get_image_features(**inputs)
            feats = F.normalize(feats.float(), dim=-1).cpu()

            for image_id, feat in zip(valid_ids, feats):
                cache[image_id] = feat.unsqueeze(0)

        except Exception as exc:
            print(f"[WARN] SigLIP batch: {exc}")

    return cache


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
    gallery_ids: List[str],
    gallery_feats: torch.Tensor,
    get_image_feature,
    get_text_feature,
    model_name: str,
    alpha: float = 0.5,
    subset_protocol: bool = False,
):
    """
    **Evaluate a model whose image/text embeddings live in the SAME space.**

    **score = alpha * sim(text, gallery)**
          **+ (1-alpha) * sim(reference_image, gallery)**
    """
    id2idx = build_id_index(gallery_ids)

    # Cache reference image embeddings and text embeddings.
    ref_cache: Dict[str, Optional[torch.Tensor]] = {}
    text_cache: Dict[str, Optional[torch.Tensor]] = {}

    for item in tqdm(data, desc=f"{model_name}: cache queries"):
        ref_id = item["reference_id"]
        caption = normalize_prompt(item["caption"])

        if ref_id not in ref_cache:
            image = load_image(image_folder, ref_id)
            ref_cache[ref_id] = (
                None if image is None
                else get_image_feature(image)
            )

        if caption not in text_cache:
            text_cache[caption] = (
                get_text_feature(caption)
                if caption
                else None
            )

    results = init_results()
    total = 0
    skipped = 0

    for item in tqdm(data, desc=f"{model_name}: ranking"):
        ref_id = item["reference_id"]
        positives = get_relevant_ids(item)

        ref_feat = ref_cache.get(ref_id)
        text_feat = text_cache.get(normalize_prompt(item["caption"]))

        if ref_feat is None or text_feat is None:
            skipped += 1
            continue

        ref_feat = F.normalize(ref_feat.float(), dim=-1)
        text_feat = F.normalize(text_feat.float(), dim=-1)

        sims_text = gallery_feats @ text_feat.squeeze(0)
        sims_ref = gallery_feats @ ref_feat.squeeze(0)

        # Both components are cosine similarities in the same embedding
        # space. Do NOT use per-query min-max normalization by default:
        # it changes the geometry and can amplify outliers.
        sims = alpha * sims_text + (1.0 - alpha) * sims_ref

        restrict_ids = (
            item.get("members", [])
            if subset_protocol
            else None
        )

        ranked_ids = rank_from_sims(
            sims,
            gallery_ids,
            id2idx,
            exclude_ids=[ref_id],
            restrict_ids=restrict_ids,
        )

        update_metrics(results, ranked_ids, positives)
        total += 1

    return results, total, skipped


def evaluate_clip_beta(
    data,
    gallery_ids,
    gallery_feats,
    clip_model,
    image_features_cache,
    generated_captions,
    alphas,
    betas,
    subset_protocol=False,
):
    """
    **Correct implementation of the user's CLIP-beta idea.**

    **q1 = beta * relative_text + (1-beta) * generated_text**
    **q2 = beta * reference_image + (1-beta) * generated_text**

    **final_score = alpha * sim(gallery,q1)**
                **+ (1-alpha) * sim(gallery,q2)**
    """
    id2idx = build_id_index(gallery_ids)

    rel_cache = {}
    gen_cache = {}

    for item in tqdm(data, desc="CLIP-beta: text cache"):
        rel = normalize_prompt(item["caption"])
        if rel not in rel_cache:
            rel_cache[rel] = clip_text_embedding(
                rel, clip_model
            ) if rel else None

        ref_id = item["reference_id"]
        gen = generated_captions.get(ref_id, "").strip()
        if gen and ref_id not in gen_cache:
            gen_cache[ref_id] = clip_text_embedding(
                gen, clip_model
            )

    outputs = {}

    for beta in betas:
        for alpha in alphas:
            results = init_results()
            total = 0
            skipped = 0

            for item in tqdm(
                data,
                desc=f"CLIP-beta a={alpha:.2f} b={beta:.2f}",
            ):
                ref_id = item["reference_id"]
                rel = normalize_prompt(item["caption"])
                gen = generated_captions.get(ref_id, "").strip()

                t_rel = rel_cache.get(rel)
                r = image_features_cache.get(ref_id)
                t_gen = gen_cache.get(ref_id)

                if t_rel is None or r is None:
                    skipped += 1
                    continue

                if t_gen is None:
                    q1 = t_rel
                    q2 = r
                else:
                    q1 = beta * t_rel + (1.0 - beta) * t_gen
                    q2 = beta * r + (1.0 - beta) * t_gen

                q1 = F.normalize(q1.float(), dim=-1)
                q2 = F.normalize(q2.float(), dim=-1)

                sims1 = gallery_feats @ q1.squeeze(0)
                sims2 = gallery_feats @ q2.squeeze(0)
                sims = alpha * sims1 + (1.0 - alpha) * sims2

                restrict_ids = (
                    item.get("members", [])
                    if subset_protocol
                    else None
                )

                ranked_ids = rank_from_sims(
                    sims,
                    gallery_ids,
                    id2idx,
                    exclude_ids=[ref_id],
                    restrict_ids=restrict_ids,
                )

                update_metrics(
                    results,
                    ranked_ids,
                    get_relevant_ids(item),
                )
                total += 1

            outputs[(alpha, beta)] = (results, total, skipped)

    return outputs


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
        choices=["clip", "searle", "open_clip", "siglip", "clip_beta"],
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
        help="Directory used to store CLIP gallery image embeddings.",
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
        "--split",
        choices=["auto", "train", "val", "test", "unknown"],
        default="auto",
        help=(
            "Dataset split. With 'auto', infer from json/image paths: "
            "dev/val -> val, train -> train, test1/test -> test."
        ),
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

    def infer_split() -> str:
        if args.split != "auto":
            return args.split
        text = f"{args.json_path} {args.image_folder}".lower()
        if "test1" in text or "/test" in text or "\\test" in text:
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
    print(f"Loading {args.dataset.upper()}...")

    detected, data = load_dataset(args.json_path)
    if detected != args.dataset:
        raise ValueError(
            f"Dataset mismatch: argument={args.dataset}, detected={detected}"
        )

    if args.dataset == "cirr":
        validate_cirr_annotations(
            data,
            require_ground_truth=args.require_cirr_gt,
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

    print(f"Queries: {len(data)}")

    # IMPORTANT: use the actual image directory as gallery.
    gallery_ids = scan_gallery_ids(args.image_folder)
    print(f"Gallery images: {len(gallery_ids)}")

    subset_protocol = args.dataset == "cirr" and args.cirr_subset

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
            results, total, skipped = evaluate_cross_modal(
                data=data,
                image_folder=args.image_folder,
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

            summary = summarize_results(results)
            results_by_alpha[alpha] = summary

            print_metric_summary(
                f"CLIP alpha={alpha:.2f}", summary, total, skipped
            )

            all_results[f"CLIP alpha={alpha:.2f}"] = summary

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
                image_features_cache_raw=clip_cache_raw,
                gallery_ids=clip_gallery_ids,
                gallery_feats=clip_gallery_feats,
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
                prediction_payload = {
                    "dataset": args.dataset,
                    "model": "SEARLE",
                    "weight": weight,
                    "num_queries": len(data),
                    "num_predictions": total,
                    "num_skipped": skipped,
                    "skip_reasons": skip_reasons,
                    "metrics_available": any(
                        x.get("positives") or x.get("target_id") for x in data
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

    # --------------------------------------------------------
    # CLIP beta
    # --------------------------------------------------------
    if "clip_beta" in args.models:
        print("\n=== CLIP-beta ===")

        generated = load_generated_captions(
            args.generated_captions_path
        )

        beta_results = evaluate_clip_beta(
            data=data,
            gallery_ids=clip_gallery_ids,
            gallery_feats=clip_gallery_feats,
            clip_model=clip_model,
            image_features_cache=clip_cache,
            generated_captions=generated,
            alphas=args.alphas,
            betas=args.betas,
            subset_protocol=subset_protocol,
        )

        for (alpha, beta), (results, total, skipped) in beta_results.items():
            summary = summarize_results(results)

            all_results[
                f"CLIP-beta a={alpha:.2f} b={beta:.2f}"
           ] = summary

            print_metric_summary(
                f"CLIP-beta a={alpha:.2f} b={beta:.2f}",
                summary, total, skipped
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

        oc_cache = build_open_clip_image_cache(
            gallery_ids,
            args.image_folder,
            oc_model,
            oc_preprocess,
        )

        oc_ids, oc_feats = stack_feature_cache(oc_cache)

        for alpha in args.alphas:
            results, total, skipped = evaluate_cross_modal(
                data=data,
                image_folder=args.image_folder,
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

            summary = summarize_results(results)
            all_results[f"OpenCLIP alpha={alpha:.2f}"] = summary

            print_metric_summary(
                f"OpenCLIP alpha={alpha:.2f}", summary, total, skipped
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

        sig_cache = build_siglip_image_cache(
            gallery_ids,
            args.image_folder,
            sig_model,
            sig_processor,
        )

        sig_ids, sig_feats = stack_feature_cache(sig_cache)

        for alpha in args.alphas:
            results, total, skipped = evaluate_cross_modal(
                data=data,
                image_folder=args.image_folder,
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

            summary = summarize_results(results)
            all_results[f"SigLIP alpha={alpha:.2f}"] = summary

            print_metric_summary(
                f"SigLIP alpha={alpha:.2f}", summary, total, skipped
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