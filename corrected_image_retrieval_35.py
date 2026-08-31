import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

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
BLIP_TEXT_MAX_LEN = 64

_IMAGE_PATH_INDEX: Dict[str, str] = {}


def set_deterministic_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_id(value) -> str:
    if value is None:
        return ""
    return os.path.splitext(str(value).strip())[0]


def normalize_prompt(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip()


def get_query_key(item: dict) -> str:
    query_id = str(item.get("query_id", "")).strip()
    if query_id:
        return query_id
    pairid = item.get("pairid")
    if pairid is not None:
        return str(pairid)
    ref = normalize_id(item.get("reference_id"))
    target = normalize_id(item.get("target_id"))
    if ref and target:
        return f"{ref}__{target}"
    return f"query_{item.get('_index', 0)}"


# ============================================================
# Dataset parsing
# ============================================================

def parse_cirr_sample(sample: dict, index: int = 0) -> dict:
    if not isinstance(sample, dict):
        raise ValueError(f"CIRR sample #{index} must be an object")

    if {"candidate_id", "caption", "group", "target_id"}.issubset(sample.keys()):
        reference_id = normalize_id(sample["candidate_id"])
        target_id = normalize_id(sample["target_id"])
        caption = normalize_prompt(sample.get("caption", ""))
        group_raw = sample.get("group")
        if not isinstance(group_raw, list):
            raise ValueError(f"CIRR sample #{index}: group must be a list")
        members = list(dict.fromkeys(normalize_id(x) for x in group_raw if normalize_id(x)))
        if not reference_id or not target_id or not members:
            raise ValueError(f"CIRR sample #{index}: invalid reference/target/group")
        if reference_id not in members:
            raise ValueError(f"CIRR sample #{index}: candidate_id not in group")
        if target_id not in members:
            raise ValueError(f"CIRR sample #{index}: target_id not in group")

        query_id = f"{reference_id}__{target_id}__{index}"
        return {
            "_index": index,
            "query_id": query_id,
            "pairid": query_id,
            "annotation_format": "candidate_group",
            "candidate_id": reference_id,
            "reference_id": reference_id,
            "target_id": target_id,
            "target_hard": target_id,
            "target_soft": {},
            "caption": caption,
            "group": members,
            "members": members,
            "positives": [target_id],
            "img_set_id": None,
            "reference_rank": None,
            "target_rank": None,
        }

    if "reference" not in sample or "caption" not in sample:
        raise ValueError(f"CIRR sample #{index}: unsupported schema")

    img_set = sample.get("img_set")
    if not isinstance(img_set, dict) or not isinstance(img_set.get("members"), list):
        raise ValueError(f"CIRR sample #{index}: img_set.members is required")

    reference_id = normalize_id(sample.get("reference"))
    caption = normalize_prompt(sample.get("caption"))
    members = list(dict.fromkeys(normalize_id(x) for x in img_set["members"] if normalize_id(x)))
    target_hard = normalize_id(sample.get("target_hard"))
    positives = [target_hard] if target_hard else []

    target_soft = sample.get("target_soft")
    if isinstance(target_soft, dict):
        positives.extend(normalize_id(x) for x in target_soft.keys() if normalize_id(x))
    elif isinstance(target_soft, list):
        positives.extend(normalize_id(x) for x in target_soft if normalize_id(x))
    positives = list(dict.fromkeys(x for x in positives if x))

    pairid = sample.get("pairid")
    query_id = str(pairid) if pairid is not None else f"{reference_id}__{index}"

    return {
        "_index": index,
        "query_id": query_id,
        "pairid": pairid if pairid is not None else query_id,
        "annotation_format": "cap_rc2",
        "candidate_id": reference_id,
        "reference_id": reference_id,
        "target_id": target_hard or None,
        "target_hard": target_hard or None,
        "target_soft": target_soft if isinstance(target_soft, (dict, list)) else {},
        "caption": caption,
        "group": members,
        "members": members,
        "positives": positives,
        "img_set_id": img_set.get("id"),
        "reference_rank": img_set.get("reference_rank"),
        "target_rank": img_set.get("target_rank"),
    }


def load_dataset(json_path: str) -> Tuple[str, List[dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON list")

    first = data[0]
    if isinstance(first, dict) and {"candidate_id", "caption", "group", "target_id"}.issubset(first.keys()):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]

    if isinstance(first, dict) and "reference" in first and isinstance(first.get("img_set"), dict):
        return "cirr", [parse_cirr_sample(x, i) for i, x in enumerate(data)]

    raise ValueError(
        "This corrected BLIP script is intended for CIRR. "
        f"First keys: {list(first.keys()) if isinstance(first, dict) else 'N/A'}"
    )


def get_relevant_ids(item: dict) -> Set[str]:
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
        relevant.update(normalize_id(x) for x in soft if normalize_id(x))

    relevant.update(normalize_id(x) for x in item.get("positives") or [] if normalize_id(x))
    return relevant


def validate_cirr_annotations(data: List[dict], require_ground_truth: bool) -> None:
    missing_members = []
    missing_ref = []
    missing_caption = []
    missing_gt = []

    for item in data:
        if not item.get("members"):
            missing_members.append(item["pairid"])
        if not item.get("reference_id"):
            missing_ref.append(item["pairid"])
        if not normalize_prompt(item.get("caption", "")):
            missing_caption.append(item["pairid"])
        if not get_relevant_ids(item):
            missing_gt.append(item["pairid"])

    print("\n=== CIRR annotation check ===")
    print(f"Queries                 : {len(data)}")
    print(f"Queries with GT         : {len(data) - len(missing_gt)}")
    print(f"Queries without GT      : {len(missing_gt)}")
    print(f"Queries without members : {len(missing_members)}")
    print(f"Queries without reference: {len(missing_ref)}")
    print(f"Queries without caption : {len(missing_caption)}")

    if missing_members:
        raise RuntimeError(f"{len(missing_members)} queries have no candidate group")
    if missing_ref:
        raise RuntimeError(f"{len(missing_ref)} queries have no reference image")
    if require_ground_truth and missing_gt:
        raise RuntimeError("Ground truth is required, but target information is missing")


def build_image_path_index(image_folder: str) -> Dict[str, str]:
    root = Path(image_folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")

    index: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.stem, str(p))
    return index


def scan_gallery_ids(image_folder: str) -> List[str]:
    global _IMAGE_PATH_INDEX
    _IMAGE_PATH_INDEX = build_image_path_index(image_folder)
    if not _IMAGE_PATH_INDEX:
        raise RuntimeError(f"No images found in {image_folder}")
    return sorted(_IMAGE_PATH_INDEX.keys())


def find_image_path(image_folder: str, image_id: str) -> Optional[str]:
    global _IMAGE_PATH_INDEX
    image_id = normalize_id(image_id)
    if not _IMAGE_PATH_INDEX:
        _IMAGE_PATH_INDEX = build_image_path_index(image_folder)

    for candidate in (image_id, image_id.zfill(12) if image_id.isdigit() else image_id):
        path = _IMAGE_PATH_INDEX.get(candidate)
        if path and Path(path).is_file():
            return path
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


def validate_gallery_coverage(data: List[dict], gallery_ids: List[str]) -> None:
    gallery = set(gallery_ids)
    missing_refs = sorted({x["reference_id"] for x in data if x.get("reference_id") and x["reference_id"] not in gallery})
    missing_targets = sorted({x for item in data for x in get_relevant_ids(item) if x not in gallery})
    missing_members = sorted({normalize_id(x) for item in data for x in item.get("members", []) if normalize_id(x) and normalize_id(x) not in gallery})

    print("\n=== Gallery coverage check ===")
    print(f"Gallery images         : {len(gallery_ids)}")
    print(f"Missing reference IDs  : {len(missing_refs)}")
    print(f"Missing target IDs     : {len(missing_targets)}")
    print(f"Missing group members  : {len(missing_members)}")

    if missing_refs:
        print("First missing references:", missing_refs[:20])
    if missing_targets:
        print("First missing targets:", missing_targets[:20])
    if missing_members:
        print("First missing group members:", missing_members[:20])

    if missing_refs:
        raise RuntimeError("Reference images are missing from the gallery")


# ============================================================
# Metrics
# ============================================================

def average_precision_at_k(relevant: Iterable[str], retrieved: List[str], k: int) -> float:
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


def summarize_results(results: dict) -> dict:
    mean = lambda values: float(np.mean(values)) if values else 0.0
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
    return {x: i for i, x in enumerate(gallery_ids)}


def rank_from_sims(
    sims: torch.Tensor,
    gallery_ids: List[str],
    id2idx: Dict[str, int],
    exclude_ids: Iterable[str] = (),
    restrict_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    sims = sims.detach().float().flatten().cpu()
    if sims.numel() != len(gallery_ids):
        raise ValueError(f"Similarity length {sims.numel()} != gallery size {len(gallery_ids)}")

    mask = torch.ones(len(gallery_ids), dtype=torch.bool)
    excluded = {normalize_id(x) for x in exclude_ids if normalize_id(x)}
    for i, image_id in enumerate(gallery_ids):
        if image_id in excluded:
            mask[i] = False

    if restrict_ids is not None:
        allowed = {normalize_id(x) for x in restrict_ids if normalize_id(x) in id2idx}
        for i, image_id in enumerate(gallery_ids):
            if image_id not in allowed:
                mask[i] = False

    if not bool(mask.any()):
        return []

    work = sims.clone()
    work[~mask] = -float("inf")
    order = torch.argsort(work, descending=True).tolist()
    return [gallery_ids[i] for i in order if torch.isfinite(work[i])]


# ============================================================
# Cache
# ============================================================

def safe_name(value: str) -> str:
    text = str(value).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    return text[:120]


def model_cache_path(cache_dir: str, dataset: str, split: str, model_key: str) -> Path:
    root = Path(cache_dir) / dataset.lower() / split.lower()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe_name(model_key)}_image_embeddings.pt"


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_feature_cache(path: Path, gallery_ids: List[str], dataset: str, split: str, model_key: str) -> Optional[Dict[str, torch.Tensor]]:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        if payload.get("cache_version") != 2:
            return None
        if payload.get("dataset") != dataset or payload.get("split") != split or payload.get("model_key") != model_key:
            return None
        if list(payload.get("gallery_ids", [])) != list(gallery_ids):
            return None
        features = payload.get("features")
        if not isinstance(features, dict) or set(features.keys()) != set(gallery_ids):
            return None
        return features
    except Exception as exc:
        print(f"[WARN] Invalid cache {path}: {exc}")
        return None


def build_feature_cache(
    image_ids: List[str],
    image_folder: str,
    encode_batch,
    cache_path: Path,
    model_key: str,
    dataset: str,
    split: str,
    batch_size: int,
    force_rebuild: bool,
    checkpoint_every_batches: int = 8,
) -> Dict[str, torch.Tensor]:
    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")

    if force_rebuild:
        for p in (cache_path, partial_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    features: Dict[str, torch.Tensor] = {}

    if partial_path.is_file() and not force_rebuild:
        try:
            payload = torch.load(partial_path, map_location="cpu", weights_only=False)
            if (
                isinstance(payload, dict)
                and payload.get("cache_version") == 2
                and payload.get("dataset") == dataset
                and payload.get("split") == split
                and payload.get("model_key") == model_key
                and list(payload.get("gallery_ids", [])) == image_ids
                and isinstance(payload.get("features"), dict)
            ):
                features.update(payload["features"])
                print(f"Resuming {model_key}: {len(features)}/{len(image_ids)}")
        except Exception as exc:
            print(f"[WARN] Could not resume partial cache: {exc}")

    def save_partial() -> None:
        atomic_torch_save(
            {
                "cache_version": 2,
                "dataset": dataset,
                "split": split,
                "model_key": model_key,
                "gallery_ids": image_ids,
                "features": features,
            },
            partial_path,
        )

    remaining = [x for x in image_ids if x not in features]
    batch_size = max(1, int(batch_size))
    checkpoint_every_batches = max(1, int(checkpoint_every_batches))

    for batch_number, start in enumerate(
        tqdm(range(0, len(remaining), batch_size), desc=f"{model_key} image embeddings"),
        start=1,
    ):
        batch_ids = remaining[start:start + batch_size]
        images: List[Image.Image] = []
        valid_ids: List[str] = []

        for image_id in batch_ids:
            image = load_image(image_folder, image_id)
            if image is None:
                raise RuntimeError(f"Cannot read gallery image: {image_id}")
            images.append(image)
            valid_ids.append(image_id)

        try:
            batch_features = encode_batch(images)
            if not torch.is_tensor(batch_features) or batch_features.ndim != 2 or batch_features.shape[0] != len(valid_ids):
                raise ValueError(f"Encoder returned invalid shape: {getattr(batch_features, 'shape', None)}")
            batch_features = F.normalize(batch_features.float(), dim=-1).cpu()
            for image_id, feature in zip(valid_ids, batch_features):
                features[image_id] = feature.unsqueeze(0)
        except Exception:
            save_partial()
            raise

        if batch_number % checkpoint_every_batches == 0:
            save_partial()
            print(f"[CACHE] {model_key}: {len(features)}/{len(image_ids)}")

    if set(features.keys()) != set(image_ids):
        missing = sorted(set(image_ids) - set(features))
        save_partial()
        raise RuntimeError(f"Cache incomplete. Missing {len(missing)} images: {missing[:20]}")

    atomic_torch_save(
        {
            "cache_version": 2,
            "dataset": dataset,
            "split": split,
            "model_key": model_key,
            "gallery_ids": image_ids,
            "features": features,
        },
        cache_path,
    )
    if partial_path.exists():
        try:
            partial_path.unlink()
        except OSError:
            pass
    print(f"Saved FULL {model_key} image cache: {cache_path}")
    return features


def stack_feature_cache(cache: Dict[str, torch.Tensor], gallery_ids: List[str]) -> torch.Tensor:
    if not cache:
        raise RuntimeError("Feature cache is empty")
    missing = [x for x in gallery_ids if x not in cache]
    if missing:
        raise RuntimeError(f"Feature cache is incomplete. Missing: {missing[:20]}")
    feats = torch.cat([cache[x].reshape(1, -1) for x in gallery_ids], dim=0).float()
    return F.normalize(feats, dim=-1)


# ============================================================
# BLIP-ITM
# ============================================================

def load_blip_itm_model(model_path: str):
    if not model_path:
        raise ValueError("--blip_path is required when using --models blip")

    from transformers import BlipForImageTextRetrieval, BlipProcessor

    local_path = Path(model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"BLIP model directory does not exist: {local_path}")
    if not (local_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in BLIP directory: {local_path}")

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

    processor = BlipProcessor.from_pretrained(str(local_path), local_files_only=True)
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


def _move_inputs(inputs: dict) -> dict:
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}


@torch.inference_mode()
def blip_image_features(images: List[Image.Image], model, processor) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = _move_inputs(inputs)

    pixel_values = inputs["pixel_values"].to(dtype=model.dtype)
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    hidden = vision_outputs.last_hidden_state if hasattr(vision_outputs, "last_hidden_state") else vision_outputs[0]

    # BLIP vision representation used by the retrieval projection.
    pooled = hidden[:, 0, :]
    projected = model.vision_proj(pooled)
    return F.normalize(projected.float(), dim=-1).cpu()


@torch.inference_mode()
def blip_text_features(texts: List[str], model, processor) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0))

    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=BLIP_TEXT_MAX_LEN,
    )
    inputs = _move_inputs(inputs)

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
    """Return probability of the positive image-text-matching class."""
    inputs = processor(
        images=images,
        text=[caption] * len(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=BLIP_TEXT_MAX_LEN,
    )
    inputs = _move_inputs(inputs)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    # This is the official BLIP-ITM path. It uses the learned ITM head,
    # not the image/text cosine representation.
    outputs = model(**inputs, use_itm_head=True)

    if hasattr(outputs, "itm_score"):
        logits = outputs.itm_score
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        logits = outputs[0]
    else:
        raise RuntimeError(f"Unsupported BLIP ITM output type: {type(outputs).__name__}")

    if logits.ndim == 2 and logits.shape[-1] == 2:
        return logits.float().softmax(dim=-1)[:, 1].cpu()
    if logits.ndim == 1:
        return logits.float().sigmoid().cpu()
    raise RuntimeError(f"Unexpected BLIP ITM score shape: {tuple(logits.shape)}")


def build_blip_gallery_cache(
    gallery_ids: List[str],
    image_folder: str,
    model,
    processor,
    cache_dir: str,
    dataset: str,
    split: str,
    model_key: str,
    batch_size: int,
    force_rebuild: bool,
) -> Dict[str, torch.Tensor]:
    path = model_cache_path(cache_dir, dataset, split, model_key)
    cached = None if force_rebuild else load_feature_cache(path, gallery_ids, dataset, split, model_key)
    if cached is not None:
        print(f"Loaded BLIP image cache: {path}")
        return cached

    def encode_batch(images):
        return blip_image_features(images, model, processor)

    return build_feature_cache(
        image_ids=gallery_ids,
        image_folder=image_folder,
        encode_batch=encode_batch,
        cache_path=path,
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
    if top_k <= 0 or not candidate_ids:
        return candidate_ids

    candidates = candidate_ids[:top_k]
    scored: List[Tuple[str, float]] = []

    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        images = []
        valid_ids = []
        for image_id in chunk:
            image = load_image(image_folder, image_id)
            if image is None:
                continue
            images.append(image)
            valid_ids.append(image_id)

        if not images:
            continue

        scores = blip_itm_score_batch(caption, images, model, processor)
        scored.extend((image_id, float(score)) for image_id, score in zip(valid_ids, scores.tolist()))

    score_map = dict(scored)
    reranked_top = sorted(candidates, key=lambda x: score_map.get(x, -float("inf")), reverse=True)
    suffix = [x for x in candidate_ids if x not in set(candidates)]
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
    itm_topk: int,
    itm_batch_size: int,
):
    id2idx = build_id_index(gallery_ids)

    # Text caching is important: many CIRR queries can share identical captions.
    unique_captions = sorted({normalize_prompt(x.get("caption", "")) for x in data if normalize_prompt(x.get("caption", ""))})
    text_cache: Dict[str, torch.Tensor] = {}
    print(f"BLIP unique captions: {len(unique_captions)}")
    for start in tqdm(range(0, len(unique_captions), 32), desc="BLIP text embeddings"):
        batch = unique_captions[start:start + 32]
        feats = blip_text_features(batch, model, processor)
        for text, feat in zip(batch, feats):
            text_cache[text] = feat.unsqueeze(0)

    ref_cache: Dict[str, Optional[torch.Tensor]] = {}
    unique_refs = sorted({x["reference_id"] for x in data})
    for ref_id in tqdm(unique_refs, desc="BLIP reference embeddings"):
        image = load_image(reference_image_folder, ref_id)
        ref_cache[ref_id] = None if image is None else blip_image_features([image], model, processor)

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
                skip_reasons["missing_reference_embedding"] = skip_reasons.get("missing_reference_embedding", 0) + 1
                continue
            if text_feat is None:
                skipped += 1
                skip_reasons["empty_caption"] = skip_reasons.get("empty_caption", 0) + 1
                continue

            try:
                # CIRR composition in the shared BLIP retrieval space.
                # alpha=1 -> caption only; alpha=0 -> reference only.
                query = F.normalize(
                    alpha * text_feat.float() + (1.0 - alpha) * ref_feat.float(),
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

                if itm_topk > 0:
                    # ITM is an optional second stage. For CIRR relative captions,
                    # it is intentionally OFF by default because the caption is not
                    # necessarily a full description of the target image.
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
                print(f"[WARN] BLIP query failed: {get_query_key(item)} -> {reason}")

        outputs[alpha] = (results, total, skipped, rankings, skip_reasons)

    return outputs


# ============================================================
# Output
# ============================================================

def print_metric_summary(label: str, summary: dict, total: int, skipped: int) -> None:
    print(
        f"{label} | MRR={summary['mrr']:.4f} | "
        f"mAP@5={summary['map5']:.4f} | mAP@10={summary['map10']:.4f} | mAP@50={summary['map50']:.4f} | "
        f"P@1={summary['prec1']:.4f} | P@5={summary['prec5']:.4f} | P@10={summary['prec10']:.4f} | P@50={summary['prec50']:.4f} | "
        f"R@1={summary['rec1']:.4f} | R@5={summary['rec5']:.4f} | R@10={summary['rec10']:.4f} | R@50={summary['rec50']:.4f} | "
        f"n={total}, skipped={skipped}"
    )


def save_predictions(
    path: Path,
    data: List[dict],
    model_name: str,
    split: str,
    rankings: Dict[str, List[str]],
    total: int,
    skipped: int,
    skip_reasons: Dict[str, int],
    alpha: float,
    itm_topk: int,
) -> None:
    predictions = []
    for item in data:
        key = get_query_key(item)
        ranking = rankings.get(key, [])
        target = normalize_id(item.get("target_id"))
        target_rank = ranking.index(target) + 1 if target and target in ranking else None
        predictions.append(
            {
                "query_id": key,
                "candidate_id": item.get("candidate_id"),
                "reference": item.get("reference_id"),
                "target_id": item.get("target_id"),
                "caption": item.get("caption", ""),
                "members": item.get("members", []),
                "ranking": ranking,
                "target_rank": target_rank,
            }
        )

    payload = {
        "dataset": "cirr",
        "split": split,
        "model": model_name,
        "alpha": alpha,
        "itm_topk": itm_topk,
        "num_queries": len(data),
        "num_predictions": total,
        "num_skipped": skipped,
        "skip_reasons": skip_reasons,
        "predictions": predictions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved: {path}")


def save_metrics(path: Path, dataset: str, split: str, subset: bool, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "split": split,
                "cirr_subset": subset,
                "metrics": results,
                "metric_order": [
                    "MRR", "mAP@5", "mAP@10", "mAP@50",
                    "P@1", "P@5", "P@10", "P@50",
                    "R@1", "R@5", "R@10", "R@50",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Metrics saved: {path}")


# ============================================================
# Main
# ============================================================

def infer_split(json_path: str, image_folder: str, requested: str) -> str:
    if requested != "auto":
        return requested
    text = f"{json_path} {image_folder}".lower().replace("\\", "/")
    if "test1" in text or "/test/" in text or text.endswith("/test"):
        return "test"
    if "val" in text or "dev" in text:
        return "val"
    if "train" in text:
        return "train"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected CIRR image retrieval with BLIP-ITM")
    parser.add_argument("--dataset", choices=["cirr"], default="cirr")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--reference_image_folder", default=None)
    parser.add_argument("--split", choices=["auto", "train", "val", "test", "unknown"], default="auto")
    parser.add_argument("--blip_path", required=True, help="Local Salesforce/blip-itm-base-coco directory")
    parser.add_argument("--embedding_cache_dir", default="./embedding_cache")
    parser.add_argument("--blip_batch_size", type=int, default=16)
    parser.add_argument("--blip_text_batch_size", type=int, default=32)
    parser.add_argument("--blip_itm_topk", type=int, default=0, help="0 disables ITM reranking; recommended default for CIRR")
    parser.add_argument("--blip_itm_batch_size", type=int, default=8)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--cirr_subset", action="store_true")
    parser.add_argument("--require_cirr_gt", action="store_true")
    parser.add_argument("--strict_evaluation", action="store_true")
    parser.add_argument("--force_rebuild_embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--metrics_output", default=None)
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    if args.torch_threads > 0 and DEVICE.type == "cpu":
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")

    if args.blip_batch_size < 1 or args.blip_text_batch_size < 1 or args.blip_itm_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    if args.blip_itm_topk < 0:
        raise ValueError("--blip_itm_topk must be >= 0")

    split = infer_split(args.json_path, args.image_folder, args.split)
    print(f"Device: {DEVICE}")
    print(f"Split : {split}")
    print(f"Model : Salesforce/blip-itm-base-coco")
    print(f"Alpha : {args.alphas}")
    print(f"ITM   : {'OFF' if args.blip_itm_topk == 0 else f'TOP-{args.blip_itm_topk}'}")

    detected, data = load_dataset(args.json_path)
    if detected != args.dataset:
        raise ValueError(f"Dataset mismatch: expected {args.dataset}, detected {detected}")

    is_new_schema = any(item.get("annotation_format") == "candidate_group" for item in data)
    require_gt = args.require_cirr_gt or is_new_schema
    validate_cirr_annotations(data, require_ground_truth=require_gt)

    data_missing = [x for x in data if not normalize_prompt(x.get("caption", ""))]
    if data_missing:
        raise RuntimeError(
            f"{len(data_missing)} queries have empty captions. "
            "Resolve captions before BLIP retrieval; do not silently invent text."
        )

    reference_image_folder = args.reference_image_folder or args.image_folder
    gallery_ids = scan_gallery_ids(args.image_folder)
    print(f"Gallery images: {len(gallery_ids)}")
    validate_gallery_coverage(data, gallery_ids)

    subset_protocol = bool(args.cirr_subset)
    print("Retrieval protocol:", "CIRR SUBSET" if subset_protocol else "FULL GALLERY")

    print("\n=== Loading BLIP-ITM ===")
    model, processor = load_blip_itm_model(args.blip_path)

    resolved_model = str(Path(args.blip_path).expanduser().resolve())
    model_hash = hashlib.sha1(resolved_model.encode("utf-8")).hexdigest()[:10]
    model_key = f"blip_itm_base_coco_{model_hash}"

    print("\n=== Building/loading BLIP gallery cache ===")
    blip_cache = build_blip_gallery_cache(
        gallery_ids=gallery_ids,
        image_folder=args.image_folder,
        model=model,
        processor=processor,
        cache_dir=args.embedding_cache_dir,
        dataset=args.dataset,
        split=split,
        model_key=model_key,
        batch_size=args.blip_batch_size,
        force_rebuild=args.force_rebuild_embeddings,
    )

    gallery_feats = stack_feature_cache(blip_cache, gallery_ids)
    print(f"Gallery embedding matrix: {tuple(gallery_feats.shape)}")

    print("\n=== Evaluating BLIP ===")
    outputs = evaluate_blip(
        data=data,
        image_folder=args.image_folder,
        reference_image_folder=reference_image_folder,
        gallery_ids=gallery_ids,
        gallery_feats=gallery_feats,
        model=model,
        processor=processor,
        alphas=args.alphas,
        subset_protocol=subset_protocol,
        itm_topk=args.blip_itm_topk,
        itm_batch_size=args.blip_itm_batch_size,
    )

    all_results = {}
    has_gt = any(bool(get_relevant_ids(x)) for x in data)

    for alpha, (results, total, skipped, rankings, skip_reasons) in outputs.items():
        label = f"BLIP-ITM alpha={alpha:.2f}"
        if skipped:
            message = f"{label}: processed={total}/{len(data)}, skipped={skipped}"
            if args.strict_evaluation:
                raise RuntimeError(message)
            print("[WARN] " + message)

        if has_gt:
            summary = summarize_results(results)
            all_results[label] = summary
            print_metric_summary(label, summary, total, skipped)
        else:
            print(f"{label}: no local GT; predictions only")

        if split == "test":
            output_path = Path(args.json_path).with_name(
                Path(args.json_path).stem
                + f"_blip_itm_a{alpha:.2f}"
                + (f"_itm{args.blip_itm_topk}" if args.blip_itm_topk > 0 else "")
                + "_predictions.json"
            )
            save_predictions(
                output_path,
                data,
                label,
                split,
                rankings,
                total,
                skipped,
                skip_reasons,
                alpha,
                args.blip_itm_topk,
            )

    if all_results:
        print("\n" + "=" * 150)
        print("FINAL RESULTS - CIRR / BLIP-ITM")
        print("=" * 150)
        for name, result in all_results.items():
            print(
                f"{name:<28} "
                f"MRR={result['mrr']:.4f} "
                f"mAP@5={result['map5']:.4f} "
                f"mAP@10={result['map10']:.4f} "
                f"mAP@50={result['map50']:.4f} "
                f"R@1={result['rec1']:.4f} "
                f"R@5={result['rec5']:.4f} "
                f"R@10={result['rec10']:.4f} "
                f"R@50={result['rec50']:.4f}"
            )

        metrics_path = Path(args.metrics_output) if args.metrics_output else Path(args.json_path).with_name(Path(args.json_path).stem + "_blip_metrics.json")
        save_metrics(metrics_path, args.dataset, split, subset_protocol, all_results)
    else:
        print("\nNo local ground-truth metrics available. Predictions were generated.")


if __name__ == "__main__":
    main()
