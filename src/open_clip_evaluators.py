"""OpenCLIP evaluation pipeline (independent from OpenAI CLIP)."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import CLIP_CAPTION_PREFIX
from .metrics import init_results, update_metrics
from .models.open_clip import get_open_clip_text_feature
from .ranking import minmax_normalize


def _caption(text: str) -> str:
    return CLIP_CAPTION_PREFIX + text


def evaluate_open_clip_alphas(
    data, model, tokenizer, image_features_cache, gallery_ids, gallery_feats, alphas
):
    text_cache = {}
    results_by_alpha = {}

    for item in tqdm(data, desc="Caching OpenCLIP text features"):
        caption_full = _caption(item["caption"])
        key = (item["reference_id"], caption_full)
        if key not in text_cache:
            text_cache[key] = get_open_clip_text_feature(caption_full, model, tokenizer)

    ref_cache = dict(image_features_cache)

    for alpha in alphas:
        results = init_results()
        total, skipped = 0, 0

        for item in tqdm(data, desc=f"OpenCLIP alpha={alpha:.2f}"):
            reference_id = item["reference_id"]
            positives = {item["target_id"]}
            caption_full = _caption(item["caption"])

            text_feat = text_cache.get((reference_id, caption_full))
            ref_feat = ref_cache.get(reference_id)
            if text_feat is None or ref_feat is None:
                skipped += 1
                continue

            sims_txt = torch.matmul(gallery_feats, text_feat.squeeze(0).float().T)
            sims_ref = torch.matmul(gallery_feats, ref_feat.squeeze(0).float().T)
            sims = alpha * sims_txt + (1.0 - alpha) * sims_ref
            ranked_idx = torch.argsort(sims.squeeze(-1), descending=True).cpu().tolist()
            ranked_ids = [gallery_ids[i] for i in ranked_idx]

            update_metrics(results, ranked_ids, positives)
            total += 1

        results_by_alpha[alpha] = (results, total, skipped)

    return results_by_alpha


def evaluate_open_clip_alphas_separate(
    data, model, tokenizer, image_features_cache, gallery_ids, gallery_feats, alphas
):
    text_cache = {}
    results_by_alpha = {}

    for item in tqdm(data, desc="Caching OpenCLIP text features (separate)"):
        caption_full = _caption(item["caption"])
        key = (item["reference_id"], caption_full)
        if key not in text_cache:
            text_cache[key] = get_open_clip_text_feature(caption_full, model, tokenizer)

    ref_cache = dict(image_features_cache)

    for alpha in alphas:
        results = init_results()
        total, skipped = 0, 0

        for item in tqdm(data, desc=f"OpenCLIP-sep alpha={alpha:.2f}"):
            reference_id = item["reference_id"]
            positives = {item["target_id"]}
            caption_full = _caption(item["caption"])

            text_feat = text_cache.get((reference_id, caption_full))
            ref_feat = ref_cache.get(reference_id)
            if text_feat is None or ref_feat is None:
                skipped += 1
                continue

            sims_txt = torch.matmul(gallery_feats, text_feat.squeeze(0).float().T)
            sims_ref = torch.matmul(gallery_feats, ref_feat.squeeze(0).float().T)

            sims_txt_n = minmax_normalize(sims_txt)
            sims_ref_n = minmax_normalize(sims_ref)
            sims = alpha * sims_txt_n + (1.0 - alpha) * sims_ref_n

            ranked_idx = torch.argsort(sims, descending=True).cpu().tolist()
            ranked_ids = [gallery_ids[i] for i in ranked_idx]

            update_metrics(results, ranked_ids, positives)
            total += 1

        results_by_alpha[alpha] = (results, total, skipped)

    return results_by_alpha


def evaluate_open_clip_beta(
    data,
    model,
    tokenizer,
    image_features_cache,
    gallery_ids,
    gallery_feats,
    generated_captions,
    alphas,
    betas,
):
    rel_text_cache = {}
    gen_text_cache = {}

    for item in tqdm(data, desc="Caching OpenCLIP-beta text features"):
        rel_caption = _caption(item["caption"])
        if rel_caption not in rel_text_cache:
            rel_text_cache[rel_caption] = get_open_clip_text_feature(rel_caption, model, tokenizer)

        ref_id = item["reference_id"]
        gen_cap = generated_captions.get(ref_id, "")
        if gen_cap and ref_id not in gen_text_cache:
            gen_text_cache[ref_id] = get_open_clip_text_feature(_caption(gen_cap), model, tokenizer)

    ref_cache = dict(image_features_cache)
    results_by_params = {}

    for beta in betas:
        for alpha in alphas:
            results = init_results()
            total, skipped = 0, 0

            for item in tqdm(data, desc=f"OpenCLIP-beta a={alpha:.2f} b={beta:.2f}"):
                reference_id = item["reference_id"]
                positives = {item["target_id"]}
                rel_caption = _caption(item["caption"])

                t_rel = rel_text_cache.get(rel_caption)
                t_gen = gen_text_cache.get(reference_id)
                r = ref_cache.get(reference_id)

                if t_rel is None or r is None:
                    skipped += 1
                    continue
                if t_gen is None:
                    q1, q2 = t_rel, r
                else:
                    q1 = beta * t_rel + (1.0 - beta) * t_gen
                    q2 = beta * r + (1.0 - beta) * t_gen

                q1 = F.normalize(q1, dim=-1)
                q2 = F.normalize(q2, dim=-1)

                s1 = torch.matmul(gallery_feats, q1.squeeze(0).float().T)
                s2 = torch.matmul(gallery_feats, q2.squeeze(0).float().T)
                sims = alpha * s1 + (1.0 - alpha) * s2

                ranked_idx = torch.argsort(sims.squeeze(-1), descending=True).cpu().tolist()
                ranked_ids = [gallery_ids[i] for i in ranked_idx]

                update_metrics(results, ranked_ids, positives)
                total += 1

            results_by_params[(alpha, beta)] = (results, total, skipped)

    return results_by_params
