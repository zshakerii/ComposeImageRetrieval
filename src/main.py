import os
import argparse
import traceback

import torch

from .config import (
    DEFAULT_ALPHAS,
    DEFAULT_BETAS,
    DEFAULT_MODELS,
    DEFAULT_OPEN_CLIP_MODEL,
    DEFAULT_OPEN_CLIP_PRETRAINED,
    OPEN_CLIP_MODELS,
    device,
)
from .datasets import load_dataset
from .io_utils import load_generated_captions
from .metrics import summarize_results
from .cache import (
    build_clip_image_cache,
    build_generic_image_cache,
    build_open_clip_image_cache,
    build_siglip_batch_cache,
    build_qwen_batch_cache,
    stack_feature_cache,
)

from .models.clip import get_clip_image_feature,get_clip_text_feature,load_clip_model
from .models.qwen import get_qwen_text_feature,get_qwen_image_feature,load_qwen_model
from .models.searle import load_searle_model,load_searle_config
from .models.lava import get_lava_text_feature,get_lava_image_feature,load_lava_model
from .models.open_clip import load_open_clip_model,get_open_clip_image_feature,get_open_clip_text_feature
from .models.blip import get_blip_text_feature,get_blip_image_feature,load_blip_model
from .models.siglip import get_siglip_image_feature,get_siglip_text_feature,load_siglip_model


from .evaluators import (
    evaluate_clip_alphas,
    evaluate_clip_alphas_separate,
    evaluate_clip_beta,
    evaluate_searle,
    evaluate_generic,
)
from .open_clip_evaluators import (
    evaluate_open_clip_alphas,
    evaluate_open_clip_alphas_separate,
    evaluate_open_clip_beta,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Unified Evaluation for CIRR and CIRCO")
    parser.add_argument("--dataset", type=str, choices=["cirr", "circo"], required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--qwen_model_path", type=str, default=None)
    parser.add_argument("--blip_model_path", type=str, default=None)
    parser.add_argument("--siglip_path", type=str, default=None)
    parser.add_argument("--generated_captions_path", type=str, default=None)
    parser.add_argument("--open_clip_model_name", type=str, default=DEFAULT_OPEN_CLIP_MODEL)
    parser.add_argument("--open_clip_pretrained", type=str, default=DEFAULT_OPEN_CLIP_PRETRAINED)
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    return parser


def collect_image_ids(data):
    image_ids = set()
    for item in data:
        image_ids.add(item["reference_id"])
        image_ids.add(item["target_id"])
        for member in item.get("members", []):
            image_ids.add(member)
    return image_ids


def print_results_table(args, all_results):
    print("\n" + "=" * 140)
    print(f"📋 جدول مقایسه‌ای {args.dataset.upper()} (Open-set)")
    print("=" * 140)

    header = (
        f"{'Model/Alpha':<24} {'MRR':<10} {'mAP@5':<10} {'mAP@10':<10} {'mAP@50':<10} "
        f"{'Prec@1':<10} {'Prec@5':<10} {'Prec@10':<10} {'Prec@50':<10} "
        f"{'Rec@1':<10} {'Rec@5':<10} {'Rec@10':<10} {'Rec@50':<10}"
    )
    print(header)
    print("-" * len(header))

    def row(name, r):
        print(
            f"{name:<24} {r['mrr']:<10.4f} {r['map5']:<10.4f} {r['map10']:<10.4f} {r['map50']:<10.4f} "
            f"{r['prec1']:<10.4f} {r['prec5']:<10.4f} {r['prec10']:<10.4f} {r['prec50']:<10.4f} "
            f"{r['rec1']:<10.4f} {r['rec5']:<10.4f} {r['rec10']:<10.4f} {r['rec50']:<10.4f}"
        )

    for alpha in sorted(args.alphas):
        key = f"CLIP alpha={alpha:.2f}"
        if key in all_results:
            row(key, all_results[key])

    for alpha in sorted(args.alphas):
        key = f"CLIP-sep alpha={alpha:.2f}"
        if key in all_results:
            row(key, all_results[key])

    for alpha in sorted(args.alphas):
        key = f"OpenCLIP alpha={alpha:.2f}"
        if key in all_results:
            row(key, all_results[key])

    for alpha in sorted(args.alphas):
        key = f"OpenCLIP-sep alpha={alpha:.2f}"
        if key in all_results:
            row(key, all_results[key])

    for beta in sorted(args.betas):
        for alpha in sorted(args.alphas):
            key = f"CLIP-beta a={alpha:.2f} b={beta:.2f}"
            if key in all_results:
                row(key, all_results[key])

    for beta in sorted(args.betas):
        for alpha in sorted(args.alphas):
            key = f"OpenCLIP-beta a={alpha:.2f} b={beta:.2f}"
            if key in all_results:
                row(key, all_results[key])

    for model_name in ["Searle", "Qwen", "Blip", "LLaVA", "SigLIP"]:
        if model_name in all_results:
            row(model_name, all_results[model_name])


def main():
    args = build_arg_parser().parse_args()

    print(f"🔄 بارگذاری دیتاست {args.dataset.upper()}...")
    detected_dataset, data = load_dataset(args.json_path)
    if detected_dataset != args.dataset:
        raise ValueError(f"Dataset mismatch: arg={args.dataset}, detected={detected_dataset}")
    print(f"✅ تعداد {len(data)} نمونه بارگذاری شد.")

    image_ids = collect_image_ids(data)
    dataset_type = args.dataset

    all_results = {}
    clip_model = None
    gallery_ids = gallery_feats = None
    image_features_cache = image_features_cache_raw = None

    need_clip_cache = any(m in args.models for m in ("clip", "searle", "clip_sep", "clip_beta"))

    # نکته: توابع ارزیابی CLIP از clip.tokenize استفاده می‌کنند، پس مدل باید
    # از openai/CLIP بارگذاری شود (نه open_clip). بارگذاری تکراری اسکریپت اصلی حذف شد.
    if need_clip_cache:
        print("🔄 بارگذاری مدل CLIP و ساخت کش تصویر...")
        clip_model, preprocess = load_clip_model()
        image_features_cache, image_features_cache_raw = build_clip_image_cache(
            image_ids, args.image_folder, clip_model, preprocess, dataset_type
        )
        gallery_ids, gallery_feats = stack_feature_cache(image_features_cache)
    else:
        print("⏭️  مدل‌های مبتنی بر CLIP انتخاب نشده‌اند؛ کش CLIP ساخته نمی‌شود.")

    need_open_clip_cache = any(m in args.models for m in OPEN_CLIP_MODELS)
    open_clip_model = open_clip_tokenizer = None
    oc_gallery_ids = oc_gallery_feats = None
    open_clip_image_cache = None

    if need_open_clip_cache:
        print("🔄 بارگذاری مدل OpenCLIP و ساخت کش تصویر...")
        open_clip_model, open_clip_preprocess, open_clip_tokenizer = load_open_clip_model(
            args.open_clip_model_name, args.open_clip_pretrained
        )
        open_clip_image_cache = build_open_clip_image_cache(
            image_ids, args.image_folder, open_clip_model, open_clip_preprocess, dataset_type
        )
        oc_gallery_ids, oc_gallery_feats = stack_feature_cache(open_clip_image_cache)
        print(f"✅ تعداد {len(open_clip_image_cache)} تصویر OpenCLIP پردازش شد.")
    else:
        print("⏭️  مدل‌های OpenCLIP انتخاب نشده‌اند؛ کش OpenCLIP ساخته نمی‌شود.")

    if "open_clip" in args.models:
        print("\n🔄 ارزیابی OpenCLIP")
        oc_results = evaluate_open_clip_alphas(
            data, open_clip_model, open_clip_tokenizer, open_clip_image_cache,
            oc_gallery_ids, oc_gallery_feats, args.alphas,
        )
        for alpha, (results, _, _) in oc_results.items():
            all_results[f"OpenCLIP alpha={alpha:.2f}"] = summarize_results(results)

    if "open_clip_sep" in args.models:
        print("\n🔄 ارزیابی OpenCLIP (Separate + Normalized)")
        oc_sep_results = evaluate_open_clip_alphas_separate(
            data, open_clip_model, open_clip_tokenizer, open_clip_image_cache,
            oc_gallery_ids, oc_gallery_feats, args.alphas,
        )
        for alpha, (results, _, _) in oc_sep_results.items():
            all_results[f"OpenCLIP-sep alpha={alpha:.2f}"] = summarize_results(results)

    if "open_clip_beta" in args.models:
        print("\n🔄 ارزیابی OpenCLIP-Beta")
        if not args.generated_captions_path:
            print("❌ برای open_clip_beta باید --generated_captions_path مشخص شود.")
        else:
            generated_captions = load_generated_captions(args.generated_captions_path)
            print(f"✅ تعداد {len(generated_captions)} caption تولیدشده بارگذاری شد.")
            oc_beta_results = evaluate_open_clip_beta(
                data, open_clip_model, open_clip_tokenizer, open_clip_image_cache,
                oc_gallery_ids, oc_gallery_feats,
                generated_captions, args.alphas, args.betas,
            )
            for (alpha, beta), (results, _, _) in oc_beta_results.items():
                all_results[f"OpenCLIP-beta a={alpha:.2f} b={beta:.2f}"] = summarize_results(results)

    if "clip" in args.models:
        print("\n🔄 ارزیابی CLIP")
        clip_results = evaluate_clip_alphas(
            data, clip_model, image_features_cache, gallery_ids, gallery_feats, args.alphas
        )
        for alpha, (results, _, _) in clip_results.items():
            all_results[f"CLIP alpha={alpha:.2f}"] = summarize_results(results)

    if "clip_sep" in args.models:
        print("\n🔄 ارزیابی CLIP (Separate + Normalized)")
        clip_sep_results = evaluate_clip_alphas_separate(
            data, clip_model, image_features_cache, gallery_ids, gallery_feats, args.alphas
        )
        for alpha, (results, _, _) in clip_sep_results.items():
            all_results[f"CLIP-sep alpha={alpha:.2f}"] = summarize_results(results)

    if "clip_beta" in args.models:
        print("\n🔄 ارزیابی CLIP-Beta")
        if not args.generated_captions_path:
            print("❌ برای clip_beta باید --generated_captions_path مشخص شود.")
        else:
            generated_captions = load_generated_captions(args.generated_captions_path)
            print(f"✅ تعداد {len(generated_captions)} caption تولیدشده بارگذاری شد.")
            beta_results = evaluate_clip_beta(
                data, clip_model, image_features_cache, gallery_ids, gallery_feats,
                generated_captions, args.alphas, args.betas
            )
            for (alpha, beta), (results, _, _) in beta_results.items():
                all_results[f"CLIP-beta a={alpha:.2f} b={beta:.2f}"] = summarize_results(results)

    if "searle" in args.models:
        print("\n🔄 ارزیابی Searle")
        try:
            _SEARLE_CFG = os.path.join(os.path.dirname(__file__), "configs", "models", "searle.yaml")
            searle_config = load_searle_config(_SEARLE_CFG)
            searle, encode_with_pseudo_tokens = load_searle_model(searle_config, device=device)
            results, total, skipped = evaluate_searle(
                data, searle, encode_with_pseudo_tokens, clip_model,
                image_features_cache, image_features_cache_raw, gallery_ids, gallery_feats
            )
            all_results["Searle"] = summarize_results(results)
        except Exception as e:
            print(f"❌ خطا در ارزیابی Searle: {e}")
            traceback.print_exc()

    if "qwen" in args.models:
        print("\n🔄 ارزیابی Qwen")
        try:
            model, processor = load_qwen_model(args.qwen_model_path, device)
            qwen_image_features = build_qwen_batch_cache(
                image_ids, args.image_folder, model, processor,
                batch_size=8, dataset_type=dataset_type,
            )
            print(f"✅ تعداد {len(qwen_image_features)} تصویر Qwen پردازش شد.")
            if qwen_image_features:
                q_ids, q_feats = stack_feature_cache(qwen_image_features)
                results, total, skipped = evaluate_generic(
                    data, args.image_folder, qwen_image_features, q_ids, q_feats,
                    lambda img: get_qwen_image_feature(img, model, processor, device),
                    lambda txt: get_qwen_text_feature(txt, model, processor, device),
                    model_name="Qwen", alpha=0.5, dataset_type=dataset_type,
                )
                print(f"✅ Qwen: {total} ارزیابی، {skipped} رد شد.")
                all_results["Qwen"] = summarize_results(results)
            else:
                print("❌ هیچ ویژگی تصویری Qwen استخراج نشد!")
        except Exception as e:
            print(f"❌ خطا در ارزیابی Qwen: {e}")
            traceback.print_exc()

    if "blip" in args.models:
        print("\n🔄 ارزیابی Blip")
        try:
            model, processor = load_blip_model(args.blip_model_path)
            blip_image_features = build_generic_image_cache(
                image_ids, args.image_folder,
                lambda img: get_blip_image_feature(img, model, processor),
                dataset_type=dataset_type,
            )
            print(f"✅ تعداد {len(blip_image_features)} تصویر Blip پردازش شد.")
            if blip_image_features:
                b_ids, b_feats = stack_feature_cache(blip_image_features)
                results, total, skipped = evaluate_generic(
                    data, args.image_folder, blip_image_features, b_ids, b_feats,
                    lambda img: get_blip_image_feature(img, model, processor),
                    lambda txt: get_blip_text_feature(txt, model, processor),
                    model_name="Blip", alpha=0.5, dataset_type=dataset_type,
                )
                print(f"✅ Blip: {total} ارزیابی، {skipped} رد شد.")
                all_results["Blip"] = summarize_results(results)
            else:
                print("⚠️ هیچ ویژگی تصویری برای Blip استخراج نشد.")

            del model, processor
            if "blip_image_features" in locals():
                del blip_image_features
            if "b_ids" in locals():
                del b_ids, b_feats
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ خطا در ارزیابی Blip: {e}")
            traceback.print_exc()

    if "siglip" in args.models:
        print("\n🔄 ارزیابی SigLIP")
        try:
            model, processor = load_siglip_model(args.siglip_path)
            siglip_image_features = build_siglip_batch_cache(
                image_ids, args.image_folder, model, processor,
                dataset_type=dataset_type, batch_size=16,
            )
            print(f"✅ تعداد {len(siglip_image_features)} تصویر SigLIP پردازش شد.")
            if siglip_image_features:
                s_ids, s_feats = stack_feature_cache(siglip_image_features)
                results, total, skipped = evaluate_generic(
                    data, args.image_folder, siglip_image_features, s_ids, s_feats,
                    lambda img: get_siglip_image_feature(img, model, processor),
                    lambda txt: get_siglip_text_feature(txt, model, processor),
                    model_name="SigLIP", alpha=0.5, dataset_type=dataset_type,
                )
                print(f"✅ SigLIP: {total} ارزیابی، {skipped} رد شد.")
                all_results["SigLIP"] = summarize_results(results)
            else:
                print("⚠️ هیچ ویژگی تصویری برای SigLIP استخراج نشد.")

            del model, processor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ خطا در ارزیابی SigLIP: {e}")
            traceback.print_exc()

    if "lava" in args.models:
        print("\n🔄 ارزیابی LLaVA")
        try:
            model, processor = load_lava_model()
            lava_image_features = build_generic_image_cache(
                image_ids, args.image_folder,
                lambda img: get_lava_image_feature(img, model, processor),
                dataset_type=dataset_type,
            )
            if lava_image_features:
                l_ids, l_feats = stack_feature_cache(lava_image_features)
                results, total, skipped = evaluate_generic(
                    data, args.image_folder, lava_image_features, l_ids, l_feats,
                    lambda img: get_lava_image_feature(img, model, processor),
                    lambda txt: get_lava_text_feature(txt, model, processor),
                    model_name="LLaVA", alpha=0.5, dataset_type=dataset_type,
                )
                all_results["LLaVA"] = summarize_results(results)
        except Exception as e:
            print(f"❌ خطا در ارزیابی LLaVA: {e}")
            traceback.print_exc()

    print_results_table(args, all_results)


if __name__ == "__main__":
    main()
