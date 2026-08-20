def load_generated_captions(json_path):
    """
    خروجی همیشه: { image_id(str): caption(str) }
    فرمت‌های پشتیبانی‌شده: dict کلیددار با image_id، یا list از dict ها.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_keys = ("image_id", "img_id", "reference_img_id", "id", "reference")
    cap_keys = ("caption", "generated_caption", "text")
    captions = {}

    def _norm_id(raw):
        return os.path.splitext(str(raw))[0]

    if isinstance(data, dict):
        for raw_id, value in data.items():
            img_id = _norm_id(raw_id)
            if isinstance(value, dict):
                caption = next((value[k] for k in cap_keys if value.get(k)), "")
            elif isinstance(value, str):
                caption = value
            else:
                caption = ""
            captions[img_id] = caption
        return captions

    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            raw_id = next((entry[k] for k in id_keys if entry.get(k) is not None), None)
            if raw_id is None:
                continue
            captions[_norm_id(raw_id)] = next(
                (entry[k] for k in cap_keys if entry.get(k)), ""
            )
        return captions

    raise ValueError(f"فرمت ناشناخته‌ی generated captions: {type(data)}")



MAP_K_VALUES = (5, 10, 50)

def init_results():
    return {
        "prec": {k: [] for k in K_VALUES},
        "rec":  {k: [] for k in K_VALUES},
        "map":  {k: [] for k in MAP_K_VALUES},
        "mrr":  [],
    }

def update_metrics(results, ranked_ids, positives):
    n_pos = len(positives)
    for k in K_VALUES:
        top_k = ranked_ids[:k]
        hits = sum(1 for img in top_k if img in positives)
        results["prec"][k].append(hits / k if k else 0.0)
        results["rec"][k].append(hits / n_pos if n_pos else 0.0)

    for k in MAP_K_VALUES:
        results["map"][k].append(average_precision_at_k(positives, ranked_ids, k))
    results["mrr"].append(mean_reciprocal_rank(positives, ranked_ids))





    image_ids = set()
    for item in data:
        image_ids.add(item["reference_id"])
        image_ids.add(item["target_id"])
        for member in item.get("members", []):
            image_ids.add(member)

    dataset_type = args.dataset

    need_clip_cache = any(m in args.models for m in ("clip", "searle", "clip_sep", "clip_beta"))
    need_open_clip = "open_clip" in args.models

    clip_model = preprocess = None
    open_clip_model = open_clip_preprocess = None
    image_features_cache = image_features_cache_raw = None
    gallery_ids = gallery_feats = None

    if need_open_clip:
        print("🔄 بارگذاری مدل open_clip ...")
        open_clip_model, open_clip_preprocess = load_open_clip_model()

    if need_clip_cache:
        print("🔄 بارگذاری مدل CLIP...")
        clip_model, preprocess = load_clip_model()
        image_features_cache, image_features_cache_raw = build_clip_image_cache(
            image_ids, args.image_folder, clip_model, preprocess, dataset_type
        )
        gallery_ids, gallery_feats = stack_feature_cache(image_features_cache)



    header = (f"{'Model/Alpha':<26} {'MRR':<9} {'mAP@5':<9} {'mAP@10':<9} {'mAP@50':<9} "
              f"{'Prec@1':<9} {'Prec@5':<9} {'Prec@10':<9} {'Prec@50':<9} "
              f"{'Rec@1':<9} {'Rec@5':<9} {'Rec@10':<9} {'Rec@50':<9}")

    def _row(label, r):
        cols = ["mrr", "map5", "map10", "map50",
                "prec1", "prec5", "prec10", "prec50",
                "rec1", "rec5", "rec10", "rec50"]
        vals = " ".join(f"{r.get(c, float('nan')):<9.4f}" for c in cols)
        print(f"{label:<26} {vals}")














ref_index = sample.get("img_set", {}).get("reference_rank", 0)
target_id = members[ref_index]      # ← این reference را برمی‌گرداند



def parse_cirr_sample(sample):
    reference_id = os.path.splitext(sample["reference"])[0]
    target_id    = os.path.splitext(sample["target_hard"])[0]   # ← منبع درست
    members = [os.path.splitext(m)[0]
               for m in sample.get("img_set", {}).get("members", [])]
    return {
        "reference_id": reference_id,
        "caption": sample.get("caption", "").strip(),
        "target_id": target_id,
        "positives": [target_id],      # CIRR: تک target
        "members": members,            # فقط برای پروتکل subset
    }
def parse_circo_sample(sample):
    gt_ids = [str(x) for x in sample.get("gt_img_ids", [])]
    target_id = str(sample.get("target_img_id"))
    if target_id not in gt_ids:
        gt_ids = [target_id] + gt_ids
    return {
        "reference_id": str(sample.get("reference_img_id")),
        "caption": sample.get("relative_caption", "").strip(),
        "target_id": target_id,
        "positives": gt_ids,          # ← همهٔ groundtruthها
        "members": gt_ids,
    }


positives = set(item.get("positives") or [item["target_id"]])



--------------------------


def build_id_index(gallery_ids):
    return {str(g): i for i, g in enumerate(gallery_ids)}

def rank_from_sims(sims, gallery_ids, id2idx,
                   exclude_ids=(), restrict_ids=None):
    """sims: 1-D tensor هم‌طول gallery_ids"""
    sims = sims.detach().float().clone().squeeze()

    if restrict_ids is not None:                   # پروتکل subset در CIRR
        mask = torch.full_like(sims, float("-inf"))
        for rid in restrict_ids:
            j = id2idx.get(str(rid))
            if j is not None:
                mask[j] = sims[j]
        sims = mask

    for ex in exclude_ids:                         # حذف تصویر مرجع
        j = id2idx.get(str(ex))
        if j is not None:
            sims[j] = float("-inf")

    order = torch.argsort(sims, descending=True).cpu().tolist()
    return [gallery_ids[i] for i in order if sims[i] != float("-inf")]



in evaluate_clip_alphas

sims = alpha * sims_txt + (1.0 - alpha) * sims_ref
ranked_ids = rank_from_sims(
    sims, gallery_ids, id2idx,
    exclude_ids=[item["reference_id"]],
    restrict_ids=item["members"] if SUBSET_PROTOCOL else None,
)
update_metrics(results, positives, ranked_ids)






def build_cirr_gallery(split_json):          # split.rc2.val.json
    with open(split_json, encoding="utf-8") as f:
        mapping = json.load(f)               # {"dev-0-0-img0": "./dev/...png", ...}
    return sorted({os.path.splitext(os.path.basename(v))[0] for v in mapping.values()})
    # انتظار: ۲۲۹۷ تصویر برای val

def build_circo_gallery(img_dir):            # COCO unlabeled2017
    exts = (".jpg", ".jpeg", ".png")
    return sorted({os.path.splitext(f)[0] for f in os.listdir(img_dir)
                   if f.lower().endswith(exts)})
    # انتظار: ~۱۲۳٬۴۰۳ تصویر








# ۱) خودبازیابی: بدون ماسک reference باید rank-1 خودش باشد
assert rank_from_sims(gallery_feats @ gallery_feats[k], gallery_ids, id2idx)[0] == gallery_ids[k]
# ۲) با ماسک، reference نباید در لیست باشد
assert item["reference_id"] not in ranked_ids
# ۳) پوشش groundtruth
assert all(p in id2idx for p in positives), "gt در gallery نیست"
# ۴) اندازه
print(len(gallery_ids))   # CIRR val: 2297




# قبل از argsort، امتیاز مرجع را بی‌اثر کن
ref_idx = id2idx.get(item["reference_id"])
if ref_idx is not None:
    sims[ref_idx] = -float("inf")
order = sims.argsort(descending=True)






def rank_by_similarity(gallery_feats, gallery_ids, query_feat, exclude_id=None, id2idx=None):
    sims = torch.matmul(gallery_feats, query_feat...)     # خط 803 (بدون تغییر)
    if exclude_id is not None and id2idx is not None:      # ← اضافه شود
        mask_reference_(sims, id2idx, exclude_id)
    ranked_idx = torch.argsort(sims, descending=True)      # خط 804
    return [gallery_ids[i] for i in ranked_idx]            # خط 805
