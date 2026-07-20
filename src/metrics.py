import numpy as np

from .config import K_VALUES, MAP_K_VALUES


def init_results():
    return {
        "prec": {k: [] for k in K_VALUES},
        "rec": {k: [] for k in K_VALUES},
        "map": {k: [] for k in MAP_K_VALUES},
        "mrr": [],
    }


def average_precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    score = 0.0
    num_hits = 0.0
    for i, img in enumerate(retrieved_k, start=1):
        if img in relevant:
            num_hits += 1.0
            score += num_hits / i
    if len(relevant) == 0:
        return 0.0
    return score / min(len(relevant), k)


def mean_reciprocal_rank(relevant, retrieved):
    for rank, img in enumerate(retrieved, start=1):
        if img in relevant:
            return 1.0 / rank
    return 0.0


def update_metrics(results, ranked_ids, positives):
    n_pos = len(positives)
    for k in K_VALUES:
        top_k = ranked_ids[:k]
        hits = sum(1 for img in top_k if img in positives)
        results["prec"][k].append(hits / k)
        results["rec"][k].append(hits / n_pos if n_pos else 0.0)

    # FIX (باگ ۱): mAP برای همه‌ی kها از جمله 50 محاسبه و append می‌شود.
    # قبلاً فقط 5 و 10 پر می‌شد و map@50 همیشه خالی می‌ماند.
    for k in results["map"]:
        results["map"][k].append(average_precision_at_k(positives, ranked_ids, k))

    results["mrr"].append(mean_reciprocal_rank(positives, ranked_ids))


def summarize_results(results):
    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "mrr": _mean(results["mrr"]),
        "map5": _mean(results["map"][5]),
        "map10": _mean(results["map"][10]),
        "map50": _mean(results["map"][50]),
        "prec1": _mean(results["prec"][1]),
        "prec5": _mean(results["prec"][5]),
        "prec10": _mean(results["prec"][10]),
        "prec50": _mean(results["prec"][50]),
        "rec1": _mean(results["rec"][1]),
        "rec5": _mean(results["rec"][5]),
        "rec10": _mean(results["rec"][10]),
        "rec50": _mean(results["rec"][50]),
    }
