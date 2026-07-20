import torch


def rank_by_similarity(query_feat, gallery_feats):
    sims = torch.matmul(gallery_feats, query_feat.squeeze(0).float().T).squeeze(-1)
    ranked_idx = torch.argsort(sims, descending=True)
    return ranked_idx.cpu().tolist()


def minmax_normalize(sims):
    """نرمال‌سازی per-query در بازه [0,1]"""
    s = sims.squeeze(-1)
    s_min = s.min()
    s_max = s.max()
    if (s_max - s_min) < 1e-8:
        return torch.zeros_like(s)
    return (s - s_min) / (s_max - s_min)
