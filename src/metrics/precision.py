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
