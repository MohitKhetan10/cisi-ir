import math
import numpy as np

def precision_at_k(ranked_ids, relevant_set, k=10):
    return sum(1 for d in ranked_ids[:k] if d in relevant_set) / k

def dcg_at_k(ranked_ids, relevant_set, k=10):
    return sum((1 if d in relevant_set else 0) / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))

def ndcg_at_k(ranked_ids, relevant_set, k=10):
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_set))))
    return dcg_at_k(ranked_ids, relevant_set, k) / (ideal if ideal > 0 else 1.0)

def average_precision(ranked_ids, relevant_set):
    hits, ap = 0, 0.0
    for i, d in enumerate(ranked_ids, 1):
        if d in relevant_set:
            hits += 1
            ap += hits / i
    return ap / max(1, len(relevant_set))

def reciprocal_rank(ranked_ids, relevant_set):
    for i, d in enumerate(ranked_ids, 1):
        if d in relevant_set:
            return 1.0 / i
    return 0.0

def evaluate_runs(runs, qrels, k=10):
    aps, ps, ndcgs, mrrs = [], [], [], []
    for qid, ranked in runs.items():
        ranked_ids = [docid for docid, _ in ranked]
        relevant_set = qrels.get(qid, set())
        if not relevant_set:
            continue
        aps.append(average_precision(ranked_ids, relevant_set))
        ps.append(precision_at_k(ranked_ids, relevant_set, k))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant_set, k))
        mrrs.append(reciprocal_rank(ranked_ids, relevant_set))
    return {
        'MAP': float(np.mean(aps)),
        f'Precision@{k}': float(np.mean(ps)),
        f'nDCG@{k}': float(np.mean(ndcgs)),
        'MRR': float(np.mean(mrrs))
    }
