"""Information-retrieval and classification metrics.

Pure functions over plain Python containers - no numpy, no sklearn. Each one is
explicit about its edge cases, because that is precisely where evaluation code
usually lies to you:

* a query with **no** relevant documents cannot have a recall; it returns
  ``float('nan')`` and is excluded from macro-averages rather than silently
  counted as 1.0 (or 0.0);
* duplicate document ids in a ranking are collapsed, keeping the best rank;
* ``precision@k`` divides by ``k`` (``strict_k=True``, the standard IR
  definition) so returning fewer than ``k`` documents is correctly penalised.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "dedupe",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "average_precision_at_k",
    "ndcg_at_k",
    "retrieval_metrics",
    "macro_average",
    "bootstrap_ci",
    "binary_classification_metrics",
    "roc_auc",
    "threshold_sweep",
    "token_f1",
]

NAN = float("nan")


def dedupe(items: Sequence[str]) -> List[str]:
    """Remove repeats while preserving first occurrence (i.e. the best rank)."""
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _validate_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    return k


def _relevant_set(relevant: Iterable[str] | Mapping[str, int]) -> Dict[str, int]:
    if isinstance(relevant, Mapping):
        return {str(d): int(g) for d, g in relevant.items() if int(g) > 0}
    return {str(d): 1 for d in relevant}


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str] | Mapping[str, int],
    k: int,
    *,
    strict_k: bool = True,
) -> float:
    """Fraction of the top-``k`` results that are relevant.

    With ``strict_k=True`` (default) the denominator is always ``k``: a system
    that returns 2 documents when asked for 5 is penalised, which is the
    behaviour you want when comparing pipelines. Set ``strict_k=False`` to
    divide by the number actually returned.
    """
    _validate_k(k)
    rel = _relevant_set(relevant)
    ranking = dedupe(retrieved)[:k]
    denom = k if strict_k else len(ranking)
    if denom == 0:
        return 0.0
    hits = sum(1 for doc_id in ranking if doc_id in rel)
    return hits / denom


def recall_at_k(
    retrieved: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int
) -> float:
    """Fraction of all relevant documents found in the top ``k``.

    Returns ``nan`` when the query has no relevant documents - undefined, not
    zero. :func:`macro_average` skips ``nan`` values.
    """
    _validate_k(k)
    rel = _relevant_set(relevant)
    if not rel:
        return NAN
    ranking = dedupe(retrieved)[:k]
    hits = sum(1 for doc_id in ranking if doc_id in rel)
    return hits / len(rel)


def f1_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str] | Mapping[str, int],
    k: int,
    *,
    strict_k: bool = True,
) -> float:
    """Harmonic mean of precision@k and recall@k."""
    p = precision_at_k(retrieved, relevant, k, strict_k=strict_k)
    r = recall_at_k(retrieved, relevant, k)
    if math.isnan(r):
        return NAN
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate_at_k(
    retrieved: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int
) -> float:
    """1.0 if at least one relevant document appears in the top ``k``.

    Often the metric that actually matters: can the generator possibly be right?
    """
    _validate_k(k)
    rel = _relevant_set(relevant)
    if not rel:
        return NAN
    return 1.0 if any(doc_id in rel for doc_id in dedupe(retrieved)[:k]) else 0.0


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: Optional[int] = None
) -> float:
    """1 / rank of the first relevant document (0.0 if none within ``k``)."""
    rel = _relevant_set(relevant)
    if not rel:
        return NAN
    ranking = dedupe(retrieved)
    if k is not None:
        ranking = ranking[: _validate_k(k)]
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved: Sequence[str], relevant: Iterable[str] | Mapping[str, int], k: int
) -> float:
    """Mean of precision@i taken at every relevant hit, normalised by
    ``min(|relevant|, k)`` - the standard truncated AP used for MAP@k."""
    _validate_k(k)
    rel = _relevant_set(relevant)
    if not rel:
        return NAN
    ranking = dedupe(retrieved)[:k]
    hits = 0
    total = 0.0
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id in rel:
            hits += 1
            total += hits / i
    denom = min(len(rel), k)
    return total / denom if denom else 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str] | Mapping[str, int],
    k: int,
    *,
    exponential: bool = True,
) -> float:
    """Normalised discounted cumulative gain with graded relevance.

    Gain is ``2**rel - 1`` (``exponential=True``, the TREC convention) or the
    raw grade; discount is ``1 / log2(rank + 1)``.
    """
    _validate_k(k)
    rel = _relevant_set(relevant)
    if not rel:
        return NAN
    ranking = dedupe(retrieved)[:k]

    def gain(grade: int) -> float:
        return (2.0 ** grade - 1.0) if exponential else float(grade)

    dcg = sum(
        gain(rel.get(doc_id, 0)) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranking, start=1)
    )
    ideal_grades = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(gain(g) / math.log2(rank + 1) for rank, g in enumerate(ideal_grades, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    retrieved: Sequence[str],
    relevant: Iterable[str] | Mapping[str, int],
    ks: Sequence[int] = (1, 3, 5, 10),
    *,
    strict_k: bool = True,
) -> Dict[str, float]:
    """Compute the full metric family at every requested cut-off.

    Keys look like ``precision@5``, ``recall@5``, ``ndcg@10``, plus a single
    un-truncated ``mrr``.
    """
    rel = _relevant_set(relevant)
    ranking = dedupe(retrieved)
    out: Dict[str, float] = {}
    for k in sorted({_validate_k(int(k)) for k in ks}):
        out[f"precision@{k}"] = precision_at_k(ranking, rel, k, strict_k=strict_k)
        out[f"recall@{k}"] = recall_at_k(ranking, rel, k)
        out[f"f1@{k}"] = f1_at_k(ranking, rel, k, strict_k=strict_k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(ranking, rel, k)
        out[f"map@{k}"] = average_precision_at_k(ranking, rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranking, rel, k)
    out["mrr"] = reciprocal_rank(ranking, rel)
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def macro_average(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Average each metric across queries, ignoring ``nan`` (undefined) values."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts[key]}


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 13,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Thirty queries is a small test set, so a point estimate on its own is
    misleading; this quantifies how much of a metric change is just noise.
    """
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return (NAN, NAN)
    if len(clean) == 1:
        return (clean[0], clean[0])
    rng = random.Random(seed)
    n = len(clean)
    means: List[float] = []
    for _ in range(max(int(resamples), 1)):
        total = 0.0
        for _ in range(n):
            total += clean[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(int(math.floor(alpha * len(means))), 0)]
    hi = means[min(int(math.ceil((1.0 - alpha) * len(means))) - 1, len(means) - 1)]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Classification metrics (used to score the hallucination detector itself)
# --------------------------------------------------------------------------- #


def binary_classification_metrics(
    y_true: Sequence[bool], y_pred: Sequence[bool]
) -> Dict[str, float]:
    """Precision/recall/F1/accuracy/balanced-accuracy plus the raw confusion cells."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": float(n),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": (tp + tn) / n if n else 0.0,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def roc_auc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """ROC AUC via the rank (Mann-Whitney U) identity, with correct tie handling.

    Returns ``nan`` when one class is missing, since AUC is undefined there.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must be the same length")
    pos = sum(1 for t in y_true if t)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return NAN
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        for m in range(i, j + 1):
            ranks[order[m]] = avg_rank
        i = j + 1
    rank_sum = sum(r for r, t in zip(ranks, y_true) if t)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def threshold_sweep(
    y_true: Sequence[bool], scores: Sequence[float], *, steps: int = 101
) -> List[Dict[str, float]]:
    """Evaluate the detector at every threshold on a uniform grid over [0, 1].

    Used both for picking an operating point and for drawing the
    precision/recall-vs-threshold curve in the HTML report.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must be the same length")
    steps = max(int(steps), 2)
    rows: List[Dict[str, float]] = []
    for i in range(steps):
        thr = i / (steps - 1)
        preds = [s >= thr for s in scores]
        row = binary_classification_metrics(y_true, preds)
        row["threshold"] = thr
        rows.append(row)
    return rows


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 between a generated answer and a gold answer.

    A cheap, deterministic proxy for answer quality; it deliberately ignores
    word order, which is what you want for short factual answers.
    """
    from .text import tokenize  # local import keeps this module import-light

    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts: Dict[str, int] = {}
    for t in pred:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    overlap = 0
    for t in ref:
        if pred_counts.get(t, 0) > 0:
            pred_counts[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)
