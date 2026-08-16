"""Evaluation orchestration.

:func:`evaluate` is the single entry point that ties the pieces together:

    dataset -> retriever -> IR metrics -> grounding checker -> EvalReport

Two properties are treated as requirements rather than nice-to-haves:

* **Reproducibility.** No randomness anywhere except the seeded bootstrap.
  Re-running on the same inputs produces identical numbers.
* **Honest aggregation.** Metrics are macro-averaged per query (so a query with
  many relevant documents cannot dominate), undefined values are dropped rather
  than coerced to zero, and every headline number ships with a bootstrap
  confidence interval.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import Dataset
from .grounding import GroundingChecker, GroundingConfig
from .index import Retriever, build_retriever
from .metrics import (
    binary_classification_metrics,
    bootstrap_ci,
    macro_average,
    retrieval_metrics,
    roc_auc,
    threshold_sweep,
)
from .types import CONTRADICTED, SUPPORTED, EvalReport, GroundingResult, QueryResult

__all__ = ["evaluate", "EvalOptions"]

DEFAULT_KS: Tuple[int, ...] = (1, 3, 5, 10)
HEADLINE_METRICS = ("precision@5", "recall@5", "ndcg@5", "hit_rate@5", "mrr")


class EvalOptions:
    """Container for run configuration (kept simple and explicit)."""

    def __init__(
        self,
        *,
        ks: Sequence[int] = DEFAULT_KS,
        retriever: str = "bm25",
        context_k: Optional[int] = None,
        strict_k: bool = True,
        bootstrap: int = 1000,
        seed: int = 13,
        grounding: Optional[GroundingConfig] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
    ):
        ks = sorted({int(k) for k in ks})
        if not ks or any(k <= 0 for k in ks):
            raise ValueError("ks must be a non-empty list of positive integers")
        self.ks = tuple(ks)
        self.retriever = retriever
        self.context_k = int(context_k) if context_k else min(5, max(self.ks))
        self.strict_k = strict_k
        self.bootstrap = max(int(bootstrap), 0)
        self.seed = int(seed)
        self.grounding = grounding or GroundingConfig()
        self.retriever_kwargs = dict(retriever_kwargs or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ks": list(self.ks),
            "retriever": self.retriever,
            "context_k": self.context_k,
            "strict_k": self.strict_k,
            "bootstrap": self.bootstrap,
            "seed": self.seed,
            "grounding": self.grounding.to_dict(),
            "retriever_kwargs": self.retriever_kwargs,
        }


def evaluate(
    dataset: Dataset,
    options: Optional[EvalOptions] = None,
    *,
    retriever: Optional[Retriever] = None,
) -> EvalReport:
    """Run the full evaluation and return an :class:`EvalReport`.

    Parameters
    ----------
    dataset:
        Corpus + labelled queries (+ optional answers under test).
    options:
        Run configuration. Defaults evaluate BM25 at k = 1/3/5/10.
    retriever:
        Bring your own retriever - anything exposing ``search(query, k)``.
        When provided, ``options.retriever`` is ignored.
    """
    opts = options or EvalOptions()
    engine = retriever or build_retriever(opts.retriever, dataset.documents, **opts.retriever_kwargs)
    max_k = max(max(opts.ks), opts.context_k)

    checker: Optional[GroundingChecker] = None
    if dataset.answers:
        checker = GroundingChecker(dataset.documents, opts.grounding)

    results: List[QueryResult] = []
    for query in dataset.queries:
        started = time.perf_counter()
        hits = engine.search(query.question, max_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranking = [h.doc_id for h in hits]
        metrics = retrieval_metrics(ranking, query.relevant, opts.ks, strict_k=opts.strict_k)

        grounding: Optional[GroundingResult] = None
        answers = dataset.answers_for(query.id)
        if checker is not None and answers:
            context_ids = ranking[: opts.context_k]
            context_docs = [d for d in (engine.get(doc_id) for doc_id in context_ids) if d]
            grounding = checker.check(
                answers[0].answer,
                context_docs,
                query_id=query.id,
                gold_hallucinated=answers[0].hallucinated,
                gold_answer=query.gold_answer,
            )
            # Record how much gold evidence actually reached the generator, so a
            # grounding flag can be attributed to the right root cause.
            gold_ids = query.relevant_ids
            if gold_ids:
                found = sum(1 for doc_id in gold_ids if doc_id in context_ids)
                grounding.context_recall = found / len(gold_ids)
            grounding.context_doc_ids = list(context_ids)

        results.append(
            QueryResult(
                query_id=query.id,
                question=query.question,
                tags=list(query.tags),
                hits=hits[: max(opts.ks)],
                metrics=metrics,
                n_relevant=len(query.relevant_ids),
                latency_ms=latency_ms,
                grounding=grounding,
            )
        )

    retrieval = macro_average([r.metrics for r in results])
    retrieval["avg_latency_ms"] = (
        sum(r.latency_ms for r in results) / len(results) if results else 0.0
    )

    ci: Dict[str, Sequence[float]] = {}
    if opts.bootstrap:
        for name in _ci_metric_names(opts.ks):
            values = [r.metrics[name] for r in results if name in r.metrics]
            lo, hi = bootstrap_ci(values, resamples=opts.bootstrap, seed=opts.seed)
            ci[name] = (lo, hi)

    detector = _score_detector(results, opts)
    by_tag = _by_tag(results)

    return EvalReport(
        config={
            **opts.to_dict(),
            "retriever_name": getattr(engine, "name", opts.retriever),
            "dataset": dataset.name,
        },
        retrieval=retrieval,
        retrieval_ci=ci,
        detector=detector,
        by_tag=by_tag,
        queries=results,
        corpus_stats=dataset.stats(),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def _ci_metric_names(ks: Sequence[int]) -> List[str]:
    names = ["mrr"]
    for k in ks:
        names.extend([f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"])
    return names


def _by_tag(results: Sequence[QueryResult]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, float]]] = {}
    for result in results:
        for tag in result.tags or ["untagged"]:
            buckets.setdefault(tag, []).append(result.metrics)
    out: Dict[str, Dict[str, float]] = {}
    for tag, rows in sorted(buckets.items()):
        agg = macro_average(rows)
        agg["n_queries"] = float(len(rows))
        out[tag] = agg
    return out


def _score_detector(results: Sequence[QueryResult], opts: EvalOptions) -> Dict[str, Any]:
    """Score the hallucination detector against the gold answer-level labels."""
    grounded = [r.grounding for r in results if r.grounding is not None]
    if not grounded:
        return {}

    labeled = [g for g in grounded if g.gold_hallucinated is not None]
    n_claims = sum(len(g.claims) for g in grounded)
    n_flagged_claims = sum(len(g.flagged_claims) for g in grounded)
    summary: Dict[str, Any] = {
        "n_answers": len(grounded),
        "n_labeled": len(labeled),
        "n_claims": n_claims,
        "n_flagged_claims": n_flagged_claims,
        "mean_faithfulness": round(
            sum(g.faithfulness for g in grounded) / len(grounded), 4
        ),
        "mean_risk": round(sum(g.risk for g in grounded) / len(grounded), 4),
        "risk_threshold": opts.grounding.risk_threshold,
        "flag_rate": round(sum(1 for g in grounded if g.flagged) / len(grounded), 4),
    }
    sims = [g.answer_similarity for g in grounded if g.answer_similarity is not None]
    if sims:
        summary["mean_answer_token_f1"] = round(sum(sims) / len(sims), 4)

    if not labeled:
        return summary

    y_true = [bool(g.gold_hallucinated) for g in labeled]
    scores = [g.risk for g in labeled]
    y_pred = [g.flagged for g in labeled]

    summary.update(
        {k: round(v, 4) for k, v in binary_classification_metrics(y_true, y_pred).items()}
    )
    summary["roc_auc"] = round(roc_auc(y_true, scores), 4)

    sweep = threshold_sweep(y_true, scores, steps=101)
    best = max(sweep, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    summary["best_threshold"] = round(best["threshold"], 3)
    summary["best_f1"] = round(best["f1"], 4)
    summary["sweep"] = [
        {
            "threshold": round(row["threshold"], 3),
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1": round(row["f1"], 4),
        }
        for row in sweep
    ]
    summary["errors"] = [
        {
            "query_id": g.query_id,
            "kind": "false_negative" if g.gold_hallucinated else "false_positive",
            "risk": round(g.risk, 4),
            "faithfulness": round(g.faithfulness, 4),
            "retrieval_starved": g.retrieval_starved,
        }
        for g in labeled
        if bool(g.gold_hallucinated) != g.flagged
    ]

    # Root-cause attribution. A flag on an answer whose gold evidence never
    # reached the context is a retrieval failure wearing a grounding costume;
    # reporting the two together hides which half of the pipeline to fix.
    starved = [g for g in labeled if g.retrieval_starved]
    summary["retrieval_starved_answers"] = len(starved)
    summary["flags_from_retrieval_miss"] = sum(1 for g in starved if g.flagged)
    grounded_ctx = [g for g in labeled if not g.retrieval_starved]
    if grounded_ctx and len(grounded_ctx) != len(labeled):
        conditioned = binary_classification_metrics(
            [bool(g.gold_hallucinated) for g in grounded_ctx],
            [g.flagged for g in grounded_ctx],
        )
        summary["given_correct_retrieval"] = {
            "n": len(grounded_ctx),
            "precision": round(conditioned["precision"], 4),
            "recall": round(conditioned["recall"], 4),
            "f1": round(conditioned["f1"], 4),
        }

    # Claim-level diagnostics against optional per-claim gold labels.
    claim_stats = _claim_level(labeled)
    if claim_stats:
        summary["claim_level"] = claim_stats
    return summary


def _claim_level(grounded: Sequence[GroundingResult]) -> Dict[str, Any]:
    counts = {SUPPORTED: 0, "unsupported": 0, CONTRADICTED: 0}
    for g in grounded:
        for claim in g.claims:
            counts[claim.verdict] = counts.get(claim.verdict, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {
        "counts": counts,
        "share_supported": round(counts.get(SUPPORTED, 0) / total, 4),
        "share_contradicted": round(counts.get(CONTRADICTED, 0) / total, 4),
    }
