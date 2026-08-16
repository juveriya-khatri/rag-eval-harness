"""rag-eval-harness - verification for RAG pipelines.

Score retrieval quality (precision/recall/nDCG@k) and flag likely hallucinations
against a small labelled test set, with zero required dependencies.

Quick start
-----------
>>> from ragval import load_dataset, evaluate
>>> report = evaluate(load_dataset("bundled"))
>>> round(report.retrieval["recall@5"], 3)  # doctest: +SKIP
0.95
"""

from __future__ import annotations

__version__ = "0.1.0"

from .dataset import Dataset, load_dataset, load_jsonl
from .evaluate import EvalOptions, evaluate
from .gate import make_baseline, run_gate
from .grounding import GroundingChecker, GroundingConfig
from .index import BM25Retriever, HybridRetriever, Retriever, TfidfRetriever, build_retriever
from .metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
    reciprocal_rank,
)
from .report import render_terminal, write_html, write_json, write_junit
from .types import AnswerCase, Document, EvalReport, Hit, QueryCase, QueryResult

__all__ = [
    "__version__",
    "Dataset",
    "load_dataset",
    "load_jsonl",
    "evaluate",
    "EvalOptions",
    "run_gate",
    "make_baseline",
    "GroundingChecker",
    "GroundingConfig",
    "Retriever",
    "BM25Retriever",
    "TfidfRetriever",
    "HybridRetriever",
    "build_retriever",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "retrieval_metrics",
    "render_terminal",
    "write_html",
    "write_json",
    "write_junit",
    "Document",
    "QueryCase",
    "AnswerCase",
    "Hit",
    "QueryResult",
    "EvalReport",
]
