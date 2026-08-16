"""Core data structures shared across the harness.

Everything here is a plain dataclass with an explicit ``to_dict``/``from_dict``
pair so that results round-trip losslessly through JSON. No third-party
dependencies, no implicit schema magic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "Document",
    "QueryCase",
    "AnswerCase",
    "Hit",
    "ClaimVerdict",
    "GroundingResult",
    "QueryResult",
    "EvalReport",
    "SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
]

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"


def _as_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"missing required field {field_name!r}")
    if not isinstance(value, str):
        return str(value)
    return value


@dataclass(frozen=True)
class Document:
    """A single retrievable unit (a chunk, passage, or whole document)."""

    id: str
    text: str
    title: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def indexable_text(self) -> str:
        """Title is repeated once so title terms carry a mild weight boost."""
        return f"{self.title}\n{self.text}" if self.title else self.text

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "text": self.text, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Document":
        return cls(
            id=_as_str(raw.get("id"), "id"),
            text=_as_str(raw.get("text", ""), "text"),
            title=str(raw.get("title", "") or ""),
            meta=dict(raw.get("meta", {}) or {}),
        )


@dataclass(frozen=True)
class QueryCase:
    """A labeled query: the question plus graded relevance judgements.

    ``relevant`` maps ``doc_id -> gain`` where the gain is a non-negative
    integer (0 = not relevant, 1 = partially relevant, 2 = highly relevant).
    Binary datasets simply use ``1`` everywhere.
    """

    id: str
    question: str
    relevant: Dict[str, int] = field(default_factory=dict)
    gold_answer: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def relevant_ids(self) -> List[str]:
        return [doc_id for doc_id, gain in self.relevant.items() if gain > 0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "relevant": dict(self.relevant),
            "gold_answer": self.gold_answer,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QueryCase":
        relevant_raw = raw.get("relevant", raw.get("relevant_doc_ids", {}))
        relevant: Dict[str, int] = {}
        if isinstance(relevant_raw, Mapping):
            for doc_id, gain in relevant_raw.items():
                relevant[str(doc_id)] = int(gain)
        else:  # a bare list of ids means binary relevance
            for doc_id in relevant_raw or []:
                relevant[str(doc_id)] = 1
        return cls(
            id=_as_str(raw.get("id"), "id"),
            question=_as_str(raw.get("question", raw.get("query", "")), "question"),
            relevant=relevant,
            gold_answer=str(raw.get("gold_answer", "") or ""),
            tags=[str(t) for t in (raw.get("tags", []) or [])],
        )


@dataclass(frozen=True)
class AnswerCase:
    """A generated answer under test, with optional ground-truth labels.

    ``hallucinated`` is the answer-level gold label used to score the detector.
    ``claim_labels`` optionally pins the expected verdict for individual claims
    and is used for the claim-level diagnostic breakdown.
    """

    query_id: str
    answer: str
    system: str = "system"
    hallucinated: Optional[bool] = None
    claim_labels: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "system": self.system,
            "hallucinated": self.hallucinated,
            "claim_labels": list(self.claim_labels),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AnswerCase":
        label = raw.get("hallucinated")
        if isinstance(label, str):
            label = label.strip().lower() in {"1", "true", "yes", "y"}
        return cls(
            query_id=_as_str(raw.get("query_id", raw.get("id")), "query_id"),
            answer=_as_str(raw.get("answer", ""), "answer"),
            system=str(raw.get("system", "system") or "system"),
            hallucinated=None if label is None else bool(label),
            claim_labels=[dict(c) for c in (raw.get("claim_labels", []) or [])],
            note=str(raw.get("note", "") or ""),
        )


@dataclass(frozen=True)
class Hit:
    """One retrieved document with its score and 1-based rank."""

    doc_id: str
    score: float
    rank: int
    title: str = ""
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "title": self.title,
            "snippet": self.snippet,
        }


@dataclass
class ClaimVerdict:
    """Verdict for one atomic claim extracted from an answer."""

    claim: str
    verdict: str
    support: float
    reasons: List[str] = field(default_factory=list)
    evidence_doc_id: str = ""
    evidence: str = ""

    @property
    def flagged(self) -> bool:
        return self.verdict in (UNSUPPORTED, CONTRADICTED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "support": round(self.support, 4),
            "reasons": list(self.reasons),
            "evidence_doc_id": self.evidence_doc_id,
            "evidence": self.evidence,
        }


@dataclass
class GroundingResult:
    """Answer-level grounding assessment."""

    query_id: str
    answer: str
    claims: List[ClaimVerdict] = field(default_factory=list)
    faithfulness: float = 1.0
    risk: float = 0.0
    flagged: bool = False
    gold_hallucinated: Optional[bool] = None
    answer_similarity: Optional[float] = None
    context_recall: Optional[float] = None
    context_doc_ids: List[str] = field(default_factory=list)

    @property
    def flagged_claims(self) -> List[ClaimVerdict]:
        return [c for c in self.claims if c.flagged]

    @property
    def retrieval_starved(self) -> bool:
        """True when none of the query's gold evidence reached the context.

        A flag raised in this state is a *retrieval* failure being reported by
        the grounding checker, not a fabrication by the generator - and the two
        need completely different fixes.
        """
        return self.context_recall is not None and self.context_recall <= 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "faithfulness": round(self.faithfulness, 4),
            "risk": round(self.risk, 4),
            "flagged": self.flagged,
            "gold_hallucinated": self.gold_hallucinated,
            "answer_similarity": (
                None if self.answer_similarity is None else round(self.answer_similarity, 4)
            ),
            "context_recall": (
                None if self.context_recall is None else round(self.context_recall, 4)
            ),
            "context_doc_ids": list(self.context_doc_ids),
            "retrieval_starved": self.retrieval_starved,
        }


@dataclass
class QueryResult:
    """Everything the harness computed for one query."""

    query_id: str
    question: str
    tags: List[str] = field(default_factory=list)
    hits: List[Hit] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    n_relevant: int = 0
    latency_ms: float = 0.0
    grounding: Optional[GroundingResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "tags": list(self.tags),
            "hits": [h.to_dict() for h in self.hits],
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "n_relevant": self.n_relevant,
            "latency_ms": round(self.latency_ms, 3),
            "grounding": None if self.grounding is None else self.grounding.to_dict(),
        }


@dataclass
class EvalReport:
    """Top-level result object produced by :func:`ragval.evaluate.evaluate`."""

    config: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, float] = field(default_factory=dict)
    retrieval_ci: Dict[str, Sequence[float]] = field(default_factory=dict)
    detector: Dict[str, Any] = field(default_factory=dict)
    by_tag: Dict[str, Dict[str, float]] = field(default_factory=dict)
    queries: List[QueryResult] = field(default_factory=list)
    corpus_stats: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "config": self.config,
            "corpus_stats": self.corpus_stats,
            "retrieval": {k: round(v, 6) for k, v in self.retrieval.items()},
            "retrieval_ci": {k: [round(x, 6) for x in v] for k, v in self.retrieval_ci.items()},
            "detector": self.detector,
            "by_tag": {
                tag: {k: round(v, 6) for k, v in vals.items()} for tag, vals in self.by_tag.items()
            },
            "queries": [q.to_dict() for q in self.queries],
        }
