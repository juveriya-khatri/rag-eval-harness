"""Dataset loading and validation.

The on-disk format is three JSONL files - ``corpus.jsonl``, ``queries.jsonl``
and (optionally) ``answers.jsonl``. JSONL is used deliberately: it diffs
cleanly in git, streams without loading everything into memory, and survives a
single malformed line with a precise error message instead of a stack trace.

:func:`validate` catches the label bugs that silently corrupt evaluation
results - dangling document ids, duplicate ids, queries with no relevance
judgements, answers pointing at queries that do not exist.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .types import AnswerCase, Document, QueryCase

__all__ = ["Dataset", "load_jsonl", "load_dataset", "bundled_path", "ValidationError"]


class ValidationError(ValueError):
    """Raised when a dataset is structurally unusable."""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, reporting the offending line number on bad JSON.

    Blank lines and ``#`` comment lines are skipped so datasets can be
    annotated by hand.
    """
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{os.path.basename(path)}:{lineno}: invalid JSON ({exc.msg})"
                ) from exc
            if not isinstance(obj, dict):
                raise ValidationError(
                    f"{os.path.basename(path)}:{lineno}: expected a JSON object, got {type(obj).__name__}"
                )
            rows.append(obj)
    return rows


@dataclass
class Dataset:
    """A corpus plus its labelled queries and (optionally) answers under test."""

    documents: List[Document] = field(default_factory=list)
    queries: List[QueryCase] = field(default_factory=list)
    answers: List[AnswerCase] = field(default_factory=list)
    name: str = "dataset"

    # -- lookups ------------------------------------------------------------ #

    @property
    def doc_ids(self) -> List[str]:
        return [d.id for d in self.documents]

    def answers_for(self, query_id: str) -> List[AnswerCase]:
        return [a for a in self.answers if a.query_id == query_id]

    @property
    def systems(self) -> List[str]:
        seen: List[str] = []
        for a in self.answers:
            if a.system not in seen:
                seen.append(a.system)
        return seen

    def stats(self) -> Dict[str, Any]:
        lengths = [len(d.text.split()) for d in self.documents]
        labels = [a.hallucinated for a in self.answers if a.hallucinated is not None]
        tags = sorted({t for q in self.queries for t in q.tags})
        return {
            "name": self.name,
            "n_documents": len(self.documents),
            "n_queries": len(self.queries),
            "n_answers": len(self.answers),
            "n_labeled_answers": len(labels),
            "n_hallucinated": sum(1 for x in labels if x),
            "avg_doc_words": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            "avg_relevant_per_query": (
                round(sum(len(q.relevant_ids) for q in self.queries) / len(self.queries), 2)
                if self.queries
                else 0.0
            ),
            "tags": tags,
        }

    # -- validation --------------------------------------------------------- #

    def validate(self, *, strict: bool = True) -> List[str]:
        """Return a list of warnings; raise :class:`ValidationError` on fatal issues.

        Fatal: duplicate ids, empty corpus/queries, relevance judgements that
        point at documents which do not exist, answers for unknown queries.
        Warning: queries with no relevance judgements, empty document text.
        """
        problems: List[str] = []
        warnings: List[str] = []

        if not self.documents:
            problems.append("corpus is empty")
        if not self.queries:
            problems.append("no queries provided")

        doc_ids = set()
        for doc in self.documents:
            if doc.id in doc_ids:
                problems.append(f"duplicate document id: {doc.id!r}")
            doc_ids.add(doc.id)
            if not doc.text.strip():
                warnings.append(f"document {doc.id!r} has empty text")

        query_ids = set()
        for query in self.queries:
            if query.id in query_ids:
                problems.append(f"duplicate query id: {query.id!r}")
            query_ids.add(query.id)
            if not query.question.strip():
                problems.append(f"query {query.id!r} has an empty question")
            if not query.relevant:
                warnings.append(f"query {query.id!r} has no relevance judgements (recall undefined)")
            for doc_id, gain in query.relevant.items():
                if doc_id not in doc_ids:
                    problems.append(
                        f"query {query.id!r} references unknown document {doc_id!r}"
                    )
                if gain < 0:
                    problems.append(f"query {query.id!r} has a negative gain for {doc_id!r}")

        for answer in self.answers:
            if answer.query_id not in query_ids:
                problems.append(
                    f"answer for unknown query {answer.query_id!r} "
                    f"(system={answer.system!r})"
                )

        if problems and strict:
            bullet = "\n  - ".join(problems[:20])
            more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
            raise ValidationError(f"dataset validation failed:\n  - {bullet}{more}")
        return warnings + problems if not strict else warnings


def bundled_path() -> str:
    """Absolute path to the dataset shipped inside the package."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data")


def load_dataset(
    path: str = "bundled",
    *,
    corpus: Optional[str] = None,
    queries: Optional[str] = None,
    answers: Optional[str] = None,
    strict: bool = True,
) -> Dataset:
    """Load a dataset from a directory (or the bundled one).

    ``path`` may be ``"bundled"`` for the packaged demo data or a directory
    containing ``corpus.jsonl`` / ``queries.jsonl`` / ``answers.jsonl``.
    Individual files can be overridden with the keyword arguments.
    """
    if path == "bundled":
        base = bundled_path()
        name = "bundled"
    else:
        base = os.path.abspath(path)
        name = os.path.basename(base.rstrip(os.sep)) or "dataset"
        if not os.path.isdir(base):
            raise FileNotFoundError(f"dataset directory not found: {base}")

    corpus_path = corpus or os.path.join(base, "corpus.jsonl")
    queries_path = queries or os.path.join(base, "queries.jsonl")
    answers_path = answers or os.path.join(base, "answers.jsonl")

    documents = [Document.from_dict(r) for r in load_jsonl(corpus_path)]
    query_cases = [QueryCase.from_dict(r) for r in load_jsonl(queries_path)]
    answer_cases: List[AnswerCase] = []
    if os.path.exists(answers_path):
        answer_cases = [AnswerCase.from_dict(r) for r in load_jsonl(answers_path)]

    dataset = Dataset(
        documents=documents, queries=query_cases, answers=answer_cases, name=name
    )
    dataset.validate(strict=strict)
    return dataset
