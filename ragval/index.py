"""Retrievers.

Three interchangeable, pure-Python retrievers plus a fusion wrapper:

* :class:`BM25Retriever`  - Okapi BM25 over an inverted index (default)
* :class:`TfidfRetriever` - cosine similarity on L2-normalised log-tf x idf
* :class:`HybridRetriever` - reciprocal rank fusion of any set of retrievers

All of them implement the same tiny :class:`Retriever` interface, so plugging in
your own production retriever (or the optional embedding plugin) is a matter of
providing ``search(query, k) -> List[Hit]``.

Scoring only touches documents that share at least one term with the query, so
cost is O(sum of posting-list lengths) rather than O(corpus). Ties break on
``doc_id`` to keep rankings byte-for-byte reproducible.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .text import tokenize
from .types import Document, Hit

__all__ = [
    "Retriever",
    "BM25Retriever",
    "TfidfRetriever",
    "HybridRetriever",
    "build_retriever",
    "RETRIEVERS",
]

Tokenizer = Callable[[str], List[str]]


def _default_tokenizer(text: str) -> List[str]:
    return tokenize(text)


def _snippet(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Retriever:
    """Minimal retriever interface.

    Subclasses must implement :meth:`search`. ``name`` is recorded in the report
    so results are always traceable to the system that produced them.
    """

    name: str = "retriever"

    def __init__(self, documents: Sequence[Document], *, tokenizer: Optional[Tokenizer] = None):
        self.documents: List[Document] = list(documents)
        self.tokenizer: Tokenizer = tokenizer or _default_tokenizer
        self.doc_ids: List[str] = [d.id for d in self.documents]
        self._by_id: Dict[str, Document] = {d.id: d for d in self.documents}
        if len(self._by_id) != len(self.documents):
            dupes = [i for i, c in Counter(self.doc_ids).items() if c > 1]
            raise ValueError(f"duplicate document ids in corpus: {sorted(dupes)[:5]}")

    # -- public API -------------------------------------------------------- #

    def search(self, query: str, k: int = 5) -> List[Hit]:  # pragma: no cover
        raise NotImplementedError

    def get(self, doc_id: str) -> Optional[Document]:
        return self._by_id.get(doc_id)

    def __len__(self) -> int:
        return len(self.documents)

    # -- helpers ----------------------------------------------------------- #

    def _top_k(self, scores: Dict[int, float], k: int) -> List[Hit]:
        """Rank by score desc, then doc_id asc, and materialise :class:`Hit`s."""
        if k <= 0 or not scores:
            return []
        ranked = heapq.nsmallest(
            min(k, len(scores)),
            scores.items(),
            key=lambda item: (-item[1], self.doc_ids[item[0]]),
        )
        hits: List[Hit] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if score <= 0:
                continue
            doc = self.documents[idx]
            hits.append(
                Hit(
                    doc_id=doc.id,
                    score=float(score),
                    rank=rank,
                    title=doc.title,
                    snippet=_snippet(doc.text),
                )
            )
        return hits


class BM25Retriever(Retriever):
    """Okapi BM25.

    ``k1`` controls term-frequency saturation and ``b`` the length
    normalisation. Defaults (1.5 / 0.75) are the standard robust settings.
    The idf uses the ``log(1 + ...)`` form, which is always positive and so
    avoids the negative-score pathology of the classic formulation on terms
    appearing in more than half the corpus.
    """

    name = "bm25"

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Tokenizer] = None,
    ):
        super().__init__(documents, tokenizer=tokenizer)
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b

        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for idx, doc in enumerate(self.documents):
            terms = self.tokenizer(doc.indexable_text)
            self.doc_len.append(len(terms))
            for term, tf in Counter(terms).items():
                self.postings[term].append((idx, tf))

        n_docs = max(len(self.documents), 1)
        total_len = sum(self.doc_len)
        self.avg_len: float = (total_len / n_docs) if total_len else 1.0
        self.idf: Dict[str, float] = {
            term: math.log(1.0 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def search(self, query: str, k: int = 5) -> List[Hit]:
        terms = self.tokenizer(query)
        if not terms or not self.documents:
            return []
        scores: Dict[int, float] = defaultdict(float)
        k1, b, avg_len = self.k1, self.b, self.avg_len
        for term, q_tf in Counter(terms).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for idx, tf in plist:
                denom = tf + k1 * (1.0 - b + b * (self.doc_len[idx] / avg_len))
                if denom <= 0:
                    continue
                scores[idx] += idf * (tf * (k1 + 1.0)) / denom
        return self._top_k(scores, k)


class TfidfRetriever(Retriever):
    """Cosine similarity over L2-normalised ``(1 + log tf) * idf`` vectors."""

    name = "tfidf"

    def __init__(self, documents: Sequence[Document], *, tokenizer: Optional[Tokenizer] = None):
        super().__init__(documents, tokenizer=tokenizer)
        n_docs = max(len(self.documents), 1)
        df: Counter = Counter()
        raw_tf: List[Counter] = []
        for doc in self.documents:
            counts = Counter(self.tokenizer(doc.indexable_text))
            raw_tf.append(counts)
            df.update(counts.keys())
        self.idf: Dict[str, float] = {
            term: math.log((1.0 + n_docs) / (1.0 + d)) + 1.0 for term, d in df.items()
        }
        self.vectors: List[Dict[str, float]] = []
        self.postings: Dict[str, List[int]] = defaultdict(list)
        for idx, counts in enumerate(raw_tf):
            vec = {
                term: (1.0 + math.log(tf)) * self.idf[term] for term, tf in counts.items()
            }
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            vec = {term: w / norm for term, w in vec.items()}
            self.vectors.append(vec)
            for term in vec:
                self.postings[term].append(idx)

    def search(self, query: str, k: int = 5) -> List[Hit]:
        counts = Counter(self.tokenizer(query))
        if not counts or not self.documents:
            return []
        q_vec = {
            term: (1.0 + math.log(tf)) * self.idf.get(term, 0.0)
            for term, tf in counts.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(w * w for w in q_vec.values()))
        if norm == 0:
            return []
        q_vec = {t: w / norm for t, w in q_vec.items()}
        scores: Dict[int, float] = defaultdict(float)
        for term, q_w in q_vec.items():
            for idx in self.postings.get(term, ()):
                scores[idx] += q_w * self.vectors[idx][term]
        return self._top_k(scores, k)


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion over several retrievers.

    RRF is score-scale agnostic (``1 / (rrf_k + rank)``), which is exactly what
    you want when fusing a lexical retriever with a dense one whose scores live
    on a completely different scale.
    """

    name = "hybrid"

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        retrievers: Optional[Sequence[Retriever]] = None,
        rrf_k: int = 60,
        weights: Optional[Sequence[float]] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        super().__init__(documents, tokenizer=tokenizer)
        self.retrievers: List[Retriever] = list(retrievers) if retrievers else [
            BM25Retriever(documents, tokenizer=tokenizer),
            TfidfRetriever(documents, tokenizer=tokenizer),
        ]
        if not self.retrievers:
            raise ValueError("HybridRetriever needs at least one sub-retriever")
        self.rrf_k = max(int(rrf_k), 1)
        if weights is None:
            weights = [1.0] * len(self.retrievers)
        if len(weights) != len(self.retrievers):
            raise ValueError("weights length must match retrievers length")
        self.weights = list(weights)
        self.name = "hybrid(" + "+".join(r.name for r in self.retrievers) + ")"

    def search(self, query: str, k: int = 5) -> List[Hit]:
        if k <= 0:
            return []
        pool = max(k * 4, 20)
        fused: Dict[str, float] = defaultdict(float)
        for retriever, weight in zip(self.retrievers, self.weights):
            for hit in retriever.search(query, pool):
                fused[hit.doc_id] += weight / (self.rrf_k + hit.rank)
        if not fused:
            return []
        index_of = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        scores = {index_of[doc_id]: s for doc_id, s in fused.items() if doc_id in index_of}
        return self._top_k(scores, k)


RETRIEVERS: Dict[str, type] = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
    "hybrid": HybridRetriever,
}


def build_retriever(name: str, documents: Sequence[Document], **kwargs) -> Retriever:
    """Factory used by the CLI. Raises a helpful error on unknown names."""
    key = (name or "bm25").strip().lower()
    if key == "embedding":
        from .plugins.embeddings import EmbeddingRetriever  # lazy: optional dependency

        return EmbeddingRetriever(documents, **kwargs)
    if key not in RETRIEVERS:
        raise ValueError(
            f"unknown retriever {name!r}; choose from {sorted(RETRIEVERS) + ['embedding']}"
        )
    return RETRIEVERS[key](documents, **kwargs)
