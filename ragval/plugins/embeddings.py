"""Optional dense-retrieval adapter (``pip install "rag-eval-harness[embeddings]"``).

Shows how to swap the lexical retriever for a dense one without touching the
evaluation code: implement ``search(query, k) -> List[Hit]`` and the metrics,
report and gate all keep working. Pair it with the lexical retriever through
:class:`ragval.index.HybridRetriever` for the usual hybrid setup.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from ..index import Retriever
from ..types import Document, Hit

__all__ = ["EmbeddingRetriever"]

_INSTALL_HINT = (
    "EmbeddingRetriever needs sentence-transformers. Install it with:\n"
    '    pip install "rag-eval-harness[embeddings]"'
)


class EmbeddingRetriever(Retriever):
    """Cosine similarity over sentence-transformer embeddings.

    Embeddings are computed once at construction time and cached in memory.
    Set ``normalize=True`` (default) so the dot product is a cosine similarity.
    """

    name = "embedding"

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        encoder: Optional[Any] = None,
    ):
        super().__init__(documents)
        self.model_name = model
        self.normalize = normalize
        if encoder is not None:
            self.encoder = encoder  # dependency injection keeps this testable
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(_INSTALL_HINT) from exc
            self.encoder = SentenceTransformer(model)
        self.name = f"embedding({model.split('/')[-1]})"
        texts = [d.indexable_text for d in self.documents]
        self.matrix: List[List[float]] = (
            self._encode(texts, batch_size) if texts else []
        )

    def _encode(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        vectors = self.encoder.encode(list(texts), batch_size=batch_size)
        out: List[List[float]] = []
        for vec in vectors:
            values = [float(x) for x in vec]
            if self.normalize:
                norm = math.sqrt(sum(v * v for v in values)) or 1.0
                values = [v / norm for v in values]
            out.append(values)
        return out

    def search(self, query: str, k: int = 5) -> List[Hit]:
        if not query.strip() or not self.matrix:
            return []
        q_vec = self._encode([query])[0]
        scores: Dict[int, float] = {}
        for idx, doc_vec in enumerate(self.matrix):
            score = sum(a * b for a, b in zip(q_vec, doc_vec))
            if score > 0:
                scores[idx] = score
        return self._top_k(scores, k)
