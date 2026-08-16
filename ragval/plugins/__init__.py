"""Optional adapters.

Nothing in here is imported by the core harness at module load time. Each
adapter fails with an actionable install message if its extra is missing, so a
plain ``pip install rag-eval-harness`` stays dependency-free.

    pip install "rag-eval-harness[embeddings]"   # dense retrieval
    pip install "rag-eval-harness[llm]"          # LLM-as-judge grounding
"""

from __future__ import annotations

__all__ = ["embeddings", "llm_judge"]
