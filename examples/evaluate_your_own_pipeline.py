"""Score *your* retriever with this harness.

The only contract is ``search(query, k) -> List[Hit]``. Everything else - the
metrics, the confidence intervals, the grounding checks, the HTML report and
the CI gate - comes for free.

Run it with:

    python examples/evaluate_your_own_pipeline.py
"""

from __future__ import annotations

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragval import EvalOptions, evaluate, load_dataset  # noqa: E402
from ragval.index import Retriever  # noqa: E402
from ragval.report import render_terminal, write_html  # noqa: E402
from ragval.types import Hit  # noqa: E402


class MyRetriever(Retriever):
    """A deliberately naive retriever: exact substring matching on the query.

    Swap the body of ``search`` for a call into your vector database, your
    hosted search API, or your production RAG service. Return ``Hit`` objects
    ordered best-first and the harness handles the rest.
    """

    name = "substring-baseline"

    def search(self, query: str, k: int = 5) -> List[Hit]:
        needles = [w for w in query.lower().split() if len(w) > 4]
        scored = []
        for doc in self.documents:
            haystack = doc.indexable_text.lower()
            score = sum(1 for needle in needles if needle in haystack)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            Hit(doc_id=doc.id, score=float(score), rank=rank, title=doc.title, snippet=doc.text[:180])
            for rank, (score, doc) in enumerate(scored[:k], start=1)
        ]


def main() -> int:
    dataset = load_dataset("bundled")

    print("=" * 72)
    print("Baseline: BM25 (built in)")
    print("=" * 72)
    bm25 = evaluate(dataset, EvalOptions(bootstrap=0))
    print(render_terminal(bm25, color=False))

    print("=" * 72)
    print("Candidate: your retriever")
    print("=" * 72)
    mine = evaluate(dataset, EvalOptions(bootstrap=0), retriever=MyRetriever(dataset.documents))
    print(render_terminal(mine, color=False))

    print("=" * 72)
    print(f"{'metric':<14}{'bm25':>10}{'yours':>10}{'delta':>10}")
    print("=" * 72)
    for name in ("precision@5", "recall@5", "ndcg@5", "mrr"):
        a, b = bm25.retrieval[name], mine.retrieval[name]
        print(f"{name:<14}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}")

    write_html(mine, "out/my_pipeline.html")
    print("\nwrote out/my_pipeline.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
