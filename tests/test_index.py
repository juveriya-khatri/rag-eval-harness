"""Retriever behaviour, determinism and degenerate inputs."""

from __future__ import annotations

import pytest

from ragval.index import BM25Retriever, HybridRetriever, TfidfRetriever, build_retriever
from ragval.types import Document

DOCS = [
    Document(id="d1", title="Battery storage", text="Grid-scale battery storage reached 12.4 gigawatt hours in 2024."),
    Document(id="d2", title="Wind auction", text="The offshore wind auction cleared at 48 per megawatt hour."),
    Document(id="d3", title="Solar credit", text="The rooftop solar tax credit remains at 30 percent through 2032."),
    Document(id="d4", title="Battery costs", text="Average installed battery cost fell to 231 per kilowatt hour."),
]


@pytest.fixture(params=["bm25", "tfidf", "hybrid"])
def retriever(request):
    return build_retriever(request.param, DOCS)


class TestCommonBehaviour:
    def test_finds_the_obvious_document(self, retriever):
        hits = retriever.search("battery storage gigawatt hours", 3)
        assert hits and hits[0].doc_id == "d1"

    def test_respects_k(self, retriever):
        assert len(retriever.search("battery", 1)) == 1
        assert len(retriever.search("battery", 10)) <= len(DOCS)

    def test_ranks_are_sequential_from_one(self, retriever):
        hits = retriever.search("battery cost", 4)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_scores_are_descending(self, retriever):
        scores = [h.score for h in retriever.search("battery cost kilowatt", 4)]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.search("", 5) == []
        assert retriever.search("   ", 5) == []

    def test_out_of_vocabulary_query_returns_nothing(self, retriever):
        assert retriever.search("zzzzqqq nonexistentterm", 5) == []

    def test_k_zero_or_negative(self, retriever):
        assert retriever.search("battery", 0) == []
        assert retriever.search("battery", -1) == []

    def test_deterministic_across_runs(self, retriever):
        first = [(h.doc_id, round(h.score, 9)) for h in retriever.search("battery hour", 4)]
        second = [(h.doc_id, round(h.score, 9)) for h in retriever.search("battery hour", 4)]
        assert first == second

    def test_get_returns_documents(self, retriever):
        assert retriever.get("d1").title == "Battery storage"
        assert retriever.get("missing") is None

    def test_unicode_and_punctuation_query(self, retriever):
        assert isinstance(retriever.search("battery — storage (12.4)!", 3), list)


class TestBM25Specifics:
    def test_title_terms_are_searchable(self):
        hits = BM25Retriever(DOCS).search("wind auction", 1)
        assert hits[0].doc_id == "d2"

    def test_idf_is_never_negative(self):
        retriever = BM25Retriever(DOCS)
        assert all(v >= 0 for v in retriever.idf.values())

    def test_rare_terms_outrank_common_ones(self):
        retriever = BM25Retriever(DOCS)
        # "battery" appears in two documents, "kilowatt" in one.
        assert retriever.idf["kilowatt"] > retriever.idf["battery"]

    @pytest.mark.parametrize("kwargs", [{"k1": -1.0}, {"b": 1.5}, {"b": -0.1}])
    def test_invalid_parameters_rejected(self, kwargs):
        with pytest.raises(ValueError):
            BM25Retriever(DOCS, **kwargs)

    def test_empty_corpus(self):
        assert BM25Retriever([]).search("anything", 5) == []

    def test_duplicate_ids_rejected(self):
        dupes = [Document(id="x", text="one"), Document(id="x", text="two")]
        with pytest.raises(ValueError, match="duplicate document ids"):
            BM25Retriever(dupes)

    def test_document_with_empty_text(self):
        docs = DOCS + [Document(id="d5", text="")]
        hits = BM25Retriever(docs).search("battery", 5)
        assert "d5" not in [h.doc_id for h in hits]

    def test_snippets_are_truncated(self):
        long_doc = [Document(id="L", text="word " * 500)]
        hit = BM25Retriever(long_doc).search("word", 1)[0]
        assert len(hit.snippet) <= 221


class TestHybrid:
    def test_fuses_sub_retrievers(self):
        hybrid = HybridRetriever(DOCS)
        assert "bm25" in hybrid.name and "tfidf" in hybrid.name

    def test_custom_weights_validated(self):
        with pytest.raises(ValueError):
            HybridRetriever(DOCS, weights=[1.0])

    def test_single_sub_retriever_reproduces_its_order(self):
        base = BM25Retriever(DOCS)
        hybrid = HybridRetriever(DOCS, retrievers=[base])
        query = "battery cost per kilowatt hour"
        assert [h.doc_id for h in hybrid.search(query, 3)] == [
            h.doc_id for h in base.search(query, 3)
        ]


class TestFactory:
    def test_unknown_name_is_actionable(self):
        with pytest.raises(ValueError, match="unknown retriever"):
            build_retriever("magic", DOCS)

    def test_defaults_to_bm25(self):
        assert isinstance(build_retriever("", DOCS), BM25Retriever)

    def test_names_are_case_insensitive(self):
        assert isinstance(build_retriever("TFIDF", DOCS), TfidfRetriever)
