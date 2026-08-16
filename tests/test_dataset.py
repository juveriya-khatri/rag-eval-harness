"""Dataset loading and the validation that stops bad labels reaching the metrics."""

from __future__ import annotations

import json

import pytest

from ragval.dataset import Dataset, ValidationError, load_dataset, load_jsonl
from ragval.types import AnswerCase, Document, QueryCase


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


class TestLoadJsonl:
    def test_skips_blanks_and_comments(self, tmp_path):
        path = tmp_path / "a.jsonl"
        path.write_text('# a note\n\n{"id": "1"}\n', encoding="utf-8")
        assert load_jsonl(str(path)) == [{"id": "1"}]

    def test_reports_the_bad_line_number(self, tmp_path):
        path = tmp_path / "a.jsonl"
        path.write_text('{"id": "1"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(ValidationError, match="a.jsonl:2"):
            load_jsonl(str(path))

    def test_rejects_non_objects(self, tmp_path):
        path = tmp_path / "a.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="expected a JSON object"):
            load_jsonl(str(path))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(str(tmp_path / "nope.jsonl"))


class TestParsing:
    def test_binary_relevance_list_accepted(self):
        query = QueryCase.from_dict({"id": "q", "question": "?", "relevant": ["a", "b"]})
        assert query.relevant == {"a": 1, "b": 1}

    def test_graded_relevance_mapping_accepted(self):
        query = QueryCase.from_dict({"id": "q", "question": "?", "relevant": {"a": 2}})
        assert query.relevant_ids == ["a"]

    def test_zero_gain_is_not_relevant(self):
        query = QueryCase.from_dict({"id": "q", "question": "?", "relevant": {"a": 0}})
        assert query.relevant_ids == []

    def test_query_alias_accepted(self):
        assert QueryCase.from_dict({"id": "q", "query": "hello"}).question == "hello"

    def test_string_labels_coerced(self):
        assert AnswerCase.from_dict({"query_id": "q", "answer": "a", "hallucinated": "true"}).hallucinated is True
        assert AnswerCase.from_dict({"query_id": "q", "answer": "a", "hallucinated": "no"}).hallucinated is False

    def test_missing_label_stays_none(self):
        assert AnswerCase.from_dict({"query_id": "q", "answer": "a"}).hallucinated is None

    def test_missing_required_field(self):
        with pytest.raises(ValueError, match="missing required field"):
            Document.from_dict({"text": "no id"})

    def test_title_boosts_indexable_text(self):
        doc = Document(id="d", title="Title", text="Body")
        assert doc.indexable_text == "Title\nBody"
        assert Document(id="d", text="Body").indexable_text == "Body"


class TestValidation:
    def make(self, **overrides):
        base = dict(
            documents=[Document(id="d1", text="one"), Document(id="d2", text="two")],
            queries=[QueryCase(id="q1", question="?", relevant={"d1": 1})],
            answers=[],
        )
        base.update(overrides)
        return Dataset(**base)

    def test_valid_dataset_passes(self):
        assert self.make().validate() == []

    def test_dangling_document_reference(self):
        dataset = self.make(queries=[QueryCase(id="q1", question="?", relevant={"ghost": 1})])
        with pytest.raises(ValidationError, match="unknown document"):
            dataset.validate()

    def test_duplicate_document_ids(self):
        dataset = self.make(documents=[Document(id="d", text="a"), Document(id="d", text="b")])
        with pytest.raises(ValidationError, match="duplicate document id"):
            dataset.validate()

    def test_duplicate_query_ids(self):
        dupes = [QueryCase(id="q", question="?"), QueryCase(id="q", question="?")]
        with pytest.raises(ValidationError, match="duplicate query id"):
            self.make(queries=dupes).validate()

    def test_answer_for_unknown_query(self):
        dataset = self.make(answers=[AnswerCase(query_id="ghost", answer="a")])
        with pytest.raises(ValidationError, match="unknown query"):
            dataset.validate()

    def test_empty_corpus_is_fatal(self):
        with pytest.raises(ValidationError, match="corpus is empty"):
            self.make(documents=[]).validate()

    def test_unjudged_query_is_a_warning_not_an_error(self):
        dataset = self.make(queries=[QueryCase(id="q1", question="?", relevant={})])
        warnings = dataset.validate()
        assert any("no relevance judgements" in w for w in warnings)

    def test_non_strict_mode_collects_instead_of_raising(self):
        dataset = self.make(queries=[QueryCase(id="q1", question="?", relevant={"ghost": 1})])
        problems = dataset.validate(strict=False)
        assert any("unknown document" in p for p in problems)


class TestBundled:
    def test_bundled_dataset_loads_and_validates(self):
        dataset = load_dataset("bundled")
        stats = dataset.stats()
        assert stats["n_documents"] >= 40
        assert stats["n_queries"] >= 30
        assert stats["n_labeled_answers"] == stats["n_answers"]
        assert 0 < stats["n_hallucinated"] < stats["n_labeled_answers"]

    def test_every_answer_maps_to_a_query(self):
        dataset = load_dataset("bundled")
        query_ids = {q.id for q in dataset.queries}
        assert all(a.query_id in query_ids for a in dataset.answers)

    def test_missing_directory_reports_clearly(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/for/tests")

    def test_answers_file_is_optional(self, tmp_path):
        write_jsonl(tmp_path / "corpus.jsonl", [{"id": "d1", "text": "hello world"}])
        write_jsonl(tmp_path / "queries.jsonl", [{"id": "q1", "question": "hello", "relevant": {"d1": 1}}])
        dataset = load_dataset(str(tmp_path))
        assert dataset.answers == []
