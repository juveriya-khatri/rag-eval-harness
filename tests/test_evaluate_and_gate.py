"""End-to-end evaluation, the quality gate and report writers."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import pytest

from ragval import EvalOptions, evaluate, load_dataset
from ragval.gate import DEFAULT_CONFIG, load_config, make_baseline, run_gate
from ragval.index import BM25Retriever
from ragval.report import render_terminal, write_html, write_json, write_junit
from ragval.types import Document, Hit


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("bundled")


@pytest.fixture(scope="module")
def report(dataset):
    return evaluate(dataset, EvalOptions(bootstrap=200))


class TestEvaluate:
    def test_covers_every_query(self, report, dataset):
        assert len(report.queries) == len(dataset.queries)

    def test_headline_metrics_present_and_in_range(self, report):
        for name in ("precision@5", "recall@5", "ndcg@5", "hit_rate@5", "mrr"):
            assert 0.0 <= report.retrieval[name] <= 1.0

    def test_confidence_intervals_bracket_the_estimate(self, report):
        for name, (lo, hi) in report.retrieval_ci.items():
            if math.isnan(lo):
                continue
            assert lo <= report.retrieval[name] + 1e-9
            assert hi >= report.retrieval[name] - 1e-9

    def test_detector_scored_against_labels(self, report):
        det = report.detector
        assert det["n_labeled"] > 0
        assert det["tp"] + det["fp"] + det["tn"] + det["fn"] == det["n_labeled"]
        assert 0.0 <= det["f1"] <= 1.0

    def test_retrieval_root_cause_is_attributed(self, report):
        det = report.detector
        assert "retrieval_starved_answers" in det
        # Any error attributed to a retrieval miss must really have empty context recall.
        starved = {q.query_id for q in report.queries if q.grounding and q.grounding.retrieval_starved}
        for err in det["errors"]:
            if err["retrieval_starved"]:
                assert err["query_id"] in starved

    def test_tag_breakdown_partitions_the_queries(self, report):
        assert set(report.by_tag) >= {"aviation", "clinical", "cloud", "energy"}
        assert all(v["n_queries"] > 0 for v in report.by_tag.values())

    def test_run_is_reproducible(self, dataset):
        def strip_wall_clock(payload):
            """Latency and timestamps are the only non-deterministic fields."""
            payload.pop("generated_at", None)
            payload["retrieval"].pop("avg_latency_ms", None)
            for query in payload["queries"]:
                query.pop("latency_ms", None)
            return payload

        a = strip_wall_clock(evaluate(dataset, EvalOptions(bootstrap=100, seed=5)).to_dict())
        b = strip_wall_clock(evaluate(dataset, EvalOptions(bootstrap=100, seed=5)).to_dict())
        assert a == b

    def test_custom_retriever_is_accepted(self, dataset):
        engine = BM25Retriever(dataset.documents)
        result = evaluate(dataset, EvalOptions(bootstrap=0), retriever=engine)
        assert result.config["retriever_name"] == "bm25"

    def test_a_retriever_returning_nothing_scores_zero_not_crash(self, dataset):
        class Empty(BM25Retriever):
            name = "empty"

            def search(self, query, k=5):
                return []

        result = evaluate(dataset, EvalOptions(bootstrap=0), retriever=Empty(dataset.documents))
        assert result.retrieval["recall@5"] == 0.0

    def test_dataset_without_answers_skips_the_detector(self, dataset):
        from ragval.dataset import Dataset

        stripped = Dataset(documents=dataset.documents, queries=dataset.queries, answers=[])
        assert evaluate(stripped, EvalOptions(bootstrap=0)).detector == {}

    @pytest.mark.parametrize("ks", [[], [0], [-3]])
    def test_invalid_cutoffs_rejected(self, ks):
        with pytest.raises(ValueError):
            EvalOptions(ks=ks)


class TestGate:
    def test_passing_thresholds(self):
        results = {"retrieval": {"recall@5": 0.9}, "detector": {"f1": 0.8}}
        config = {"thresholds": {"retrieval": {"recall@5": 0.8}, "detector": {"f1": 0.7}}}
        assert run_gate(results, config).passed

    def test_failing_threshold_names_the_metric(self):
        results = {"retrieval": {"recall@5": 0.5}}
        config = {"thresholds": {"retrieval": {"recall@5": 0.8}}}
        outcome = run_gate(results, config)
        assert not outcome.passed
        assert outcome.failures[0].name == "retrieval.recall@5"

    def test_missing_metric_fails_loudly(self):
        outcome = run_gate({}, {"thresholds": {"retrieval": {"recall@5": 0.8}}})
        assert not outcome.passed
        assert "missing" in outcome.failures[0].detail

    def test_regression_within_tolerance_passes(self):
        results = {"retrieval": {"recall@5": 0.79}}
        baseline = {"metrics": {"retrieval.recall@5": 0.80}}
        config = {"thresholds": {}, "regression_tolerance": 0.02}
        assert run_gate(results, config, baseline).passed

    def test_regression_beyond_tolerance_fails(self):
        results = {"retrieval": {"recall@5": 0.70}}
        baseline = {"metrics": {"retrieval.recall@5": 0.80}}
        config = {"thresholds": {}, "regression_tolerance": 0.02}
        outcome = run_gate(results, config, baseline)
        assert not outcome.passed
        assert "-0.1000" in outcome.failures[0].detail

    def test_improvement_never_fails(self):
        results = {"retrieval": {"recall@5": 0.95}}
        baseline = {"metrics": {"retrieval.recall@5": 0.80}}
        assert run_gate(results, {"thresholds": {}}, baseline).passed

    def test_baseline_freezes_the_configured_metrics(self, report):
        baseline = make_baseline(report.to_dict())
        assert "retrieval.recall@5" in baseline["metrics"]
        assert baseline["retriever"] == report.config["retriever_name"]

    def test_round_trip_baseline_then_gate(self, report):
        results = report.to_dict()
        baseline = make_baseline(results)
        assert run_gate(results, {"thresholds": {}}, baseline).passed

    def test_config_file_overrides_defaults(self, tmp_path):
        path = tmp_path / "gate.json"
        path.write_text(json.dumps({"regression_tolerance": 0.5}), encoding="utf-8")
        assert load_config(str(path))["regression_tolerance"] == 0.5

    def test_default_config_used_when_no_path(self):
        assert load_config(None)["regression_tolerance"] == DEFAULT_CONFIG["regression_tolerance"]

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/gate.json")


class TestReportWriters:
    def test_terminal_summary_mentions_the_key_numbers(self, report):
        text = render_terminal(report, color=False)
        assert "Retrieval quality" in text
        assert "Hallucination detection" in text
        assert "\033[" not in text  # colour disabled means no escape codes

    def test_json_round_trips(self, report, tmp_path):
        path = write_json(report, str(tmp_path / "nested" / "results.json"))
        payload = json.loads(open(path, encoding="utf-8").read())
        assert payload["retrieval"]["recall@5"] == pytest.approx(report.retrieval["recall@5"])
        assert len(payload["queries"]) == len(report.queries)

    def test_html_is_self_contained(self, report, tmp_path):
        path = write_html(report, str(tmp_path / "report.html"))
        html = open(path, encoding="utf-8").read()
        assert html.startswith("<!doctype html>")
        assert "<svg" in html
        assert "http://" not in html and "https://" not in html  # no CDN, works offline

    def test_html_escapes_hostile_content(self, tmp_path, dataset):
        from ragval.dataset import Dataset

        evil = Dataset(
            documents=dataset.documents,
            queries=[type(dataset.queries[0])(id="<script>", question="<img onerror=x>", relevant={})],
            answers=[],
        )
        path = write_html(evaluate(evil, EvalOptions(bootstrap=0)), str(tmp_path / "x.html"))
        html = open(path, encoding="utf-8").read()
        assert "<script>alert" not in html
        assert "&lt;img onerror=x&gt;" in html

    def test_junit_is_wellformed(self, report, tmp_path):
        path = write_junit(report, str(tmp_path / "junit.xml"))
        root = ET.parse(path).getroot()
        assert root.tag == "testsuites"
        names = {suite.get("name") for suite in root}
        assert "retrieval" in names and "grounding" in names

    def test_junit_includes_gate_results(self, report, tmp_path):
        gate = run_gate(report.to_dict(), {"thresholds": {"retrieval": {"recall@5": 1.1}}}).to_dict()
        path = write_junit(report, str(tmp_path / "junit.xml"), gate=gate)
        root = ET.parse(path).getroot()
        gate_suite = [s for s in root if s.get("name") == "gate"][0]
        assert int(gate_suite.get("failures")) >= 1
