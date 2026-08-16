"""CLI contract (arguments, exit codes, artifacts) and the optional adapters."""

from __future__ import annotations

import json
import os

import pytest

from ragval.cli import main
from ragval.plugins.llm_judge import LLMJudgeChecker, _parse_judgement
from ragval.types import CONTRADICTED, SUPPORTED, UNSUPPORTED, Document


class TestRunCommand:
    def test_run_writes_every_artifact(self, tmp_path, capsys):
        out = tmp_path / "out"
        code = main(
            [
                "run", "--dataset", "bundled", "--k", "1,5",
                "--json", str(out / "results.json"),
                "--html", str(out / "report.html"),
                "--junit", str(out / "junit.xml"),
                "--bootstrap", "50", "--no-color",
            ]
        )
        assert code == 0
        for name in ("results.json", "report.html", "junit.xml"):
            assert (out / name).exists() and (out / name).stat().st_size > 0
        payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert set(payload["config"]["ks"]) == {1, 5}

    def test_quiet_suppresses_the_summary(self, tmp_path, capsys):
        main(["run", "--dataset", "bundled", "--bootstrap", "0", "--quiet"])
        assert "Retrieval quality" not in capsys.readouterr().out

    def test_thresholds_are_overridable(self, tmp_path):
        out = tmp_path / "r.json"
        main(["run", "--dataset", "bundled", "--bootstrap", "0", "--quiet",
              "--risk-threshold", "0.9", "--json", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["config"]["grounding"]["risk_threshold"] == 0.9

    @pytest.mark.parametrize("retriever", ["bm25", "tfidf", "hybrid"])
    def test_each_retriever_runs(self, retriever, tmp_path):
        out = tmp_path / f"{retriever}.json"
        assert main(["run", "--dataset", "bundled", "--retriever", retriever,
                     "--bootstrap", "0", "--quiet", "--json", str(out)]) == 0
        assert json.loads(out.read_text(encoding="utf-8"))["retrieval"]["recall@5"] > 0

    def test_bad_dataset_path_exits_two(self, capsys):
        assert main(["run", "--dataset", "/no/such/dir", "--quiet"]) == 2
        assert "error:" in capsys.readouterr().err

    def test_invalid_k_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            main(["run", "--k", "one,two"])


class TestGateCommand:
    def _results(self, tmp_path):
        path = tmp_path / "results.json"
        main(["run", "--dataset", "bundled", "--bootstrap", "0", "--quiet", "--json", str(path)])
        return path

    def test_gate_passes_against_its_own_baseline(self, tmp_path):
        results = self._results(tmp_path)
        baseline = tmp_path / "baseline.json"
        assert main(["baseline", "--results", str(results), "--out", str(baseline)]) == 0
        assert main(["gate", "--results", str(results), "--baseline", str(baseline)]) == 0

    def test_gate_fails_on_an_impossible_threshold(self, tmp_path):
        results = self._results(tmp_path)
        config = tmp_path / "gate.json"
        config.write_text(json.dumps({"thresholds": {"retrieval": {"recall@5": 1.01}}}), encoding="utf-8")
        assert main(["gate", "--results", str(results), "--config", str(config)]) == 1

    def test_gate_writes_its_outcome(self, tmp_path):
        results = self._results(tmp_path)
        outcome = tmp_path / "gate-out.json"
        main(["gate", "--results", str(results), "--json", str(outcome)])
        assert "checks" in json.loads(outcome.read_text(encoding="utf-8"))

    def test_run_can_fail_the_build_on_a_gate_failure(self, tmp_path):
        config = tmp_path / "gate.json"
        config.write_text(json.dumps({"thresholds": {"retrieval": {"recall@5": 1.01}}}), encoding="utf-8")
        code = main(["run", "--dataset", "bundled", "--bootstrap", "0", "--quiet",
                     "--config", str(config), "--fail-under-gate"])
        assert code == 1


class TestOtherCommands:
    def test_validate_reports_the_stats(self, capsys):
        assert main(["validate", "--dataset", "bundled"]) == 0
        assert "structurally valid" in capsys.readouterr().out

    def test_sweep_finds_an_operating_point(self, capsys):
        assert main(["sweep", "--dataset", "bundled"]) == 0
        assert "best threshold" in capsys.readouterr().out

    def test_sweep_from_a_results_file(self, tmp_path, capsys):
        path = tmp_path / "r.json"
        main(["run", "--dataset", "bundled", "--bootstrap", "0", "--quiet", "--json", str(path)])
        assert main(["sweep", "--results", str(path)]) == 0

    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            main([])


class FakeJudge:
    """Deterministic stand-in for an LLM, so the adapter is testable offline."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class TestLLMJudgeAdapter:
    CONTEXT = [Document(id="s1", title="Rest rules", text="Minimum rest before a duty period is 10 hours.")]

    def test_parses_json_wrapped_in_prose(self):
        verdict, confidence, _ = _parse_judgement(
            'Sure! {"verdict": "supported", "confidence": 0.9, "reason": "stated"} Hope that helps.'
        )
        assert verdict == SUPPORTED and confidence == pytest.approx(0.9)

    @pytest.mark.parametrize("raw", ["", "no json here", "{bad json}", '{"verdict": "maybe"}'])
    def test_malformed_replies_are_ignored(self, raw):
        assert _parse_judgement(raw)[0] is None

    def test_confidence_is_clamped(self):
        assert _parse_judgement('{"verdict": "supported", "confidence": 5}')[1] == 1.0

    def test_only_uncertain_claims_are_escalated(self):
        judge = FakeJudge('{"verdict": "supported", "confidence": 0.95, "reason": "ok"}')
        checker = LLMJudgeChecker(self.CONTEXT, judge, uncertainty_band=0.0)
        checker.check("Minimum rest before a duty period is 10 hours.", self.CONTEXT)
        assert judge.prompts == []  # band of zero means never call out

    def test_judge_can_override_a_lexical_verdict(self):
        judge = FakeJudge('{"verdict": "contradicted", "confidence": 0.9, "reason": "conflict"}')
        checker = LLMJudgeChecker(self.CONTEXT, judge, uncertainty_band=1.0)
        result = checker.check("Crews rest for ten hours between duties.", self.CONTEXT)
        assert any(c.verdict == CONTRADICTED for c in result.claims)
        assert result.risk >= 0.9

    def test_a_judge_outage_does_not_fail_the_run(self):
        class Broken:
            def complete(self, prompt):
                raise RuntimeError("503")

        checker = LLMJudgeChecker(self.CONTEXT, Broken(), uncertainty_band=1.0)
        result = checker.check("Some claim about rest periods here.", self.CONTEXT)
        assert result.claims  # degraded to the lexical verdict, no exception

    def test_call_budget_is_respected(self):
        judge = FakeJudge('{"verdict": "unsupported", "confidence": 0.5, "reason": "x"}')
        checker = LLMJudgeChecker(self.CONTEXT, judge, uncertainty_band=1.0, max_calls=1)
        checker.check("First claim about rest. Second claim about duty. Third claim about crew.", self.CONTEXT)
        assert len(judge.prompts) <= 1


class TestEmbeddingAdapter:
    def test_works_with_an_injected_encoder(self):
        from ragval.plugins.embeddings import EmbeddingRetriever

        class ToyEncoder:
            """Two-dimensional bag-of-keywords encoder."""

            def encode(self, texts, batch_size=32):
                return [[float("battery" in t.lower()), float("wind" in t.lower())] for t in texts]

        docs = [
            Document(id="a", text="battery storage deployment"),
            Document(id="b", text="offshore wind auction"),
        ]
        retriever = EmbeddingRetriever(docs, encoder=ToyEncoder())
        assert retriever.search("battery", 1)[0].doc_id == "a"
        assert retriever.search("wind", 1)[0].doc_id == "b"
        assert retriever.search("unrelated", 1) == []
