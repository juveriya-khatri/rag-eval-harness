"""Hallucination detection: each failure mode, plus the false-positive traps."""

from __future__ import annotations

import pytest

from ragval.grounding import GroundingChecker, GroundingConfig, split_claims
from ragval.types import CONTRADICTED, SUPPORTED, UNSUPPORTED, Document

CONTEXT = [
    Document(
        id="src-1",
        title="Flight crew duty limits",
        text=(
            "The 2024 revision sets a maximum duty period of 13 hours for an unaugmented crew. "
            "Minimum rest before a duty period is 10 hours. "
            "The annual flight-time cap is 900 hours."
        ),
    ),
    Document(
        id="src-2",
        title="Runway incursion statistics",
        text=(
            "Twelve monitored airports recorded 214 runway incursions in 2024, "
            "an increase of 8 percent over the 198 incursions recorded in 2023. "
            "Pilot deviations accounted for 74 percent of events."
        ),
    ),
    Document(
        id="src-3",
        title="Cabin depressurisation",
        text=(
            "Flight TQ-233 was cruising at 36000 feet. "
            "The crew initiated an emergency descent to 10000 feet, completing it in 6 minutes. "
            "In moderate turbulence the procedure is not approved."
        ),
    ),
]


@pytest.fixture(scope="module")
def checker():
    return GroundingChecker(CONTEXT)


def verdicts(result):
    return [c.verdict for c in result.claims]


class TestClaimSplitting:
    def test_sentences_become_claims(self):
        assert len(split_claims("First fact here. Second fact here.")) == 2

    def test_short_fragments_dropped(self):
        assert split_claims("Yes.") == []

    def test_long_coordinated_sentences_are_split(self):
        long_sentence = (
            "The aircraft was removed from service for four days, "
            "and the operator revised its low-visibility procedure afterwards."
        )
        assert len(split_claims(long_sentence)) == 2

    def test_short_sentences_are_not_split(self):
        assert len(split_claims("Costs rose and volumes fell.")) == 1

    def test_empty_answer(self):
        assert split_claims("") == []


class TestFaithfulAnswers:
    def test_verbatim_answer_is_supported(self, checker):
        result = checker.check(
            "Minimum rest before a duty period is 10 hours.", CONTEXT
        )
        assert verdicts(result) == [SUPPORTED]
        assert result.risk < 0.2
        assert not result.flagged

    def test_figures_drawn_from_two_sentences_of_one_document(self, checker):
        # 36000 and 10000 live in different sentences of src-3; combining them
        # is faithful and must not read as a figure conflict.
        result = checker.check(
            "The crew descended from 36000 feet to 10000 feet in 6 minutes.", CONTEXT
        )
        assert CONTRADICTED not in verdicts(result)
        assert not result.flagged

    def test_matching_negation_is_not_a_conflict(self, checker):
        result = checker.check("In moderate turbulence the procedure is not approved.", CONTEXT)
        assert CONTRADICTED not in verdicts(result)

    def test_same_direction_is_not_a_conflict(self, checker):
        result = checker.check(
            "Runway incursions rose 8 percent in 2024 to 214 events.", CONTEXT
        )
        assert CONTRADICTED not in verdicts(result)


class TestHallucinations:
    def test_fabricated_figure_in_the_same_slot(self, checker):
        result = checker.check(
            "The 2024 revision sets a maximum duty period of 14 hours for an unaugmented crew.",
            CONTEXT,
        )
        assert verdicts(result) == [CONTRADICTED]
        assert result.flagged
        assert "figure conflict" in result.claims[0].reasons[0]

    def test_reversed_direction_is_contradiction(self, checker):
        result = checker.check(
            "Runway incursions fell by 8 percent in 2024, with 214 events recorded.", CONTEXT
        )
        assert CONTRADICTED in verdicts(result)
        assert result.flagged

    def test_fabricated_entity_is_flagged(self, checker):
        result = checker.check(
            "The finding was confirmed by the Verrado Institute review panel.", CONTEXT
        )
        assert result.claims[0].verdict in (UNSUPPORTED, CONTRADICTED)
        assert result.flagged

    def test_unsourced_figure_is_flagged(self, checker):
        result = checker.check("The annual flight-time cap is 1450 hours.", CONTEXT)
        assert result.claims[0].verdict != SUPPORTED
        assert result.flagged

    def test_wholly_invented_claim(self, checker):
        result = checker.check(
            "Quarterly submarine inspections require a notarised sonar affidavit.", CONTEXT
        )
        assert verdicts(result) == [UNSUPPORTED]
        assert result.risk > 0.5


class TestEvidenceAndExplainability:
    def test_every_verdict_carries_evidence(self, checker):
        result = checker.check("Minimum rest before a duty period is 10 hours.", CONTEXT)
        claim = result.claims[0]
        assert claim.evidence_doc_id in {d.id for d in CONTEXT}
        assert claim.evidence

    def test_flags_always_carry_a_reason(self, checker):
        result = checker.check("The annual flight-time cap is 1450 hours.", CONTEXT)
        assert all(claim.reasons for claim in result.flagged_claims)

    def test_faithfulness_is_the_supported_share(self, checker):
        result = checker.check(
            "Minimum rest before a duty period is 10 hours. "
            "Quarterly submarine inspections require a notarised sonar affidavit.",
            CONTEXT,
        )
        assert result.faithfulness == pytest.approx(0.5)


class TestDegenerateInputs:
    def test_empty_answer_asserts_nothing(self, checker):
        result = checker.check("", CONTEXT)
        assert result.claims == [] and not result.flagged and result.faithfulness == 0.0

    def test_no_context_means_nothing_is_grounded(self, checker):
        result = checker.check("Minimum rest is 10 hours.", [])
        assert verdicts(result) == [UNSUPPORTED]
        assert "no context" in result.claims[0].reasons[0]

    def test_checker_without_a_corpus_still_runs(self):
        result = GroundingChecker().check("Minimum rest is 10 hours.", CONTEXT)
        assert result.claims

    def test_answer_similarity_computed_when_gold_given(self, checker):
        result = checker.check(
            "Minimum rest before a duty period is 10 hours.",
            CONTEXT,
            gold_answer="Minimum rest is 10 hours.",
        )
        assert result.answer_similarity is not None and result.answer_similarity > 0.5


class TestConfig:
    def test_invalid_thresholds_rejected(self):
        with pytest.raises(ValueError):
            GroundingConfig(support_threshold=1.5).validate()
        with pytest.raises(ValueError):
            GroundingConfig(mean_weight=-0.1).validate()

    def test_threshold_controls_flagging(self):
        answer = "The annual flight-time cap is 1450 hours."
        strict = GroundingChecker(CONTEXT, GroundingConfig(risk_threshold=0.05))
        lax = GroundingChecker(CONTEXT, GroundingConfig(risk_threshold=0.99))
        assert strict.check(answer, CONTEXT).flagged
        assert not lax.check(answer, CONTEXT).flagged

    def test_results_are_deterministic(self, checker):
        answer = "Runway incursions fell by 8 percent in 2024, with 214 events recorded."
        first = checker.check(answer, CONTEXT)
        second = checker.check(answer, CONTEXT)
        assert first.risk == second.risk and verdicts(first) == verdicts(second)
