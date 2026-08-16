"""Metric correctness, with emphasis on the edge cases that silently corrupt
evaluation: empty rankings, duplicate ids, undefined recall, graded gains."""

from __future__ import annotations

import math

import pytest

from ragval.metrics import (
    average_precision_at_k,
    binary_classification_metrics,
    bootstrap_ci,
    dedupe,
    f1_at_k,
    hit_rate_at_k,
    macro_average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    roc_auc,
    threshold_sweep,
    token_f1,
)


class TestPrecisionRecall:
    def test_perfect_ranking(self):
        assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0
        assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_strict_k_penalises_short_rankings(self):
        # Only one result returned but k=5 -> precision is 1/5, not 1/1.
        assert precision_at_k(["a"], ["a"], 5) == pytest.approx(0.2)
        assert precision_at_k(["a"], ["a"], 5, strict_k=False) == 1.0

    def test_truncation_at_k(self):
        assert precision_at_k(["x", "y", "a"], ["a"], 2) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], 3) == 1.0

    def test_empty_ranking(self):
        assert precision_at_k([], ["a"], 5) == 0.0
        assert recall_at_k([], ["a"], 5) == 0.0
        assert hit_rate_at_k([], ["a"], 5) == 0.0

    def test_recall_undefined_without_judgements(self):
        assert math.isnan(recall_at_k(["a"], [], 5))
        assert math.isnan(hit_rate_at_k(["a"], [], 5))
        assert math.isnan(reciprocal_rank(["a"], []))
        assert math.isnan(f1_at_k(["a"], [], 5))

    def test_duplicates_collapse_to_best_rank(self):
        assert dedupe(["a", "a", "b", "a"]) == ["a", "b"]
        # Padding the ranking with repeats must not inflate precision.
        assert precision_at_k(["a", "a", "a"], ["a"], 3) == pytest.approx(1 / 3)

    def test_graded_relevance_accepts_mapping_or_list(self):
        assert recall_at_k(["a"], {"a": 2, "b": 1}, 5) == pytest.approx(0.5)
        assert recall_at_k(["a"], ["a", "b"], 5) == pytest.approx(0.5)
        # A zero gain means "judged, not relevant".
        assert recall_at_k(["a"], {"a": 1, "b": 0}, 5) == 1.0

    @pytest.mark.parametrize("bad_k", [0, -1])
    def test_invalid_k_rejected(self, bad_k):
        with pytest.raises(ValueError):
            precision_at_k(["a"], ["a"], bad_k)

    def test_non_integer_k_rejected(self):
        with pytest.raises(TypeError):
            precision_at_k(["a"], ["a"], 2.5)  # type: ignore[arg-type]


class TestRankingQuality:
    def test_reciprocal_rank(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
        assert reciprocal_rank(["x"], ["a"]) == 0.0

    def test_average_precision(self):
        # Relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
        got = average_precision_at_k(["a", "x", "b"], ["a", "b"], 3)
        assert got == pytest.approx((1.0 + 2 / 3) / 2)

    def test_average_precision_normalises_by_min(self):
        # Three relevant documents but k=1: denominator is 1, not 3.
        assert average_precision_at_k(["a"], ["a", "b", "c"], 1) == 1.0

    def test_ndcg_perfect_and_reversed(self):
        rel = {"a": 2, "b": 1}
        assert ndcg_at_k(["a", "b"], rel, 2) == pytest.approx(1.0)
        assert ndcg_at_k(["b", "a"], rel, 2) < 1.0

    def test_ndcg_rewards_higher_grades_first(self):
        rel = {"a": 3, "b": 1}
        assert ndcg_at_k(["a", "b"], rel, 2) > ndcg_at_k(["b", "a"], rel, 2)

    def test_ndcg_linear_gain_option(self):
        rel = {"a": 2, "b": 1}
        assert ndcg_at_k(["b", "a"], rel, 2, exponential=False) == pytest.approx(
            (1.0 + 2 / math.log2(3)) / (2.0 + 1 / math.log2(3))
        )

    def test_ndcg_zero_when_nothing_relevant_retrieved(self):
        assert ndcg_at_k(["x", "y"], {"a": 2}, 2) == 0.0


class TestAggregation:
    def test_retrieval_metrics_keys(self):
        out = retrieval_metrics(["a"], ["a"], ks=(1, 3))
        assert set(out) == {
            "precision@1", "recall@1", "f1@1", "hit_rate@1", "map@1", "ndcg@1",
            "precision@3", "recall@3", "f1@3", "hit_rate@3", "map@3", "ndcg@3",
            "mrr",
        }

    def test_macro_average_skips_nan(self):
        rows = [{"m": 1.0}, {"m": float("nan")}, {"m": 0.0}]
        assert macro_average(rows)["m"] == pytest.approx(0.5)

    def test_macro_average_empty(self):
        assert macro_average([]) == {}

    def test_bootstrap_ci_is_deterministic_and_brackets_the_mean(self):
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        first = bootstrap_ci(values, seed=7)
        second = bootstrap_ci(values, seed=7)
        assert first == second
        assert first[0] <= sum(values) / len(values) <= first[1]

    def test_bootstrap_ci_degenerate_inputs(self):
        assert all(math.isnan(x) for x in bootstrap_ci([]))
        assert bootstrap_ci([0.5]) == (0.5, 0.5)
        assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


class TestClassification:
    def test_confusion_counts(self):
        out = binary_classification_metrics([True, True, False, False], [True, False, True, False])
        assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (1.0, 1.0, 1.0, 1.0)
        assert out["precision"] == 0.5
        assert out["recall"] == 0.5
        assert out["f1"] == 0.5
        assert out["accuracy"] == 0.5

    def test_no_predictions_gives_zero_not_crash(self):
        out = binary_classification_metrics([True, False], [False, False])
        assert out["precision"] == 0.0 and out["recall"] == 0.0 and out["f1"] == 0.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            binary_classification_metrics([True], [True, False])

    def test_roc_auc_perfect_and_inverted(self):
        assert roc_auc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert roc_auc([True, True, False, False], [0.1, 0.2, 0.8, 0.9]) == 0.0

    def test_roc_auc_handles_ties(self):
        # All scores identical -> no discrimination at all.
        assert roc_auc([True, False, True, False], [0.5] * 4) == pytest.approx(0.5)

    def test_roc_auc_undefined_for_single_class(self):
        assert math.isnan(roc_auc([True, True], [0.1, 0.9]))

    def test_threshold_sweep_monotonic_recall(self):
        rows = threshold_sweep([True, False, True], [0.9, 0.1, 0.6], steps=11)
        recalls = [r["recall"] for r in rows]
        assert recalls == sorted(recalls, reverse=True)
        assert rows[0]["threshold"] == 0.0 and rows[-1]["threshold"] == 1.0


class TestTokenF1:
    def test_identical_and_disjoint(self):
        assert token_f1("the quick brown fox", "the quick brown fox") == 1.0
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self):
        assert 0.0 < token_f1("the quick brown fox", "the quick fox") < 1.0

    def test_empty_strings(self):
        assert token_f1("", "") == 1.0
        assert token_f1("something", "") == 0.0
