"""Text processing: tokenisation, sentence splitting, numeric and entity extraction."""

from __future__ import annotations

import pytest

from ragval.text import (
    direction_conflict,
    extract_entities,
    extract_number_units,
    extract_numbers,
    normalize,
    split_sentences,
    stem,
    tokenize,
)


class TestNormalizeAndTokenize:
    def test_unicode_folding_and_quotes(self):
        assert normalize("café") == "cafe"
        assert normalize("“quoted”") == '"quoted"'
        assert normalize("a  \n b\tc") == "a b c"

    def test_empty_inputs(self):
        assert normalize("") == ""
        assert tokenize("") == []
        assert split_sentences("") == []
        assert extract_numbers("") == []
        assert extract_entities("") == []

    def test_stopwords_removed_by_default(self):
        assert "the" not in tokenize("the report")
        assert "the" in tokenize("the report", remove_stopwords=False)

    def test_thousands_separators_unified(self):
        assert tokenize("1,200 units") == tokenize("1200 units")

    def test_possessives_stripped(self):
        assert tokenize("operator's manual") == tokenize("operator manual")

    @pytest.mark.parametrize(
        "word,expected",
        [
            ("filters", "filter"),
            ("filtering", "filter"),
            ("filtered", "filter"),
            ("policies", "policy"),
            ("gas", "gas"),          # too short to strip
            ("analysis", "analysis"),  # -is protected
            ("bus", "bus"),            # -us protected
            ("pass", "pass"),          # -ss protected
            ("stopping", "stop"),      # doubled consonant undone
        ],
    )
    def test_stemmer_is_conservative(self, word, expected):
        assert stem(word) == expected

    def test_stem_never_returns_empty(self):
        for word in ["a", "an", "is", "ies", "ed", "ing"]:
            assert stem(word)


class TestSentenceSplitting:
    def test_basic(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_decimals_are_not_boundaries(self):
        assert split_sentences("The rate was 3.5 percent overall.") == [
            "The rate was 3.5 percent overall."
        ]

    def test_abbreviations_are_not_boundaries(self):
        got = split_sentences("Dr. Smith reviewed it. He agreed.")
        assert got == ["Dr. Smith reviewed it.", "He agreed."]

    def test_initials_are_not_boundaries(self):
        assert len(split_sentences("J. R. Smith signed the report.")) == 1

    def test_ellipsis_is_not_a_boundary(self):
        assert len(split_sentences("It was ... unclear at the time.")) == 1

    def test_trailing_text_without_punctuation_is_kept(self):
        assert split_sentences("First one. Second one") == ["First one.", "Second one"]


class TestNumbers:
    def test_plain_and_percent_live_in_separate_namespaces(self):
        assert (50.0, "number") in extract_numbers("50 units")
        assert (50.0, "percent") in extract_numbers("50 percent")
        assert (50.0, "number") not in extract_numbers("50%")

    def test_magnitude_words_folded_into_value(self):
        assert (1_200_000_000.0, "number") in extract_numbers("1.2 billion")
        assert (1_200_000_000.0, "number") in extract_numbers("1,200,000,000")

    def test_years_detected(self):
        assert (2024.0, "year") in extract_numbers("issued in 2024")
        assert (1200.0, "number") in extract_numbers("1200 feet")

    def test_negative_values(self):
        assert (-3.0, "number") in extract_numbers("the temperature was -3 C")

    def test_unambiguous_month_names(self):
        assert (3.0, "month") in extract_numbers("14 March 2023")
        # "may" as a modal verb must not become month 5
        assert (5.0, "month") not in extract_numbers("this may happen")

    def test_units_attach_the_measured_noun(self):
        assert (13.0, "number", "hour") in extract_number_units("a duty period of 13 hours")
        # stopwords are skipped when looking for the unit
        assert (48.0, "number", "megawatt") in extract_number_units("48 per megawatt hour")

    def test_percent_units_are_literal(self):
        assert (25.0, "percent", "percent") in extract_number_units("a credit of 25 percent")

    def test_years_carry_no_unit(self):
        assert (2024.0, "year", "") in extract_number_units("in 2024, deployment rose")

    def test_unit_awareness_separates_conflicting_slots(self):
        source = extract_number_units("failover took 11 minutes against a 2 minute target")
        claim = extract_number_units("failover completed in 4 minutes")
        assert (4.0, "number", "minute") not in source
        assert any(k == "number" and u == "minute" for _, k, u in claim)


class TestEntities:
    def test_multiword_names_and_acronyms(self):
        found = {e.lower() for e in extract_entities("Flight NR-482 landed at Halden Regional.")}
        assert "flight nr-482" in found
        assert "halden regional" in found

    def test_sentence_initial_single_words_ignored(self):
        # "Roughly" is capitalised by orthography, not because it names anything.
        assert extract_entities("Roughly half of the fleet was affected.") == []

    def test_midsentence_proper_noun_captured(self):
        found = {e.lower() for e in extract_entities("Confirmed by the Verrado cohort last year.")}
        assert any("verrado" in e for e in found)


class TestDirectionConflict:
    def test_opposite_directions_flagged(self):
        claim = set(tokenize("incursions fell by 8 percent"))
        evidence = set(tokenize("an increase of 8 percent over last year"))
        assert direction_conflict(claim, evidence)

    def test_same_direction_not_flagged(self):
        claim = set(tokenize("deployment rose by 61 percent"))
        evidence = set(tokenize("an increase of 61 percent year over year"))
        assert direction_conflict(claim, evidence) == ""

    def test_mixed_direction_sentences_are_skipped(self):
        # A sentence containing both directions is ambiguous, so never a conflict.
        claim = set(tokenize("costs rose while volumes fell"))
        evidence = set(tokenize("volumes increased"))
        assert direction_conflict(claim, evidence) == ""
