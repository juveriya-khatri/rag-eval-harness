"""Hallucination detection: is every claim in the answer entailed by the context?

The detector is deliberately *lexical and explainable* rather than a black-box
model. For each answer it:

1. **Splits** the answer into atomic claims (sentence-level, with clause
   splitting on ``and``/``while``/``whereas`` when both halves carry their own
   facts).
2. **Scores support** for each claim against the retrieved context using
   IDF-weighted term coverage - rare, informative terms must be present, common
   ones are nearly free. Coverage is computed against the single best evidence
   sentence and, at a small discount, against the best *pair* of sentences, so
   multi-hop claims are not unfairly punished.
3. **Runs three precision checks** that catch the failure modes lexical overlap
   alone misses:
   * *numeric check* - every figure, percentage, year and month in the claim
     must appear in the context. Fabricated numbers are the single most common
     RAG hallucination and the most damaging.
   * *entity check* - proper nouns and acronyms must appear in the context.
   * *polarity check* - if a claim aligns strongly with an evidence sentence but
     the negation cues or directional antonyms disagree, that is a
     **contradiction**, which is worse than a merely unsupported statement.
4. **Aggregates** to an answer-level ``risk`` score in [0, 1] combining the mean
   and the worst claim, because one badly fabricated sentence should be enough
   to flag an otherwise faithful answer.

Every verdict carries the evidence sentence and a human-readable reason, so a
flag can always be audited. Thresholds live in :class:`GroundingConfig` and can
be calibrated on labelled data with ``ragval sweep``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .text import (
    NEGATION_CUES,
    direction_conflict,
    extract_entities,
    extract_number_units,
    extract_numbers,
    normalize,
    split_sentences,
    stem,
    tokenize,
)
from .types import CONTRADICTED, SUPPORTED, UNSUPPORTED, ClaimVerdict, Document, GroundingResult

__all__ = ["GroundingConfig", "GroundingChecker", "split_claims"]

_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,\s+(?:and|but|while|whereas|although)\s+|\s+(?:whereas|while|although)\s+)",
    re.IGNORECASE,
)
_HEDGES = {
    "may", "might", "could", "possibly", "perhaps", "likely", "probably",
    "appears", "seems", "suggests", "reportedly", "approximately", "roughly",
}


@dataclass
class GroundingConfig:
    """Tunable thresholds for the detector.

    Defaults were calibrated on the bundled labelled set with ``ragval sweep``;
    ``support_threshold`` is the knob you will most likely want to move
    (raise it for higher recall on hallucinations, lower it for fewer false
    alarms).
    """

    support_threshold: float = 0.62
    contradiction_min_overlap: float = 0.45
    figure_conflict_min_overlap: float = 0.25
    direction_min_overlap: float = 0.55
    negation_min_overlap: float = 0.70
    risk_threshold: float = 0.34
    numeric_penalty: float = 0.45
    numeric_bonus: float = 0.15
    entity_penalty: float = 0.65
    multi_sentence_discount: float = 0.95
    mean_weight: float = 0.55
    min_claim_tokens: int = 2
    split_clauses: bool = True
    hedge_relief: float = 0.06
    # Ablation switches. Turning these off reduces the detector to plain
    # lexical overlap, which is how the contribution of each check was measured
    # (see the ablation table in the README).
    enable_numeric_check: bool = True
    enable_entity_check: bool = True
    enable_contradiction: bool = True

    def validate(self) -> "GroundingConfig":
        for name in ("support_threshold", "contradiction_min_overlap", "risk_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 <= self.mean_weight <= 1.0:
            raise ValueError("mean_weight must be in [0, 1]")
        return self

    def to_dict(self) -> Dict[str, float]:
        return {
            "support_threshold": self.support_threshold,
            "contradiction_min_overlap": self.contradiction_min_overlap,
            "direction_min_overlap": self.direction_min_overlap,
            "negation_min_overlap": self.negation_min_overlap,
            "risk_threshold": self.risk_threshold,
            "numeric_penalty": self.numeric_penalty,
            "numeric_bonus": self.numeric_bonus,
            "entity_penalty": self.entity_penalty,
            "mean_weight": self.mean_weight,
        }


def split_claims(answer: str, *, split_clauses: bool = True, min_tokens: int = 2) -> List[str]:
    """Break an answer into atomic, independently checkable claims."""
    claims: List[str] = []
    for sentence in split_sentences(answer):
        parts = [sentence]
        if split_clauses and len(sentence.split()) > 14:
            candidate = [p.strip(" ,;") for p in _CLAUSE_SPLIT_RE.split(sentence)]
            # Only accept the split when both halves stand alone as facts.
            if len(candidate) > 1 and all(len(p.split()) >= 4 for p in candidate):
                parts = candidate
        for part in parts:
            part = part.strip()
            if part and len(tokenize(part, remove_stopwords=False)) >= min_tokens:
                claims.append(part)
    return claims


@dataclass
class _Evidence:
    """A single context sentence, pre-tokenised once and reused."""

    doc_id: str
    text: str
    tokens: Set[str]
    numbers: Set[Tuple[float, str]]
    number_units: Set[Tuple[float, str, str]]
    entities: Set[str]
    negations: Set[str]


class GroundingChecker:
    """Scores answers against retrieved context.

    Pass the full corpus at construction time so IDF weights come from the whole
    collection rather than just the few retrieved passages - otherwise every
    term in a two-document context looks equally rare.
    """

    def __init__(
        self,
        corpus: Optional[Sequence[Document]] = None,
        config: Optional[GroundingConfig] = None,
    ):
        self.config = (config or GroundingConfig()).validate()
        self._idf: Dict[str, float] = {}
        self._default_idf: float = 1.0
        self._sentence_cache: Dict[str, List[_Evidence]] = {}
        if corpus:
            self.fit(corpus)

    # -- IDF ---------------------------------------------------------------- #

    def fit(self, corpus: Sequence[Document]) -> "GroundingChecker":
        n_docs = max(len(corpus), 1)
        df: Counter = Counter()
        for doc in corpus:
            df.update(set(tokenize(doc.indexable_text)))
        self._idf = {
            term: math.log((1.0 + n_docs) / (1.0 + d)) + 1.0 for term, d in df.items()
        }
        # Unknown terms are, by definition, not in the corpus: treat them as
        # maximally informative so a fabricated rare word cannot be free.
        self._default_idf = math.log(1.0 + n_docs) + 1.0
        return self

    def idf(self, term: str) -> float:
        return self._idf.get(term, self._default_idf)

    # -- evidence preparation ---------------------------------------------- #

    def _evidence(self, docs: Sequence[Document]) -> List[_Evidence]:
        out: List[_Evidence] = []
        for doc in docs:
            cached = self._sentence_cache.get(doc.id)
            if cached is None:
                cached = []
                body = f"{doc.title}. {doc.text}" if doc.title else doc.text
                for sentence in split_sentences(body):
                    toks = set(tokenize(sentence))
                    cached.append(
                        _Evidence(
                            doc_id=doc.id,
                            text=sentence,
                            tokens=toks,
                            numbers=set(extract_numbers(sentence)),
                            number_units=extract_number_units(sentence),
                            entities={e.lower() for e in extract_entities(sentence)},
                            negations=toks & _stemmed_negations(),
                        )
                    )
                self._sentence_cache[doc.id] = cached
            out.extend(cached)
        return out

    @staticmethod
    def _by_document(evidence: Sequence[_Evidence]) -> Dict[str, Set[Tuple[float, str, str]]]:
        """Union of ``(value, kind, unit)`` triples per source document."""
        out: Dict[str, Set[Tuple[float, str, str]]] = {}
        for ev in evidence:
            out.setdefault(ev.doc_id, set()).update(ev.number_units)
        return out

    # -- main entry point --------------------------------------------------- #

    def check(
        self,
        answer: str,
        context: Sequence[Document],
        *,
        query_id: str = "",
        gold_hallucinated: Optional[bool] = None,
        gold_answer: str = "",
    ) -> GroundingResult:
        """Assess ``answer`` against ``context`` and return a full verdict."""
        cfg = self.config
        claims = split_claims(answer, split_clauses=cfg.split_clauses, min_tokens=cfg.min_claim_tokens)
        evidence = self._evidence(context)
        context_numbers = {n for ev in evidence for n in ev.numbers}
        context_entities = {e for ev in evidence for e in ev.entities}
        context_tokens = {t for ev in evidence for t in ev.tokens}
        by_document = self._by_document(evidence)

        verdicts: List[ClaimVerdict] = []
        for claim in claims:
            verdicts.append(
                self._check_claim(
                    claim, evidence, context_numbers, context_entities, context_tokens, by_document
                )
            )

        if not verdicts:
            # An empty answer asserts nothing, so it cannot hallucinate - but it
            # is also useless, so surface it as zero faithfulness with no flag.
            return GroundingResult(
                query_id=query_id,
                answer=answer,
                claims=[],
                faithfulness=0.0,
                risk=0.0,
                flagged=False,
                gold_hallucinated=gold_hallucinated,
                answer_similarity=None,
            )

        supports = [v.support for v in verdicts]
        mean_support = sum(supports) / len(supports)
        min_support = min(supports)
        risk = cfg.mean_weight * (1.0 - mean_support) + (1.0 - cfg.mean_weight) * (1.0 - min_support)
        if any(v.verdict == CONTRADICTED for v in verdicts):
            risk = max(risk, 0.9)
        risk = min(max(risk, 0.0), 1.0)
        faithfulness = sum(1 for v in verdicts if v.verdict == SUPPORTED) / len(verdicts)

        similarity = None
        if gold_answer:
            from .metrics import token_f1

            similarity = token_f1(answer, gold_answer)

        return GroundingResult(
            query_id=query_id,
            answer=answer,
            claims=verdicts,
            faithfulness=faithfulness,
            risk=risk,
            flagged=risk >= cfg.risk_threshold,
            gold_hallucinated=gold_hallucinated,
            answer_similarity=similarity,
        )

    # -- per-claim logic ---------------------------------------------------- #

    def _check_claim(
        self,
        claim: str,
        evidence: Sequence[_Evidence],
        context_numbers: Set[Tuple[float, str]],
        context_entities: Set[str],
        context_tokens: Set[str],
        by_document: Dict[str, Set[Tuple[float, str, str]]],
    ) -> ClaimVerdict:
        cfg = self.config
        claim_tokens = set(tokenize(claim))
        if not claim_tokens:
            return ClaimVerdict(claim=claim, verdict=SUPPORTED, support=1.0, reasons=["no content terms"])

        weights = {t: self.idf(t) for t in claim_tokens}
        total_weight = sum(weights.values()) or 1.0

        if not evidence:
            return ClaimVerdict(
                claim=claim,
                verdict=UNSUPPORTED,
                support=0.0,
                reasons=["no context retrieved for this query"],
            )

        # 1. best single evidence sentence
        scored: List[Tuple[float, _Evidence]] = []
        for ev in evidence:
            covered = sum(w for t, w in weights.items() if t in ev.tokens)
            scored.append((covered / total_weight, ev))
        scored.sort(key=lambda pair: (-pair[0], pair[1].doc_id, pair[1].text))
        best_cov, best_ev = scored[0]

        # 2. best pair of sentences (multi-hop), at a small discount
        coverage = best_cov
        if best_cov < 1.0 and len(scored) > 1:
            pool = [ev for _, ev in scored[:8]]
            best_pair = best_cov
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    union = pool[i].tokens | pool[j].tokens
                    cov = sum(w for t, w in weights.items() if t in union) / total_weight
                    if cov > best_pair:
                        best_pair = cov
            coverage = max(best_cov, best_pair * cfg.multi_sentence_discount)

        reasons: List[str] = []
        support = coverage

        # 3. numeric check - fabricated figures are the highest-signal failure
        claim_numbers = set(extract_numbers(claim)) if cfg.enable_numeric_check else set()
        claim_number_units = extract_number_units(claim) if cfg.enable_contradiction else set()
        missing_numbers = [n for n in claim_numbers if not _number_present(n, context_numbers)]
        if missing_numbers:
            support *= cfg.numeric_penalty
            pretty = ", ".join(_fmt_number(n) for n in sorted(missing_numbers)[:4])
            reasons.append(f"figure(s) not present in context: {pretty}")
        elif claim_numbers:
            # Every figure in the claim is sourced. That is positive evidence of
            # grounding and offsets vocabulary drift in the surrounding prose,
            # which is what otherwise sinks a faithful paraphrase.
            support = min(1.0, support + cfg.numeric_bonus)

        # 4. entity check
        claim_entities = {e.lower() for e in extract_entities(claim)} if cfg.enable_entity_check else set()
        missing_entities = [
            e
            for e in claim_entities
            if e not in context_entities and not _entity_tokens_present(e, context_tokens)
        ]
        if missing_entities:
            support *= cfg.entity_penalty
            reasons.append("entity not in context: " + ", ".join(sorted(missing_entities)[:3]))

        # 5. hedged language is a weaker assertion, so relax slightly
        if claim_tokens & {stem(h) for h in _HEDGES}:
            support = min(1.0, support + cfg.hedge_relief)

        support = min(max(support, 0.0), 1.0)

        # 6. contradiction: opposite polarity, or a figure the source disputes
        contradiction = (
            self._contradiction_reason(claim_tokens, claim_number_units, scored, by_document)
            if cfg.enable_contradiction
            else ""
        )
        if contradiction:
            reasons.insert(0, contradiction)
            return ClaimVerdict(
                claim=claim,
                verdict=CONTRADICTED,
                support=min(support, 0.15),
                reasons=reasons,
                evidence_doc_id=best_ev.doc_id,
                evidence=best_ev.text,
            )

        if support >= cfg.support_threshold and not missing_numbers and not missing_entities:
            verdict = SUPPORTED
        else:
            verdict = UNSUPPORTED
            if not reasons:
                reasons.append(
                    f"only {support:.0%} of the claim's informative terms are covered by the context"
                )

        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            support=support,
            reasons=reasons,
            evidence_doc_id=best_ev.doc_id,
            evidence=best_ev.text,
        )

    def _contradiction_reason(
        self,
        claim_tokens: Set[str],
        claim_number_units: Set[Tuple[float, str, str]],
        scored: Sequence[Tuple[float, _Evidence]],
        by_document: Dict[str, Set[Tuple[float, str, str]]],
    ) -> str:
        """Detect figure conflicts and polarity flips against *aligned* evidence.

        Only sentences that are near-ties with the best match are considered.
        Scanning further down the ranking is how a naive implementation
        manufactures contradictions out of unrelated passages - the original
        version of this method flagged nine faithful answers that way.
        """
        cfg = self.config
        if not scored:
            return ""
        best_cov, best_ev = scored[0]

        # (a) Figure conflict, resolved at *document* scope.
        #
        # The claim is compared against the whole passage it paraphrases, not a
        # single sentence: "the crew descended from 36,000 feet to 10,000 feet"
        # draws its two figures from two different sentences of the same report
        # and is perfectly faithful. A conflict therefore requires the claim's
        # (value, unit) pair to be absent from the entire source document while
        # that document asserts a *different* value for the same unit.
        doc_units = by_document.get(best_ev.doc_id, set())
        for value, kind, unit in sorted(claim_number_units):
            if best_cov < cfg.figure_conflict_min_overlap:
                break
            if not unit:
                continue  # bare years and months carry no comparable slot
            if any(
                k == kind and u == unit and math.isclose(v, value, rel_tol=0.005, abs_tol=1e-9)
                for v, k, u in doc_units
            ):
                continue
            rivals = [v for v, k, u in doc_units if k == kind and u == unit]
            if not rivals:
                continue
            nearest = min(rivals, key=lambda v: abs(v - value))
            if _significantly_different(value, nearest):
                return (
                    f"figure conflict for '{unit}': claim states "
                    f"{_fmt_number((value, kind))}, source states "
                    f"{_fmt_number((nearest, kind))}"
                )

        # (b) and (c) are sentence-level: they only make sense against evidence
        # that is a near-restatement of the claim. Scanning further down the
        # ranking is how a naive implementation manufactures contradictions out
        # of unrelated passages.
        claim_negations = claim_tokens & _stemmed_negations()
        for cov, ev in scored[:4]:
            if cov < cfg.contradiction_min_overlap or cov < best_cov - 0.05:
                break
            direction = direction_conflict(claim_tokens, ev.tokens)
            if direction and cov >= cfg.direction_min_overlap:
                return f"directional conflict ({direction})"
            if (
                cov >= cfg.negation_min_overlap
                and bool(claim_negations) != bool(ev.negations)
                and len(claim_tokens & ev.tokens) >= 4
            ):
                cue = ", ".join(sorted(claim_negations or ev.negations)[:2])
                return f"polarity conflict with evidence (negation cue: {cue})"
        return ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_STEMMED_NEGATIONS: Optional[Set[str]] = None


def _stemmed_negations() -> Set[str]:
    global _STEMMED_NEGATIONS
    if _STEMMED_NEGATIONS is None:
        _STEMMED_NEGATIONS = {stem(w) for w in NEGATION_CUES}
    return _STEMMED_NEGATIONS


def _number_present(
    number: Tuple[float, str], context_numbers: Set[Tuple[float, str]], *, rel_tol: float = 0.005
) -> bool:
    """A figure counts as present if an equal (or near-equal) value of the same
    kind appears in the context. Small relative tolerance absorbs rounding
    (``4.7%`` vs ``4.70%``) without letting ``47%`` pass for ``4.7%``."""
    value, kind = number
    for other_value, other_kind in context_numbers:
        if other_kind != kind:
            continue
        if value == other_value:
            return True
        if math.isclose(value, other_value, rel_tol=rel_tol, abs_tol=1e-9):
            return True
    return False


def _significantly_different(a: float, b: float, *, rel_tol: float = 0.02) -> bool:
    return not math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def _entity_tokens_present(entity: str, context_tokens: Set[str]) -> bool:
    """Fall back to token-level containment for multi-word or inflected names."""
    toks = tokenize(entity, remove_stopwords=False)
    if not toks:
        return True
    return all(t in context_tokens for t in toks)


def _fmt_number(number: Tuple[float, str]) -> str:
    value, kind = number
    if kind == "percent":
        return f"{value:g}%"
    if kind == "month":
        return f"month {int(value)}"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:g}"
