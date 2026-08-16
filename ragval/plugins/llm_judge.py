"""Optional LLM-as-judge grounding adapter (``pip install "rag-eval-harness[llm]"``).

The lexical detector in :mod:`ragval.grounding` is fast, free and fully
explainable, but it cannot recognise a paraphrase that shares no vocabulary with
its evidence. This adapter escalates only the *uncertain* claims - the ones
whose support score sits in a configurable band around the decision boundary -
to an LLM, which typically means judging 10-20% of claims rather than all of
them.

Bring your own client: anything with ``complete(prompt) -> str``. That keeps the
harness free of vendor SDKs and makes the judge trivially mockable in tests.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

from ..grounding import GroundingChecker, GroundingConfig
from ..types import CONTRADICTED, SUPPORTED, UNSUPPORTED, ClaimVerdict, Document, GroundingResult

__all__ = ["JudgeClient", "LLMJudgeChecker", "PROMPT_TEMPLATE"]

PROMPT_TEMPLATE = """You are a strict factual verifier for a retrieval-augmented system.

CONTEXT (the only permissible source of truth):
---
{context}
---

CLAIM: {claim}

Decide whether the CONTEXT entails the CLAIM. Rules:
- "supported"     - every part of the claim, including all figures, dates and
                    names, is stated or directly implied by the context.
- "contradicted"  - the context asserts something incompatible with the claim.
- "unsupported"   - the context neither entails nor contradicts the claim.
Do not use outside knowledge. A claim with a figure absent from the context is
never "supported".

Reply with JSON only: {{"verdict": "...", "confidence": 0.0-1.0, "reason": "one short sentence"}}
"""


class JudgeClient(Protocol):
    """Any object with a ``complete(prompt) -> str`` method."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class LLMJudgeChecker(GroundingChecker):
    """Lexical detector first, LLM adjudication for the uncertain middle band.

    Parameters
    ----------
    client:
        Object exposing ``complete(prompt) -> str``.
    uncertainty_band:
        Claims whose lexical support falls within ``+/- band`` of
        ``support_threshold`` are escalated. Widen it to spend more tokens for
        more accuracy; set it to 0 to disable escalation entirely.
    """

    def __init__(
        self,
        corpus: Sequence[Document],
        client: JudgeClient,
        config: Optional[GroundingConfig] = None,
        *,
        uncertainty_band: float = 0.18,
        max_calls: int = 200,
    ):
        super().__init__(corpus, config)
        self.client = client
        self.uncertainty_band = max(float(uncertainty_band), 0.0)
        self.max_calls = int(max_calls)
        self.calls_made = 0

    def check(self, answer: str, context: Sequence[Document], **kwargs: Any) -> GroundingResult:
        result = super().check(answer, context, **kwargs)
        if not result.claims or self.uncertainty_band <= 0:
            return result

        context_text = "\n\n".join(
            f"[{d.id}] {d.title}\n{d.text}".strip() for d in context
        )
        threshold = self.config.support_threshold
        changed = False
        for claim in result.claims:
            if self.calls_made >= self.max_calls:
                break
            if abs(claim.support - threshold) > self.uncertainty_band:
                continue  # lexically decisive - no need to spend a call
            verdict, confidence, reason = self._judge(claim.claim, context_text)
            if verdict is None:
                continue
            self.calls_made += 1
            if verdict != claim.verdict:
                claim.reasons.append(
                    f"LLM judge overrode '{claim.verdict}' -> '{verdict}' "
                    f"(confidence {confidence:.2f}): {reason}"
                )
                claim.verdict = verdict
                claim.support = confidence if verdict == SUPPORTED else 1.0 - confidence
                changed = True

        if changed:
            supports = [c.support for c in result.claims]
            mean_support = sum(supports) / len(supports)
            min_support = min(supports)
            w = self.config.mean_weight
            risk = w * (1.0 - mean_support) + (1.0 - w) * (1.0 - min_support)
            if any(c.verdict == CONTRADICTED for c in result.claims):
                risk = max(risk, 0.9)
            result.risk = min(max(risk, 0.0), 1.0)
            result.faithfulness = sum(
                1 for c in result.claims if c.verdict == SUPPORTED
            ) / len(result.claims)
            result.flagged = result.risk >= self.config.risk_threshold
        return result

    def _judge(self, claim: str, context: str) -> Tuple[Optional[str], float, str]:
        prompt = PROMPT_TEMPLATE.format(context=context, claim=claim)
        try:
            raw = self.client.complete(prompt)
        except Exception as exc:  # noqa: BLE001 - a judge outage must not fail the run
            return None, 0.0, f"judge call failed: {exc}"
        return _parse_judgement(raw)


def _parse_judgement(raw: str) -> Tuple[Optional[str], float, str]:
    """Tolerantly parse the judge's reply (models love wrapping JSON in prose)."""
    if not raw:
        return None, 0.0, "empty response"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, 0.0, "no JSON object in response"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 0.0, "malformed JSON in response"
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {SUPPORTED, UNSUPPORTED, CONTRADICTED}:
        return None, 0.0, f"unrecognised verdict {verdict!r}"
    try:
        confidence = min(max(float(payload.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.5
    return verdict, confidence, str(payload.get("reason", ""))[:200]
