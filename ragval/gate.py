"""Quality gate: turn an evaluation into a pass/fail CI signal.

Two independent checks, both optional:

* **Absolute thresholds** - "recall@5 must be at least 0.80". Catches a
  pipeline that was never good enough in the first place.
* **Regression vs a frozen baseline** - "recall@5 may not drop by more than
  0.02 from ``baseline.json``". Catches the far more common failure: a chunking
  tweak or prompt change that quietly degrades retrieval.

The tolerance matters. Without it every run fails on bootstrap-scale noise; the
default 0.02 is roughly the half-width of the confidence interval on a 30-query
set, so only real movement trips the gate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["GateCheck", "GateResult", "run_gate", "load_config", "make_baseline", "DEFAULT_CONFIG"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "thresholds": {
        "retrieval": {"recall@5": 0.80, "ndcg@5": 0.70, "hit_rate@5": 0.90},
        "detector": {"f1": 0.70, "recall": 0.70},
    },
    "regression_tolerance": 0.02,
    "regression_metrics": [
        "retrieval.recall@5",
        "retrieval.ndcg@5",
        "retrieval.precision@5",
        "retrieval.mrr",
        "detector.f1",
    ],
}


@dataclass
class GateCheck:
    name: str
    kind: str  # "threshold" | "regression"
    value: float
    reference: float
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "value": round(self.value, 6),
            "reference": round(self.reference, 6),
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class GateResult:
    checks: List[GateCheck] = field(default_factory=list)
    passed: bool = True

    @property
    def failures(self) -> List[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failures": len(self.failures),
            "checks": [c.to_dict() for c in self.checks],
        }


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a gate config from JSON (or TOML on Python 3.11+); fall back to defaults."""
    if not path:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    if not os.path.exists(path):
        raise FileNotFoundError(f"gate config not found: {path}")
    if path.endswith(".toml"):
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError as exc:  # pragma: no cover - version dependent
            raise RuntimeError("TOML configs require Python 3.11+; use JSON instead") from exc
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged.update(raw or {})
    return merged


def _lookup(results: Mapping[str, Any], dotted: str) -> Optional[float]:
    """Resolve ``"retrieval.recall@5"`` against a results dict."""
    node: Any = results
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def make_baseline(results: Mapping[str, Any], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """Freeze the metrics a future run will be compared against."""
    names = metrics or DEFAULT_CONFIG["regression_metrics"]
    frozen: Dict[str, float] = {}
    for name in names:
        value = _lookup(results, name)
        if value is not None:
            frozen[name] = round(value, 6)
    return {
        "generated_at": results.get("generated_at", ""),
        "dataset": (results.get("config") or {}).get("dataset", ""),
        "retriever": (results.get("config") or {}).get("retriever_name", ""),
        "metrics": frozen,
    }


def run_gate(
    results: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    baseline: Optional[Mapping[str, Any]] = None,
) -> GateResult:
    """Evaluate thresholds and regressions; ``GateResult.passed`` drives exit code."""
    cfg = dict(config or DEFAULT_CONFIG)
    checks: List[GateCheck] = []

    thresholds = cfg.get("thresholds", {}) or {}
    for section, metrics in thresholds.items():
        for metric, minimum in (metrics or {}).items():
            dotted = f"{section}.{metric}"
            value = _lookup(results, dotted)
            if value is None:
                checks.append(
                    GateCheck(
                        name=dotted,
                        kind="threshold",
                        value=float("nan"),
                        reference=float(minimum),
                        passed=False,
                        detail="metric missing from results",
                    )
                )
                continue
            passed = value >= float(minimum) - 1e-9
            checks.append(
                GateCheck(
                    name=dotted,
                    kind="threshold",
                    value=value,
                    reference=float(minimum),
                    passed=passed,
                    detail="" if passed else f"{value:.4f} < required {float(minimum):.4f}",
                )
            )

    if baseline:
        tol = float(cfg.get("regression_tolerance", 0.02))
        frozen = baseline.get("metrics", baseline) or {}
        for dotted, previous in frozen.items():
            value = _lookup(results, dotted)
            if value is None:
                continue
            delta = value - float(previous)
            passed = delta >= -tol - 1e-9
            checks.append(
                GateCheck(
                    name=dotted,
                    kind="regression",
                    value=value,
                    reference=float(previous),
                    passed=passed,
                    detail=(
                        f"{delta:+.4f} vs baseline (tolerance {tol:.4f})"
                        if not passed
                        else f"{delta:+.4f}"
                    ),
                )
            )

    result = GateResult(checks=checks)
    result.passed = all(c.passed for c in checks)
    return result
