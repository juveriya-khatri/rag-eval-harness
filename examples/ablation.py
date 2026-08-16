"""Ablation study: how much does each grounding check actually contribute?

Every row turns one component off and re-scores the detector against the same
labelled answers. The numbers printed here are the ones reported in the README.

    python examples/ablation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragval import EvalOptions, evaluate, load_dataset  # noqa: E402
from ragval.grounding import GroundingConfig  # noqa: E402

VARIANTS = {
    "full detector": {},
    "no contradiction rule": {"enable_contradiction": False},
    "no numeric check": {"enable_numeric_check": False},
    "no entity check": {"enable_entity_check": False},
    "lexical overlap only": {
        "enable_contradiction": False,
        "enable_numeric_check": False,
        "enable_entity_check": False,
    },
}


def main() -> int:
    dataset = load_dataset("bundled")
    header = f"{'variant':<24}{'precision':>10}{'recall':>9}{'F1':>8}{'ROC-AUC':>9}{'FP':>4}{'FN':>4}"
    print(header)
    print("-" * len(header))
    for label, overrides in VARIANTS.items():
        config = GroundingConfig(**overrides)
        report = evaluate(dataset, EvalOptions(bootstrap=0, grounding=config))
        d = report.detector
        print(
            f"{label:<24}{d['precision']:>10.3f}{d['recall']:>9.3f}{d['f1']:>8.3f}"
            f"{d['roc_auc']:>9.3f}{int(d['fp']):>4}{int(d['fn']):>4}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
