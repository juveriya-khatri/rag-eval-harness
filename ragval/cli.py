"""Command line interface.

    ragval run       evaluate a dataset and emit terminal / JSON / HTML / JUnit output
    ragval gate      pass-fail a results file against thresholds and a baseline
    ragval baseline  freeze current results as the regression baseline
    ragval sweep     calibrate the detector threshold on labelled data
    ragval validate  check a dataset for label bugs before you trust any numbers

Exit codes: ``0`` success, ``1`` gate failed, ``2`` bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .dataset import Dataset, ValidationError, load_dataset
from .evaluate import DEFAULT_KS, EvalOptions, evaluate
from .gate import load_config, make_baseline, run_gate
from .grounding import GroundingConfig
from .index import RETRIEVERS
from .metrics import binary_classification_metrics, roc_auc, threshold_sweep
from .report import ensure_parent, render_terminal, write_html, write_json, write_junit

__all__ = ["main", "build_parser"]


def _parse_ks(value: str) -> List[int]:
    try:
        ks = sorted({int(part) for part in value.replace(" ", "").split(",") if part})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--k expects comma-separated integers, got {value!r}")
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--k values must be positive integers")
    return ks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragval",
        description="Score RAG retrieval quality and flag likely hallucinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ragval run --dataset bundled --html out/report.html\n"
            "  ragval run --dataset ./my_data --retriever hybrid --k 1,3,5,10\n"
            "  ragval gate --results out/results.json --baseline baseline.json\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"ragval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run ---------------------------------------------------------------- #
    run = sub.add_parser("run", help="evaluate a dataset")
    run.add_argument("--dataset", default="bundled", help="dataset directory, or 'bundled'")
    run.add_argument("--corpus", help="override path to corpus.jsonl")
    run.add_argument("--queries", help="override path to queries.jsonl")
    run.add_argument("--answers", help="override path to answers.jsonl")
    run.add_argument("--k", type=_parse_ks, default=list(DEFAULT_KS), help="cut-offs, e.g. 1,3,5,10")
    run.add_argument(
        "--retriever",
        default="bm25",
        choices=sorted(RETRIEVERS) + ["embedding"],
        help="retrieval backend (default: bm25)",
    )
    run.add_argument("--context-k", type=int, default=None, help="documents passed to the grounding check")
    run.add_argument("--risk-threshold", type=float, default=None, help="flag answers at or above this risk")
    run.add_argument("--support-threshold", type=float, default=None, help="per-claim support cut-off")
    run.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (0 disables)")
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--loose-k", action="store_true", help="divide precision@k by results returned, not k")
    run.add_argument("--json", dest="json_out", help="write machine-readable results here")
    run.add_argument("--html", dest="html_out", help="write the HTML dashboard here")
    run.add_argument("--junit", dest="junit_out", help="write JUnit XML here")
    run.add_argument("--config", help="gate config; when given, the gate runs after evaluation")
    run.add_argument("--baseline", help="baseline.json for the regression check")
    run.add_argument("--fail-under-gate", action="store_true", help="exit 1 when the gate fails")
    run.add_argument("--quiet", action="store_true", help="suppress the terminal summary")
    run.add_argument("--no-color", action="store_true")

    # -- gate --------------------------------------------------------------- #
    gate = sub.add_parser("gate", help="apply thresholds / regression checks to a results file")
    gate.add_argument("--results", required=True)
    gate.add_argument("--config")
    gate.add_argument("--baseline")
    gate.add_argument("--json", dest="json_out", help="write the gate outcome here")

    # -- baseline ----------------------------------------------------------- #
    base = sub.add_parser("baseline", help="freeze results as the regression baseline")
    base.add_argument("--results", required=True)
    base.add_argument("--out", default="baseline.json")

    # -- sweep -------------------------------------------------------------- #
    sweep = sub.add_parser("sweep", help="calibrate the detector risk threshold")
    sweep.add_argument("--results", help="an existing results.json")
    sweep.add_argument("--dataset", default="bundled", help="or evaluate a dataset directly")
    sweep.add_argument("--steps", type=int, default=101)
    sweep.add_argument("--top", type=int, default=10, help="rows to print around the optimum")

    # -- validate ----------------------------------------------------------- #
    val = sub.add_parser("validate", help="check a dataset for label bugs")
    val.add_argument("--dataset", default="bundled")

    return parser


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _load(args: argparse.Namespace) -> Dataset:
    return load_dataset(
        args.dataset,
        corpus=getattr(args, "corpus", None),
        queries=getattr(args, "queries", None),
        answers=getattr(args, "answers", None),
    )


def cmd_run(args: argparse.Namespace) -> int:
    dataset = _load(args)

    grounding = GroundingConfig()
    if args.risk_threshold is not None:
        grounding.risk_threshold = args.risk_threshold
    if args.support_threshold is not None:
        grounding.support_threshold = args.support_threshold
    grounding.validate()

    options = EvalOptions(
        ks=args.k,
        retriever=args.retriever,
        context_k=args.context_k,
        strict_k=not args.loose_k,
        bootstrap=args.bootstrap,
        seed=args.seed,
        grounding=grounding,
    )
    report = evaluate(dataset, options)
    results = report.to_dict()

    gate_result: Optional[Dict[str, Any]] = None
    if args.config or args.baseline:
        config = load_config(args.config)
        baseline = None
        if args.baseline:
            if not os.path.exists(args.baseline):
                print(f"warning: baseline {args.baseline} not found; skipping regression check", file=sys.stderr)
            else:
                with open(args.baseline, "r", encoding="utf-8") as handle:
                    baseline = json.load(handle)
        gate_result = run_gate(results, config, baseline).to_dict()

    if not args.quiet:
        print(render_terminal(report, color=not args.no_color and sys.stdout.isatty()))
        if gate_result:
            print(_render_gate(gate_result))

    if args.json_out:
        payload = dict(results)
        if gate_result:
            payload["gate"] = gate_result
        ensure_parent(args.json_out)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.json_out}")
    if args.html_out:
        write_html(report, args.html_out, gate=gate_result)
        print(f"wrote {args.html_out}")
    if args.junit_out:
        write_junit(report, args.junit_out, gate=gate_result)
        print(f"wrote {args.junit_out}")

    if gate_result and args.fail_under_gate and not gate_result["passed"]:
        return 1
    return 0


def _render_gate(gate: Dict[str, Any]) -> str:
    lines = ["", "Quality gate: " + ("PASS" if gate["passed"] else "FAIL")]
    for check in gate["checks"]:
        mark = "ok  " if check["passed"] else "FAIL"
        lines.append(
            f"  [{mark}] {check['kind']:<10} {check['name']:<24} "
            f"{check['value']:.4f} (ref {check['reference']:.4f}) {check.get('detail', '')}"
        )
    return "\n".join(lines)


def cmd_gate(args: argparse.Namespace) -> int:
    with open(args.results, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    config = load_config(args.config)
    baseline = None
    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
    outcome = run_gate(results, config, baseline).to_dict()
    print(_render_gate(outcome))
    if args.json_out:
        ensure_parent(args.json_out)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(outcome, handle, indent=2)
    return 0 if outcome["passed"] else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    with open(args.results, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    baseline = make_baseline(results)
    ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out} with {len(baseline['metrics'])} frozen metrics")
    for name, value in baseline["metrics"].items():
        print(f"  {name:<26} {value:.4f}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    if args.results:
        with open(args.results, "r", encoding="utf-8") as handle:
            results = json.load(handle)
        pairs = [
            (bool(q["grounding"]["gold_hallucinated"]), float(q["grounding"]["risk"]))
            for q in results.get("queries", [])
            if q.get("grounding") and q["grounding"].get("gold_hallucinated") is not None
        ]
    else:
        dataset = load_dataset(args.dataset)
        report = evaluate(dataset, EvalOptions(bootstrap=0))
        pairs = [
            (bool(q.grounding.gold_hallucinated), q.grounding.risk)
            for q in report.queries
            if q.grounding and q.grounding.gold_hallucinated is not None
        ]

    if not pairs:
        print("no labelled answers found; nothing to calibrate", file=sys.stderr)
        return 2

    y_true = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]
    rows = threshold_sweep(y_true, scores, steps=args.steps)
    best = max(rows, key=lambda r: (r["f1"], r["recall"], -r["threshold"]))

    print(f"labelled answers: {len(pairs)}  positives: {sum(y_true)}  ROC-AUC: {roc_auc(y_true, scores):.4f}")
    print(f"best threshold: {best['threshold']:.2f}  precision {best['precision']:.3f} "
          f"recall {best['recall']:.3f}  F1 {best['f1']:.3f}")
    print()
    print(f"{'thr':>5}  {'prec':>6}  {'rec':>6}  {'F1':>6}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
    centre = rows.index(best)
    half = max(args.top // 2, 1)
    window = rows[max(centre - half, 0) : centre + half + 1]
    for row in window:
        marker = " <-- best" if row is best else ""
        print(
            f"{row['threshold']:>5.2f}  {row['precision']:>6.3f}  {row['recall']:>6.3f}  "
            f"{row['f1']:>6.3f}  {int(row['tp']):>3} {int(row['fp']):>3} "
            f"{int(row['fn']):>3} {int(row['tn']):>3}{marker}"
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset, strict=True)
    warnings = dataset.validate(strict=True)
    stats = dataset.stats()
    print(f"dataset '{stats['name']}' is structurally valid")
    for key in (
        "n_documents", "n_queries", "n_answers", "n_labeled_answers",
        "n_hallucinated", "avg_doc_words", "avg_relevant_per_query",
    ):
        print(f"  {key:<24} {stats[key]}")
    print(f"  {'tags':<24} {', '.join(stats['tags']) or '-'}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


_COMMANDS = {
    "run": cmd_run,
    "gate": cmd_gate,
    "baseline": cmd_baseline,
    "sweep": cmd_sweep,
    "validate": cmd_validate,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except (ValidationError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # pragma: no cover - piping into head/less
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
