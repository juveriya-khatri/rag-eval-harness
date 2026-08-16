# RAG Evaluation Harness

**A lightweight verification harness for retrieval-augmented generation: it scores retrieval quality (precision/recall/nDCG@k), flags likely hallucinations against a labelled test set, and fails your CI build when either regresses.**

Zero required dependencies · pure Python 3.9+ · 219 tests · 95% coverage · deterministic and reproducible

```bash
pip install -e .
ragval run --dataset bundled --html out/report.html
```

---

## Table of contents

- [The problem](#the-problem)
- [What this does](#what-this-does)
- [Results on the bundled test set](#results-on-the-bundled-test-set)
- [How the hallucination detector works](#how-the-hallucination-detector-works)
- [Ablation: what each check is worth](#ablation-what-each-check-is-worth)
- [Where it fails](#where-it-fails-and-why-that-is-in-the-readme)
- [Architecture](#architecture)
- [Installation and usage](#installation-and-usage)
- [Dataset format](#dataset-format)
- [Using it on your own pipeline](#using-it-on-your-own-pipeline)
- [CI integration](#ci-integration)
- [Design decisions](#design-decisions-and-why)
- [Engineering notes](#engineering-notes)
- [Testing](#testing)
- [Roadmap](#roadmap)

---

## The problem

Shipping a RAG pipeline is easy. Knowing whether it is *right* is not.

The usual failure mode is a pipeline that runs perfectly — no exceptions, low latency, plausible prose — while quietly retrieving the wrong passages and inventing figures that were never in the source. Teams notice months later, from a user complaint. "Verification" is repeatedly named the core problem for AI teams in 2026 for exactly this reason: the systems fail silently.

Two distinct things can go wrong, and they need completely different fixes:

1. **Retrieval failure** — the evidence needed to answer never reaches the generator. No prompt engineering saves you.
2. **Grounding failure** — the evidence *was* there, and the generator asserted something it does not support.

Most tooling measures neither, or conflates the two. This harness measures both separately, attributes each flag to the right root cause, and turns the result into a build-breaking signal.

## What this does

- **Scores retrieval** with the standard IR metric family — precision@k, recall@k, F1@k, hit rate@k, MAP@k, nDCG@k (graded relevance), MRR — macro-averaged per query, each with a **bootstrap 95% confidence interval** so you can tell a real regression from noise on a 36-query set.
- **Flags likely hallucinations** by decomposing each answer into atomic claims and checking every claim against the retrieved context, with three precision checks that lexical overlap alone misses: **fabricated figures**, **fabricated entities**, and **contradictions**.
- **Explains every flag.** Each verdict carries the offending claim, a human-readable reason, and the nearest evidence sentence with its document id. Nothing is a black box score.
- **Scores the detector itself** against gold labels — precision, recall, F1, ROC-AUC, confusion matrix, and a full threshold sweep for calibration.
- **Attributes root cause.** A flag on an answer whose gold evidence never reached the context is a *retrieval* failure wearing a grounding costume. The report separates the two.
- **Gates CI.** Absolute thresholds plus regression-vs-baseline checks, JUnit XML output, and a non-zero exit code.
- **Ships a self-contained HTML report** — inline CSS, hand-rendered SVG charts, no CDN, no build step. It opens from a CI artifact bundle on an air-gapped machine.

Everything runs offline with no API keys and no model downloads. Optional adapters add dense retrieval (`sentence-transformers`) and LLM-as-judge grounding if you want them.

---

## Results on the bundled test set

40 documents across four domains (aviation safety, clinical trials, cloud SLAs and postmortems, energy policy), 36 labelled queries with graded relevance, 36 generated answers of which 15 contain a deliberately planted defect. Reproduce with `make demo`.

### Retrieval — BM25 baseline

| k | precision | recall | F1 | hit rate | MAP | nDCG |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.889 | 0.819 | 0.843 | 0.889 | 0.889 | 0.889 |
| 3 | 0.343 | 0.903 | 0.489 | 0.917 | 0.884 | 0.898 |
| 5 | 0.217 | **0.944** | 0.348 | 0.944 | 0.898 | **0.913** |
| 10 | 0.111 | 0.972 | 0.198 | 0.972 | 0.902 | 0.922 |

MRR **0.909** · mean query latency **0.12 ms** over a 40-document index.

95% bootstrap confidence intervals (1000 resamples, seeded):

| metric | estimate | 95% CI |
|---|---:|---|
| precision@5 | 0.217 | [0.189, 0.244] |
| recall@5 | 0.944 | [0.861, 1.000] |
| nDCG@5 | 0.913 | [0.826, 0.984] |
| MRR | 0.909 | [0.811, 0.981] |

> Precision@5 is low *by construction*: most queries have exactly one relevant document, so the ceiling is 0.2. This is why the harness reports the whole family rather than one headline number — precision@k is close to meaningless on a sparse-judgement set, and recall@k / nDCG@k are what actually matter.

### Retrieval — by domain

The `paraphrase` tag marks six queries deliberately written with almost no vocabulary in common with their source document. It isolates the known weakness of lexical retrieval:

| tag | queries | recall@5 | nDCG@5 |
|---|---:|---:|---:|
| aviation | 10 | 1.000 | 1.000 |
| clinical | 10 | 0.900 | 0.894 |
| cloud | 9 | 0.889 | 0.826 |
| energy | 7 | 1.000 | 0.929 |
| **paraphrase** | **6** | **0.667** | **0.488** |

Vocabulary-mismatched queries lose a third of their recall and half their ranking quality. That single number is the argument for adding dense retrieval, and the harness produces it without anyone having to guess.

### Retrieval — backend comparison

| retriever | P@1 | R@5 | R@10 | nDCG@5 | MRR | latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.889 | 0.944 | 0.972 | 0.913 | 0.909 | 0.14 ms |
| TF-IDF cosine | 0.889 | 0.944 | 0.972 | **0.918** | **0.916** | 0.13 ms |
| Hybrid (RRF) | 0.889 | 0.944 | 0.972 | 0.917 | 0.913 | 0.33 ms |

All three sit inside each other's confidence intervals — an honest read is *"no measurable difference on this corpus"*, not *"TF-IDF wins"*. Reporting the CI is what makes that conclusion available.

### Hallucination detection

36 labelled answers, 70 extracted claims, operating threshold 0.34:

| metric | value |
|---|---:|
| precision | **0.933** |
| recall | **0.933** |
| F1 | **0.933** |
| ROC-AUC | **0.997** |
| balanced accuracy | 0.942 |
| mean faithfulness | 0.657 |

Confusion matrix:

|  | flagged | not flagged |
|---|---:|---:|
| **hallucinated** | 14 (TP) | 1 (FN) |
| **faithful** | 1 (FP) | 20 (TN) |

**Root-cause attribution.** Two answers received no gold evidence at all — retrieval returned nothing relevant. Both were flagged, correctly in the sense that the answer genuinely was not supported by what the generator saw. Restricted to the 34 answers whose context *did* contain the gold evidence, the detector scores:

| conditioned on correct retrieval (n=34) | value |
|---|---:|
| precision | **1.000** |
| recall | 0.929 |
| F1 | **0.963** |

That decomposition is the most useful output the harness produces: *the grounding checker has one residual miss; the rest of the gap is a retrieval problem.*

### Risk score separation

Answer-level risk scores, hallucinated vs faithful:

```
hallucinated  ██████████████ 0.28 – 0.90   (median 0.90)
faithful      ███            0.00 – 0.26   (median 0.00)
```

ROC-AUC of 0.997 means the ranking is almost perfectly separable, so the operating threshold can be moved a long way in either direction without collapsing. `ragval sweep` prints the full precision/recall trade-off; on this set the F1-optimal threshold is 0.27 (F1 0.968) against the shipped default of 0.34, which is deliberately set slightly conservative to favour precision.

---

## How the hallucination detector works

The detector is **lexical, deterministic and explainable by design** — not a model. It runs in microseconds, costs nothing, produces the same answer on every machine, and every flag can be audited by a human in seconds. That is the right default for a verification tool; a model you cannot interrogate is a strange thing to verify another model with.

### 1. Claim decomposition

The answer is split into atomic, independently checkable claims. Sentence segmentation is abbreviation-, decimal- and initial-aware (`Dr.`, `3.5%`, `J. R. Smith`, `v1.2.3`, ellipses), and long coordinated sentences are split at `and`/`while`/`whereas` when both halves stand alone as facts. Scoring whole answers instead of claims is how detectors end up with one fabricated figure diluted by three true sentences.

### 2. IDF-weighted support scoring

Each claim is scored against every context sentence by **IDF-weighted term coverage**: rare, informative terms must be present, common ones are nearly free. IDF comes from the **whole corpus**, not the handful of retrieved passages — otherwise every term in a two-document context looks equally rare. Unknown terms get maximum IDF, so a fabricated rare word can never be cheap.

Coverage is taken against the best single evidence sentence *and*, at a 5% discount, against the best **pair** of sentences, so genuinely multi-hop claims are not punished for spanning two sentences.

### 3. Three precision checks

**Numeric check.** Every figure, percentage, year and month in the claim must appear in the context. Numbers are normalised so `1,200,000,000`, `1.2 billion` and `1200000000` unify, percentages live in a separate namespace (`50%` never silently matches the bare number `50`), and a small relative tolerance absorbs `4.7%` vs `4.70%` without letting `47%` through. Fabricated figures are the most common and most damaging RAG hallucination, so this check carries the heaviest penalty — and, symmetrically, a claim whose figures *all* check out receives a support bonus, because matching figures are positive evidence of grounding that offsets vocabulary drift in the surrounding prose.

**Entity check.** Proper-noun spans and acronyms must appear in the context. Two rules keep the false-positive rate usable without a POS tagger: a single capitalised word at the start of a sentence is ignored (that is orthography, not a name), and stopwords are never entities. Multi-word spans (`Flight NR-482`, `Halden Regional`) and acronyms (`AD-2024-07`, `ACR20`) survive both rules — precisely the strings a generator is most likely to invent.

**Contradiction check.** Three signals, each carefully scoped:

- *Figure conflict*, resolved at **document** scope with **unit awareness**. A conflict requires the claim's `(value, unit)` pair to be absent from the entire source document while that document asserts a *different* value for the same unit. This is the single most important design detail in the project — see below.
- *Directional conflict*, using opposed word **groups** (rose/fell, increase/decrease, approved/rejected) rather than antonym pairs, and only when each side is unambiguous. A sentence mentioning both an increase and a decrease is skipped rather than guessed at.
- *Polarity flip*, when a claim and a near-duplicate evidence sentence disagree on negation cues.

### 4. Aggregation

Claim support is combined into an answer-level risk score weighting the **mean** (0.55) and the **worst** claim (0.45): one badly fabricated sentence should be enough to flag an otherwise faithful answer. Any contradiction floors the risk at 0.9.

### The detail that mattered most

The first implementation compared *any* number in the claim against *any* number of the same kind in the evidence sentence. It flagged **nine faithful answers out of eighteen** — F1 0.629, worse than useless.

The reason is that a faithful claim routinely draws figures from several sentences:

> Claim: *"The crew initiated an emergency descent from 36,000 feet to 10,000 feet and completed it in 6 minutes."*
> Evidence sentence: *"The crew initiated an emergency descent to 10,000 feet, completing it in 6 minutes."*

`36,000` is missing from that sentence, so a naive check screams contradiction. It is stated one sentence earlier in the same report.

Two changes fixed it, and both are load-bearing:

1. **Unit awareness.** Figures are extracted as `(value, kind, unit)`, where the unit is the first content word after the number (stopwords skipped, stemmed): `13 hours → (13, number, "hour")`, `48 per megawatt hour → (48, number, "megawatt")`. Years and months carry no unit, because "in 2024," is followed by an arbitrary word. Now *"the claim says 14 **hours** where the source says 13 **hours**"* is distinguishable from *"the claim mentions 36,000 **feet**, which appears elsewhere in the same passage"*.
2. **Document scope for conflicts, context scope for penalties.** A figure present anywhere in the retrieved context is at least *sourced* (a soft penalty at worst). A conflict requires the claim to disagree with the specific passage it is paraphrasing.

Detector F1 went from **0.629 → 0.933**, and false positives from 12 to 1. The lesson generalises: in verification tooling, false positives are the thing that gets your tool switched off, and they come from checks whose scope is wrong rather than from checks that are too weak.

---

## Ablation: what each check is worth

Each row disables one component and re-scores the detector on the same labelled answers (`python examples/ablation.py`):

| variant | precision | recall | F1 | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| **full detector** | **0.933** | **0.933** | **0.933** | **0.997** | 1 | 1 |
| no contradiction rule | 0.917 | 0.733 | 0.815 | 0.916 | 1 | 4 |
| no numeric check | 0.875 | 0.933 | 0.903 | 0.994 | 2 | 1 |
| no entity check | 0.933 | 0.933 | 0.933 | 0.997 | 1 | 1 |
| lexical overlap only | 0.600 | 0.200 | 0.300 | 0.892 | 2 | 12 |

Reading it honestly:

- **Lexical overlap alone is not a hallucination detector.** F1 0.300, catching 3 of 15 planted defects. A fabricated figure changes almost none of a sentence's vocabulary, which is exactly why overlap-based faithfulness scores miss the failures that matter.
- **The contradiction rule carries the detector** — removing it triples the false negatives.
- **The entity check contributes nothing measurable here**, because the one fabricated entity in the set is also caught by the numeric check. It is retained because it is the only check that catches a *purely nominal* fabrication ("according to the Marrow Institute") with no numbers attached, but this test set does not isolate that case. Stated plainly rather than quietly dropped from the table.

---

## Where it fails, and why that is in the README

A verification tool that hides its own error modes is not a verification tool.

**1. Semantic inferences with no lexical footprint (the one false negative).**
Answer: *"Overall survival was also significantly improved in the experimental arm."* Source: *"Overall survival data were immature at the time of the analysis."* Refuting this requires understanding that immature data cannot support a significance claim. Every content word in the claim appears in the context, no figure is fabricated, no direction word conflicts. Risk 0.278 against a 0.34 threshold — a near miss, but a miss. **This is the case the optional LLM-judge adapter exists for**, and it escalates exactly this shape of claim: lexically decisive-looking, semantically wrong.

**2. Retrieval starvation reported as a grounding flag (the one false positive).**
For `q-36` the gold document never entered the top 5, so a faithful answer was flagged as ungrounded. The harness is *right* that the answer is unsupported by what was retrieved, and the report says so explicitly (`<- retrieval miss`) rather than blaming the generator. This is why root-cause attribution is a first-class output.

**3. Cross-document figure confusion.** If the same figure appears in an unrelated retrieved passage, the context-wide numeric penalty does not fire. Only the document-scoped conflict rule catches it. Chunk-level provenance would close this.

**4. Paraphrase-heavy retrieval.** recall@5 drops from ~0.95 to 0.667 on vocabulary-mismatched queries. Expected for a lexical retriever; the fix is the embeddings adapter, and the harness quantifies exactly how much it would need to buy you.

**5. The test set is small and synthetic.** 36 queries, one author, deliberately planted defects. The confidence intervals are wide for a reason — recall@5 is [0.861, 1.000]. These numbers demonstrate that the *harness* works; they are not a claim about performance on your corpus. Point it at your data and find out.

---

## Architecture

```
ragval/
├── types.py       Dataclasses; every result round-trips losslessly through JSON
├── text.py        Tokenisation, stemming, sentence splitting, numeric/entity extraction
├── index.py       BM25 · TF-IDF · Hybrid (RRF), behind one 3-method interface
├── metrics.py     IR + classification metrics; pure functions, explicit edge cases
├── grounding.py   Claim decomposition, support scoring, the three precision checks
├── dataset.py     JSONL loading and the validation that stops bad labels reaching metrics
├── evaluate.py    Orchestration, macro-averaging, bootstrap CIs, root-cause attribution
├── report.py      Terminal · JSON · JUnit XML · self-contained HTML with SVG charts
├── gate.py        Thresholds + regression-vs-baseline → CI exit code
├── cli.py         run · gate · baseline · sweep · validate
├── data/          The bundled labelled test set (3 JSONL files)
└── plugins/
    ├── embeddings.py   Optional dense retrieval (sentence-transformers)
    └── llm_judge.py    Optional LLM adjudication for uncertain claims only
```

Data flow:

```
corpus.jsonl ─┐
queries.jsonl ├─> Dataset ──> validate ──> Retriever.search(q, k)
answers.jsonl ┘                                    │
                                                   ├──> IR metrics ──> macro-average ──> bootstrap CI
                                                   │
                                          top-k context
                                                   │
                              GroundingChecker.check(answer, context)
                                                   │
                          claims → support → verdicts → risk ──> detector metrics
                                                   │
                                    ┌──────────────┴──────────────┐
                              EvalReport                    root-cause split
                                    │
              ┌──────────┬──────────┼──────────┬────────────┐
          terminal     JSON      HTML       JUnit      quality gate → exit code
```

**~4,100 lines of library code, ~1,200 lines of tests, zero runtime dependencies.**

---

## Installation and usage

```bash
git clone https://github.com/juveriya-khatri/rag-eval-harness
cd rag-eval-harness
pip install -e ".[dev]"      # dev extras are only pytest; the library itself needs nothing
```

Requires Python 3.9+. No compiler, no model download, no network access.

### Commands

```bash
# Evaluate the bundled set and write every artifact
ragval run --dataset bundled --k 1,3,5,10 \
    --json out/results.json --html out/report.html --junit out/junit.xml

# Your own data, hybrid retrieval, more context for the grounding check
ragval run --dataset ./my_data --retriever hybrid --context-k 8

# Check labels before trusting any number
ragval validate --dataset ./my_data

# Calibrate the detector threshold on labelled data
ragval sweep --dataset bundled

# Freeze a baseline, then gate future runs against it
ragval baseline --results out/results.json --out baseline.json
ragval gate --results out/results.json --config ragval.config.json --baseline baseline.json
```

`make demo`, `make test`, `make cov`, `make baseline`, `make gate` wrap the common paths.

### As a library

```python
from ragval import load_dataset, evaluate, EvalOptions

report = evaluate(load_dataset("bundled"), EvalOptions(ks=(1, 5, 10), retriever="hybrid"))

print(report.retrieval["recall@5"])        # 0.944
print(report.retrieval_ci["recall@5"])     # (0.861, 1.0)
print(report.detector["f1"])               # 0.933

for query in report.queries:
    for claim in (query.grounding.flagged_claims if query.grounding else []):
        print(query.query_id, claim.verdict, claim.reasons, claim.evidence_doc_id)
```

### Sample output

```
Hallucination detection
  answers=36 labeled=36 claims=70 flagged claims=21
  mean faithfulness 0.657   mean risk 0.393   flag rate 41.7%
  precision 0.933  recall 0.933  F1 0.933  ROC-AUC 0.997
  confusion: TP=14 FP=1 FN=1 TN=20  (threshold 0.34; best F1 0.968 at 0.27)
  2 answer(s) had no gold evidence in context; 2 of the flags are retrieval failures, not fabrications
  given correct retrieval (n=34): precision 1.000 recall 0.929 F1 0.963

Detector errors (2)
  q-16       false_negative  risk=0.278 faithfulness=0.500
  q-36       false_positive  risk=0.896 faithfulness=0.000  <- retrieval miss
```

And an individual flag, as it appears in the HTML report:

> **Claim** — "The 2024 revision sets a maximum duty period of 14 hours for an unaugmented crew."
> **contradicted** · support 0.15 · figure conflict for 'hour': claim states 14, source states 13
> **Nearest evidence [av-06]** — "The 2024 revision of the flight-time limitation scheme sets a maximum duty period of 13 hours for an unaugmented crew starting between 06:00 and 13:59."

---

## Dataset format

Three JSONL files in one directory. JSONL because it diffs cleanly in git, streams without loading everything into memory, and survives a malformed line with a precise error message (`queries.jsonl:14: invalid JSON`) instead of a stack trace.

**`corpus.jsonl`** — one retrievable chunk per line:

```json
{"id": "av-06", "title": "Flight crew duty and rest limits", "text": "The 2024 revision sets a maximum duty period of 13 hours...", "meta": {"domain": "aviation"}}
```

**`queries.jsonl`** — graded relevance (`2` highly relevant, `1` partially, absent/`0` not). A bare list of ids is accepted for binary judgements:

```json
{"id": "q-04", "question": "What is the maximum duty period...?", "relevant": {"av-06": 2}, "gold_answer": "13 hours...", "tags": ["aviation"]}
```

**`answers.jsonl`** — the answers under test, with gold labels:

```json
{"query_id": "q-04", "system": "baseline-rag", "answer": "The 2024 revision sets a maximum duty period of 14 hours...", "hallucinated": true, "note": "figure conflict: the regulation specifies 13 hours"}
```

`answers.jsonl` is optional — omit it to score retrieval only.

`ragval validate` catches the label bugs that silently corrupt results: dangling document references, duplicate ids, answers pointing at queries that do not exist, negative gains. Queries with no judgements are a *warning*, not an error — their recall is genuinely undefined, and the harness excludes them from the macro-average rather than counting them as 0.0 or 1.0.

### The bundled test set

| | |
|---|---|
| documents | 40 (10 each: aviation safety, clinical trials, cloud SLAs/postmortems, energy policy) |
| mean document length | 63 words |
| queries | 36, graded relevance, 6 tagged `paraphrase` |
| answers | 36 (21 faithful, 15 containing a planted defect) |

Planted defect types, chosen to cover the failure taxonomy rather than to be easy: fabricated figure, figure conflict in the same slot, directional reversal (`fell` where the source says `increase`), fabricated entity with false attribution, subtle single-digit change (`12 hours` for `10 hours`), and an unsupported semantic inference. Each answer carries a `note` explaining precisely what was planted, so the labels are auditable.

The corpora are dense with numbers on purpose — figures are where RAG hallucinations do real damage, and a test set of vague prose would not exercise the checks that matter.

---

## Using it on your own pipeline

The only contract is `search(query, k) -> List[Hit]`:

```python
from ragval import evaluate, load_dataset, EvalOptions
from ragval.index import Retriever
from ragval.types import Hit

class MyRetriever(Retriever):
    name = "production-v3"

    def search(self, query: str, k: int = 5):
        rows = my_vector_db.query(query, top_k=k)      # or your search API, or your RAG service
        return [Hit(doc_id=r.id, score=r.score, rank=i)
                for i, r in enumerate(rows, start=1)]

report = evaluate(load_dataset("./my_data"), EvalOptions(bootstrap=1000),
                  retriever=MyRetriever(dataset.documents))
```

Metrics, confidence intervals, grounding checks, the HTML report and the CI gate all come for free. `examples/evaluate_your_own_pipeline.py` runs a full A/B against the BM25 baseline and prints the deltas.

### Optional adapters

```bash
pip install "rag-eval-harness[embeddings]"   # dense retrieval
pip install "rag-eval-harness[llm]"          # LLM-as-judge grounding
```

**Dense retrieval** — `ragval run --retriever embedding`, or compose it with BM25 through `HybridRetriever` for reciprocal rank fusion.

**LLM judge** — escalates *only the uncertain claims*: those whose lexical support falls within a configurable band of the decision boundary, typically 10–20% of claims rather than all of them. Bring any client with a `complete(prompt) -> str` method, which keeps the harness free of vendor SDKs and makes the judge trivially mockable in tests. A judge outage degrades to the lexical verdict instead of failing the run.

```python
from ragval.plugins.llm_judge import LLMJudgeChecker

checker = LLMJudgeChecker(corpus, my_client, uncertainty_band=0.18, max_calls=200)
```

---

## CI integration

`.github/workflows/ci.yml` runs the test matrix (Python 3.9–3.12) and a separate quality-gate job that treats retrieval quality and hallucination detection as build-breaking properties:

```bash
ragval run --dataset bundled --config ragval.config.json \
           --baseline baseline.json --fail-under-gate
```

Two independent checks:

- **Absolute thresholds** (`recall@5 >= 0.85`, `detector F1 >= 0.85`, …) — catches a pipeline that was never good enough.
- **Regression vs a frozen baseline** with a 0.02 tolerance — catches the far more common failure: a chunking tweak or prompt change that quietly makes retrieval worse.

The tolerance is not arbitrary. It is roughly the half-width of the bootstrap confidence interval on this test set, so ordinary sampling noise cannot fail the build while a genuine regression can. Configure it in `ragval.config.json` (JSON, or TOML on 3.11+).

The gate emits JUnit XML, so every query, every labelled answer and every gate check renders as a test case in your CI's native test view, and the HTML report is uploaded as a build artifact.

---

## Design decisions, and why

**Zero required dependencies.** Anyone can clone and `make demo` in seconds, on any Python 3.9+, offline, with no version conflicts against an existing ML stack. A verification tool that is hard to install does not get installed.

**Undefined values are `NaN`, never `0.0`.** A query with no relevance judgements has no recall. Coercing that to zero silently drags down every average and hides label bugs; the harness returns `NaN` and excludes it from the macro-average. This is the single most common bug in hand-rolled evaluation scripts.

**`precision@k` divides by `k`.** A system returning 2 documents when asked for 5 is penalised, which is what you want when comparing pipelines. `--loose-k` switches to dividing by results returned.

**Macro-average over queries, never micro.** Otherwise a query with eight relevant documents dominates seven queries with one.

**Confidence intervals on every headline number.** 36 queries is a small test set. A point estimate alone invites you to celebrate a 0.01 improvement that is pure noise — as the retriever comparison above demonstrates.

**Deterministic everywhere.** No randomness except the seeded bootstrap. Ties in ranking break on document id. Same bytes in, same numbers out, on any machine. Non-reproducible evaluation is not evaluation.

**Explainable over accurate, at the same F1.** Every flag carries its reason and its evidence. When the detector is wrong you can see *why* in one line, which is what makes it fixable — and what made the 0.629 → 0.933 improvement possible.

**Root cause over blame.** Separating retrieval starvation from fabrication is what turns "your pipeline is wrong" into "fix your retriever" or "fix your prompt".

---

## Engineering notes

A few implementation details worth calling out:

**Retrieval is O(Σ posting-list lengths), not O(corpus).** BM25 scores only documents sharing a term with the query, via an inverted index built once at construction. The IDF uses the `log(1 + …)` form, which is always positive and avoids the negative-score pathology the classic formulation hits on terms appearing in more than half the corpus.

**The stemmer is deliberately conservative.** It strips only suffixes with low collapse risk and never shortens a token below three characters, so `gas`, `analysis`, `bus` and `pass` survive intact while `filters`/`filtering`/`filtered` merge. Aggressive stemming creates false term matches, which in a grounding checker means false *support* — the worst possible direction to be wrong in.

**Opposite word groups, not antonym pairs.** A pairwise table breaks the moment your stemmer maps `increase` and `increased` to different stems. Listing surface forms in a group and stemming them all sidesteps the problem entirely.

**Sentence-level evidence is cached per document** and reused across every claim and every answer, so the grounding check is linear in claims rather than quadratic.

**The HTML report has no external references at all** — verified by a test asserting the string `http` never appears in the output. Inline CSS, hand-rendered SVG, and ~40 lines of vanilla JS for table sorting and filtering.

**ROC-AUC via the rank identity with correct tie handling**, so a detector that assigns identical scores to everything scores 0.5 rather than accidentally looking good.

**All user content is HTML-escaped** in the report, with a test that feeds `<script>` and `<img onerror=…>` through the pipeline and asserts they come out inert.

---

## Testing

```bash
make test     # 219 tests, ~3 seconds
make cov      # 95% line coverage
```

| module | coverage |
|---|---:|
| `metrics.py` | 98% |
| `evaluate.py` | 98% |
| `grounding.py` | 97% |
| `index.py` | 97% |
| `types.py` | 97% |
| `report.py` | 94% |
| `cli.py` | 93% |
| `gate.py` | 92% |
| `dataset.py` | 91% |
| `text.py` | 91% |
| **total** | **95%** |

The suite is written around the cases that actually break evaluation code rather than around line coverage:

- **Metric edge cases** — empty rankings, duplicate ids (collapsed to the best rank), undefined recall, graded vs binary relevance, `strict_k` semantics, AP normalisation by `min(|R|, k)`, ties in ROC-AUC, single-class AUC, degenerate bootstraps.
- **Detector false-positive traps** — figures split across two sentences of one document, matching negations, same-direction comparisons, sentence-initial capitalisation. These are regression tests for the bugs described above.
- **Every hallucination type** — fabricated figure, same-slot conflict, directional reversal, fabricated entity, wholly invented claim.
- **Degenerate inputs** — empty corpus, empty query, empty answer, no context, out-of-vocabulary queries, `k=0`, duplicate document ids, documents with empty text.
- **Determinism** — two identical runs produce byte-identical output once wall-clock fields are removed.
- **Adversarial content** — HTML injection through query text.
- **CLI contract** — every subcommand, exit codes (`0` success, `1` gate failure, `2` bad input), artifact creation, and well-formed JUnit XML parsed back with `ElementTree`.
- **Plugins without their dependencies** — both adapters are tested with injected fakes, including judge outages and call budgets, so the optional paths are covered in a plain install.

---

## Roadmap

- Chunk-level provenance, to close the cross-document figure-confusion gap
- NLI-based entailment as a third grounding channel, calibrated against the lexical one
- Retrieval diagnostics: per-query term-level attribution of *why* a document was or was not retrieved
- Multi-system comparison in one report (A/B two pipelines side by side with paired significance tests)
- Adapters for public QA benchmarks (HotpotQA, NQ) alongside the bundled set

---

## License

MIT — see [LICENSE](LICENSE).

Built by **Juveriya Khatri**. Issues and pull requests welcome.
