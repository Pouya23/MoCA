# ICLR experiment execution guide

This codebase preserves the paper's MoCA training objective while making the
reviewer-requested evaluation protocol executable and auditable.

## What changed

- CoQA/QuAC use story/dialogue-grouped 80/10/10 splits.
- Main routing uses the last prompt-token embedding, matching the response.
- Main confidence is MoCA-1P: a validation-only logistic calibrator over
  non-special first-token confidence, nearest-centroid distance, and top-two
  routing margin.
- Test generation emits one answer, `calibrated_confidence`, and an optional
  `abstained` flag. It does not sample semantic alternatives or run NLI online.
- Ten-sample semantic entropy is retained as a separate offline diagnostic.
- Evaluations include Brier, AUROC, fixed/adaptive ECE, log loss, reliability
  bins, AURC/risk–coverage, optional selective accuracy, OOD AUROC, and FPR95.
- Reviewer controls include Vanilla FT, K-FT, matched-rank single LoRA, random
  partition, first-five-token KL, non-special uniform target, KL direction,
  pooling, embedding normalization, semantic-estimator, and NLI variants.
- Forced-wrong-route evaluation and generation/timing manifests are included.
- Seed aggregation writes JSON, CSV, and LaTeX summaries.

## 1. Install and smoke-test

```bash
cd Code
./scripts/setup.sh
source .venv/bin/activate
pytest

moca run \
  --config configs/qwen2.5-7b-coqa.yaml \
  --set data.max_examples=64 \
  --set clustering.num_clusters=2 \
  --set optimization.max_epochs=1 \
  --set evaluation.max_eval_examples=16 \
  --set experiment_name=smoke_qwen_coqa
```

The complete main pipeline is:

```text
prepare -> cluster -> train experts -> generate validation
        -> fit validation calibrator -> generate test -> evaluate test
```

## 2. Main runs

```bash
moca run --config configs/qwen2.5-7b-coqa.yaml
moca run --config configs/qwen2.5-7b-quac.yaml
moca run --config configs/qwen2.5-7b-xsum.yaml
moca run --config configs/llama3-8b-coqa.yaml
moca run --config configs/llama3-8b-quac.yaml
moca run --config configs/llama3-8b-xsum.yaml
```

Before an archival run, replace every null model/tokenizer/dataset/NLI revision
with an immutable Hugging Face commit. Use a fresh experiment name after any
training-affecting change.

## 3. Reviewer suite

```bash
SEEDS=2026,2027,2028 RANDOM_K=5 \
  ./scripts/run_reviewer_suite.sh configs/qwen2.5-7b-coqa.yaml core

./scripts/run_reviewer_suite.sh configs/qwen2.5-7b-coqa.yaml sensitivity

RUN_NAME=qwen2.5-7b-coqa_moca_s2026 \
  ./scripts/run_reviewer_suite.sh configs/qwen2.5-7b-coqa.yaml diagnostics
```

Use the silhouette-selected `k` from the main cluster manifest as `RANDOM_K`
for the random-partition control. Repeat the suite for all model/dataset pairs
that will appear in the paper.

## 4. Aggregate seeds

```bash
moca aggregate --root runs --output-prefix runs/iclr_summary
```

The output includes `iclr_summary.json`, `iclr_summary.csv`, and
`iclr_summary.tex`. Preserve all per-example JSONL files for paired bootstrap
or permutation tests.

## 5. External OOD

Create prompt-only JSONL files with stable `example_id` and `prompt` fields:

```bash
moca generate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --input-file data_external/triviaqa.jsonl \
  --output-file runs/qwen2.5-7b_coqa_moca/predictions/triviaqa.jsonl

moca generate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --input-file data_external/medqa.jsonl \
  --output-file runs/qwen2.5-7b_coqa_moca/predictions/medqa.jsonl

moca evaluate-ood \
  --config configs/qwen2.5-7b-coqa.yaml \
  --id-predictions runs/qwen2.5-7b_coqa_moca/predictions/test.jsonl \
  --ood-predictions runs/qwen2.5-7b_coqa_moca/predictions/triviaqa.jsonl \
  --ood-predictions runs/qwen2.5-7b_coqa_moca/predictions/medqa.jsonl
```

Report OOD results separately from semantic correctness because prompt-only
OOD sets do not necessarily have comparable answer references.

## 6. Third-party baselines

Do not approximate named methods. Run their official pinned implementations,
then export:

```json
{"example_id":"...","prompt":"...","generation":"...","references":["..."],"confidence":0.73}
```

Evaluate every method through the same correctness judge and metric code:

```bash
moca evaluate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --ablation provided_confidence \
  --predictions external_method.jsonl \
  --output runs/external_method/evaluation/test_metrics.json
```

Pin the method repository commit, model checkpoint, and confidence semantics in
the paper. Probability calibration should be fitted on validation—not test—if
the external method emits only an uncalibrated score.

## 7. Submission gates

Do not copy a result into the paper unless all of the following exist:

1. immutable model, tokenizer, dataset, and evaluator revisions;
2. resolved config and training/cluster fingerprints;
3. prediction JSONL and timing JSON for every seed;
4. validation calibrator artifact and fit report;
5. test metrics and per-example metrics;
6. at least three seeds or a justified paired resampling interval;
7. the exact table/figure-generation command;
8. no story/dialogue overlap between data partitions;
9. separate answer-generation and confidence-estimation latency;
10. no fitting, threshold selection, or model selection on the test split.

