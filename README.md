# MoCA: Mixture of Calibrated Adapters

This repository is a clean-room reproduction of **Long-Text Calibration for
Fine-Tuned LLMs** (NeurIPS 2026 submission). It implements the paper's MoCA
training and inference algorithm, exposes every consequential choice through
typed YAML configuration, and keeps the paper baselines and diagnostic
ablations on the same code path.

The original source code was unavailable when this repository was rebuilt.
The paper specifies the algorithm and its central hyperparameters, but omits
some engineering details needed to execute it. Published settings are encoded
directly; all reconstructed choices are listed in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). This distinction is important:
the objective and routing algorithm can be reproduced exactly from the paper,
while bitwise reproduction of the reported tables is not possible without the
omitted revisions, seeds, batching details, and evaluation implementation.

## Method

Given prompt-response pairs $\mathcal D=\{(x_i,y_i)\}_{i=1}^N$, MoCA:

1. embeds every training prompt with the frozen base LLM;
2. applies k-means to the prompt embeddings;
3. trains one LoRA expert per cluster;
4. routes a new prompt to its nearest centroid and generates from only that
   expert.

For cluster $C_i$, expert $\theta_i$ minimizes

$$
\mathcal L_i =
\mathbb E_{(x,y)\sim P_i}\left[-\log p_{\theta_i}(y_{1:T}\mid x)\right]
+ \lambda\,
\mathbb E_{x\sim\cup_{j\ne i}P_j}
\left[
D_{\mathrm{KL}}\!\left(
p_{\theta_i}(y_1\mid x)\,\|\,p_{\mathrm{unif}}
\right)
\right].
$$

The positive term is full-response negative log-likelihood on the expert's own
cluster. The cross-cluster term pushes the first response-token distribution
toward the uniform distribution on pseudo-OOD prompts. At inference time,

$$
i^*(x)=\arg\min_i \lVert e(x)-c_i\rVert_2,
$$

then one response is sampled from $p_{\theta_{i^*}}(\cdot\mid x)$.

### One-generation deployment confidence

The main configs use **MoCA-1P**, not multi-sample semantic entropy, for
deployed confidence. Each prediction records the first-token entropy (including
a non-special-token version), nearest and second-nearest centroid distances,
and routing margin. A regularized logistic calibrator is fitted on semantic
correctness labels from the validation split only. Test Brier/ECE therefore
uses one generated response and never fits on test data.

Ten-sample semantic entropy remains available through the
`semantic_entropy_evaluation` ablation as an explicitly offline diagnostic.
It is not used to justify the single-pass latency claim.

## Installation

Requirements are Python 3.10+, a recent CUDA-capable PyTorch installation, and
enough GPU memory for an unquantized 7B/8B model. The paper used NVIDIA A100
GPUs and no quantization.

```bash
./scripts/setup.sh
source .venv/bin/activate
```

The setup script creates an isolated virtual environment and installs the
package in editable mode. On a managed cluster, install the CUDA-specific
PyTorch build required by that cluster before `pip install -e .` if its module
stack does not expose one automatically.

Meta-Llama-3-8B is gated. Accept its model license on Hugging Face and make an
access token available through the standard Hugging Face mechanism, for
example:

```bash
export HF_TOKEN="..."
```

Do not commit tokens, model caches, prepared datasets, or generated runs.

## Quick start

Run the complete single-process pipeline:

```bash
./scripts/run_pipeline.sh configs/qwen2.5-7b-coqa.yaml
```

This performs data preparation, prompt embedding and clustering, sequential
expert training, routed generation, and evaluation. The equivalent CLI command
is:

```bash
moca run --config configs/qwen2.5-7b-coqa.yaml
```

For a short integration check before allocating a full run:

```bash
moca run \
  --config configs/qwen2.5-7b-coqa.yaml \
  --set data.max_examples=64 \
  --set clustering.num_clusters=2 \
  --set optimization.max_epochs=1 \
  --set evaluation.max_eval_examples=16 \
  --set experiment_name=smoke_qwen_coqa
```

The six paper model/dataset combinations are:

| Config | Base checkpoint | Dataset | Published observed $k$ |
|---|---|---:|---:|
| `configs/qwen2.5-7b-coqa.yaml` | `Qwen/Qwen2.5-7B` | CoQA | 5 |
| `configs/qwen2.5-7b-quac.yaml` | `Qwen/Qwen2.5-7B` | QuAC | 2 |
| `configs/qwen2.5-7b-xsum.yaml` | `Qwen/Qwen2.5-7B` | XSum | 6 |
| `configs/llama3-8b-coqa.yaml` | `meta-llama/Meta-Llama-3-8B` | CoQA | 5 |
| `configs/llama3-8b-quac.yaml` | `meta-llama/Meta-Llama-3-8B` | QuAC | 2 |
| `configs/llama3-8b-xsum.yaml` | `meta-llama/Meta-Llama-3-8B` | XSum | 6 |

Each file is self-contained so a saved run does not depend on an implicit
configuration inheritance system.

## Run individual stages

Stages write namespaced artifacts under the run directory derived from
`output_root/experiment_name`.

```bash
moca prepare --config configs/qwen2.5-7b-coqa.yaml
moca cluster --config configs/qwen2.5-7b-coqa.yaml
moca train-all --config configs/qwen2.5-7b-coqa.yaml

moca generate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --split validation
moca calibrate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --split validation
moca generate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --split test
moca evaluate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --split test
```

Convenience wrappers for those stages are also available:

```bash
./scripts/prepare.sh configs/qwen2.5-7b-coqa.yaml
./scripts/cluster.sh configs/qwen2.5-7b-coqa.yaml
./scripts/train_expert.sh configs/qwen2.5-7b-coqa.yaml 0
./scripts/generate_evaluate.sh configs/qwen2.5-7b-coqa.yaml
```

The full configs run the paper's silhouette selection by default. To pin the
reported CoQA value for a strict workload replay, repeat the same override at
every stage:

```bash
moca run \
  --config configs/qwen2.5-7b-coqa.yaml \
  --set clustering.num_clusters=5 \
  --set experiment_name=qwen_coqa_fixed_k5
```

With no override, clustering computes silhouette scores for
$k=2,\ldots,10$ and selects the largest score. The chosen value is recorded
in `runs/<experiment>/clusters/manifest.json`.

## Expert-parallel training on local GPUs

Experts are independent after clustering. The local launcher performs
preparation and clustering once, trains up to one expert per listed GPU
concurrently, then generates and evaluates on the first GPU:

```bash
GPU_IDS=0,1,2,3 \
  ./scripts/run_local_multigpu.sh configs/qwen2.5-7b-coqa.yaml
```

If there are more experts than GPUs, the launcher schedules them in waves. Each
expert writes its own log under `runs/<experiment>/logs/`.

Run an ablation into a distinct directory:

```bash
GPU_IDS=0,1 \
ABLATION=k_ft \
EXPERIMENT_NAME=qwen_coqa_k_ft \
  ./scripts/run_local_multigpu.sh configs/qwen2.5-7b-coqa.yaml
```

`GPU_IDS` uses physical device identifiers. `CUDA_VISIBLE_DEVICES` is used as a
fallback when `GPU_IDS` is unset.

## Slurm array launch

The one-command submitter creates a dependency chain:

```text
prepare + cluster -> expert job array -> generate + evaluate
```

For a cluster whose A100 nodes are selected with a constraint:

```bash
SBATCH_PARTITION=gpu \
SBATCH_CONSTRAINT=a100 \
SBATCH_ACCOUNT=my_project \
  ./scripts/submit_slurm.sh configs/qwen2.5-7b-coqa.yaml
```

The same launcher supports an ablation while preserving a separate run:

```bash
SBATCH_PARTITION=gpu \
SBATCH_CONSTRAINT=a100 \
ABLATION=k_ft \
EXPERIMENT_NAME=qwen_coqa_k_ft \
  ./scripts/submit_slurm.sh configs/qwen2.5-7b-coqa.yaml
```

`SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `SBATCH_QOS`, and `SBATCH_CONSTRAINT` are
optional. The job files request one GPU and are intentionally easy to adapt to
site-specific module loading and resource names:

- `scripts/slurm_prepare_cluster.sh`
- `scripts/slurm_dispatch.sh`
- `scripts/slurm_train_array.sh`
- `scripts/slurm_generate_evaluate.sh`

The submitter first runs preparation and clustering. A lightweight dependent
dispatcher then reads `clusters/manifest.json`, creates array indices
`0..k-1`, and submits evaluation after the array succeeds. The lower-level
manual equivalent is:

```bash
prep_job="$(
  sbatch --parsable \
    --export=ALL,CONFIG_PATH="$PWD/configs/qwen2.5-7b-coqa.yaml" \
    scripts/slurm_prepare_cluster.sh
)"

# After the preparation job has completed, inspect the manifest. If it reports
# k=5, submit:
train_job="$(
  sbatch --parsable \
    --dependency="afterok:${prep_job}" \
    --array=0-4 \
    --export=ALL,CONFIG_PATH="$PWD/configs/qwen2.5-7b-coqa.yaml" \
    scripts/slurm_train_array.sh
)"

sbatch \
  --dependency="afterok:${train_job}" \
  --export=ALL,CONFIG_PATH="$PWD/configs/qwen2.5-7b-coqa.yaml" \
  scripts/slurm_generate_evaluate.sh
```

## Ablations

List the registered ablations:

```bash
moca list-ablations
```

Apply an ablation to any complete experiment config and always change the
experiment name so artifacts cannot overwrite the reference run:

```bash
moca run \
  --config configs/llama3-8b-quac.yaml \
  --ablation k_ft \
  --set experiment_name=llama_quac_k_ft
```

| Name | Change |
|---|---|
| `paper` | Paper MoCA objective; no-op marker |
| `k_ft` | Routed cluster experts with in-cluster NLL only |
| `vanilla_ft` | One adapter on all examples with NLL only |
| `first_5_tokens` | Regularize the first five response positions |
| `uniform_to_model_kl` | Reverse $D_{\mathrm{KL}}(p\|U)$ to $D_{\mathrm{KL}}(U\|p)$ |
| `uniform_cluster_ood` | Sample an OOD cluster uniformly before an example |
| `normalized_embeddings` | L2-normalize embeddings before k-means/routing |
| `last_token_pooling` | Use the last prompt-token embedding (main configs) |
| `mean_token_pooling` | Mean-pooling comparison requested by reviewers |
| `non_special_uniform` | Put the uniform target only on non-special tokens |
| `frequency_semantic_entropy` | Replace Kuhn Eq. 4 with sample-frequency entropy |
| `transitive_semantic_clustering` | Replace greedy classes with transitive closure |
| `threshold_nli` | Replace entailment argmax with a probability threshold |
| `semantic_entropy_evaluation` | Offline ten-sample semantic-entropy diagnostic |
| `first_token_only_confidence` | Raw, uncalibrated one-pass entropy score |
| `sequence_probability_confidence` | Length-normalized sequence-probability baseline |
| `provided_confidence` | Shared evaluation of an external baseline's confidence field |
| `random_partition` | Balanced random-partition control; requires fixed `k` |
| `matched_rank_80` | Rank-80 matched-capacity control |

`k_ft`, `vanilla_ft`, random partition, and matched rank are reviewer-requested
controls. The remaining variants are diagnostic extensions. The files under
`configs/ablations/` are standalone
Qwen2.5-7B/CoQA examples; for another model/dataset, apply the named ablation
to one of the six full configs as shown above.

## Configuration overrides

`--set` accepts typed YAML values and can be repeated:

```bash
moca train-expert \
  --config configs/qwen2.5-7b-xsum.yaml \
  --expert-id 0 \
  --set sampling.in_batch_size=1 \
  --set sampling.out_batch_size=1 \
  --set optimization.gradient_accumulation_steps=16
```

Repeat the same overrides at every separately invoked stage. Resolved configs
and content fingerprints are written with router and training artifacts.
Inference refuses incomplete, stale, or mixed expert checkpoints.

Existing expert checkpoints are never overwritten implicitly. Prefer a new
`experiment_name`; to intentionally restart one in place, pass:

```bash
moca train-expert \
  --config configs/qwen2.5-7b-coqa.yaml \
  --expert-id 0 \
  --set runtime.overwrite_existing=true
```

## Output layout

A typical run is organized as:

```text
runs/<experiment>/
  resolved_config.yaml           complete config and fingerprint
  data/                         prepared train/validation/test records
  clusters/
    manifest.json               selected k and clustering provenance
    centroids.npy
    train_assignments.npy
    train.jsonl
    validation.jsonl
    test.jsonl
  adapters/
    expert_0/
      best/                     PEFT adapter checkpoint
      resolved_config.yaml
      training_manifest.json
    ...
  predictions/
    validation.jsonl            one-response validation predictions
    validation_timing.json      generation-only timing/provenance
    test.jsonl                  routed single-response outputs
    test_timing.json            generation-only timing/provenance
    test_semantic.jsonl         cached semantic-entropy samples
    test_semantic.manifest.json cache provenance and content fingerprint
  evaluation/
    moca_1p_calibrator.json      validation-fitted single-pass calibrator
    moca_1p_calibration_fit.json validation fit diagnostics
    test_metrics.json           aggregate metrics
    test_metrics_examples.jsonl per-example evaluation records
```

The precise generated filenames are emitted by the CLI when each stage
completes. Cluster and adapter manifests contain enough provenance to reject
incompatible routing, data, or checkpoint configurations.

## Reviewer experiment suite

Run the three-seed main factorial and matched controls:

```bash
SEEDS=2026,2027,2028 RANDOM_K=5 \
  ./scripts/run_reviewer_suite.sh configs/qwen2.5-7b-coqa.yaml core
```

Run `k`, lambda, first-five-token, non-special-target, and pooling sensitivity:

```bash
./scripts/run_reviewer_suite.sh configs/qwen2.5-7b-coqa.yaml sensitivity
```

Run the offline semantic-entropy diagnostic and forced-wrong-route stress test
for an existing run:

```bash
RUN_NAME=qwen2.5-7b-coqa_moca_s2026 \
  ./scripts/run_reviewer_suite.sh configs/qwen2.5-7b-coqa.yaml diagnostics
```

Aggregate all seeded metrics into JSON, CSV, and a LaTeX table:

```bash
moca aggregate --root runs --output-prefix runs/iclr_summary
```

For genuine external OOD data, prepare prompt-only JSONL files, generate each
with `moca generate --input-file ... --output-file ...`, then compare them with
ID predictions:

```bash
moca evaluate-ood \
  --config configs/qwen2.5-7b-coqa.yaml \
  --id-predictions runs/qwen2.5-7b_coqa_moca/predictions/test.jsonl \
  --ood-predictions runs/qwen2.5-7b_coqa_moca/predictions/triviaqa.jsonl \
  --ood-predictions runs/qwen2.5-7b_coqa_moca/predictions/medqa.jsonl
```

The evaluator reports Brier, AUROC, fixed/adaptive ECE, binary log loss,
reliability bins, AURC/risk–coverage, optional selective accuracy, OOD AUROC,
FPR95, and router-distance/margin OOD diagnostics.

External methods such as SEP, LUQ, Laplace-LoRA, Bayesian-LoRA, or verbal
confidence can be evaluated without altering their implementations. Export one
JSONL row per example containing `example_id`, `prompt`, `generation`,
`references`, and a probability-valued `confidence`, then run:

```bash
moca evaluate \
  --config configs/qwen2.5-7b-coqa.yaml \
  --ablation provided_confidence \
  --predictions external_method.jsonl \
  --output runs/external_method/test_metrics.json
```

This keeps correctness labeling and every calibration metric identical across
methods. The external method and checkpoint commit must still be pinned in the
experiment manifest; this repository does not silently reimplement third-party
algorithms under approximate names.

## Repository structure

```text
src/moca/
  data/             dataset loading, prompt templates, deterministic splitting
  embeddings.py     frozen-base prompt embeddings
  clustering.py     k selection, centroids, routing artifacts
  losses.py         sequence NLL and first-token uniform KL
  modeling.py       tokenizer, base model, and PEFT adapter lifecycle
  training.py       isolated expert training
  inference.py      hard routing and single-expert generation
  evaluation/       semantic equivalence, calibration, QA, and ROUGE metrics
  ablations.py      named, conflict-checked config transforms
  cli/              command-line orchestration
configs/            six complete paper experiment configs
configs/ablations/  runnable ablation examples
scripts/            local and Slurm launchers
tests/              dependency-light unit and integration tests
```

## Development checks

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

The lightweight tests do not download 7B/8B checkpoints. A real GPU smoke run
is still required to validate a site's CUDA, model-access, and memory setup.

For the exact paper/reconstruction boundary, dataset caveats, and a release
checklist, read [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
