# Reproducibility notes

This codebase was reconstructed from the submitted manuscript after the
original implementation was lost. It distinguishes:

- **paper-specified behavior**, which is implemented directly;
- **paper-observed values**, which are documented as expected cluster counts;
  and
- **reconstructed defaults**, which are necessary to run the method but are not
  stated in the manuscript.

That distinction prevents an undocumented guess from being mistaken for a
published experimental condition.

## Paper-specified implementation

| Component | Setting stated in the paper | Implementation |
|---|---|---|
| Base models | non-chat Qwen2.5-7B and Llama-3-8B | `Qwen/Qwen2.5-7B`, `meta-llama/Meta-Llama-3-8B` |
| Datasets | CoQA, QuAC, XSum | Hugging Face loaders with the Appendix E prompt forms |
| Split | 80/10/10 train/validation/test | deterministic group split for CoQA/QuAC; document split for XSum |
| Prompting | single-shot; no in-context examples | one task template per record |
| Embeddings | frozen base LLM embedding \(e(x)\) | prompt embeddings from the selected base model |
| Clustering | k-means on training prompt embeddings | Euclidean k-means |
| \(k\) selection | maximum silhouette score over \(k=2,\ldots,10\) | `num_clusters: null` in the six reference configs |
| Observed \(k\) | CoQA 5, QuAC 2, XSum 6 | documented expected values; optional fixed overrides |
| Experts | one LoRA adapter per cluster | independent PEFT checkpoints |
| LoRA | rank 16, alpha 32, query/value projections in every attention layer | `q_proj`, `v_proj`, `r=16`, `alpha=32` |
| ID term | full-sequence response NLL | response-only, teacher-forced sequence NLL |
| Pseudo-OOD term | \(D_{\mathrm{KL}}(p_{\theta_i}(y_1\mid x)\|p_{\mathrm{unif}})\) | first response-token model-to-uniform KL |
| KL weight | \(\lambda=0.1\) | `objective.lambda_kl: 0.1` |
| Pseudo-OOD data | union of all clusters other than \(C_i\) | complement of the expert's cluster |
| Optimizer | AdamW | AdamW |
| Learning rate | \(2\times10^{-4}\) | `0.0002` |
| Schedule | cosine decay | cosine |
| Duration | until convergence, no more than 10 epochs | early-stopped with a 10-epoch cap |
| Precision/hardware | A100 GPUs, no quantization | unquantized model; BF16 reference runtime |
| Routing | nearest centroid by \(\ell_2\) distance | hard, deterministic nearest-centroid route |
| Deployment generation | one routed expert and one response | no expert ensemble or adapter averaging |
| Evaluation | Kuhn et al. semantic entropy defaults; Brier/AUROC and ROUGE-L F1 for QA; semantic Brier and ROUGE-1 for XSum | Original semantic-entropy protocol retained as an offline ablation; main configs use validation-fitted MoCA-1P confidence |
| Temperature scaling | not used | no post-hoc temperature scaling |

The Appendix B wording says the observed cluster counts are “for the datasets,
respectively”; following the dataset order in Section 6.1, this code maps them
to CoQA 5, QuAC 2, and XSum 6.

The cited semantic protocol was cross-checked against
[Kuhn et al. (2023)](https://arxiv.org/abs/2302.09664) and its
[official 2023 implementation](https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/compute_confidence_measure.py).

## Objective sign

Equation (1), its surrounding prose, and the proof all require minimizing a
positive \(D_{\mathrm{KL}}(p\|U)\), whose minimum is the uniform distribution.
Algorithm 1's extracted line 8 can visually appear to place a minus sign before
the minibatch average. Taking that sign literally would maximize the KL and
contradict the method and proof. This implementation follows Equation (1) and
minimizes:

\[
\mathcal L_{\mathrm{NLL}} + 0.1\,D_{\mathrm{KL}}(p\|U).
\]

## Reconstructed defaults

The manuscript does not state the following values. They are explicit in every
full config and can be changed without editing Python:

| Area | Reconstructed default | Why it is explicit |
|---|---|---|
| Checkpoint/data revisions | latest available (`null`) | no commit hashes or revisions are reported |
| Random seed | 2026 | the appendix reports three seeded CoQA runs but not the seed values |
| Split unit | group for CoQA/QuAC; example/document for XSum | prevents story/dialogue leakage across partitions |
| Dialogue history | excluded | Appendix E shows story/context plus the current question |
| Prompt limit | 1536 tokens, left truncation | context limit and truncation policy are omitted |
| Response limit | 256 training tokens | response cap is omitted |
| Token boundaries | tokenizer-native prompt special tokens, one leading response space, append EOS | BOS/EOS and answer-separator handling are omitted |
| Embedding layer/pooling | last hidden layer, last prompt token | matches the NeurIPS response; mean pooling is a registered ablation |
| Embedding normalization | off | the paper states k-means/Euclidean distance but not normalization |
| Embedding batch/dtype | 8 / float32 output | omitted |
| k-means details | k-means++, 10 starts, 300 iterations, tolerance \(10^{-4}\) | omitted |
| Silhouette sample cap | 10,000 | controls memory/time on XSum |
| LoRA dropout/bias/init | 0 / none / PEFT default | omitted |
| In/OOD batch sizes | 2 / 2 | omitted |
| OOD sampling | empirical complement, with replacement | the paper says sample from the union but not the finite sampler |
| NLL reduction | per-sequence sum, then batch mean | matches the full-sequence objective; minibatch reduction is omitted |
| AdamW details | betas 0.9/0.999, epsilon \(10^{-8}\), zero weight decay | only optimizer name and learning rate are reported |
| Warmup | none | omitted |
| Gradient accumulation | 8 | omitted |
| Gradient clipping | 1.0 | omitted |
| Early stopping | patience 2 on validation objective | “until convergence” is not operationally defined |
| Model/runtime dtype | BF16 | appropriate for the reported A100 hardware; exact precision is omitted |
| Generation | sampling at temperature 1, top-p 1, top-k 0, max 128 new tokens | Algorithm 1 says “sampled” but gives no decoding parameters |
| Semantic samples | 10 | a concrete interpretation of semantic entropy's default protocol |
| Semantic estimator | Kuhn et al. (2023) Eq. 4: classwise log-sum-exp followed by the negative mean over unique classes | the manuscript cites the protocol rather than restating the estimator |
| Semantic equivalence | representative-greedy bidirectional entailment with `microsoft/deberta-large-mnli`, entailment argmax, prompt context, and 512-token left truncation | follows the cited 2023 implementation; checkpoint revision and truncation details remain unreported |
| Deployment confidence | validation-fitted logistic map over non-special first-token confidence, centroid distance, and routing margin | one-generation protocol added in response to reviewer concern; never fit on test |
| Offline semantic confidence | \(\exp(-\max(0,H_{\mathrm{semantic}}))\) | retained only for historical comparison; its absolute mapping is not used as the main calibration claim |
| QA correctness | semantic equivalence against references | paper states semantic equality but not an exact classifier |
| XSum correctness/calibration rule | semantic-confidence protocol plus ROUGE-1 | the cited protocol is not restated algorithmically |

These choices are reasonable and auditable, but they are not claims about the
authors' deleted implementation. Results sensitive to them should be reported
as reproduction results, not exact reruns of the submitted tables.

Table 1 labels ROUGE-L F1 as “Acc.” The evaluator therefore writes `accuracy`
as an alias of `rouge_l_f1`; the binary semantic-equivalence rate used as the
Brier/AUROC target is separately named `semantic_correctness_rate`.

## Reselecting \(k\) versus fixing the reported value

The six main configs execute the paper's selection procedure. The selected
count is therefore discovered after clustering and drives the local/Slurm
expert arrays.

To pin the reported CoQA value instead:

```bash
moca run \
  --config configs/qwen2.5-7b-coqa.yaml \
  --set clustering.num_clusters=5 \
  --set experiment_name=qwen_coqa_fixed_k5
```

The selected score, all candidate scores, k-means settings, model provenance,
and training-example fingerprint are saved under
`runs/qwen_coqa_fixed_k5/clusters/manifest.json`. With a fixed count, the
manifest records that count but does not contain a multi-\(k\) selection sweep.

## Baselines and ablations

All methods use the same data, prompt formatting, embeddings, routing, LoRA
construction, optimizer, checkpointing, generation, and evaluation unless an
ablation explicitly touches that setting.

- `paper`: clustered experts with NLL plus first-token
  \(D_{\mathrm{KL}}(p\|U)\).
- `k_ft`: clustered experts and nearest-centroid routing, with NLL only.
- `vanilla_ft`: a single LoRA adapter trained on the whole training split,
  with NLL only.
- `frequency_semantic_entropy`: sample-frequency class entropy instead of the
  cited Kuhn Eq. 4 estimator.
- `transitive_semantic_clustering`: pairwise transitive closure instead of
  representative-greedy semantic classes.
- `threshold_nli`: an entailment-probability threshold instead of entailment
  argmax.

The ablation registry records the exact config paths changed by every named
variant and rejects overlapping transforms. Use a new `experiment_name` for
every run.

## Data and prompt fidelity

Appendix E gives these zero-demonstration templates:

```text
CoQA
Story: {STORY}
Question: {QUESTION}
Answer:

QuAC
Context: {CONTEXT}
Question: {QUESTION}
Answer:

XSum
Document: {DOCUMENT}
Summary:
```

The manuscript says 80/10/10 but does not specify whether conversational turns
from one story/article are kept together. The corrected reference configs use
`split_unit: group` for CoQA and QuAC so no story/dialogue crosses partitions.
XSum keeps its one-example-per-document split unit. This is a deliberate
leakage-prevention correction and should be stated in the paper.

Hugging Face datasets and model repositories can change. For an archival run,
fill in:

- `model.revision`;
- `model.tokenizer_revision`;
- `data.dataset_revision`;
- `evaluation.nli_revision`.

Record model/dataset license acceptance separately from the experiment
artifacts.

## Determinism

The runtime seeds Python, NumPy, and PyTorch. Expert \(i\) uses a deterministic
offset from the run seed so experts do not share an identical random stream.
Set:

```yaml
runtime:
  deterministic: true
```

for stronger deterministic enforcement. Some CUDA kernels, generation
sampling, distributed scheduling, and upstream model code may still vary
across hardware or library versions. Report mean and standard deviation over
multiple seeds for comparisons; the paper reports three seeded runs for its
CoQA stability table without publishing the seed values.

## Resource expectations

The paper used unquantized A100 GPUs. Each expert is trained independently, so
wall-clock time scales well with the number of available GPUs while aggregate
GPU-hours still scale with \(k\). Main MoCA-1P evaluation generates one answer
per example and uses the NLI model only to create offline correctness labels.
The semantic-entropy diagnostic additionally draws multiple generations and
must report that sampling/NLI cost separately.

If memory is insufficient, lower the in/OOD batch sizes and increase gradient
accumulation to preserve the effective update size:

```bash
moca train-expert \
  --config configs/qwen2.5-7b-xsum.yaml \
  --expert-id 0 \
  --set sampling.in_batch_size=1 \
  --set sampling.out_batch_size=1 \
  --set optimization.gradient_accumulation_steps=16
```

This is an engineering adaptation, not a setting reported in the paper.

## Reproduction checklist

Before producing final tables:

1. pin all model, tokenizer, dataset, and NLI revisions;
2. archive the resolved config and its fingerprint;
3. retain `clusters/manifest.json` and verify the selected/fixed \(k\);
4. verify every `adapters/expert_<i>/best/` checkpoint exists;
5. record package versions, GPU model, CUDA version, and precision;
6. use a unique run directory per method and seed;
7. confirm temperature scaling remains disabled;
8. evaluate every method on the identical test record IDs;
9. report the reconstructed choices above alongside the results;
10. run at least three seeds when comparing against the paper's stability
    claims.
