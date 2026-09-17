from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

import moca.cli.evaluate as evaluate_cli
from moca.calibration import SinglePassCalibrator, fit_logistic_calibrator
from moca.config import EvaluationConfig, ExperimentConfig
from moca.evaluation import (
    BidirectionalNLIScorer,
    auroc_score,
    brier_score,
    categorical_entropy,
    expected_calibration_error,
    false_positive_rate_at_recall,
    correctness_against_references,
    entropy_to_confidence,
    evaluate_generation_jsonl,
    evaluate_generation_records,
    kuhn_eq4_semantic_entropy,
    max_rouge_l_f1,
    mean_entropy_by_position,
    reliability_bins,
    risk_coverage_curve,
    rouge_1_f1,
    rouge_l_f1,
    semantic_class_probabilities,
    semantic_cluster_ids,
    semantic_clusters,
    semantic_confidence,
    semantic_entropy,
    spearman_correlation,
    token_position_entropy,
)


def test_brier_and_auroc_are_dependency_free_and_tie_aware() -> None:
    assert brier_score([0.9, 0.2], [1, 0]) == pytest.approx(0.025)
    assert auroc_score([0.9, 0.2], [1, 0]) == pytest.approx(1.0)
    assert auroc_score([0.5, 0.5], [1, 0]) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="positive and one negative"):
        auroc_score([0.2, 0.4], [0, 0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.1], [1])


def test_reliability_selective_and_ood_metrics() -> None:
    confidence = [0.9, 0.8, 0.2, 0.1]
    outcomes = [1, 1, 0, 0]
    assert expected_calibration_error(confidence, outcomes, num_bins=2) == pytest.approx(0.15)
    assert sum(item["count"] for item in reliability_bins(confidence, outcomes)) == 4
    curve, aurc = risk_coverage_curve(confidence, outcomes)
    assert curve[0]["risk"] == 0.0
    assert aurc >= 0.0
    assert false_positive_rate_at_recall([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 0.0
    assert spearman_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_single_pass_calibrator_fits_and_round_trips() -> None:
    rows = [
        {
            "non_special_first_token_confidence": index / 39,
            "routing_distance": 1.0 - index / 39,
            "routing_margin": index / 39,
        }
        for index in range(40)
    ]
    outcomes = [int(index >= 20) for index in range(40)]
    fitted = fit_logistic_calibrator(rows, outcomes)
    restored = SinglePassCalibrator.from_dict(fitted.to_dict())
    probabilities = restored.predict(rows)
    assert fitted.converged
    assert min(probabilities) < 0.1
    assert max(probabilities) > 0.9
    assert brier_score(probabilities, outcomes) < 0.05


def test_rouge_metrics_and_multiple_reference_maximum() -> None:
    prediction = "the cat sat on the mat"
    reference = "the cat is on the mat"
    assert rouge_1_f1(prediction, reference) == pytest.approx(5 / 6)
    assert rouge_l_f1(prediction, reference) == pytest.approx(5 / 6)
    assert max_rouge_l_f1(
        prediction,
        ["entirely unrelated", prediction],
    ) == pytest.approx(1.0)


def test_bidirectional_nli_uses_both_directions_and_loads_lazily() -> None:
    observed_pairs: list[tuple[str, str]] = []

    def predictor(pairs: list[tuple[str, str]]) -> list[float]:
        observed_pairs.extend(pairs)
        values: list[float] = []
        for premise, hypothesis in pairs:
            premise_answer = premise.rsplit("Answer: ", 1)[-1]
            hypothesis_answer = hypothesis.rsplit("Answer: ", 1)[-1]
            values.append(0.9 if {premise_answer, hypothesis_answer} == {"cat", "feline"} else 0.2)
        return values

    scorer = BidirectionalNLIScorer(
        predictor=predictor,
        threshold=0.8,
        include_prompt=True,
    )
    assert not scorer.is_loaded
    assert scorer.equivalent("cat", "feline", prompt="What animal?")
    assert not scorer.equivalent("cat", "dog", prompt="What animal?")
    assert not scorer.is_loaded
    assert len(observed_pairs) == 4
    assert all(pair[0].startswith("What animal?") for pair in observed_pairs)


def test_semantic_clustering_supports_kuhn_greedy_and_transitive_ablation() -> None:
    related = {frozenset(("a", "b")), frozenset(("b", "c"))}

    def scorer(left: str, right: str) -> bool:
        return frozenset((left, right)) in related

    assert semantic_cluster_ids(["a", "b", "c", "d"], scorer) == [0, 0, 1, 2]
    assert semantic_clusters(
        ["a", "b", "c", "d"],
        scorer,
        algorithm="transitive_closure",
    ) == [
        [0, 1, 2],
        [3],
    ]


def test_semantic_entropy_supports_frequency_and_model_log_probability() -> None:
    labels = [0, 0, 1]
    frequency_probabilities = semantic_class_probabilities(labels)
    assert frequency_probabilities == pytest.approx({0: 2 / 3, 1: 1 / 3})
    assert semantic_entropy(labels) == pytest.approx(
        -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
    )

    log_probabilities = [math.log(0.2), math.log(0.3), math.log(0.5)]
    model_probabilities = semantic_class_probabilities(
        labels,
        log_probabilities,
    )
    assert model_probabilities == pytest.approx({0: 0.5, 1: 0.5})
    assert semantic_entropy(labels, log_probabilities) == pytest.approx(math.log(2))
    assert semantic_confidence(
        labels,
        log_probabilities,
        mapping="maximum_class_probability",
    ) == pytest.approx(0.5)
    assert entropy_to_confidence(math.log(2)) == pytest.approx(0.5)


def test_kuhn_eq4_estimator_averages_semantic_class_log_likelihoods() -> None:
    labels = [0, 0, 1]
    log_probabilities = [math.log(0.1), math.log(0.2), math.log(0.7)]
    expected = -(math.log(0.3) + math.log(0.7)) / 2
    frequency_weighted = -(2 * math.log(0.3) + math.log(0.7)) / 3

    estimate = kuhn_eq4_semantic_entropy(
        labels,
        log_probabilities,
    )
    assert estimate == pytest.approx(expected)
    assert estimate != pytest.approx(frequency_weighted)
    # This likelihood-based Eq. 4 estimate is not cluster-frequency entropy.
    assert kuhn_eq4_semantic_entropy(
        labels,
        log_probabilities,
    ) != pytest.approx(semantic_entropy(labels))


def test_kuhn_confidence_clips_negative_finite_sample_estimate() -> None:
    report = evaluate_generation_records(
        [
            {
                "prediction": "a",
                "references": ["a"],
                "samples": [
                    {
                        "text": "a",
                        "sequence_log_probability": math.log(0.8),
                        "generated_token_count": 1,
                    },
                    {
                        "text": "a",
                        "sequence_log_probability": math.log(0.8),
                        "generated_token_count": 1,
                    },
                ],
            }
        ],
        semantic_estimator="kuhn_eq4",
    )
    assert report["examples"][0]["semantic_entropy"] < 0
    assert report["examples"][0]["confidence"] == pytest.approx(1.0)


def test_correctness_checks_every_reference() -> None:
    assert correctness_against_references(
        "  PARIS ",
        ["London", "Paris"],
    )
    assert not correctness_against_references("Rome", ["London", "Paris"])

    equivalent = lambda left, right: left[0] == right[0]
    assert correctness_against_references(
        "kitten",
        ["cat", "kangaroo"],
        equivalent,
    )


def test_dataset_jsonl_evaluation(tmp_path) -> None:
    path = tmp_path / "generations.jsonl"
    rows = [
        {
            "example_id": "correct",
            "prompt": "Capital of France?",
            "prediction": "Paris",
            "references": ["Paris"],
            "semantic_samples": ["Paris", "Paris"],
        },
        {
            "example_id": "wrong",
            "prompt": "Capital of France?",
            "prediction": "Lyon",
            "references": ["Paris"],
            "semantic_samples": ["Lyon", "Marseille"],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = evaluate_generation_jsonl(
        path,
        semantic_estimator="frequency",
    )
    assert report["metrics"]["num_examples"] == 2
    assert report["metrics"]["brier"] == pytest.approx(0.125)
    assert report["metrics"]["auroc"] == pytest.approx(1.0)
    assert report["metrics"]["accuracy"] == pytest.approx(0.5)
    assert report["examples"][1]["semantic_entropy"] == pytest.approx(math.log(2))
    assert report["examples"][1]["semantic_class_ids"] == [0, 1]


def test_dataset_model_probability_estimator_uses_sample_log_probs() -> None:
    rows = [
        {
            "example_id": "weighted",
            "prediction": "a",
            "references": ["a"],
            "samples": [
                {
                    "text": "a",
                    "sequence_log_probability": 2 * math.log(0.2),
                    "generated_token_count": 2,
                },
                {
                    "text": "a",
                    "sequence_log_probability": 2 * math.log(0.3),
                    "generated_token_count": 2,
                },
                {
                    "text": "b",
                    "sequence_log_probability": 2 * math.log(0.5),
                    "generated_token_count": 2,
                },
            ],
        }
    ]
    report = evaluate_generation_records(
        rows,
        semantic_estimator="model_probability",
        length_normalize_log_probabilities=True,
    )
    assert report["examples"][0]["semantic_entropy"] == pytest.approx(math.log(2))
    assert report["metrics"]["auroc"] is None


def test_token_position_entropy_and_variable_length_aggregation() -> None:
    logits = [
        [[0.0, 0.0], [100.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]
    values = token_position_entropy(
        logits,
        attention_mask=[[1, 1], [1, 0]],
    )
    assert values[0] == pytest.approx(math.log(2))
    assert values[1] == pytest.approx(0.0, abs=1e-12)

    probability_entropy = categorical_entropy(
        [0.25, 0.75],
        input_type="probabilities",
    )
    assert float(probability_entropy) == pytest.approx(
        -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    )
    assert mean_entropy_by_position([[1.0, 2.0], [3.0]]) == pytest.approx([2.0, 2.0])


def test_table_accuracy_is_rouge_l_not_semantic_correctness_rate() -> None:
    always_equivalent = lambda left, right: True
    report = evaluate_generation_records(
        [
            {
                "prediction": "alpha",
                "references": ["omega"],
                "semantic_samples": ["alpha", "alpha"],
            }
        ],
        always_equivalent,
        semantic_estimator="frequency",
    )

    assert report["metrics"]["semantic_correctness_rate"] == pytest.approx(1.0)
    assert report["metrics"]["rouge_l_f1"] == pytest.approx(0.0)
    assert report["metrics"]["accuracy"] == pytest.approx(0.0)


def test_semantic_generation_cache_reuses_and_invalidates(
    tmp_path,
    monkeypatch,
) -> None:
    config = ExperimentConfig(
        output_root=str(tmp_path),
        experiment_name="cache-test",
        evaluation=EvaluationConfig(
            semantic_samples=2,
            cache_generations=True,
        ),
    )
    source_path = tmp_path / "source.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    source_rows = [
        {
            "example_id": "one",
            "prompt": "Question",
            "generation": "answer-a",
            "references": ["reference"],
        }
    ]
    calls: list[tuple[str, bool, int]] = []

    def fake_add_semantic_samples(config, rows, *, force=False):
        calls.append(
            (
                rows[0]["generation"],
                force,
                config.evaluation.semantic_samples,
            )
        )
        for row in rows:
            row["semantic_samples"] = [
                {
                    "text": f"{row['generation']}-sample-{index}",
                    "sequence_log_probability": -1.0,
                    "generated_token_count": 1,
                }
                for index in range(config.evaluation.semantic_samples)
            ]
        return rows

    monkeypatch.setattr(
        evaluate_cli,
        "_add_semantic_samples",
        fake_add_semantic_samples,
    )

    first = evaluate_cli._semantic_rows_with_cache(
        config,
        source_rows,
        source_path=source_path,
        semantic_path=semantic_path,
        skip_semantic_sampling=False,
    )
    assert len(calls) == 1
    assert semantic_path.exists()
    manifest_path = evaluate_cli._semantic_cache_manifest_path(semantic_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_fingerprint"]
    assert manifest["config_fingerprint"] == config.fingerprint()
    assert manifest["cache_fingerprint"]

    second = evaluate_cli._semantic_rows_with_cache(
        config,
        source_rows,
        source_path=source_path,
        semantic_path=semantic_path,
        skip_semantic_sampling=True,
    )
    assert second == first
    assert len(calls) == 1

    changed_source = [dict(source_rows[0], generation="answer-b")]
    evaluate_cli._semantic_rows_with_cache(
        config,
        changed_source,
        source_path=source_path,
        semantic_path=semantic_path,
        skip_semantic_sampling=False,
    )
    assert calls[-1] == ("answer-b", True, 2)
    assert len(calls) == 2

    changed_config = replace(
        config,
        evaluation=replace(config.evaluation, semantic_samples=3),
    )
    evaluate_cli._semantic_rows_with_cache(
        changed_config,
        changed_source,
        source_path=source_path,
        semantic_path=semantic_path,
        skip_semantic_sampling=False,
    )
    assert calls[-1] == ("answer-b", True, 3)
    assert len(calls) == 3

    semantic_path.write_text('{"corrupt": true}\n', encoding="utf-8")
    evaluate_cli._semantic_rows_with_cache(
        changed_config,
        changed_source,
        source_path=source_path,
        semantic_path=semantic_path,
        skip_semantic_sampling=False,
    )
    assert len(calls) == 4


def test_skip_sampling_rejects_an_incompatible_cache(
    tmp_path,
    monkeypatch,
) -> None:
    config = ExperimentConfig(
        output_root=str(tmp_path),
        experiment_name="cache-test",
        evaluation=EvaluationConfig(
            semantic_samples=2,
            cache_generations=True,
        ),
    )
    source_path = tmp_path / "source.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    source_rows = [
        {
            "example_id": "one",
            "prompt": "Question",
            "generation": "answer-a",
            "references": ["reference"],
        }
    ]

    def fake_add_semantic_samples(config, rows, *, force=False):
        for row in rows:
            row["semantic_samples"] = [
                {"text": f"sample-{index}"} for index in range(config.evaluation.semantic_samples)
            ]
        return rows

    monkeypatch.setattr(
        evaluate_cli,
        "_add_semantic_samples",
        fake_add_semantic_samples,
    )
    evaluate_cli._semantic_rows_with_cache(
        config,
        source_rows,
        source_path=source_path,
        semantic_path=semantic_path,
        skip_semantic_sampling=False,
    )

    with pytest.raises(ValueError, match="No compatible"):
        evaluate_cli._semantic_rows_with_cache(
            config,
            [dict(source_rows[0], generation="changed")],
            source_path=source_path,
            semantic_path=semantic_path,
            skip_semantic_sampling=True,
        )
