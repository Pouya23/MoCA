from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from moca.config import DataConfig, ExperimentConfig, TokenizationConfig
from moca.data import (
    COQA_PROMPT_TEMPLATE,
    QUAC_PROMPT_TEMPLATE,
    XSUM_PROMPT_TEMPLATE,
    ResponseOnlyCausalLMCollator,
    deterministic_split,
    expand_record,
    load_prepared_splits,
    load_raw_dataset,
    prepare_data,
    use_official_splits,
)
from moca.records import PromptResponse


class PromptTests(unittest.TestCase):
    def test_paper_no_icl_templates_are_exact(self) -> None:
        self.assertEqual(
            COQA_PROMPT_TEMPLATE.format(story="S", question="Q"),
            "Story: S\nQuestion: Q\nAnswer:",
        )
        self.assertEqual(
            QUAC_PROMPT_TEMPLATE.format(context="C", question="Q"),
            "Context: C\nQuestion: Q\nAnswer:",
        )
        self.assertEqual(
            XSUM_PROMPT_TEMPLATE.format(document="D"),
            "Document: D\nSummary:",
        )


class ExpansionTests(unittest.TestCase):
    def test_coqa_expands_turns_and_preserves_references_and_group(self) -> None:
        row = {
            "id": "conversation-1",
            "source": "wikipedia",
            "story": "A story.",
            "questions": ["First?", "Second?"],
            "answers": {
                "input_text": ["one", "two"],
                "answer_start": [0, 4],
                "answer_end": [3, 7],
            },
            "additional_answers": {
                "annotator": [
                    {"input_text": "the first"},
                    {"input_text": "the second"},
                ]
            },
        }
        examples = expand_record(row, "coqa", split="train")
        self.assertEqual(len(examples), 2)
        self.assertEqual({item.group_id for item in examples}, {"conversation-1"})
        self.assertEqual(examples[0].references, ("one", "the first"))
        self.assertEqual(
            examples[1].prompt,
            "Story: A story.\nQuestion: Second?\nAnswer:",
        )

        with_history = expand_record(
            row,
            "coqa",
            split="train",
            include_dialogue_history=True,
        )
        self.assertEqual(
            with_history[1].prompt,
            "Story: A story.\nQuestion: First?\nAnswer: one\nQuestion: Second?\nAnswer:",
        )

    def test_quac_uses_original_response_and_all_reference_answers(self) -> None:
        row = {
            "dialogue_id": "dialogue-1",
            "context": "A passage.",
            "questions": ["Where?"],
            "turn_ids": ["dialogue-1-q0"],
            "answers": [{"texts": ["there", "in that place"], "answer_starts": [0, 0]}],
            "orig_answers": {"texts": ["over there"], "answer_starts": [0]},
        }
        example = expand_record(row, "quac", split="validation")[0]
        self.assertEqual(example.example_id, "dialogue-1-q0")
        self.assertEqual(example.group_id, "dialogue-1")
        self.assertEqual(example.response, "over there")
        self.assertEqual(
            example.references,
            ("over there", "there", "in that place"),
        )

    def test_xsum_expands_one_document(self) -> None:
        row = {"id": "bbc-1", "document": "News body.", "summary": "One sentence."}
        example = expand_record(row, "xsum", split="test")[0]
        self.assertEqual(example.example_id, "bbc-1")
        self.assertEqual(example.group_id, "bbc-1")
        self.assertEqual(example.prompt, "Document: News body.\nSummary:")
        self.assertEqual(example.references, ("One sentence.",))


def _example(index: int, group: str | None = None, split: str = "all") -> PromptResponse:
    return PromptResponse(
        example_id=f"e{index}",
        prompt=f"p{index}",
        response=f"r{index}",
        references=(f"r{index}",),
        split=split,
        group_id=group or f"g{index}",
    )


class SplittingTests(unittest.TestCase):
    def test_example_split_is_reproducible_and_80_10_10(self) -> None:
        examples = [_example(index) for index in range(10)]
        first = deterministic_split(examples, seed=7)
        second = deterministic_split(list(reversed(examples)), seed=7)
        self.assertEqual([len(first[name]) for name in first], [8, 1, 1])
        self.assertEqual(
            {name: {item.example_id for item in rows} for name, rows in first.items()},
            {name: {item.example_id for item in rows} for name, rows in second.items()},
        )

    def test_group_split_never_leaks_a_dialogue(self) -> None:
        examples = [
            _example(0, "a"),
            _example(1, "a"),
            _example(2, "b"),
            _example(3, "b"),
            _example(4, "c"),
            _example(5, "c"),
            _example(6, "d"),
            _example(7, "d"),
            _example(8, "e"),
            _example(9, "e"),
        ]
        splits = deterministic_split(examples, seed=11, split_unit="group")
        owner: dict[str, str] = {}
        for split, rows in splits.items():
            for row in rows:
                if row.group_id in owner:
                    self.assertEqual(owner[row.group_id], split)
                owner[row.group_id] = split
        self.assertEqual(set(owner), {"a", "b", "c", "d", "e"})

    def test_official_split_aliases_are_preserved(self) -> None:
        examples = [
            _example(0, split="train"),
            _example(1, split="dev"),
            _example(2, split="test"),
        ]
        splits = use_official_splits(examples)
        self.assertEqual([len(splits[name]) for name in splits], [1, 1, 1])
        self.assertEqual(splits["validation"][0].split, "validation")


class PersistenceTests(unittest.TestCase):
    def test_local_jsonl_prepare_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "xsum.jsonl"
            with source.open("w", encoding="utf-8") as handle:
                for index in range(10):
                    handle.write(
                        json.dumps(
                            {
                                "id": f"id-{index}",
                                "document": f"doc-{index}",
                                "summary": f"sum-{index}",
                            }
                        )
                        + "\n"
                    )
            data = DataConfig(name="xsum", local_jsonl=str(source))
            config = ExperimentConfig(
                experiment_name="unit",
                output_root=str(root / "runs"),
                data=data,
            )
            splits = prepare_data(config)
            self.assertEqual([len(splits[name]) for name in splits], [8, 1, 1])
            reloaded = load_prepared_splits(config)
            self.assertEqual(
                {name: [item.to_dict() for item in rows] for name, rows in splits.items()},
                {name: [item.to_dict() for item in rows] for name, rows in reloaded.items()},
            )
            for split in ("train", "validation", "test"):
                self.assertTrue((config.run_dir / "data" / f"{split}.jsonl").is_file())

    def test_huggingface_loader_is_injected_and_receives_revision(self) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def loader(*args: object, **kwargs: object) -> dict[str, list[dict[str, str]]]:
            calls.append((args, kwargs))
            return {"train": []}

        config = DataConfig(
            name="quac",
            dataset_id="allenai/quac",
            dataset_config="plain_text",
            dataset_revision="abc123",
        )
        self.assertEqual(load_raw_dataset(config, load_dataset_fn=loader), {"train": []})
        self.assertEqual(
            calls,
            [(("allenai/quac", "plain_text"), {"revision": "abc123"})],
        )


class _FakeTensor(list):
    pass


class _FakeTorch(types.ModuleType):
    long = object()

    def tensor(self, values: list[list[int]], *, dtype: object) -> _FakeTensor:
        self.last_dtype = dtype
        return _FakeTensor(values)


class _CharacterTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    padding_side = "right"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [ord(character) - 96 for character in text]


class CollatorTests(unittest.TestCase):
    def test_masks_prompt_and_padding_with_left_prompt_truncation(self) -> None:
        config = TokenizationConfig(
            max_prompt_tokens=3,
            max_response_tokens=3,
            prompt_truncation_side="left",
            response_prefix="",
            append_eos=True,
            include_eos_in_nll=True,
            pad_to_multiple_of=4,
        )
        collator = ResponseOnlyCausalLMCollator(_CharacterTokenizer(), config)
        fake_torch = _FakeTorch("torch")
        with patch.dict(sys.modules, {"torch": fake_torch}):
            batch = collator(
                [
                    {"prompt": "abcd", "response": "xyz"},
                    {"prompt": "q", "response": "r"},
                ]
            )
        self.assertEqual(batch["input_ids"][0], [2, 3, 4, 24, 25, 99, 0, 0])
        self.assertEqual(
            batch["labels"][0],
            [-100, -100, -100, 24, 25, 99, -100, -100],
        )
        self.assertEqual(batch["attention_mask"][1], [1, 1, 1, 0, 0, 0, 0, 0])

    def test_can_exclude_appended_eos_from_loss(self) -> None:
        config = replace(
            TokenizationConfig(),
            max_prompt_tokens=2,
            max_response_tokens=2,
            response_prefix="",
            include_eos_in_nll=False,
            pad_to_multiple_of=1,
        )
        fake_torch = _FakeTorch("torch")
        with patch.dict(sys.modules, {"torch": fake_torch}):
            labels = ResponseOnlyCausalLMCollator(_CharacterTokenizer(), config)(
                [{"prompt": "a", "response": "bc"}]
            )["labels"][0]
        self.assertEqual(labels, [-100, 2, -100])

    def test_leading_response_space_is_supervised(self) -> None:
        config = replace(
            TokenizationConfig(),
            max_prompt_tokens=1,
            max_response_tokens=3,
            response_prefix=" ",
            pad_to_multiple_of=1,
        )
        fake_torch = _FakeTorch("torch")
        with patch.dict(sys.modules, {"torch": fake_torch}):
            batch = ResponseOnlyCausalLMCollator(
                _CharacterTokenizer(),
                config,
            )([{"prompt": "a", "response": "x"}])
        self.assertEqual(batch["input_ids"][0], [1, -64, 24, 99])
        self.assertEqual(batch["labels"][0], [-100, -64, 24, 99])


if __name__ == "__main__":
    unittest.main()
