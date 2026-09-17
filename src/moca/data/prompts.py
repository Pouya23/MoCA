from __future__ import annotations

from collections.abc import Iterable

# Appendix E of the paper.  These are deliberately single-example prompts:
# the filled examples shown in the appendix illustrate separate prompts and
# are not in-context demonstrations.
COQA_PROMPT_TEMPLATE = "Story: {story}\nQuestion: {question}\nAnswer:"
QUAC_PROMPT_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer:"
XSUM_PROMPT_TEMPLATE = "Document: {document}\nSummary:"

PROMPT_TEMPLATES = {
    "coqa": COQA_PROMPT_TEMPLATE,
    "quac": QUAC_PROMPT_TEMPLATE,
    "xsum": XSUM_PROMPT_TEMPLATE,
}

_DATASET_ALIASES = {
    "coqa": "coqa",
    "stanfordnlp/coqa": "coqa",
    "quac": "quac",
    "allenai/quac": "quac",
    "xsum": "xsum",
    "edinburghnlp/xsum": "xsum",
}


def canonical_dataset_name(name: str) -> str:
    """Return the canonical short name for a supported dataset."""

    normalized = name.strip().lower()
    try:
        return _DATASET_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported dataset {name!r}; expected one of {sorted(PROMPT_TEMPLATES)}"
        ) from error


def _format_history(history: Iterable[tuple[str, str]]) -> str:
    turns: list[str] = []
    for question, answer in history:
        turns.extend((f"Question: {question}", f"Answer: {answer}"))
    return "\n".join(turns)


def format_coqa_prompt(
    story: str,
    question: str,
    history: Iterable[tuple[str, str]] = (),
) -> str:
    """Format a CoQA prompt, optionally inserting earlier dialogue turns."""

    history_text = _format_history(history)
    if not history_text:
        return COQA_PROMPT_TEMPLATE.format(story=story, question=question)
    return f"Story: {story}\n{history_text}\nQuestion: {question}\nAnswer:"


def format_quac_prompt(
    context: str,
    question: str,
    history: Iterable[tuple[str, str]] = (),
) -> str:
    """Format a QuAC prompt, optionally inserting earlier dialogue turns."""

    history_text = _format_history(history)
    if not history_text:
        return QUAC_PROMPT_TEMPLATE.format(context=context, question=question)
    return f"Context: {context}\n{history_text}\nQuestion: {question}\nAnswer:"


def format_xsum_prompt(document: str) -> str:
    """Format the paper's no-ICL XSum prompt."""

    return XSUM_PROMPT_TEMPLATE.format(document=document)


def format_prompt(
    dataset_name: str,
    source_text: str,
    question: str | None = None,
    history: Iterable[tuple[str, str]] = (),
) -> str:
    """Format a supported paper prompt through a common interface."""

    name = canonical_dataset_name(dataset_name)
    if name == "xsum":
        if question is not None:
            raise ValueError("XSum prompts do not take a question")
        if tuple(history):
            raise ValueError("XSum prompts do not have dialogue history")
        return format_xsum_prompt(source_text)
    if question is None:
        raise ValueError(f"{name} prompts require a question")
    if name == "coqa":
        return format_coqa_prompt(source_text, question, history)
    return format_quac_prompt(source_text, question, history)
