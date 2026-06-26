"""Prompt construction and SST-method dispatch.

``METHODS`` is the canonical ordered list of evaluated pipelines, shared by
every notebook so the variants never drift apart.
"""

from __future__ import annotations

from .keyword import keyword_aware_sst_filter
from .sst import (
    make_full_sst,
    make_no_attributes_sst,
    make_no_relations_sst,
    make_objects_only_sst,
    sst_to_prompt_caption,
    sst_to_prompt_compact,
    sst_to_prompt_structured,
)

METHODS = [
    "full_sst",
    "caption_style_sst",
    "keyword_aware_sst",
    "compact_keyword_sst",
    "no_relations_sst",
    "no_attributes_sst",
    "objects_only_sst",
    "question_only",   # language-prior baseline — no scene info
]


def build_standard_prompt(question: str, sst_text: str) -> str:
    return f"""You are answering a visual question using only the structured scene information below.

Structured scene:
{sst_text}

Question:
{question}

Rules:
- Answer using only the structured scene information.
- Give only the final answer.
- Do not explain.
- If the answer is not available, answer unknown.

Final answer:""".strip()


def build_compact_prompt(question: str, sst_text: str) -> str:
    return (f"Scene: {sst_text}\nQ: {question}\n"
            "Answer only with the final short answer. If unknown, say unknown.\nA:")


def build_question_only_prompt(question: str) -> str:
    """Language-prior baseline: the LLM answers from the question alone, with
    no scene information. Accuracy above this floor is the genuine
    contribution of the SST scene representation."""
    return (
        "Answer the following visual question as best you can.\n"
        "You do not have access to the image.\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "- Give only the final answer.\n"
        "- Do not explain.\n"
        "- If you cannot answer, say unknown.\n\n"
        "Final answer:"
    )


def build_method_prompt(sample: dict, method: str):
    """Return ``(prompt, sst_prompt)`` for the given method."""
    sst = sample["sst"]
    if method == "full_sst":
        sp = sst_to_prompt_structured(make_full_sst(sst))
        p = build_standard_prompt(sample["question"], sp)
    elif method == "caption_style_sst":
        sp = sst_to_prompt_caption(make_full_sst(sst))
        p = build_standard_prompt(sample["question"], sp)
    elif method == "keyword_aware_sst":
        sp = sst_to_prompt_structured(keyword_aware_sst_filter(sample))
        p = build_standard_prompt(sample["question"], sp)
    elif method == "compact_keyword_sst":
        sp = sst_to_prompt_compact(keyword_aware_sst_filter(sample))
        p = build_compact_prompt(sample["question"], sp)
    elif method == "no_relations_sst":
        sp = sst_to_prompt_structured(make_no_relations_sst(sst))
        p = build_standard_prompt(sample["question"], sp)
    elif method == "no_attributes_sst":
        sp = sst_to_prompt_structured(make_no_attributes_sst(sst))
        p = build_standard_prompt(sample["question"], sp)
    elif method == "objects_only_sst":
        sp = sst_to_prompt_structured(make_objects_only_sst(sst))
        p = build_standard_prompt(sample["question"], sp)
    elif method == "question_only":
        sp = "no scene information"
        p = build_question_only_prompt(sample["question"])
    else:
        raise ValueError(f"Unknown method: {method}")
    return p, sp
