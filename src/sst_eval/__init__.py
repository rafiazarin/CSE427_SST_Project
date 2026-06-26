"""Shared utilities for the SST (Scene-to-Structured-Text) evaluation project.

Single source of truth for the helpers that the three evaluation notebooks
used to duplicate. Import what you need, e.g.::

    from sst_eval import normalize_answer, build_method_prompt, METHODS

The notebooks add ``src/`` to ``sys.path`` and then ``from sst_eval import *``.
"""

from __future__ import annotations

from .keyword import (
    STOPWORDS,
    SYNONYMS,
    extract_terms,
    item_matches,
    keyword_aware_sst_filter,
    normalize_term,
    obj_name_from_fact,
)
from .llm import ollama_available, ollama_query, require_ollama
from .normalize import (
    ARTICLES,
    NUMBER_WORDS,
    TOKEN_ENCODER,
    count_tokens,
    is_lenient_correct,
    normalize_answer,
)
from .prompts import (
    METHODS,
    build_compact_prompt,
    build_method_prompt,
    build_question_only_prompt,
    build_standard_prompt,
)
from .sst import (
    LOW_VALUE_OBJECTS,
    SST_FIELDS,
    WEAK_RELATIONS,
    get_semantic_type,
    gqa_scene_to_clean_sst,
    make_full_sst,
    make_no_attributes_sst,
    make_no_relations_sst,
    make_objects_only_sst,
    sst_to_prompt_caption,
    sst_to_prompt_compact,
    sst_to_prompt_structured,
    validate_sample,
)
from .stats import bootstrap_ci, exact_mcnemar_p, mcnemar_from_pairs

# Human-readable names used in summary tables and figures.
METHOD_NAMES = {
    "full_sst": "Full SST",
    "caption_style_sst": "Caption-Style SST",
    "keyword_aware_sst": "Keyword-Aware SST",
    "compact_keyword_sst": "Compact Keyword SST",
    "no_relations_sst": "No-Relations SST",
    "no_attributes_sst": "No-Attributes SST",
    "objects_only_sst": "Objects-Only SST",
    "question_only": "Question-Only (no scene)",
}

__all__ = [
    "ARTICLES", "NUMBER_WORDS", "TOKEN_ENCODER", "count_tokens",
    "is_lenient_correct", "normalize_answer",
    "STOPWORDS", "SYNONYMS", "extract_terms", "item_matches",
    "keyword_aware_sst_filter", "normalize_term", "obj_name_from_fact",
    "LOW_VALUE_OBJECTS", "SST_FIELDS", "WEAK_RELATIONS", "get_semantic_type",
    "gqa_scene_to_clean_sst", "make_full_sst", "make_no_attributes_sst",
    "make_no_relations_sst", "make_objects_only_sst", "sst_to_prompt_caption",
    "sst_to_prompt_compact", "sst_to_prompt_structured", "validate_sample",
    "METHODS", "METHOD_NAMES", "build_compact_prompt", "build_method_prompt",
    "build_question_only_prompt", "build_standard_prompt",
    "ollama_available", "ollama_query", "require_ollama",
    "bootstrap_ci", "exact_mcnemar_p", "mcnemar_from_pairs",
]
