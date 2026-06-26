"""Answer normalisation, lenient matching, and token counting.

These helpers were previously copy-pasted across all three evaluation
notebooks. They live here so there is a single source of truth.

`normalize_answer` is intentionally free of a pandas dependency so this
module can be imported in lightweight contexts (tests, scripts) without
pulling in the scientific stack.
"""

from __future__ import annotations

import re

# ── Optional subword tokenizer ─────────────────────────────────────────────
# tiktoken's cl100k_base is a GPT-family tokenizer. It is used only as a
# stable, locally-available proxy for subword token counts. See
# count_tokens() for the important caveat about comparing these counts to a
# VLM's *visual* token budget.
try:  # pragma: no cover - depends on optional install
    import tiktoken

    TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    TOKEN_ENCODER = None

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
ARTICLES = {"a", "an", "the"}

_UNKNOWN_MARKERS = [
    "unknown", "cannot determine", "can't determine",
    "not enough information", "not available", "does not provide",
    "not provided",
]


def _is_missing(x) -> bool:
    """True for None / NaN-like values without requiring pandas."""
    if x is None:
        return True
    # NaN is the only value that is not equal to itself.
    return isinstance(x, float) and x != x


def normalize_answer(x) -> str:
    """Lower-case, strip boilerplate prefixes/suffixes, map number words,
    drop articles, and collapse unanswerable phrasings to "unknown"."""
    if _is_missing(x):
        return ""
    x = str(x).lower().strip()
    for p in ["final answer:", "answer:", "the answer is", "it is", "it's", "i think"]:
        if x.startswith(p):
            x = x[len(p):].strip()
    x = re.sub(r"[(][^)]*[)]", "", x).strip()
    x = x.split("\n")[0].strip()
    for sep in [" because ", " since ", " based on ", " as "]:
        if sep in x:
            x = x.split(sep)[0].strip()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    tokens = [NUMBER_WORDS.get(t, t) for t in x.split() if t not in ARTICLES]
    joined = " ".join(tokens)
    for unk in _UNKNOWN_MARKERS:
        if unk in joined:
            return "unknown"
    return joined.strip()


def is_lenient_correct(true_answer, pred_answer) -> int:
    """Looser-than-exact match.

    Note: for multi-word ground truth this uses substring containment, which
    can over-credit a verbose prediction. Exact-match accuracy is the primary
    headline metric for that reason; lenient accuracy is a secondary signal.
    """
    t = normalize_answer(true_answer)
    p = normalize_answer(pred_answer)
    if p == t:
        return 1
    if p == "unknown":
        return 0
    if t in {"yes", "no"}:
        return int(p == t)
    pt = set(p.split())
    tt = t.split()
    if len(tt) == 1:
        return int(tt[0] in pt)
    return int(t in p)


def count_tokens(text) -> int:
    """Subword token count via tiktoken when available, else whitespace count.

    Caveat: this counts *text* subword tokens with a GPT-family encoder. It is
    not the same unit as a VLM's CLIP visual tokens (e.g. LLaVA's 576), nor is
    it Mistral's own tokenizer. Use it for relative comparison between SST
    variants; treat any "text token vs. visual token" ratio as approximate.
    """
    if TOKEN_ENCODER is not None:
        return len(TOKEN_ENCODER.encode(str(text)))
    return len(str(text).split())
