"""Local LLM access via Ollama.

Importing this module never fails just because ``ollama`` is missing — that is
checked explicitly via :func:`require_ollama` so evaluation scripts can fail
loudly *before* spending hours producing empty predictions.
"""

from __future__ import annotations

import time

try:  # pragma: no cover - depends on optional install
    import ollama
except Exception:  # pragma: no cover
    ollama = None


def ollama_available() -> bool:
    return ollama is not None


def require_ollama() -> None:
    """Raise immediately if Ollama is not importable.

    Call this once before an evaluation run. Previously a missing ``ollama``
    package was swallowed per-row, silently recording every prediction as an
    (incorrect) empty string and poisoning the aggregate accuracy.
    """
    if ollama is None:
        raise RuntimeError(
            "The 'ollama' package is not installed (pip install ollama) or the "
            "Ollama server is unreachable. Refusing to start an evaluation that "
            "would record every row as a failure. Install Ollama and pull the "
            "required models (e.g. `ollama pull mistral`)."
        )


def ollama_query(prompt: str, model: str = "mistral"):
    """Query a local Ollama model. Returns ``(answer_string, latency_ms)``."""
    if ollama is None:
        raise RuntimeError("ollama package is not installed. Run: pip install ollama")
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 20},
    )
    ms = (time.perf_counter() - t0) * 1000
    return resp["message"]["content"].strip(), ms
