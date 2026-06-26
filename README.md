# Token-Efficient Visual Reasoning via Scene-to-Structured-Text (SST) Compression

Can structured semantic text replace visual tokens in vision-language models without sacrificing reasoning accuracy?

This project evaluates replacing image inputs to VLMs with compact structured text representations — objects, attributes, spatial relations, and visible text derived from scene graphs — and measures how much token compression is possible before reasoning accuracy degrades.

---

## Key Finding

**Keyword-Aware SST matches Full SST accuracy (57.0% vs 56.5% exact) while cutting input tokens by ~46% — a difference that is not statistically significant (McNemar p = 0.747).**

More aggressive compression and component ablations cause large, statistically significant accuracy drops, revealing which parts of a scene representation actually drive visual reasoning.

---

## Results

Evaluated on 1,000 balanced GQA validation questions (the same paired set across every variant). All SST methods use Mistral 7B; LLaVA 7B serves as the image-based baseline. Numbers below are reproduced directly from `outputs/results/gqa_15000_sst_eval_summary.csv` (regenerate with `python scripts/recompute_results.py`).

| Method | Exact Acc. | Lenient Acc. | Avg Tokens | vs Full SST |
|---|---|---|---|---|
| LLaVA 7B (image) | 59.8% | 59.9% | 576 visual | — |
| Full SST | 56.5% | 58.7% | 289 | baseline |
| Caption-Style SST | 56.0% | 58.9% | 287 | −0.5 pp |
| **Keyword-Aware SST** | **57.0%** | **60.4%** | **157** | **+0.5 pp *** |
| Compact Keyword SST | 45.5% | 51.5% | 116 | −11.0 pp ✗ |
| No-Relations SST | 44.2% | 46.1% | 155 | −12.3 pp ✗ |
| No-Attributes SST | 15.0% | 16.6% | 249 | −41.5 pp ✗ |
| Objects-Only SST | 2.5% | 2.9% | 105 | −54.0 pp ✗ |

`*` not statistically significant vs Full SST at p < 0.05 (McNemar paired test, p = 0.747)  
`✗` statistically significant degradation (McNemar p < 0.001)

**Compression ratio:** Keyword-Aware SST uses ~1.8× fewer tokens than Full SST and **~3.67× fewer** than LLaVA's 576 visual-token reference, while landing within 2.8 pp of LLaVA's accuracy. (Token counts are text subword tokens; see the note in `src/sst_eval/normalize.py` on comparing these to visual tokens.)

---

## What the Ablation Reveals

- **Objects alone are not enough** — 2.5% vs 56.5% with full SST
- **Attributes are the most informative field** — removing them causes the largest single drop (56.5% → 15.0%)
- **Relations matter** — removing them drops accuracy to 44.2%
- **Question-aware filtering** can safely remove ~half the scene tokens with no significant loss
- **Readable structure** matters — aggressive serialization (Compact Keyword SST) hurts more than token count alone suggests

---

## Approach

### Standard VLM pipeline
```
Image → Visual Encoder (576 tokens) → LLM → Answer
```

### SST pipeline
```
Image → Scene Graph → Structured Text (~105–289 tokens) → LLM → Answer
```

An SST representation looks like:
```json
{
  "objects":    ["bus", "person", "stop_sign"],
  "attributes": ["bus: red", "person: standing"],
  "relations":  ["person near bus", "stop_sign right of bus"],
  "counts":     ["3 persons"],
  "text":       ["STOP"]
}
```

The **keyword-aware variant** filters fields by question-relevant terms before constructing the prompt, cutting irrelevant scene content while preserving reasoning signal.

---

## Evaluation Pipeline

Three notebooks cover the full evaluation:

**`sst_eval_main.ipynb`** — Core ablation on GQA  
8 SST variants + LLaVA baseline + question-only language-prior floor. Includes McNemar significance tests, bootstrap CIs, latency measurements, and error type breakdown (colour, spatial, counting, existence, attribute, action).

**`sst_eval_blip2.ipynb`** — BLIP-2 OPT-2.7B baseline  
Second VLM comparison point. BLIP-2's Q-Former compresses vision to 32 query tokens — an interesting contrast to both LLaVA (576 tokens) and SST text representations.

**`sst_eval_vqav2.ipynb`** — VQAv2 generalization  
Tests the fully automatic detected-SST pipeline (YOLO + colour extraction + EasyOCR) on VQAv2/COCO — no oracle scene graphs. Measures how much performance transfers to a different benchmark and a real-world detection setup.

---

## Reproduction

**Requirements**
```bash
pip install -r requirements.txt
```

**Ollama** (for Mistral and LLaVA inference):
```bash
ollama pull mistral && ollama pull llava
ollama serve
```

**Data:** GQA from [cs.stanford.edu/people/dorarad/gqa](https://cs.stanford.edu/people/dorarad/gqa/download.html) → `data/gqa_raw/`. VQAv2 download instructions are in `sst_eval_vqav2.ipynb`.

---

## Project Structure

```
├── notebooks/
│   ├── sst_eval_main.ipynb       # GQA ablation — core results
│   ├── sst_eval_blip2.ipynb      # BLIP-2 baseline
│   └── sst_eval_vqav2.ipynb      # VQAv2 generalization
├── src/sst_eval/                 # Shared package: normalisation, SST builders,
│                                 #   keyword filter, prompts, Ollama client, stats
├── scripts/
│   ├── recompute_results.py      # Rebuild result CSVs from per-row data
│   └── regenerate_figures.py     # Redraw figures from the corrected CSVs
├── tests/                        # Unit tests for the shared helpers
├── data/processed/               # Preprocessed GQA samples
├── outputs/results/              # Result CSVs and figures
└── paper/                        # Full paper (LaTeX)
```

The notebooks import every shared helper from `src/sst_eval` rather than
re-defining them, so the evaluation logic has a single source of truth. If the
result CSVs are ever regenerated from raw per-row output, run
`python scripts/recompute_results.py && python scripts/regenerate_figures.py`.

---

## Limitations

Current results use GQA ground-truth scene graphs, making this an **oracle ceiling analysis** — an upper bound on what structured semantic compression can achieve. The detected-SST pipeline (YOLO + colour + OCR) closes part of this gap but not all of it; that gap is studied explicitly in `sst_eval_vqav2.ipynb`. Evaluations use locally-run 7B models, which are weaker than frontier VLMs.
