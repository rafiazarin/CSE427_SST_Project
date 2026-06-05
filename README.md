# Token-Efficient Visual Reasoning via Scene-to-Structured-Text (SST) Compression

Can structured semantic text replace visual tokens in vision-language models without sacrificing reasoning accuracy?

This project evaluates replacing image inputs to VLMs with compact structured text representations — objects, attributes, spatial relations, and visible text derived from scene graphs — and measures how much token compression is possible before reasoning accuracy degrades.

---

## Key Finding

**Keyword-Aware SST achieves 51% token reduction with only a 4.5 pp accuracy drop versus Full SST — a difference that is not statistically significant (McNemar p = 0.078).**

More aggressive compression and component ablations cause significant accuracy drops, revealing which parts of a scene representation actually drive visual reasoning.

---

## Results

Evaluated on 200 balanced GQA validation samples. All SST methods use Mistral 7B; LLaVA 7B serves as the image-based baseline.

| Method | Exact Acc. | Lenient Acc. | Avg Tokens | vs Full SST |
|---|---|---|---|---|
| LLaVA 7B (image) | 59.8% | 64.5% | 576 visual | — |
| Full SST | 54.5% | 58.5% | 441 | baseline |
| Caption-Style SST | 53.0% | 58.5% | 444 | −1.5 pp |
| **Keyword-Aware SST** | **50.0%** | **57.0%** | **216** | **−4.5 pp *** |
| Compact Keyword SST | 41.0% | 52.0% | 175 | −13.5 pp ✗ |
| No-Relations SST | 43.0% | 43.0% | 154 | −11.5 pp ✗ |
| No-Attributes SST | 40.0% | 43.0% | 404 | −14.5 pp ✗ |
| Objects-Only SST | 22.5% | 24.0% | 106 | −32.0 pp ✗ |

`*` not statistically significant at p < 0.05 (McNemar paired test)  
`✗` statistically significant degradation

**Compression ratio:** Keyword-Aware SST uses ~2.0× fewer tokens than Full SST and ~2.7× fewer than LLaVA's visual token count.

---

## What the Ablation Reveals

- **Objects alone are not enough** — 22.5% vs 54.5% with full SST
- **Attributes are the most informative field** — removing them causes the largest single drop
- **Relations matter** — spatial information contributes significantly to reasoning accuracy
- **Question-aware filtering** can safely remove ~half the scene tokens with negligible loss
- **Readable structure** matters — aggressive serialization hurts more than token count alone suggests

---

## Approach

### Standard VLM pipeline
```
Image → Visual Encoder (576 tokens) → LLM → Answer
```

### SST pipeline
```
Image → Scene Graph → Structured Text (~106–444 tokens) → LLM → Answer
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
├── data/processed/               # Preprocessed GQA samples
├── outputs/results/              # Result CSVs and figures
├── paper/                        # Full paper (LaTeX)
└── src/                          # Shared utilities
```

---

## Limitations

Current results use GQA ground-truth scene graphs, making this an **oracle ceiling analysis** — an upper bound on what structured semantic compression can achieve. The detected-SST pipeline (YOLO + colour + OCR) closes part of this gap but not all of it; that gap is studied explicitly in `sst_eval_vqav2.ipynb`. Evaluations use locally-run 7B models, which are weaker than frontier VLMs.
