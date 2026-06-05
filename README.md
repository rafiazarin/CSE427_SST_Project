# Token-Efficient Visual Reasoning via Scene-to-Structured-Text (SST) Compression

Can structured semantic text replace dense visual tokens in vision-language models without sacrificing reasoning accuracy?

This project investigates replacing image inputs to VLMs with compact, structured text representations derived from scene graphs and automatic image analysis — evaluating the accuracy–efficiency tradeoff on standard VQA benchmarks.

---

## Core Idea

Standard VLM pipeline:
```
Image → Visual Encoder (576 tokens) → LLM → Answer
```

SST pipeline:
```
Image → Scene Graph / Detector → Structured Text (~150–440 tokens) → LLM → Answer
```

Instead of feeding raw visual tokens, we represent the image as a structured semantic text (SST) containing objects, attributes, spatial relations, counts, and visible text — then query a language model directly.

---

## Results (Preliminary — 200 GQA samples)

| Method | Exact Acc. | Lenient Acc. | Avg Tokens | Token Reduction |
|---|---|---|---|---|
| Full SST | 54.5% | 58.5% | 441 | — |
| Caption-Style SST | 53.0% | 58.5% | 444 | −0.6% |
| **Keyword-Aware SST** | **50.0%** | **57.0%** | **216** | **51.2%** |
| Compact Keyword SST | 41.0% | 52.0% | 175 | 60.3% |
| No-Relations SST | 43.0% | 43.0% | 154 | 65.2% |
| No-Attributes SST | 40.0% | 43.0% | 404 | 8.5% |
| Objects-Only SST | 22.5% | 24.0% | 106 | 75.9% |
| Question-Only (no scene) | — | — | — | — |
| LLaVA 7B (image baseline) | — | — | 576 visual | — |
| BLIP-2 OPT-2.7B | — | — | 32 (Q-Former) | — |

> Large-scale evaluation (15,000 GQA samples, VQAv2 generalization, BLIP-2 baseline) is currently running. This table will be updated with full results.

**Key finding so far:** Keyword-Aware SST achieves ~51% token reduction vs. Full SST with only a 4.5 pp accuracy drop — a difference that paired McNemar tests found non-significant at p < 0.05.

---

## What This Is (and Isn't)

**This is:** A ceiling analysis of how well structured semantic representations can preserve visual reasoning signal, plus a real-world detector-based pipeline using YOLO + colour extraction + EasyOCR.

**This is not:** A claim that SST fully replaces VLMs in production. The oracle-SST results represent an upper bound; the detector-based (real-world) pipeline has a larger accuracy gap, which is studied explicitly.

---

## Project Structure

```
CSE427_SST_project/
├── notebooks/
│   ├── sst_eval_main.ipynb        # Core SST ablation study on GQA (15k samples)
│   ├── sst_eval_vqav2.ipynb       # Generalization benchmark — VQAv2 / COCO
│   ├── sst_eval_blip2.ipynb       # BLIP-2 OPT-2.7B baseline comparison
│   └── archive/
│       └── sec10_group11_...ipynb # Original course project notebook
├── data/
│   ├── gqa_raw/                   # GQA questions + scene graphs (not tracked)
│   ├── gqa_images_subset/         # GQA image subset (not tracked)
│   └── processed/                 # Preprocessed sample JSONs
├── outputs/
│   └── results/                   # CSVs and figures from evaluation runs
├── paper/
│   └── cse427_sst_report.tex      # Full paper (LaTeX)
├── src/                           # Shared utilities (TODO: refactor from notebooks)
├── requirements.txt
└── README.md
```

---

## Notebooks

### `sst_eval_main.ipynb`
The primary evaluation notebook. Covers:
- 8 SST variants (ablation study over objects, attributes, relations, question-aware filtering)
- LLaVA 7B image baseline
- Question-only language-prior baseline
- McNemar paired significance tests + bootstrap confidence intervals
- Error type analysis (colour, spatial, counting, existence, attribute, action)
- Latency and token compression measurements

### `sst_eval_vqav2.ipynb`
Generalization study on VQAv2 (COCO val2014). Key differences:
- No oracle scene graphs — uses fully automatic detected SST (YOLO + colour + EasyOCR)
- VQAv2 soft accuracy metric (official)
- Balanced sampling across answer types (yes/no, number, other)
- Cross-benchmark comparison plot vs. GQA results

### `sst_eval_blip2.ipynb`
BLIP-2 OPT-2.7B as a second VLM baseline:
- Runs on the exact same GQA question IDs as the main evaluation
- BLIP-2 Q-Former compresses vision to 32 query tokens (vs. LLaVA's 576)
- 3-way comparison: BLIP-2 vs. LLaVA vs. best SST method vs. question-only floor

---

## Reproduction

### Requirements
```bash
pip install -r requirements.txt
```

You will also need [Ollama](https://ollama.ai) running locally with `mistral` and `llava` pulled:
```bash
ollama pull mistral
ollama pull llava
ollama serve
```

For BLIP-2, additional dependencies:
```bash
pip install torch torchvision transformers accelerate Pillow
```

### Data
Download GQA from [https://cs.stanford.edu/people/dorarad/gqa/download.html](https://cs.stanford.edu/people/dorarad/gqa/download.html) and place under `data/gqa_raw/`. For VQAv2, see the download cell in `sst_eval_vqav2.ipynb`.

### Run order
```
sst_eval_main.ipynb   →   sst_eval_blip2.ipynb   →   sst_eval_vqav2.ipynb
```
`sst_eval_blip2` and `sst_eval_vqav2` read from `sst_eval_main` checkpoints for aligned question IDs.

---

## Ablation Takeaways

- **Attributes are the most important field** — removing them causes the largest accuracy drop
- **Objects alone are insufficient** (22.5% vs. 54.5%)
- **Relations matter** — removing them causes a significant drop (McNemar p = 0.003)
- **Question-aware filtering** halves token count with negligible accuracy loss
- **Readable structure** matters — compact serialization hurts more than expected

---

## Citation / Reference

If you use this work, please cite it as:

```
SST: Token-Efficient Visual Reasoning via Scene-to-Structured-Text Compression
[Paper in preparation]
```

---

## Status

| Component | Status |
|---|---|
| GQA oracle SST ablation (200 samples) | ✅ Complete |
| LLaVA baseline | ✅ Complete |
| Large-scale GQA run (15k samples) | 🔄 In progress |
| Question-only baseline | 🔄 In progress |
| BLIP-2 baseline | 🔄 In progress |
| VQAv2 generalization | 🔄 In progress |
| Automatic detector-based SST (real-world) | 🔄 In progress |
| ArXiv preprint | 📋 Planned |
