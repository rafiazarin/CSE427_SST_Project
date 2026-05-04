# Token-Efficient Multimodal Reasoning via Scene-to-Structured-Text (SST) Compression

## Overview

This project studies whether images can be replaced with compact structured semantic text representations for visual reasoning tasks.

Instead of feeding raw images directly into a vision-language model, the project converts visual information into a structured text format called **Scene-to-Structured-Text (SST)**. SST represents an image using objects, attributes, relations, counts, and visible text.

The main goal is to evaluate whether this compressed semantic representation can preserve visual reasoning accuracy while reducing token usage.

---

## Research Question

Can structured semantic text representations preserve visual reasoning accuracy while substantially reducing token usage compared with richer visual or semantic inputs?

---

## Motivation

Vision-language models process images using dense visual tokens. This can increase:

- computational cost
- memory usage
- context window usage
- inference latency

However, many visual reasoning questions depend mainly on semantic information rather than raw pixel-level detail.

For example, to answer a question such as:

> What color is the bus?

The model may only need to know:

```text
object: bus
attribute: bus is red
```

This motivates the idea of replacing dense visual input with a compact structured semantic representation.

---

## Core Idea

Traditional direct vision-language model pipeline:

```text
Image + Question → Vision-Language Model → Answer
```

Proposed SST-based pipeline:

```text
Scene Graph / SST + Question → Language Model → Answer
```

Question-aware SST pipeline:

```text
Filtered SST + Question → Language Model → Answer
```

---

## Dataset

This project uses the **GQA dataset**.

GQA is a visual reasoning dataset built from Visual Genome scene graphs. It is well suited for this project because it contains:

- image-question-answer examples
- scene graphs
- objects
- object attributes
- relations between objects
- compositional reasoning questions
- functional programs for questions

The current SST experiments use GQA scene graphs to construct structured semantic text representations.

---

## SST Representation

An SST representation contains structured fields such as:

```json
{
  "objects": ["bus", "person", "stop_sign"],
  "counts": ["person: 3"],
  "attributes": ["bus: red"],
  "relations": ["person near bus", "stop_sign right of bus"],
  "text": ["STOP"]
}
```

---

## Methods Compared

### 1. Direct VLM Baseline

```text
Image + Question → LLaVA → Answer
```

This baseline tests how a local vision-language model performs when given the actual image and question.

### 2. Full SST

```text
Full Scene Graph SST + Question → Mistral → Answer
```

This method uses the full structured semantic representation derived from the GQA scene graph.

### 3. Keyword-Aware SST

```text
Question-Relevant SST + Question → Mistral → Answer
```

This method filters the SST based on keywords from the question. It keeps objects, attributes, counts, and relations most relevant to the question.

### 4. Compact Keyword-Aware SST

This method uses a more compressed serialization format with shorter field names such as:

```text
obj:
cnt:
attr:
rel:
```

This is used as an aggressive compression ablation.

### 5. Ablation Variants

Additional ablations test which semantic components matter most:

- Objects-only SST
- No-relations SST
- No-attributes SST
- Caption-style SST

These help determine whether objects, attributes, relations, and readable structure are important for reasoning.

---

## Current Experimental Setup

The current SST experiments use:

```text
GQA Scene Graph → SST → Mistral
```

This means the current SST branch is an **oracle-SST evaluation**.

It does **not yet** use a fully automatic pipeline such as:

```text
Raw Image → Object Detection / OCR / Relation Extraction → SST
```

This limitation is important. The current results should be interpreted as a semantic-compression study, not as a complete automatic image-to-SST system.

---

## Main Results on 200 Balanced GQA Samples

| Method | Exact Accuracy | Lenient Accuracy | Avg Tokens | Token Reduction |
|---|---:|---:|---:|---:|
| Full SST | 54.5% | 58.5% | 441.44 | 0.00% |
| Caption-Style SST | 53.0% | 58.5% | 443.96 | -0.57% |
| Keyword-Aware SST | 50.0% | 57.0% | 215.53 | 51.18% |
| Compact Keyword SST | 41.0% | 52.0% | 175.10 | 60.33% |
| No-Relations SST | 43.0% | 43.0% | 153.79 | 65.16% |
| No-Attributes SST | 40.0% | 43.0% | 403.84 | 8.52% |
| Objects-Only SST | 22.5% | 24.0% | 106.24 | 75.93% |

---

## Key Finding

Keyword-aware SST reduced average input tokens by approximately **51.18%** compared with Full SST while preserving most reasoning accuracy.

The exact accuracy was:

```text
Full SST:           54.5%
Keyword-Aware SST:  50.0%
```

This is a drop of only **4.5 percentage points** while reducing token usage by about half.

A paired McNemar-style comparison did not find this difference statistically significant at `p < 0.05`.

---

## Statistical Comparison

McNemar-style paired tests against Full SST:

| Method Compared to Full SST | Accuracy | p-value | Interpretation |
|---|---:|---:|---|
| Caption-Style SST | 53.0% | 0.549 | Not significantly worse |
| Keyword-Aware SST | 50.0% | 0.078 | Not significantly worse at p < 0.05 |
| Compact Keyword SST | 41.0% | 0.000025 | Significantly worse |
| No-Relations SST | 43.0% | 0.002667 | Significantly worse |
| No-Attributes SST | 40.0% | 0.000009 | Significantly worse |
| Objects-Only SST | 22.5% | approximately 0 | Significantly worse |

---

## Interpretation

The results suggest that structured semantic text can preserve much of the reasoning signal needed for GQA-style visual reasoning.

The strongest result so far is:

```text
Keyword-aware SST gives about 51% token reduction with only a small accuracy drop.
```

More aggressive compression methods reduce tokens further but hurt accuracy.

This suggests that:

- objects alone are not enough
- relations matter
- attributes matter
- readable structure matters
- question-aware filtering can remove irrelevant information efficiently

---

## LLaVA Baseline

A direct local LLaVA baseline is being added:

```text
Image + Question → LLaVA → Answer
```

This baseline provides a direct image-based comparison against SST-based reasoning.

The LLaVA baseline allows comparison across:

- answer accuracy
- inference latency
- image-based reasoning behavior
- SST-based reasoning behavior
- failure cases

Current status:

- 200-image subset downloaded successfully
- LLaVA 5-sample smoke test completed
- Full 200-sample LLaVA evaluation in progress

---

## Clean Research Claim

In an oracle-SST evaluation on 200 balanced GQA validation examples, Keyword-Aware SST reduced average input tokens by 51.18% compared with Full SST while preserving most reasoning accuracy. Its exact accuracy was 50.0% compared with 54.5% for Full SST, and the paired McNemar-style test did not find a statistically significant difference at `p < 0.05`.

More aggressive compression and component ablations caused statistically significant accuracy drops, indicating that structured semantic content and readable organization are important for visual reasoning.

---

## What This Project Does Not Claim

This project does **not** claim that SST fully replaces all vision-language models.

This project does **not** claim that the current SST system can automatically process arbitrary raw images.

The current SST results are based on ground-truth GQA scene graphs, so they should be understood as an oracle semantic-compression study.

---

## Limitations

Current limitations include:

- SST is built from GQA ground-truth scene graphs.
- The pipeline is not yet fully automatic from raw images.
- The sample size is currently 200 balanced validation examples.
- Results may depend on the local LLM and prompting strategy.
- GQA-style questions may not represent all visual reasoning tasks.
- Scene graphs may contain information that an automatic detector would miss.
- Direct token comparison between image tokens and text tokens is approximate because VLM visual token accounting differs by model.

---

## Future Work

Important future directions include:

1. Build a fully automatic SST pipeline:

```text
Image → Object Detection + OCR + Relation Extraction → SST
```

2. Add object detection using YOLO or Detectron2.

3. Add OCR using EasyOCR or a similar OCR system.

4. Add relation extraction using bounding-box geometry or a learned relation model.

5. Scale evaluation beyond 200 samples.

6. Compare against stronger VLMs such as BLIP-2, InstructBLIP, or larger multimodal models.

7. Perform deeper error analysis by question type:

- spatial reasoning
- counting
- object recognition
- attribute reasoning
- relation reasoning
- yes/no questions

8. Study hybrid methods that combine small visual inputs with SST.

---

## Project Structure

```text
data/
  processed/
    real_gqa_sst_50.json
    real_gqa_sst_200_balanced.json

notebooks/
  week3_eda_and_models.ipynb

outputs/
  results/
    real_gqa_200_publish_methods_detailed.csv
    real_gqa_200_publish_methods_summary.csv
    final_200_sample_report_table.csv
    final_200_sample_accuracy_bootstrap_ci.csv
    final_200_sample_paired_comparison_vs_full_sst.csv
    final_200_sample_semantic_type_table.csv
    final_200_sample_mcnemar_vs_full_sst.csv
    llava_5_sample_smoke_test.csv

src/
  reusable project code

paper/
  project summary and report files
```

---

## Recommended Lightweight Project Bundle

The full local project may be very large because it contains raw GQA data and image files.

For sharing, uploading, or archiving, keep only the lightweight research bundle:

```text
data/processed/
outputs/results/
notebooks/final_experiments.ipynb
paper/
README.md
requirements.txt
```

Do not include:

```text
large raw image zips
full GQA image folders
temporary cache files
duplicate notebooks
```

---


---

## Main Takeaway

Structured semantic compression can substantially reduce token usage for visual reasoning tasks while preserving much of the accuracy, especially when the SST is filtered using question-aware keyword matching.

The current evidence supports SST as a promising efficiency-oriented research direction, but the oracle scene-graph setup must be extended to a real automatic image-to-SST pipeline before making broader claims.