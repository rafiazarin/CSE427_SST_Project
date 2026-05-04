# Token-Efficient Multimodal Reasoning via Scene-to-Structured-Text Compression

## Abstract

Vision-language models process images using dense visual representations, which can increase computational cost, latency, and context usage. However, many visual reasoning tasks depend primarily on semantic information such as objects, attributes, relations, counts, and visible text. This project investigates whether structured semantic text can preserve visual reasoning accuracy while reducing token usage. We propose Scene-to-Structured-Text (SST), a compact representation of visual scenes derived from GQA scene graphs. Using a balanced 200-sample subset of GQA validation examples, we compare Full SST, Keyword-Aware SST, Compact SST, and several ablations using local Mistral as the reasoning model. We also evaluate a direct image-based LLaVA baseline through Ollama. Keyword-Aware SST reduces average input tokens by 51.18% compared with Full SST while achieving 50.0% exact accuracy compared with 54.5% for Full SST. A local LLaVA baseline achieves 55.5% exact accuracy. Error analysis shows that LLaVA is stronger on object-centric questions, while SST variants show strengths on relation and attribute-oriented questions. These results suggest that structured semantic compression is a promising direction for token-efficient multimodal reasoning, but the current study remains limited by its oracle use of GQA ground-truth scene graphs.

## 1. Introduction

Vision-language models have become increasingly capable at answering questions about images. These systems often process images using dense visual tokens or image embeddings, which can increase computational cost, memory usage, latency, and context window consumption. While this dense visual representation is useful for many tasks, not every visual reasoning problem requires full pixel-level detail.

Many visual question answering tasks depend mainly on semantic information. For example, to answer “What color is the bus?”, the model may only need to know that a bus exists and that its color is red. Similarly, to answer “Is the person near the car?”, the model may only need object and relation information. This motivates the idea of replacing dense visual inputs with compact structured semantic text.

This project studies Scene-to-Structured-Text (SST) compression. SST represents visual information using structured fields such as objects, counts, attributes, relations, and visible text. Instead of giving an image directly to a vision-language model, the system gives this structured text representation to a language model for reasoning.

The main research question is:

Can structured semantic text representations preserve visual reasoning accuracy while substantially reducing token usage compared with richer visual or semantic inputs?

This project evaluates the question using the GQA dataset, which is well suited for compositional visual reasoning because it includes scene graphs, objects, attributes, relations, questions, answers, and semantic question types.

## 2. Related Work

Vision-language models such as VisualBERT, ViLBERT, LXMERT, BLIP-2, Flamingo, and LLaVA combine visual and textual information to perform multimodal reasoning. These models typically represent images using visual features, image patches, or object-region embeddings. While this enables strong performance, it can require many visual tokens and significant computation.

Scene graph research provides another way to represent images. A scene graph describes objects, their attributes, and their relationships. GQA builds on this idea by generating compositional questions from Visual Genome scene graphs. Because GQA questions are grounded in scene graphs, it is particularly suitable for testing whether structured semantic representations can support visual reasoning.

This project differs from standard VLM work by testing whether structured text derived from scene graphs can replace or complement raw visual inputs. It also differs from visual token pruning methods because it does not merely reduce dense visual tokens; instead, it converts semantic scene information into text for LLM-based reasoning.

## 3. Dataset

The project uses the GQA dataset. GQA is a visual reasoning dataset based on Visual Genome scene graphs. It contains image-question-answer examples that test object recognition, attributes, relations, counting, and compositional reasoning.

The project uses:

- GQA validation scene graphs
- GQA balanced validation questions
- A balanced 200-sample subset for the main experiments

The balanced subset contains 200 examples divided across semantic question types. Each example includes an image id, question, ground-truth answer, semantic type, and an SST representation derived from the corresponding GQA scene graph.

## 4. Methodology

### 4.1 Scene-to-Structured-Text Representation

The SST representation stores visual scene information as structured text fields:

- objects
- counts
- attributes
- relations
- visible text / OCR

Example:

{
  "objects": ["bus", "person", "stop_sign"],
  "counts": ["person: 3"],
  "attributes": ["bus: red"],
  "relations": ["person near bus", "stop_sign right of bus"],
  "text": ["STOP"]
}

The current SST is built from GQA ground-truth scene graphs. Therefore, this study is an oracle-SST semantic compression experiment rather than a fully automatic raw-image-to-SST pipeline.

### 4.2 Full SST

Full SST uses the complete cleaned scene-graph-derived representation. It includes all retained objects, counts, attributes, and relations after basic cleaning. No question-specific filtering is applied.

### 4.3 Keyword-Aware SST

Keyword-Aware SST filters the full SST according to terms in the question. It extracts meaningful question terms, matches them to relevant objects, keeps attributes and counts for matched objects, and keeps relations involving matched objects. It also includes connected objects when necessary.

The goal is to reduce irrelevant information while preserving enough semantic structure to answer the question.

### 4.4 Compact Keyword SST

Compact Keyword SST uses the same keyword-aware filtering idea but serializes the result more aggressively using shorter field labels such as obj, cnt, attr, and rel. This method tests whether stronger compression harms reasoning performance.

### 4.5 Ablations

The project also evaluates component ablations:

- No-Relations SST
- No-Attributes SST
- Objects-Only SST
- Caption-Style SST

These ablations test which parts of the structured representation are most important.

### 4.6 Direct VLM Baseline

A local LLaVA model is used as a direct image-based baseline:

Image + Question → LLaVA → Answer

Only the 198 unique images needed for the 200-sample GQA subset were downloaded, avoiding the full GQA image archive. LLaVA was run locally through Ollama.

### 4.7 Reasoning Model

The SST-based methods use local Mistral through Ollama:

SST + Question → Mistral → Answer

The prompt asks the model to answer concisely using the provided structured scene information.

## 5. Experimental Setup

The main experiments use a balanced 200-sample subset of GQA validation examples.

The evaluated methods are:

1. LLaVA Image + Question
2. Full SST + Mistral
3. Caption-Style SST + Mistral
4. Keyword-Aware SST + Mistral
5. Compact Keyword SST + Mistral
6. No-Relations SST + Mistral
7. No-Attributes SST + Mistral
8. Objects-Only SST + Mistral

Evaluation metrics include:

- Exact-match accuracy
- Lenient accuracy
- Average input text tokens for SST methods
- Token reduction relative to Full SST
- Latency
- Paired statistical comparison
- Error analysis by semantic type

Token counts are computed for text prompts using tiktoken. Direct comparison between text tokens and VLM visual tokens is approximate because local LLaVA visual token accounting is not directly exposed.

## 6. Results

### 6.1 SST Results

| Method | Exact Accuracy | Lenient Accuracy | Avg Tokens | Token Reduction vs Full SST |
|---|---:|---:|---:|---:|
| Full SST | 54.5% | 58.5% | 441.44 | 0.00% |
| Caption-Style SST | 53.0% | 58.5% | 443.96 | -0.57% |
| Keyword-Aware SST | 50.0% | 57.0% | 215.53 | 51.18% |
| Compact Keyword SST | 41.0% | 52.0% | 175.10 | 60.33% |
| No-Relations SST | 43.0% | 43.0% | 153.79 | 65.16% |
| No-Attributes SST | 40.0% | 43.0% | 403.84 | 8.52% |
| Objects-Only SST | 22.5% | 24.0% | 106.24 | 75.93% |

Keyword-Aware SST reduced token usage by 51.18% while dropping exact accuracy by only 4.5 percentage points compared with Full SST.

### 6.2 LLaVA Baseline

| Method | Exact Accuracy | Lenient Accuracy | Avg Latency |
|---|---:|---:|---:|
| LLaVA Image + Question | 55.5% | 56.5% | 5.74 sec |

LLaVA achieved the highest exact accuracy overall, but the gap between LLaVA and Full SST was small on the 200-sample subset.

### 6.3 Final Comparison

| Method | Input Type | Model | Exact Accuracy | Lenient Accuracy | Avg Tokens | Avg Latency |
|---|---|---|---:|---:|---:|---:|
| LLaVA Image + Question | Raw image + question | LLaVA | 55.5% | 56.5% | N/A | 5.74 sec |
| Full SST | Oracle SST text | Mistral | 54.5% | 58.5% | 441.44 | N/A |
| Keyword-Aware SST | Oracle SST text | Mistral | 50.0% | 57.0% | 215.53 | N/A |

The results suggest that SST-based reasoning can be competitive with a local image-based VLM baseline in this oracle setting, while Keyword-Aware SST substantially reduces text token usage.

## 7. Statistical Analysis

A paired McNemar-style comparison was performed against Full SST.

| Method Compared to Full SST | Exact Accuracy | p-value | Interpretation |
|---|---:|---:|---|
| Caption-Style SST | 53.0% | 0.549 | Not significantly worse |
| Keyword-Aware SST | 50.0% | 0.078 | Not significantly worse at p < 0.05 |
| Compact Keyword SST | 41.0% | 0.000025 | Significantly worse |
| No-Relations SST | 43.0% | 0.002667 | Significantly worse |
| No-Attributes SST | 40.0% | 0.000009 | Significantly worse |
| Objects-Only SST | 22.5% | approximately 0 | Significantly worse |

The Keyword-Aware SST result is important because it shows substantial compression without a statistically significant drop relative to Full SST at the 0.05 level.

## 8. Error Analysis

### 8.1 LLaVA vs Keyword-Aware SST

| Category | Count |
|---|---:|
| Both correct | 62 |
| Both wrong | 51 |
| LLaVA only correct | 49 |
| Keyword-Aware SST only correct | 38 |

This shows that the two approaches have complementary strengths. Keyword-Aware SST correctly answered 38 examples that LLaVA missed, while LLaVA correctly answered 49 examples that Keyword-Aware SST missed.

### 8.2 Semantic Type Analysis

| Type | n | LLaVA Accuracy | Keyword SST Accuracy | Full SST Accuracy |
|---|---:|---:|---:|---:|
| attr | 40 | 47.5% | 55.0% | 52.5% |
| cat | 40 | 60.0% | 70.0% | 77.5% |
| global | 40 | 45.0% | 40.0% | 42.5% |
| obj | 40 | 82.5% | 27.5% | 32.5% |
| rel | 40 | 42.5% | 57.5% | 67.5% |

LLaVA is strongest on object-centric questions, while SST variants perform better on some attribute and relation questions. This suggests that structured scene information may be especially useful for semantic and relational reasoning, while raw image models are stronger for direct object recognition.

## 9. Discussion

The results support the idea that structured semantic text can preserve much of the reasoning information needed for GQA-style visual reasoning. Full SST performs close to LLaVA, and Keyword-Aware SST reduces token usage by more than half while maintaining competitive accuracy.

However, the results also show that aggressive compression can hurt performance. Objects-Only SST performs poorly, indicating that object names alone are insufficient. Removing relations and attributes also causes large drops, showing that structured semantic components are important.

The comparison with LLaVA suggests that SST should not be framed as a complete replacement for vision-language models. Instead, SST may be useful as a semantic compression layer or hybrid reasoning component. LLaVA and SST make different errors, suggesting potential for combining visual and structured semantic inputs.

## 10. Limitations

The most important limitation is that SST is currently built from GQA ground-truth scene graphs. Therefore, the current pipeline does not automatically extract SST from raw images. This makes the SST branch an oracle semantic-compression study.

Other limitations include:

- Evaluation uses only 200 balanced GQA validation examples.
- Local Mistral and LLaVA may not represent the strongest available models.
- Prompt design may affect results.
- Token comparisons are exact for SST text but only approximate relative to VLM visual token usage.
- GQA-style reasoning may not generalize to all visual reasoning tasks.
- Automatic extraction errors are not measured because YOLO/OCR/relation extraction are not yet implemented.

## 11. Future Work

Future work should build a fully automatic SST pipeline:

Image → Object Detection + OCR + Relation Extraction → SST

Possible components include:

- YOLOv8 or Detectron2 for object detection
- EasyOCR for visible text extraction
- Bounding-box heuristics for spatial relations
- Color or attribute extraction
- Learned relation extraction models

Future experiments should compare:

1. Direct LLaVA image baseline
2. Oracle SST
3. Automatic SST
4. Keyword-Aware Automatic SST
5. Hybrid image + SST models

Scaling the evaluation to 1,000 or more examples and adding stronger VLM baselines would also make the results more robust.

## 12. Conclusion

This project evaluated Scene-to-Structured-Text compression for token-efficient multimodal reasoning. In an oracle GQA scene-graph setting, Keyword-Aware SST reduced average input tokens by 51.18% compared with Full SST while preserving much of the reasoning accuracy. A direct LLaVA image baseline achieved slightly higher exact accuracy, but error analysis showed that SST and LLaVA have complementary strengths.

The results suggest that structured semantic compression is a promising direction for efficient visual reasoning. However, the current system should be viewed as an oracle semantic-compression study. A fully automatic image-to-SST pipeline is needed before making broader claims about replacing raw image inputs in practical VLM systems.

## References

Hudson, D. A., & Manning, C. D. (2019). GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.

Li, L. H., Yatskar, M., Yin, D., Hsieh, C., & Chang, K. (2019). VisualBERT: A Simple and Performant Baseline for Vision and Language.

Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks.

Tan, H., & Bansal, M. (2019). LXMERT: Learning Cross-Modality Encoder Representations from Transformers.

Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning / LLaVA.

Xu, D., Zhu, Y., Choy, C. B., & Fei-Fei, L. (2017). Scene Graph Generation by Iterative Message Passing.