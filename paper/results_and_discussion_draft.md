# Results and Discussion Draft

## Experimental Setup

The final evaluation was conducted on a balanced 200-sample subset of the GQA validation split. The project compares two main reasoning settings. The first is a direct vision-language baseline, where LLaVA receives the raw image and question. The second is an oracle Scene-to-Structured-Text (SST) setting, where GQA scene graphs are converted into structured text containing objects, counts, attributes, relations, and text/OCR-like entries, and Mistral answers the question using only this structured representation.

This distinction is important: the SST branch does not yet implement a fully automatic raw image-to-SST pipeline. Instead, it evaluates whether compact structured semantic representations can preserve visual reasoning accuracy when high-quality scene graph information is already available. Therefore, the current SST results should be interpreted as an oracle semantic-compression study rather than a complete deployable vision pipeline.

## Main Accuracy Results

The direct LLaVA image-question baseline achieved 55.5% exact accuracy and 56.5% lenient accuracy. Full SST with Mistral achieved 54.5% exact accuracy and 58.5% lenient accuracy. This shows that the oracle SST representation is competitive with the direct local VLM baseline on this 200-sample GQA subset.

The strongest compressed SST method was Keyword-Aware SST. It achieved 50.0% exact accuracy and 57.0% lenient accuracy while reducing average input tokens from 441.44 to 215.53. This corresponds to a 51.18% token reduction compared with Full SST. The exact accuracy drop from Full SST was 4.5 percentage points, while the lenient accuracy drop was only 1.5 percentage points.

Caption-Style SST achieved 53.0% exact accuracy and 58.5% lenient accuracy, which is close to Full SST. However, it did not reduce token usage; its average token count was 443.96, slightly higher than Full SST. This suggests that converting structured SST into prose can preserve readability, but it does not provide the desired compression benefit.

## Token-Efficiency Tradeoff

The main contribution of the project is the accuracy-compression tradeoff demonstrated by Keyword-Aware SST. Full SST achieved the highest SST exact accuracy, but required 441.44 average input tokens. Keyword-Aware SST reduced this to 215.53 average tokens while preserving most of the reasoning performance.

More aggressive compression produced larger token reductions but caused clear accuracy degradation. Compact Keyword-Aware SST reduced tokens by 60.33%, but exact accuracy dropped to 41.0%. Objects-Only SST achieved the largest token reduction at 75.93%, but exact accuracy collapsed to 22.5%. This confirms that object lists alone are not sufficient for many GQA-style reasoning questions; attributes, relations, and readable structure are important.

## Ablation Analysis

The ablation results show which SST components are most important. Removing relations reduced exact accuracy to 43.0%, and removing attributes reduced exact accuracy to 40.0%. Objects-Only SST performed worst, reaching only 22.5% exact accuracy. These results support the claim that the benefit of SST comes not merely from naming visible objects, but from preserving structured semantic evidence such as attributes and relationships.

The No-Attributes result is especially informative because it retained a relatively high token count of 403.84 tokens but still performed poorly. This indicates that accuracy is not determined only by prompt length; the type and relevance of retained semantic information matter.

## Latency Results

Latency was measured on the local hardware used for the experiment, so it should be interpreted as system-dependent rather than as a universal property of the models. LLaVA averaged 5.74 seconds per query. Full SST with Mistral averaged 6.62 seconds, while Keyword-Aware SST averaged 2.12 seconds. This suggests that reducing SST prompt length can also reduce local inference time, although latency depends on hardware, model loading, Ollama runtime behavior, and prompt length.

## Comparison with Direct VLM Baseline

LLaVA slightly outperformed Full SST in exact accuracy, with 55.5% compared with 54.5%. However, Full SST achieved higher lenient accuracy, with 58.5% compared with LLaVA's 56.5%. These results should not be interpreted as SST replacing direct VLMs. Instead, they show that structured semantic text can be competitive for GQA-style reasoning when high-quality scene graph information is available.

The comparison also has an important measurement limitation: text token counts are meaningful for SST prompts, but the raw image input used by LLaVA is not directly comparable using the same text-token metric. Therefore, the strongest token-efficiency claim should be made within the SST family, especially Full SST versus Keyword-Aware SST.

## Key Finding

The main finding is that Keyword-Aware SST provides the best balance between compression and reasoning performance. On the 200-sample balanced GQA subset, it reduced average input tokens by 51.18% compared with Full SST while retaining most of the accuracy. More aggressive compression methods reduced token usage further but caused substantially larger accuracy drops. This supports the hypothesis that question-aware semantic filtering can remove irrelevant scene information while preserving much of the evidence needed for visual reasoning.

## Limitations

The most important limitation is that the SST pipeline currently uses GQA scene graphs as input. This makes the SST branch an oracle semantic-compression evaluation. A real deployed system would require automatic image-to-SST extraction using object detection, OCR, attribute recognition, and relation extraction. Errors in those extraction stages would likely reduce downstream reasoning accuracy.

The evaluation is also limited to 200 balanced validation examples. This is appropriate for a course-scale local-model experiment, but a stronger research paper would require evaluation on a larger subset and possibly multiple random seeds or multiple balanced splits. The models were also run locally through Ollama, so performance and latency may differ across machines.

Finally, answer normalization affects exact and lenient accuracy. Some model outputs may be semantically correct but phrased differently from the ground-truth answer. This is why both exact and lenient accuracy are reported.

## Final Claim

In an oracle-SST evaluation on 200 balanced GQA validation examples, Keyword-Aware SST reduced average input tokens by 51.18% compared with Full SST while preserving most reasoning accuracy. Its exact accuracy was 50.0%, compared with 54.5% for Full SST and 55.5% for the direct LLaVA image-question baseline. The ablation results show that over-compression and removal of key semantic components substantially harm accuracy, indicating that structured attributes, relations, and readable semantic organization are important for visual reasoning.
