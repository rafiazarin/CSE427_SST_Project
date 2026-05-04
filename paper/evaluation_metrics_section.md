## Evaluation Metrics

The final experiments are evaluated using both task-performance and efficiency metrics. Exact accuracy measures the percentage of predictions that exactly match the normalized ground-truth answer. Lenient accuracy allows minor acceptable answer-format variations after normalization, which is useful because generative models may produce semantically correct answers with slightly different wording.

For efficiency, average input tokens are reported for all SST-based methods. Token reduction is computed relative to Full SST as:

Token Reduction (%) = ((Tokens_Full_SST - Tokens_Method) / Tokens_Full_SST) × 100

Average latency measures the mean inference time per question in seconds on the local experimental machine. Since latency depends on local hardware and Ollama runtime behavior, it is reported as an implementation-level efficiency measure rather than a universal model property.

Additional derived metrics are included for interpretability. Exact correct count and lenient correct count show how many of the 200 questions were answered correctly. Accuracy delta versus Full SST shows how much performance is lost or gained compared with the oracle SST reference. Accuracy delta versus LLaVA compares each method against the direct image-question VLM baseline. Tokens saved versus Full SST and correct answers per 1000 input tokens summarize the compression-efficiency tradeoff among SST methods.

For statistical reliability, the project also reports bootstrap confidence intervals and McNemar-style paired comparisons where available. These paired comparisons are important because all methods are evaluated on the same 200 examples.
