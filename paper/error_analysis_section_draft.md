## Error and Disagreement Analysis

A paired disagreement analysis was performed across LLaVA, Full SST, and Keyword-Aware SST on the same 200 GQA validation examples. This analysis is more appropriate than a confusion matrix because the task is open-ended visual question answering rather than fixed-label classification.

The Full SST versus Keyword-Aware SST comparison directly measures the effect of semantic compression. Cases where Full SST is correct but Keyword-Aware SST is wrong indicate possible compression loss, where question-aware filtering may have removed useful semantic evidence. Cases where Keyword-Aware SST is correct but Full SST is wrong suggest that filtering can sometimes help by removing distracting or noisy scene-graph information.

The corrected disagreement analysis uses sample-level alignment rather than question-text matching, avoiding errors caused by repeated question templates in GQA. The resulting files provide both aggregate disagreement categories and qualitative examples for manual inspection.
