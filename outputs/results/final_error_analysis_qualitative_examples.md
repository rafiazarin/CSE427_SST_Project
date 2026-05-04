| sample_index | semantic_type | question | answer | disagreement_category | full_vs_keyword_category | LLaVA Prediction | Full SST Prediction | Keyword-Aware SST Prediction | Caption-Style SST Prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | attr | What material do you think is the boot, rubber or leather? | rubber | All three correct | Both correct | Rubber | rubber | Rubber | rubber |
| 3 | attr | Is that bus teal and long? | yes | All three correct | Both correct | Yes | Yes | Yes | Yes |
| 2 | attr | Is the camera in the bottom part or in the top of the photo? | top | All three wrong | Both wrong | Bottom | Unknown | Unknown (The structured scene does not provide information about the camera's location in the photo.) | Unknown |
| 5 | attr | Does this plane look white and huge? | no | All three wrong | Both wrong | Yes | Unknown | Unknown | Unknown |
| 6 | attr | Which color is the floor? | brown | Full + Keyword correct; LLaVA wrong | Both correct | White | brown | brown | brown |
| 10 | attr | Is the catcher on the left side? | no | Full + Keyword correct; LLaVA wrong | Both correct | Yes | No | No | No |
| 15 | attr | Where in the photo is the cat, in the top or in the bottom? | bottom | LLaVA + Full correct; Keyword wrong | Full correct, Keyword wrong - possible compression loss | Bottom | bottom | Unknown (since there's no information about the cat's vertical position in the structured scene | bottom |
| 51 | cat | Which kind of clothing is black? | dress | LLaVA + Full correct; Keyword wrong | Full correct, Keyword wrong - possible compression loss | Dress | dress (black) | Black | dress (black) |
| 25 | attr | Are the flags on the left side? | no | LLaVA + Keyword correct; Full wrong | Keyword correct, Full wrong - filtering may help | No | Yes | No | Yes |
| 29 | attr | Is the man in the bottom part of the picture? | yes | LLaVA + Keyword correct; Full wrong | Keyword correct, Full wrong - filtering may help | Yes | No | Yes | No |
| 7 | attr | On which side of the photo is the full boat? | right | Only Full SST correct | Full correct, Keyword wrong - possible compression loss | Left | Right | Unknown (The structured scene does not provide information about the location of the boat in the photo.) | Right |
| 110 | global | Which place is it? | parking lot | Only Full SST correct | Full correct, Keyword wrong - possible compression loss | Road | parking lot | Unknown (The structured scene does not provide location information.) | Unknown (The structured scene does not provide location information) |
| 4 | attr | On which side of the image is the toilet paper? | right | Only Keyword-Aware SST correct | Keyword correct, Full wrong - filtering may help | Left | Left | Right | Left |
| 0 | attr | On which side of the picture is the chimney? | right | Only LLaVA correct | Both wrong | Right | Left | Left | Left |
| 8 | attr | Is the color of the boat the same as the sky? | yes | Only LLaVA correct | Both wrong | Yes | no | No | No |