# Task 3 — Open-Book QA Changes (stimuli.py)

## 1. Easy questions replaced — all 6 passages

The original easy questions were answerable from general knowledge without reading the passage (e.g. publication year, named concepts). Replaced with questions that require reading the specific passage text.

| Item | Old question | New question | Gold | Accept |
|---|---|---|---|---|
| `qa_lit_pride` | "How many sisters are in the Bennet family?" | "According to the passage, what two words describe the personalities of Lydia and Kitty?" | "Flighty and immature." | `["flighty", "immature"]` |
| `qa_lit_gatsby` | "In what year was The Great Gatsby published?" | "Where did Gatsby and Daisy originally meet in their youth?" | "Kentucky." | `["kentucky"]` |
| `qa_sci_blackhole` | "What is the name of the 'surface' that hides the singularity at the centre of a black hole?" | "According to the passage, what is the Schwarzschild radius of a black hole with a mass 10 times that of the Sun?" | "30 km." | `["30 km", "30km", "30 kilometers", "30 kilometres"]` |
| `qa_sci_evolution` | "What are the raw material for evolution, according to the passage?" | "According to the passage, through what four factors can differential reproduction occur?" | "Survival rates, fertility, mating success, or other life cycle aspects." | `["survival rates", "fertility", "mating success", "life cycle"]` |
| `qa_hist_frenchrev` | "Why did the rulers of Europe seek to raise money by taxing the nobles and clergy?" | "What role did the rulers claim to justify taxing the privileged classes?" | "Enlightened despots." | `["enlightened despot", "enlightened despots"]` |
| `qa_hist_printing` | "In what year did Gutenberg use his press to print an edition of the Bible?" | "What mechanism did Gutenberg's press use to exert pressure on the paper?" | "A long handle turning a heavy wooden screw." | `["wooden screw", "long handle", "heavy screw", "handle to turn"]` |

## 2. Hard question accept lists — all 6 passages

Replaced single-word accepts and over-generic strings with discriminating multi-word phrases. Rules applied:
- Minimum 2 words per accept string (exception: unique proper nouns and specific years)
- Removed strings ≤ 3 characters
- Removed strings that appear in any on-topic response regardless of correctness

| Item | Old accept | New accept |
|---|---|---|
| `qa_lit_pride` | `["wealth", "rank", "conventional", "society"]` | `["wealth and rank", "conventional views", "distaste for", "social status"]` |
| `qa_lit_gatsby` | `["old money", "old-money", "east egg"]` | `["old money", "old-money"]` |
| `qa_sci_blackhole` | `["double", "proportional", "twice", "increase"]` | `["would double", "also double", "roughly double", "proportional to the mass", "twice as"]` |
| `qa_sci_evolution` | `["natural selection", "increase in frequency", "frequency", "over generations"]` | `["natural selection", "increase in frequency", "natural selection favors", "selection favors"]` |
| `qa_hist_frenchrev` | `["refusal to pay", "tax", "king of great britain", "backlash"]` | `["refusal to pay", "king of great britain", "refused to pay", "refusal to pay a tax"]` |
| `qa_hist_printing` | `["jikji", "korea", "1377"]` | `["jikji", "1377", "in korea", "korea in 1377"]` |

Notable decisions:
- `qa_lit_gatsby` hard: dropped `"east egg"` — the question names East Egg explicitly, so it appears in any response. `"old money"` alone is sufficient since the question asks which society triumphs.
- `qa_sci_blackhole` hard: removed `"increase"` (appears in any answer describing the radius changing at all) and added `"also double"` after a smoke run revealed the model produced "would also double" which broke the exact `"would double"` substring match.
- `qa_hist_frenchrev` hard: removed `"tax"` (3 chars) and `"backlash"` (appears verbatim in the question text).
- `qa_hist_printing` hard: `"jikji"` and `"1377"` retained as single words — they are unique proper nouns/years that only appear in correct responses.

## 3. Medium accept list fix — qa_hist_printing

Dropped bare `"wine"` and `"olive"` from the accept list; kept only the compound phrases.

| Before | After |
|---|---|
| `["wine-and-olive", "wine and olive", "wine", "olive"]` | `["wine-and-olive", "wine and olive"]` |
