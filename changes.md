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

---

# Task 2 — Summarization Changes (stimuli.py + pipeline.py)

## 1. Key claim lists tightened — 4 passages (stimuli.py)

Removed over-generic single-word variants that would appear in any on-topic summary regardless of whether the model preserved the specific claim. Two passages needed no changes (sum_prose_frenchrev, sum_sci_dna).

### sum_prose_industrial

| Claim | Before | After | Reason |
|---|---|---|---|
| 2 | `["factory system", "factory"]` | `["factory system", "large-scale industry", "mechanized manufacturing", "mass production"]` | Dropped bare `"factory"`; expanded with synonyms after smoke test showed model said "large-scale industry" not "factory system" — correct substance, wrong phrase |
| 4 | `["women and children", "women", "children"]` | `["women and children"]` | Dropped singles — `"women"` and `"children"` appear in any on-topic response |
| 5 | `["first industrial", "second industrial", "two"]` | `["first industrial", "second industrial", "first lasted", "second lasted", "first phase", "second phase"]` | Dropped `"two"`; expanded after smoke test showed model wrote "the first lasted...the second from" without repeating "Industrial" in each clause |

### sum_math_calculus

| Claim | Before | After | Reason |
|---|---|---|---|
| 4 | `["area under", "area"]` | `["area under", "area under the curve", "area function"]` | Dropped bare `"area"` — matches any calculus response; added `"area function"` after smoke test showed model wrote "area function F(t) under the curve" which broke the original `"area under"` substring |

### sum_math_pythagorean

| Claim | Before | After | Reason |
|---|---|---|---|
| 3 | `["a2 + b2", "a² + b²", "squares"]` | `["a2 + b2", "a² + b²"]` | Dropped `"squares"` — matches any geometry response |
| 5 | `["babylonian", "older", "euclid", "triples"]` | `["babylonian", "euclid", "pythagorean triple", "triples"]` | Dropped `"older"` — appears in any response noting antiquity; added `"pythagorean triple"` as more specific variant |

### sum_sci_tectonics

| Claim | Before | After | Reason |
|---|---|---|---|
| 3 | `["plate", "plates"]` | `["tectonic plates", "lithosphere is broken", "several plates"]` | `"plate"` matches "plate tectonics" in any response trivially; replaced with compound phrases requiring the model to describe the plate structure specifically |
| 4 | `["converge", "diverge", "boundaries", "slip past"]` | `["converge", "diverge", "slip past"]` | Dropped `"boundaries"` — appears in any plate tectonics response |

## 2. pipeline.py changes

### _headline() — ROUGE-L surfaced
Added `"summarization_mean_rouge_l"` to the returned dict, pulling from `tasks["summarization"]["mean_rouge_l"]`. Same pattern as all other headline metrics.

### task_summarization — token budget raised
| Mode | Before | After | Reason |
|---|---|---|---|
| smoke | 64 | 200 | 64-token cap was cutting summaries mid-sentence; frenchrev truncated before France/26M claim fired |
| full run | 256 | 350 | 256 left frenchrev truncated in full runs too; 350 gives all passages room to complete |

## 3. Environment
Installed `rouge-score==0.1.2` (+ `nltk==3.9.4`). Previously `_rouge_l()` silently returned `None` on every call; all ROUGE fields were null in results.
