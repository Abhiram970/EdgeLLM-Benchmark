# Task 1 — Context Retention Redesign (stimuli.py + pipeline.py)

## 1. Scripts fully replaced — stimuli.py

Replaced all 5 `CONTEXT_RETENTION_SCRIPTS` (old format: plant/distractor/probe steps, static facts, single final probe) with `RETENTION_SCRIPTS` (new format: realistic multi-turn conversations, structured current/stale fact tables, two scored checkpoints).

### Key structural changes

| Before | After |
|---|---|
| `CONTEXT_RETENTION_SCRIPTS` | `RETENTION_SCRIPTS` |
| `script: [{type, content/ask/expect}]` | `turns: [{role, content, is_probe?}]` |
| Facts as flat string dict | Facts as `{current: [...], stale: [...]}` per key |
| Single final probe per script | Two probes: `"mid"` (turn ~8) and `"synthesis"` (final turn) |
| Distractor prose inserts | Domain-relevant questions the model actually answers |
| No update tracking | Mid-conversation fact updates (e.g. pet age, grant code) that supersede earlier values |

### New scripts

| Script | Character | Facts | Updated facts |
|---|---|---|---|
| `ret_hydrologist` | Helena Vintner | name, job, city, pet (name/kind/age), project codename, funding code | pet_age 4→5, funding_code HL-4482→HL-4490 |
| `ret_chef` | Dario Pellegrini | name, job, city, signature dish, opened year, book title, deadline | book_title Embers→Coals, deadline October→August |
| `ret_astronomer` | Saanvi Rao | name, job, city, telescope, target, grant code | target Kepler-442→TOI-700, grant_code AR-3398→AR-3405 |
| `ret_archivist` | Otto Lindqvist | name, job, city, collection name, oldest doc year, deadline, budget code | deadline winter→spring, budget_code MC-2291→MC-2305 |
| `ret_engineer` | Freya Sandberg | name, job, city, bridge name, bridge length, deadline, contract code | deadline December→October, contract_code BR-5571→BR-5588 |

Each script is 14 turns: ~6 domain-question distractors, 1 plant turn, 2-3 update turns, plus the 2 probe turns. Mid probe fires at turn ~8; synthesis probe is always the final turn.

Probe text uses neutral "Give me a quick status update on what we've covered about..." phrasing (no "remember"/"memory" wording, which triggered AI-refusal responses on some models).

## 2. Pipeline — new scoring functions (pipeline.py)

### _fact_check — new helper
Mirrors `answer_match` logic (normalized substring/token check) but takes `(response_text, fact_variants)` for use with the new fact structure. Single words >3 chars: substring match on normalized text; hyphenated phrases or multi-word phrases: raw substring on lowercased text; short/numeric tokens: whole-token match.

### _score_probe — new helper
Scores a probe response against a list of fact IDs. Per-fact status: `"correct"` (current variant hit), `"stale_contamination"` (stale variant hit, current missed), or `"missing"` (neither). Returns `{per_fact, score, stale_count, total_facts}`.

### _run_retention_script — replaced
Old: iterated `script` steps by type, tracked `probe_scores`, computed coherence horizon.
New: iterates `turns`, sends every turn through `ollama_chat`, calls `_score_probe` when `is_probe` is set. Assistant messages use `resp.get("text", "")` (stripped, not raw) so reasoning traces don't accumulate in context.

### task_context_retention — replaced
| | Before | After |
|---|---|---|
| max_tokens smoke | 64 | 80 |
| max_tokens full | 512 | 500 |
| score | mean probe score across all scripts | weighted mean: 0.3 × mid + 0.7 × synthesis per script, then mean |
| headline metrics | `score`, `mean_coherence_horizon` | `score`, `stale_contamination_rate` |

`stale_contamination_rate` = total stale fact hits at synthesis / total synthesis facts checked, across all scripts. Computed from synthesis probes only (mid-probe stale is uninformative for facts not yet updated at that point).

Smoke mode runs first 2 scripts only.

## 3. _headline() — stale_contamination_rate surfaced (pipeline.py)

Added `"context_retention_stale_rate"` to the headline dict alongside the existing `"context_retention"` score. Both are now leaderboard columns.

## 4. Coherence horizon removed

The old `coherence_horizon` metric (max depth of any probe scoring 1.0) is gone. It was directionally wrong — a model that passes probe 1, fails probe 2, then recovers probe 3 got an artificially high horizon. The two-checkpoint mid/synthesis structure replaces it with a more informative signal.

## 5. Baseline results — qwen25-1b5

| Metric | Value |
|---|---|
| overall score | 0.346 |
| stale_contamination_rate | 0.029 |

| Script | mid | synthesis | syn_stale | weighted |
|---|---|---|---|---|
| ret_hydrologist | 0.625 | 0.750 | 1 | 0.712 |
| ret_chef | 0.200 | 0.286 | 0 | 0.200 |
| ret_astronomer | 0.333 | 0.333 | 0 | 0.333 |
| ret_archivist | 0.143 | 0.286 | 0 | 0.243 |
| ret_engineer | 0.143 | 0.286 | 0 | 0.243 |

Notable: `pet_age` stale contamination on ret_hydrologist — model retains "4-year-old" after being told Rusty turned 5. ret_chef mid probe triggers AI-refusal ("I don't have information about your restaurant") on qwen25-1b5 despite neutral probe phrasing; synthesis partially recovers.

---

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

---

# Task 4 — Structured Output Changes (pipeline.py)

## 1. _v_json — strict now fails on fenced JSON

Previously, `_v_json` extracted JSON from a `` ```json ``` `` fence and passed it as strict=True. The JSON prompts say "Return ONLY valid JSON, no other text," so fenced output is a real formatting failure.

**Fix:** If `re.fullmatch` finds a fence, strict is set to False without attempting to parse. Lenient path (`re.search` for any `{...}`) is unchanged.

| Before | After |
|---|---|
| Fenced JSON → strict=True | Fenced JSON → strict=False, lenient=True |

**Observed on qwen25-1b5 full run:** `json_person` and `json_nested` both produced fenced output and now correctly score strict=False, lenient=True. strict_compliance_rate dropped from 0.667 to 0.333, which is the accurate number.

## 2. _v_markdown — real strict/lenient split

Previously `_v_markdown` returned `ok, ok` — strict and lenient were identical, the gap was never captured.

**Fix:** Lenient = has header + bold + bullet anywhere in text. Strict = lenient AND `non_md_lines <= 2` (at most 2 lines that aren't Markdown-structural). `non_md_lines` is now surfaced in `details`.

| Signal | Before | After |
|---|---|---|
| strict | same as lenient | lenient AND prose lines ≤ 2 |
| lenient | has_header + has_bold + has_bullet | unchanged |
| details | 3 bools | 3 bools + `non_md_lines` count |

---

# Task 5 — Creative Generation Changes (stimuli.py + pipeline.py)

## 1. Per-item token budgets (stimuli.py + pipeline.py)

Replaced the single blanket `max_tokens = 96 if smoke else 512` with per-item budgets stored in `CREATIVE_TASKS`. Smoke mode uses `min(item["max_tokens"], 150)` instead of a flat 96.

| Item | Full budget | Smoke budget |
|---|---|---|
| article | 350 | 150 |
| poem_question_word | 180 | 150 |
| short_story | 280 | 150 |
| acrostic | 100 | 100 |
| dialogue | 220 | 150 |
| six_word_story | 40 | 40 |

**Rationale:** The old 96-token smoke cap cut the article to ~70 words and the short story before arc completion. The old 512-token full cap was unnecessarily large for short items. Per-item budgets let short items (acrostic, six_word_story) stay tight while longer items get room to complete.

## 2. poem_unrhymed → poem_question_word (stimuli.py + pipeline.py)

Replaced the unrhymed poem item. The original had a "no rhyme" constraint that was never programmatically checked, making it unverifiable.

**New item:** 12-line poem where every line must begin with a question word (Who, What, Where, When, Why, How). Do not rhyme.

**Constraint changes:**

| Before | After |
|---|---|
| `"constraints": ["12 lines", "no rhyme", "vivid imagery"]` | `"constraints": ["12 lines", "each line starts with question word"]` |
| `"target_lines": 12` | `"question_word_lines": True` |
| pipeline: `line_count_exact` only | pipeline: `line_count_exact` + `all_lines_start_question_word` |

`question_word_lines: True` triggers two separate checks in the pipeline: `line_count_exact` (== 12) and `all_lines_start_question_word` (every non-empty line's first word, lowercased, is in `{"who", "what", "where", "when", "why", "how"}`). Stored as separate keys so a model writing 6 correct-prefix lines doesn't score the same as one writing 12.

---

# Task 6 — Code Generation Changes (stimuli.py + pipeline.py)

## 1. Per-item token budgets (stimuli.py + pipeline.py)

Replaced the single blanket `max_tokens = 128 if smoke else 512` with per-item budgets stored in `CODE_TASKS`. Smoke mode uses `min(item["max_tokens"], 150)`.

| Item | Full budget | Smoke budget |
|---|---|---|
| py_process_csv | 300 | 150 |
| py_is_palindrome | 250 | 150 |
| py_fizzbuzz | 250 | 150 |
| py_two_sum | 250 | 150 |
| perl_error_counter | 280 | 150 |
| csharp_unity_controller | 700 | 150 |

**Rationale:** The old 128-token smoke cap truncated `py_is_palindrome` mid-function (confirmed: syntax error from incomplete output). The old 512-token full cap was too tight for `csharp_unity_controller` (WASD + mouse look + jump + gravity runs long). `csharp_unity_controller` now gets 700 tokens on full runs.

## 2. Full redesign — dropped Python and C#, replaced with 3 Perl + 3 Ruby (stimuli.py + pipeline.py)

The original CODE_TASKS had 4 Python problems and 1 Perl + 1 C# problem. Python problems are too easy for the 3B/4B tier (all pass on 1.5B models), and the C# static check (`MonoBehaviour` + `Update` + `Input.GetAxis`) was trivially gameable noise. Replaced the entire list with 3 Perl and 3 Ruby problems that require real language knowledge, file I/O, and idiomatic constructs.

### New CODE_TASKS

| Name | Language | Test method | Budget |
|---|---|---|---|
| `perl_word_count` | Perl | `exec_perl` (ordered) | 320 |
| `perl_csv_filter` | Perl | `exec_perl` | 320 |
| `perl_regex_extract` | Perl | `exec_perl` | 320 |
| `ruby_array_dedupe_sort` | Ruby | `exec_ruby` | 280 |
| `ruby_hash_group` | Ruby | `exec_ruby` | 320 |
| `ruby_class_stack` | Ruby | `exec_ruby` | 350 |

All items are execution-tested against fixtures. Each takes a filename via `$ARGV[0]` / `ARGV[0]` so the harness controls input without touching stdin.

### _strip_code_fence — re.fullmatch → re.search (pipeline.py)

Changed from `re.fullmatch` to `re.search` so it correctly extracts fenced code even when the model wraps the fence in preamble or postamble prose. Also updated the language prefix list.

| Before | After |
|---|---|
| `re.fullmatch(...)` — fails if any prose surrounds fence | `re.search(...)` — extracts fence from anywhere in output |
| prefix list: python, csharp, perl, cs, c# | prefix list: python, ruby, perl, rb, pl |

### _exec_ruby — new function (pipeline.py)

Added `_exec_ruby` mirroring `_exec_perl`: writes code to `s.rb`, fixture to `f.txt`, runs `ruby s.rb f.txt`, checks expected substrings. Includes hardcoded fallback to `C:\Ruby33-x64\bin\ruby.exe` for when PATH hasn't been refreshed in the pipeline process.

### _exec_perl — ordered substring check (pipeline.py)

Added `ordered: bool = False` parameter. When `True`, checks that all expected substrings are present **and** appear in the given order (by string index). Used by `perl_word_count` so a model that outputs the correct counts in the wrong rank order fails.

### task_code_generation — updated dispatch (pipeline.py)

Removed the Python execution path. Added Ruby dispatch mirroring Perl. Updated `executed_pass_at_1` exclusion list to also skip `"ruby-not-installed"`.

### perl_word_count fixture tightened (stimuli.py)

| | Fixture | Expected |
|---|---|---|
| Before | `"the cat sat on the mat\nthe cat ran\nthe dog sat\n"` | `the: 4`, `cat: 2`, `sat: 2` (tied — order ambiguous) |
| After | `"the the the the cat cat cat sat sat dog mat\n"` | `the: 4`, `cat: 3`, `sat: 2` (strict ranking, `ordered: True`) |

The old fixture passed a model using `sort { $a <=> $b }` (wrong numeric comparator) because Perl's fallback string sort coincidentally produced the right words in the right order. The new fixture correctly fails it.

### Baseline results (qwen25-1b5 vs gemma3-4b)

| Item | qwen25-1b5 | gemma3-4b |
|---|---|---|
| `perl_word_count` | ✗ wrong comparator, output unordered | ✗ bare `k` / `break` — compile error under `use strict` |
| `perl_csv_filter` | ✗ regex assumes digit-only first field | ✓ |
| `perl_regex_extract` | ✗ prints `$_` instead of `$&` | ✗ `$#ARGV != 1` requires 2 args, dies immediately |
| `ruby_array_dedupe_sort` | ✗ calls `.split` on CSV-parsed Array | ✓ |
| `ruby_hash_group` | ✓ | ✗ calls `.is_alphabetical` (not a Ruby method) |
| `ruby_class_stack` | ✓ | ✓ |
| **pass_at_1** | **0.333** | **0.500** |
