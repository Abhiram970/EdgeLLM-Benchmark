# EdgeLM — 1.5B Track Context Dump (our side)

> Raw research context for the team / paper lead. **Not a narrative** — this is the
> full set of what we did, the findings, the numbers, the roadblocks, the decisions,
> and the caveats. Lead decides framing and what goes in the manuscript.
> Scope: **our 1.5B-track work only** (5 models + the Ollama harness we built).
> Excludes teammates' 3B/4B runs and shared aggregate files.
> All runs: **2026-07-01**, Ollama, Q4_K_M quantization, single consumer machine.
> Scores on **[0, 1]** unless stated.

---

## A. What we built

- **`pipeline.py`** — Ollama-track benchmark runner + all six task scorers. Calls
  local Ollama (`/api/generate`, `/api/chat`); auto-registers GGUF weights into
  Ollama; splits `<think>` reasoning from final answer; per-call logging of token
  counts, `reasoning_truncated`, `hit_token_cap`, latency, tok/s.
- **`stimuli.py`** — single auditable store of **all human-authored test content**
  (verbatim Encyclopaedia Britannica passages + human gold answers / key-claims).
  No AI-generated stimuli (contamination defense).
- **`dl_daemon.py`** — Q4_K_M GGUF downloader from Hugging Face.
- **`judge.py`** — separate qualitative judge (Tasks 2 & 5) using **Claude Sonnet**
  via the Lava.so proxy, 1–10 rubric; reads result JSONs read-only, writes
  `<model>_qualitative.json`.
- **Per-model decode policy** (`DECODE` table): reasoning models sampled
  T=0.6/top_p=0.95 with large token budgets; non-reasoning greedy.

### Tasks implemented (6) and scoring method

| # | Task | Scorer | Tier |
|---|------|--------|------|
| 1 | Context retention under load | 14-turn scripts; two probes (mid ~turn 8, synthesis = final), weighted 0.3/0.7; exact-match fact recall + **stale-contamination** flag on mid-conversation fact updates | objective |
| 2 | Cross-domain summarization | ROUGE-L vs human reference + programmatic **key-claim survival** per domain; prose→technical faithfulness drop | objective (+ Sonnet judge) |
| 3 | Open-book comprehension | graded literal/inferential/synthesis QA; fuzzy/exact match vs human gold; answer-determinable from passage | objective |
| 4 | Structured output | JSON / Markdown / code-only / numbered-list / CSV; **strict** (no preamble/fences) vs **lenient** parse rate | fully objective |
| 5 | Creative generation | objective constraint checks inline; **quality** via Sonnet judge (1–10) | judge-hard |
| 6 | Code generation | execution-based pass@1 — **3 Perl + 3 Ruby** problems run against fixtures | fully objective |

### Stimulus inventory (human-authored)
- T1: 5 retention scripts (14 turns each, with fact updates) — 2 scored probes/script
- T2: 6 summarization passages (2 prose / 2 math / 2 science)
- T3: 6 passages × 3 graded questions = 18 QA items
- T4: 6 structured-format tasks
- T5: 6 constrained creative tasks
- T6: 6 code problems (3 Perl, 3 Ruby), execution-tested

### Models run (the 1.5B controlled study)
| id | lineage | type |
|----|---------|------|
| `qwen25-1b5` | Qwen2.5-1.5B-Instruct | instruct (non-reasoning) — control |
| `deepseek-r1-1b5` | R1-distill on Qwen2.5-Math-1.5B | reasoning (`<think>`) — base of RL pair |
| `deepscaler-1b5` | RL fine-tune of deepseek-r1-1b5 | reasoning — **clean RL delta pair** |
| `nemotron-1b5` | OpenReasoning-Nemotron-1.5B (Qwen2.5-1.5B, R1-0528 distill) | reasoning |
| `llama32-1b` | Llama-3.2-1B-Instruct | instruct (non-reasoning) — cross-family point |

Four of five share/closely-share a Qwen2.5-1.5B backbone → near-controlled study of
post-training recipes on one base. `deepseek-r1-1b5 → deepscaler-1b5` is a clean
base→finetune pair (the delta = the RL effect).

---

## B. Results (all 5 models, 2026-07-01)

### B.1 Headline scores

| Model | Ctx ret. | Summ (claim-surv) | ROUGE-L | Open-book | Struct strict/lenient | Creative | Code pass@1 |
|-------|:--------:|:-----------------:|:-------:|:---------:|:---------------------:|:--------:|:-----------:|
| qwen25-1b5      | 0.604 | 0.83 | 0.412 | 0.944 | 0.333 / 0.833 | 0.833 | 0.167 |
| deepseek-r1-1b5 | 0.243 | 0.83 | 0.422 | 0.722 | 0.333 / 0.833 | 0.167 | 0.167 |
| deepscaler-1b5  | 0.411 | 0.83 | 0.402 | 0.833 | 0.333 / 0.833 | 0.417 | 0.000 |
| nemotron-1b5    | 0.143 | 0.60 | 0.210 | 0.278 | 0.000 / 0.167 | 0.000 | 0.000 |
| llama32-1b      | 0.421 | 0.87 | 0.351 | 1.000 | 0.500 / 1.000 | 0.833 | 0.333 |

(Summ column = mean key-claim survival across the 3 domains.)

### B.2 Open-book accuracy by difficulty tier (easy / medium / hard)

| Model | easy | medium | hard |
|-------|:----:|:------:|:----:|
| qwen25-1b5      | 1.00 | 1.00 | 0.83 |
| deepseek-r1-1b5 | 0.83 | 0.83 | 0.50 |
| deepscaler-1b5  | 0.83 | 1.00 | 0.67 |
| nemotron-1b5    | 0.33 | 0.17 | 0.33 |
| llama32-1b      | 1.00 | 1.00 | 1.00 |

### B.3 Summarization key-claim survival, prose / math / science + drop

| Model | prose | math | science | drop (prose − mean(tech)) |
|-------|:-----:|:----:|:-------:|:-------------------------:|
| qwen25-1b5      | 0.9 | 0.8 | 0.8 | +0.10 |
| deepseek-r1-1b5 | 0.8 | 0.9 | 0.8 | −0.05 |
| deepscaler-1b5  | 0.8 | 0.8 | 0.9 | −0.05 |
| nemotron-1b5    | 0.6 | 0.7 | 0.5 |  0.00 |
| llama32-1b      | 1.0 | 0.9 | 0.7 | +0.20 |

### B.4 Code pass@1 (execution-based) + reasoning-truncation diagnostics

| Model | Perl | Ruby | pass@1 | truncated-think | empty answers |
|-------|:----:|:----:|:------:|:---------------:|:-------------:|
| qwen25-1b5      | 0/3 | 1/3 | 0.167 | 0  | 0/36 |
| deepseek-r1-1b5 | 0/3 | 1/3 | 0.167 | 0  | 1/36 |
| deepscaler-1b5  | 0/3 | 0/3 | 0.000 | 5  | 0/36 |
| nemotron-1b5    | 0/3 | 0/3 | 0.000 | 21 | 6/36 |
| llama32-1b      | 1/3 | 1/3 | 0.333 | 0  | 0/36 |

### B.5 Context-retention stale-contamination rate
qwen25-1b5 0.0 · deepseek-r1-1b5 0.029 · deepscaler-1b5 0.0 · nemotron-1b5 0.057 · llama32-1b 0.029

### B.6 Qualitative judge — Claude Sonnet, 1–10 (Tasks 2 & 5)

Run with `judge.py` (Sonnet via Lava, 1–10 rubric, with per-item rationales).
Task 2 = summarization faithfulness vs the source passage; Task 5 = creative
quality (voice/coherence/fluency, ignoring constraint satisfaction). Mean over the
6 items per task. Files: `results/<model>_qualitative.json`.

| Model | T2 faithfulness (mean /10) | per-item | T5 quality (mean /10) | per-item |
|-------|:--------------------------:|----------|:---------------------:|----------|
| qwen25-1b5      | 7.83 | 9,8,6,7,8,9 | 4.17 | 5,3,3,5,5,4 |
| deepseek-r1-1b5 | 6.33 | 3,8,7,8,4,8 | 2.00 | 3,2,2,3,1,1 |
| deepscaler-1b5  | 6.50 | 5,9,8,6,3,8 | 2.83 | 4,2,3,3,2,3 |
| nemotron-1b5    | 5.20 | 4,9,4,3,—,6 | 1.33 | 1,1,3,1,1,1 |
| llama32-1b      | 7.17 | 6,9,8,7,7,6 | 4.33 | 6,4,4,5,3,4 |

(nemotron T2 has one null — judge returned no parseable score for a truncated/empty
summary item.)

---

## C. Findings

- **F1.** Non-reasoning instruct models lead the track. llama32-1b and qwen25-1b5
  top open-book (1.000, 0.944), creative constraints (0.833 each), structured
  strict (0.500, 0.333), and code (0.333, 0.167). The 3 reasoning-distilled models
  trail on all format/instruction-bound tasks.
- **F2.** Strict-vs-lenient structured gap is the diagnostic. The 3 reasoning models
  all hit 0.833 *lenient* but only 0.333 *strict* — they emit the correct artifact
  wrapped in reasoning preambles / stray code fences that fail strict parsing.
  Instruct models close the gap (llama 0.5/1.0, qwen 0.333/0.833).
- **F3.** RL effect (deepscaler vs r1-distill, same base) is broadly positive:
  context retention 0.243→0.411, open-book 0.722→0.833, creative 0.167→0.417;
  code flat-to-down (0.167→0.000, within sampling noise). RL broadens capability
  beyond its competition-math target.
- **F4.** Truncated reasoning is a real failure mode, not a budget artifact.
  nemotron-1b5: 21/36 responses never closed `</think>` (still reasoning at the
  context limit → empty final answer; 6 fully empty), driving its near-zero scores.
  deepscaler-1b5: milder (5/36). **No model hit the token-generation cap**
  (`hit_token_cap=0` everywhere after we raised budgets to 14×), so the cause is
  context-window non-convergence / looping, not insufficient tokens.
- **F5.** Decoupling signal weak/reversed at 1.5B (our 6 passages). Predicted
  prose→technical faithfulness drop does not appear: reasoning models show −0.05
  (slightly favor technical); instruct models +0.10 / +0.20. No prose-favoring gap
  at this scale on this set.
- **F6.** Code is hard for all at 1.5B. pass@1 ∈ {0.0, 0.167, 0.333}. Perl 0–1/3,
  Ruby 0–1/3. The Perl/Ruby execution tasks are discriminating — failures are real
  language errors (wrong comparators, hardcoded filenames, non-existent methods),
  not boilerplate gaming.
- **F7.** Best overall on our track: **llama32-1b** (top or tied-top on open-book,
  structured, creative, code; mid-pack on retention). Worst: **nemotron-1b5**
  (truncation-dominated).
- **F8.** Reasoning-tag hygiene: our 3 reasoning models use standard
  `<think>...</think>` (cleanly split). None show the tag-less "Thinking Process:"
  prose leak seen in some larger models. Answer fields are reasoning-free except
  the truncated-think empties (F4).
- **F9.** Sonnet qualitative judge (B.6) corroborates the objective findings.
  *Faithfulness* (Task 2) is moderate-to-good for all but nemotron: instruct models
  top (qwen 7.83, llama 7.17), reasoning models mid (deepseek 6.33, deepscaler
  6.50), nemotron lowest (5.20, truncation-driven). *Creative quality* (Task 5) is
  low across the board (max 4.33) and cleanly separates instruct (llama 4.33, qwen
  4.17) from reasoning-distilled (deepscaler 2.83, deepseek 2.00, nemotron 1.33) —
  i.e. reasoning post-training degrades open-ended creative writing at 1.5B, not
  just format compliance. Judge = Claude Sonnet (not a small open judge, per
  methodology). Note Task-5 quality is a known judge-hard category — treat as a
  calibrated signal, not ground truth.

---

## D. Roadblocks / issues hit (and how handled)

- **R1. Ollama silent context truncation.** Ollama defaults `num_ctx=2048`
  regardless of model window; this truncated the 14-turn retention scripts and
  produced degenerate 1–2-word answers at the synthesis probe. **Fixed** by setting
  `num_ctx=32768` on every call. (Materially changed Task 1 scores.)
- **R2. Reasoning-model truncated-think.** See F4. Raised `max_factor` token budget
  8×→14× for reasoning models; this removed *cap-hit* truncations but nemotron
  still fails to converge within the window. Remaining issue is model behavior, not
  config. Logged explicitly via `reasoning_truncated`.
- **R3. ROUGE silently disabled.** `rouge_score` was not installed, so `_rouge_l`
  returned `None` for every item (all ROUGE fields null). **Fixed** by installing
  `rouge-score` (+`nltk`); ROUGE-L now reported.
- **R4. Task 4 strict JSON false-positives.** `_v_json` was passing fenced
  ```` ```json ```` output as strict=True although the prompt says "ONLY valid
  JSON." **Fixed**: fenced output now strict=False, lenient=True (lowered observed
  strict rates to the accurate value).
- **R5. Task 3 determinability leak.** Original "easy" questions were answerable
  from general knowledge without the passage. **Replaced** all 6 with passage-
  specific questions; tightened hard-question accept-lists (≥2-word phrases, removed
  strings appearing in the question itself or any on-topic answer).
- **R6. Code-task interpreter availability (Windows).** Native Python doesn't have
  Perl/Ruby on PATH. Perl resolved via Git-bundled fallback; **Ruby installed**
  (3.3.11) and found via fallback path. Without this, Task 6 silently reports
  "not-installed" and skips. (Flagged: teammates on other machines must verify
  Perl+Ruby or those items skip.)
- **R7. Llama model variant.** The originally-shared link was the *base*, gated,
  safetensors `meta-llama/Llama-3.2-1B` (no chat template, won't run on chat tasks).
  Switched to **Llama-3.2-1B-Instruct** Q4_K_M GGUF (bartowski) — the correct
  artifact for the benchmark.

---

## E. Decisions / methodology notes

- Stimuli are 100% human-authored (Britannica + canonical CS exercises). An LLM may
  *judge* but never *author* a stimulus.
- Scoring is tiered: T4/T6 fully objective (parser / execution); T1/T3 objective
  (match); T2 objective + Sonnet judge; T5 judge-hard (Sonnet, not a small model —
  small open judges unreliable for creative writing).
- Reasoning models sampled (T=0.6/top_p=0.95) per their model cards → single-run
  numbers carry sampling variance; report mean ± CI over seeds for final tables.
- `deepseek-r1-1b5 → deepscaler-1b5` reported as a controlled pair, not two
  independent points.
- Quantization pinned and reported (Q4_K_M) — part of the consumer-hardware story.

---

## F. Caveats / open items for the paper

- **Sampling variance**: reasoning-model rows are single-run; recommend mean ± CI
  over multiple seeds before publishing exact numbers.
- **Small item counts** (human-authored): report confidence intervals so 1-point
  gaps aren't over-read.
- **Mid-probe stale false positives** (Task 1): the mid probe scores against the
  *final* fact set, so a fact updated *after* the mid probe can be wrongly flagged
  stale. Affects mid-probe `stale_count` only; synthesis number is unaffected.
- **nemotron-1b5 budget sensitivity**: its near-zero scores are truncation-driven;
  a larger context window / different stop handling might recover some capability —
  worth a caveat rather than reading it as pure capability.
- **Decoupling result is scale/sample-bound**: F5 is on 6 passages × 5 models at
  1.5B; whether it holds across the full roster + larger passage set is open.

---

## G. Artifacts (our side)

Code: `pipeline.py` (Ollama runner + 6 task scorers), `stimuli.py` (human-authored
test content), `dl_daemon.py` (GGUF downloader), `judge.py` (Sonnet qualitative
judge, Tasks 2 & 5).

Per-model objective results (full 6-task, each with a `headline` block):
`results/<model>.json` for `qwen25-1b5`, `deepseek-r1-1b5`, `deepscaler-1b5`,
`nemotron-1b5`, `llama32-1b`.

Per-model qualitative judge results (Sonnet 1–10, Tasks 2 & 5, with rationales):
`results/<model>_qualitative.json` for the same five models.

(No aggregate/leaderboard files in this drop — read the per-model JSONs directly;
each is self-contained.)
