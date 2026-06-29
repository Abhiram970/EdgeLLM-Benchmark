# EdgeLM — Project Context (1.5B Track)

> Drop this file in your repo (e.g. as `CONTEXT.md`, or rename to `CLAUDE.md` so Claude Code auto-loads it). It captures everything from the design discussion so an agent can help build the 1.5B-model evaluation harness without re-deriving the project.

---

## 1. What EdgeLM is

EdgeLM is a **capability-first benchmark for sub-4B language models on consumer hardware**. Existing edge-LLM benchmarks are systems-first (throughput, energy, memory) and collapse task quality into a single multiple-choice accuracy number. EdgeLM instead decomposes *generative* ability at the sub-4B scale into **six task families**, each scored separately, with **human-authored stimuli** and **task-matched scoring** (deterministic parsing / code execution where possible, calibrated judging elsewhere).

Framing claim the benchmark tests: the **reasoning–knowledge decoupling hypothesis** — verifiable reasoning (math, code, constraint satisfaction) compresses into a compact parametric core, while broad knowledge does not. Prediction: sub-4B models stay competitive on reasoning/format tasks and degrade on knowledge-heavy comprehension and summarization.

There are three companion documents (design-level, not needed to code but useful for the paper): `EdgeLM_related_work.md`, `EdgeLM_methodology.md`, and the LNCS paper draft (`main.tex` / `references.bib`).

---

## 2. Your track: the four 1.5B models

| Short id | HF repo | Type | Key run settings |
|---|---|---|---|
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | **standard instruct** (no `<think>`) | greedy (temp 0), system prompt OK |
| `r1-distill-1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | **reasoning** (`<think>`) | temp 0.6 / top_p 0.95, **no system prompt**, force `<think>` |
| `deepscaler-1.5b` | `agentica-org/DeepScaleR-1.5B-Preview` | **reasoning** (`<think>`) | same as r1-distill |
| `openreasoning-nemotron-1.5b` | `nvidia/OpenReasoning-Nemotron-1.5B` | **reasoning** (`<think>`) | temp 0.6 / top_p 0.95, system prompt OK |

**Two facts that shape the whole track:**

1. **Three of the four share a Qwen2.5-1.5B backbone** (Qwen2.5-1.5B-Instruct; OpenReasoning-Nemotron is Qwen2.5-1.5B distilled from R1-0528; the R1-Distill→DeepScaleR pair is on Qwen2.5-Math-1.5B). So the 1.5B track is close to a **controlled study of four post-training recipes on one base**: plain instruct vs. R1-style distillation vs. distillation+RL vs. NVIDIA reasoning distillation. Report it that way — it's a clean scientific story and far more interesting than four scattered points.
2. **`deepscaler-1.5b` is an RL fine-tune of `r1-distill-1.5b`** (same base). This is a clean base→finetune pair: the delta between them *is* the RL effect. Label it explicitly; don't treat them as independent.

---

## 3. The six task categories (what you're scoring)

1. **Context Retention Under Load** — multi-turn coherence as context accumulates. Plant a tracked state (named entities/facts + mid-conversation updates) interleaved with human-written distractor prose; probe at increasing turn depth. Two signals per probe: probe accuracy (exact-match recall, objective) and coherence (judge/heuristic). **Headline = coherence horizon** (turns/tokens until breakdown) + degradation curve. **Must separate** "context-window overflow" from "coherence collapse with content still in-window" (see §4).
2. **Cross-Domain Summarization** — summarize three human-written passages (prose / mathematical / scientific) to a fixed compression target. Rubric judge on faithfulness, coverage, coherence; faithfulness weighted for math/science, plus a programmatic "key claims survive" check. **Headline = per-domain score + prose→math/science faithfulness drop** (the decoupling signal).
3. **Open-Book Comprehension** — graded-difficulty questions over a provided human-written passage in three domains (literary HS-level / scientific / historical). Tiers: literal → inferential → synthesis. **Critical: answer-determinability** — answer must come from the passage, not parametric knowledge (validate by answering with the passage withheld). Mostly multiple-choice (exact-match) + some short-answer (judge).
4. **Structured Output Compliance** — force JSON / Markdown / code-only / custom formats, under format-only and format-under-load conditions. **Fully objective**: parser/validator. Report **strict** pass rate (no prose preamble, no stray fences) plus lenient; the gap characterizes failure.
5. **Creative Generation** — article / poem / short story under explicit constraints. Two layers: constraint satisfaction (checklist, objective) + quality (judged). **Quality is the automation trap** — small open judges are unreliable for creative writing; use a stronger judge / reward model / human slice, pairwise + position-permuted, pre-registered rubric.
6. **Code Generation** — same problems in Python + an obscure language (Perl or Pascal), executed. **Execution-based pass@1** (objective). **Headline = pass@1 per language + Python→obscure drop** (generalization signal). Sandbox the executor (no network, timeouts, pinned interpreter/compiler).

**Standing rule:** all *stimuli* are human-authored — no AI-generated passages, questions, or conversation scripts. This is the contamination defense. It does **not** forbid an LLM *judge* (a judge reads human references + model outputs; it injects no synthetic stimulus). So a Claude Code agent should never generate test content, but may help build judge pipelines.

---

## 4. Implementation rules that make or break small-model eval

These are non-obvious and they matter more at 1.5B than at frontier scale. The provided `runner.py` already handles them.

- **Chat templates.** Always use `tokenizer.apply_chat_template(...)`. A wrong/absent template disproportionately tanks small models and silently corrupts comparisons. This is the #1 source of unfair results.
- **Reasoning models emit `<think>...</think>`.** Three of your four do. Consequences: (a) budget **large** `max_new_tokens` (these were trained for up to ~64K output; use 16–32K for real runs); (b) **sample, don't greedy** — temp ≈ 0.6, top_p 0.95; greedy decoding makes the R1-distill family loop/degenerate; (c) **split reasoning from the final answer** before scoring; (d) handle the **truncated-think** case (no closing tag because it hit the cap) — that's a real failure mode, not a parse bug.
- **DeepSeek R1-distill specifics:** no system prompt (put instructions in the user turn), and force the output to start with `<think>\n` so it can't skip reasoning. Applies to `r1-distill-1.5b` and `deepscaler-1.5b`.
- **Determinism vs. sampling tension.** The benchmark wants reproducibility, but the reasoning models degenerate at temp 0. Resolution: keep the recommended sampling, fix the seed, and for sampled models **run multiple samples and report mean ± CI** (also matches how these models report pass@1 = avg@k).
- **Quantization.** Pin the dtype and **report it** (BF16/FP16 vs Q4/Q8) — it changes outputs and is part of the consumer-hardware story.
- **Context-window logging.** Log each model's native window (`model.config.max_position_embeddings`) and the running token count every turn. Without it you cannot separate "ran out of room" (capacity) from "got confused" (quality) in the retention category — and these 1.5B reasoning models burn context fast because the `<think>` traces accumulate.
- **Confidence intervals.** Human-authored item counts are small → report CIs everywhere so a one-point gap isn't over-read.

---

## 5. Scoring tiers (how much to trust each category's automation)

| Category | Primary scoring | Tier |
|---|---|---|
| Context retention | exact-match probe + coherence check | objective + judge |
| Summarization | rubric judge (+ claim-survival check) | small-judge-reliable |
| Open-book comprehension | multiple-choice / short-answer | objective (mostly) |
| Structured output | parser / validator (strict) | **fully objective** |
| Creative generation | pairwise, calibrated judge / human | **judge-hard** |
| Code generation | execution pass@1 | **fully objective** |

Small open-weight LLM judges are validated ≈ human for comprehension/summarization but **fail for creative writing** — don't reuse the same judge across all categories.

---

## 6. Repo layout & how to run

```
edgelm_1p5b/
  models.py          # model registry + per-model ModelSpec (the 4 configs)
  runner.py          # load / chat-template / generate / split reasoning / log tokens
  download.py        # pre-fetch the 4 model weights
  requirements.txt   # deps
  CONTEXT.md         # this file
```

Setup and smoke test:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional but helps with rate limits (none of the 4 are gated):
# huggingface-cli login

python download.py            # pull all four sets of weights
python runner.py --smoke      # plumbing check on all 4 (small token cap, fast)
python runner.py --model deepscaler-1.5b   # one model, full token budget
```

Use it from your category harnesses:

```python
from models import get
from runner import load, generate

spec = get("r1-distill-1.5b")
tok, model = load(spec)                       # dtype pinned to bf16 inside
messages = [{"role": "user", "content": "..."}]   # system prompt auto-dropped for this model
res = generate(tok, model, spec, messages, seed=0)
# res has: answer, reasoning, prompt_tokens, new_tokens, context_window,
#          ctx_exceeded, hit_token_cap, reasoning_truncated, tok_per_s
```

**Hardware:** a single consumer GPU (≥8 GB) runs these one at a time in BF16 (~3 GB each). CPU works but is slow, because the reasoning models emit thousands of tokens — validate with `--smoke` (capped) first. The runner loads one model, runs, frees VRAM, then the next.

**Scaling up to full runs:** for the real benchmark (many prompts × samples), swap the transformers path for **vLLM** (`pip install vllm`) — it batches and handles long reasoning outputs far faster. Keep the same `ModelSpec` settings (`temperature`, `top_p`, `max_new_tokens`) and set `--max-model-len` to each model's context window. The transformers runner here is the low-friction starting point and the reference for correct templating/parsing.

> Note: the model-download path was not executed in the environment where this was written (no GPU, restricted egress), but the code is syntax-checked and the reasoning-split logic is unit-tested. Run `--smoke` once on your machine to confirm weights load.

---

## 7. Full 15-model roster (cross-track awareness)

Qwen-derived: Qwen2.5-1.5B, Qwen2.5-Coder-3B, Qwen3.5-4B, VibeThinker-3B (post-train of Qwen2.5-Coder-3B), Crow-4B (Qwen3.5-4B; Opus-4.6-trace distill). Llama-derived: Llama-3.2-3B → Hermes-3-Llama-3.2-3B (clean pair). Reasoning-distill: DeepSeek-R1-Distill-1.5B → DeepScaleR-1.5B (clean pair), OpenReasoning-Nemotron-1.5B. Other bases: Phi-4-mini, Ministral-3B, SmolLM3-3B, Gemma 3 4B, GLM-Edge-4B.

Three controlled comparisons across the whole benchmark: Llama-3.2-3B↔Hermes-3, R1-Distill-1.5B↔DeepScaleR (yours), Qwen2.5-Coder-3B↔VibeThinker — each isolates post-training on a fixed base.

---

## 8. Decisions already made (so an agent doesn't re-litigate them)

- **Stimuli are human-authored**, sourced from existing human-written datasets, not model-generated. (Datasets per category are in `EdgeLM_related_work.md` §5.)
- **Crow-4B is kept** as a normal roster entry (not removed, not quarantined); its provenance is disclosed factually.
- **Scoring is tiered** (objective / small-judge / judge-hard), not uniform.
- **The benchmark is framed as a test of reasoning–knowledge decoupling**, with per-category predictions.
- **Lineage pairs are reported as controlled comparisons**, not independent points.
