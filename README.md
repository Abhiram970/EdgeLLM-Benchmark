# EdgeLM — A Capability-First Benchmark for Sub-4B LLMs on Consumer Hardware

EdgeLM evaluates small (sub-4B) language models across **six task families**, each
scored separately, using **human-authored stimuli** and **task-matched scoring**
(deterministic parsers and code execution where outputs are objectively checkable,
calibrated judging elsewhere). It tests the *reasoning–knowledge decoupling
hypothesis*: verifiable reasoning compresses into a compact parametric core while
broad knowledge does not.

Models are served locally through **[Ollama](https://ollama.com)** — no GPU
cloud, no API keys for the models under test. Everything runs on consumer
hardware.

## The six tasks

| # | Task | What it measures | Scoring |
|---|------|------------------|---------|
| 1 | Context retention | Recall of planted facts across turns, under distractor load | exact-match probe recall |
| 2 | Cross-domain summarization | Faithful compression of prose / math / science passages | ROUGE-L + key-claim survival |
| 3 | Open-book comprehension | Graded literal/inferential/synthesis QA over a passage | fuzzy/exact match vs. gold |
| 4 | Structured output | JSON / Markdown / code-only / numbered-list / CSV compliance | parser/validator (strict + lenient) |
| 5 | Creative generation | Article / poem / story / acrostic / dialogue under constraints | objective constraint checks (+ optional judge) |
| 6 | Code generation | Python (executed) + Perl (executed) + C# (static) | execution-based pass@1 |

All stimuli live in [`stimuli.py`](stimuli.py) — the single, auditable home for
every passage, question, and prompt. **All stimuli are human-authored** (sourced
from Encyclopaedia Britannica and canonical CS exercises); the benchmark never
generates its own test content.

## Repository layout

```
pipeline.py        # the benchmark runner (Ollama REST client + all 6 task scorers)
stimuli.py         # all human-authored test content (passages, questions, prompts, gold answers)
dl_daemon.py       # downloads Q4_K_M GGUF weights from Hugging Face into ./models/
models.py          # transformers-track ModelSpec registry (reference; not used by the Ollama pipeline)
download.py        # transformers-track weight prefetch (reference)
requirements.txt   # Python dependencies
main.tex           # the paper (Springer LNCS)
references.bib     # paper bibliography
CONTEXT.md         # full design rationale
```

## Prerequisites

- **Python 3.9+**
- **Ollama** installed and running — install from <https://ollama.com>, then make
  sure the daemon is up (the desktop app, or `ollama serve`). The pipeline talks
  to it at `http://localhost:11434`.
- For Task 6: **Perl** on `PATH` (for the Perl execution test). If Perl is absent,
  that one problem falls back to a static check automatically. Python execution
  always works.

## Setup

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

## Step 1 — Download model weights (GGUF)

`dl_daemon.py` pulls Q4_K_M GGUF files from Hugging Face into `./models/`,
auto-detecting the right quant file per repo. Edit the `MODELS` list inside it to
choose which models to fetch (uncomment the ones you want), then run:

```bash
python dl_daemon.py
```

The default 1.5B track downloads four files:

| GGUF saved as | Hugging Face source |
|---------------|---------------------|
| `models/qwen25-1b5.gguf` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| `models/deepseek-r1-1b5.gguf` | `mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` |
| `models/deepscaler-1b5.gguf` | `bartowski/agentica-org_DeepScaleR-1.5B-Preview-GGUF` |
| `models/nemotron-1b5.gguf` | `mradermacher/OpenReasoning-Nemotron-1.5B-GGUF` |

> GGUF weights are large (the four above total ~4 GB) and are **not** committed to
> the repo (`.gitignore` excludes `models/`). Re-run `dl_daemon.py` on a fresh
> clone.

## Step 2 — Run the benchmark

The pipeline registers each GGUF with Ollama automatically on first use
(`ollama create` from a generated Modelfile), so you don't need to `ollama pull`
anything manually.

```bash
# run all models listed in MODELS (auto-skips any without a GGUF), all 6 tasks
python pipeline.py

# run a single model
python pipeline.py qwen25-1b5

# fast plumbing check (tiny token caps, fewer items) — use this first
python pipeline.py --smoke qwen25-1b5

# only specific tasks (1–6), e.g. structured output + code
python pipeline.py --tasks 4,6 qwen25-1b5

# also run the optional LLM-judge pass (creative quality + summarization faithfulness)
python pipeline.py --judge
```

Which models run is controlled by the `MODELS` list near the top of
[`pipeline.py`](pipeline.py). The default is the four 1.5B models; add an entry
once you've downloaded its GGUF.

### Notes that matter for fair results

- **Reasoning models are slow.** `deepseek-r1-1b5`, `deepscaler-1b5`, and
  `nemotron-1b5` emit long `<think>` traces; a full run takes a while. The
  pipeline samples them (T=0.6, top_p=0.95), splits the reasoning from the final
  answer, and gives them a larger token budget — all configured per model in the
  `DECODE` table.
- **Don't use `--smoke` for real numbers.** Smoke mode only runs the first three
  items per task, so the prose→technical faithfulness drop (the decoupling
  signal) can come out `null`. Run without `--smoke` for paper-grade results.
- **Optional judge.** The `--judge` pass uses an Ollama model
  (`EDGELM_JUDGE_MODEL`, default `qwen25-1b5`). Small open judges are unreliable
  for creative writing — point it at a stronger model for serious use:
  ```bash
  EDGELM_JUDGE_MODEL=gemma3-4b python pipeline.py --judge
  ```

## Step 3 — Read the results

Outputs land in `results/` (gitignored, regenerated each run):

```
results/<model>.json     # full per-model results: every prompt, response, and score
results/all_results.json # all models combined
results/leaderboard.json # compact headline metrics per model
```

The `headline` block in each per-model JSON gives the at-a-glance scores:
context retention, summarization claim-survival by domain (+ faithfulness drop),
open-book accuracy (overall and by difficulty tier), structured-output
strict/lenient rates, creative constraint-pass rate, and code pass@1.

## Adding a model

1. Add `(repo_id, "local-name")` to `MODELS` in `dl_daemon.py` and run it.
2. Add `"local-name"` to the `MODELS` list in `pipeline.py`.
3. If it's a reasoning model that emits `<think>`, add a row to the `DECODE`
   table in `pipeline.py` (sampling + larger `max_factor`).
4. `python pipeline.py local-name`

## Adding or editing stimuli

All test content is in [`stimuli.py`](stimuli.py), grouped by task. Keep stimuli
**human-authored** — copy real passages from a reputable human-written source and
write the gold answers / key-claims yourself. Do not generate stimuli with an LLM.
