"""
EdgeLM Benchmark Pipeline (Ollama track)
=========================================
Evaluates sub-4B GGUF models served by Ollama across the six EdgeLM task
categories (CONTEXT.md). Consumer-hardware / Ollama counterpart to the
transformers reference runner (runner.py).

Tasks (all stimuli human-authored, see stimuli.py):
  1. Context retention under load  -> probe recall at increasing turn depth
  2. Cross-domain summarization     -> ROUGE-L + key-claim survival, per domain
  3. Open-book comprehension        -> graded literal/inferential/synthesis QA
  4. Structured output compliance   -> strict vs lenient parser/validator
  5. Creative generation            -> objective constraints inline; quality via judge/human
  6. Code generation                -> Python/Perl executed, C# static; pass@1

Reasoning models (R1-style <think>) degenerate at greedy decoding and need
sampling + reasoning/answer splitting + large token budgets (CONTEXT.md §4); see
the DECODE table.

Run:
    python pipeline.py                  # all models in MODELS that have a GGUF / are loaded
    python pipeline.py qwen25-1b5       # one model
    python pipeline.py --smoke          # tiny token caps, fast plumbing check
    python pipeline.py --tasks 4,6 qwen25-1b5   # only some tasks
    python pipeline.py --judge          # also run the LLM-judge pass (Task 5/2)

Deps: pip install requests psutil rouge-score tqdm
"""

import json
import time
import subprocess
import sys
import re
import shutil
import argparse
import requests
import psutil
import os
import tempfile
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from stimuli import (
    SUMMARIZATION_PASSAGES, OPEN_BOOK_ITEMS, RETENTION_SCRIPTS,
    STRUCTURED_TASKS, CREATIVE_TASKS, CODE_TASKS,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
RESULTS_DIR = Path("./results")
MODELS_DIR = Path("../models")
RESULTS_DIR.mkdir(exist_ok=True)

# Default roster = the four downloaded 1.5B GGUFs (CONTEXT.md §2 controlled study).
# Any model without a GGUF / not loaded in Ollama is auto-skipped, so you can add
# more here as you download them.
MODELS = [
    "qwen25-1b5",
    "granite4-1b",
    "deepseek-r1-1b5",
    "deepscaler-1b5",
    "llama3.2:3b",
    "qwen2.5-coder:3b",
    "ministral-3:3b",
    "phi4-mini",
    "vibethinker-3b",
    "hermes3-llama32-3b",
    "smollm3-3b",
    # 4B track
    "gemma3-4b",
    "glm-edge-4b",
    "crow-4b",
    "qwen35-4b",
]

# Optional LLM judge for Task 5 (creative quality) and Task 2 (summarization).
# Off by default; enable with --judge. Uses an Ollama model by default; point at a
# stronger judge for paper-grade results (CONTEXT.md §5: small open judges are
# unreliable for creative writing).
JUDGE_MODEL = os.environ.get("EDGELM_JUDGE_MODEL", "llama3.2:3b")

# ── PER-MODEL DECODING SPEC (CONTEXT.md §4) ──────────────────────────────────
DECODE = {
    "default":         {"temperature": 0.0, "top_p": 1.0,  "think": False, "max_factor": 1.0},
    "deepseek-r1-1b5": {"temperature": 0.6, "top_p": 0.95, "think": True,  "max_factor": 8.0,
                        "task_max_factors": {"creative_generation": 14.0, "code_generation": 14.0}},
    "deepscaler-1b5":  {"temperature": 0.6, "top_p": 0.95, "think": True,  "max_factor": 8.0,
                        "task_max_factors": {"creative_generation": 14.0, "code_generation": 14.0}},
    "nemotron-1b5":    {"temperature": 0.6, "top_p": 0.95, "think": True,  "max_factor": 8.0,
                        "task_max_factors": {"creative_generation": 14.0, "code_generation": 14.0}},
    # roster models for future downloads:
    "vibethinker-3b":  {"temperature": 0.6, "top_p": 0.95, "think": True,  "max_factor": 8.0},
    "crow-4b":         {"temperature": 0.6, "top_p": 0.95, "think": True,  "max_factor": 8.0},
    "qwen35-4b":       {"temperature": 0.6, "top_p": 0.95, "think": True,  "max_factor": 8.0,
                        "task_max_factors": {"creative_generation": 14.0}},
}


def decode_spec(model: str) -> dict:
    return DECODE.get(model, DECODE["default"])


def split_reasoning(text: str):
    """Return (reasoning, answer, truncated). truncated=True if <think> opened but
    never closed (model hit the token cap mid-thought; CONTEXT.md §4).
    Also handles qwen3-style plain-prose thinking (starts with 'Thinking Process:')."""
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        return think.replace("<think>", "").strip(), answer.strip(), False
    if "<think>" in text:
        return text.replace("<think>", "").strip(), "", True
    # qwen3 plain-prose reasoning: no tags, starts with "Thinking Process:"
    if text.lstrip().startswith("Thinking Process:"):
        import re
        # Creative writing: final answer markers like *Final Article: or *Revised Draft:
        m = list(re.finditer(r'\n\s*\*(?:Final [A-Za-z ]+|Revised (?:Draft|Text)|Trimmed)[:\*]', text))
        if m:
            candidate = text[m[-1].end():].strip()
            if len(candidate) > 20:
                return text[:m[-1].start()].strip(), candidate, False
        # Code generation: last complete code fence block
        fence_blocks = list(re.finditer(r'```[^\n]*\n.*?```', text, flags=re.S))
        if fence_blocks:
            last_block = fence_blocks[-1]
            candidate = text[last_block.start():].strip()
            if len(candidate) > 10:
                return text[:last_block.start()].strip(), candidate, False
        return text.strip(), "", True
    return "", text.strip(), False


# ── OLLAMA HELPERS ───────────────────────────────────────────────────────────

def _options(model: str, max_tokens: int, task: str = "") -> dict:
    spec = decode_spec(model)
    factor = spec["max_factor"]
    if task:
        factor = spec.get("task_max_factors", {}).get(task, factor)
    opt = {
        "num_predict": int(max_tokens * factor),
        "temperature": spec["temperature"],
        "num_ctx": 32768,
    }
    if spec["temperature"] > 0:
        opt["top_p"] = spec["top_p"]
    return opt


def ollama_generate(model: str, prompt: str, system: str = "", max_tokens: int = 512,
                    timeout: int = 600, task: str = "") -> dict:
    spec = decode_spec(model)
    options = _options(model, max_tokens, task=task)
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    if system:
        payload["system"] = system
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": str(e), "latency_s": time.perf_counter() - t0, "text": "", "raw": ""}
    latency = time.perf_counter() - t0
    raw = data.get("response", "")
    reasoning, answer, truncated = split_reasoning(raw) if spec["think"] else ("", raw, False)
    eval_count = data.get("eval_count", 0); eval_dur = data.get("eval_duration", 1)
    return {
        "text": answer, "raw": raw, "reasoning_chars": len(reasoning),
        "reasoning_truncated": truncated, "latency_s": latency, "eval_count": eval_count,
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "tokens_per_sec": eval_count / max(eval_dur / 1e9, 1e-9),
        "hit_token_cap": eval_count >= options["num_predict"],
    }


def ollama_chat(model: str, messages: list, max_tokens: int = 512, timeout: int = 600,
                task: str = "") -> dict:
    spec = decode_spec(model)
    options = _options(model, max_tokens, task=task)
    payload = {"model": model, "messages": messages, "stream": False, "options": options}
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": str(e), "latency_s": time.perf_counter() - t0, "text": "", "raw": ""}
    latency = time.perf_counter() - t0
    raw = data.get("message", {}).get("content", "")
    reasoning, answer, truncated = split_reasoning(raw) if spec["think"] else ("", raw, False)
    eval_count = data.get("eval_count", 0); eval_dur = data.get("eval_duration", 1)
    return {
        "text": answer, "raw": raw, "reasoning_chars": len(reasoning),
        "reasoning_truncated": truncated, "latency_s": latency, "eval_count": eval_count,
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "tokens_per_sec": eval_count / max(eval_dur / 1e9, 1e-9),
        "hit_token_cap": eval_count >= options["num_predict"],
    }


def get_memory_usage() -> dict:
    ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, encoding="utf-8", errors="replace")
        vram_used, vram_total = result.stdout.strip().split(", ")
        return {"ram_mb": ram_mb, "vram_used_mb": int(vram_used), "vram_total_mb": int(vram_total)}
    except Exception:
        return {"ram_mb": ram_mb, "vram_used_mb": -1, "vram_total_mb": -1}


def load_model(model_name: str) -> bool:
    """Ensure model is loaded in Ollama (create from GGUF if needed). Returns False
    (skip) if neither an Ollama entry nor a local GGUF exists."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        existing = [m["name"] for m in r.json().get("models", [])]
        if model_name in existing or f"{model_name}:latest" in existing:
            return True
    except Exception:
        pass
    gguf_path = MODELS_DIR / f"{model_name}.gguf"
    if not gguf_path.exists():
        print(f"  - skip {model_name}: no Ollama entry and no GGUF at {gguf_path}")
        return False
    modelfile_path = MODELS_DIR / f"{model_name}.Modelfile"
    modelfile_path.write_text(f'FROM {gguf_path.absolute()}\n')
    print(f"  Creating Ollama model {model_name}...")
    result = subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)],
                            capture_output=True, text=True, timeout=300,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"  X Failed to create {model_name}: {result.stderr}")
        return False
    print(f"  + {model_name} ready")
    return True


# ── SCORING UTILITIES ────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def claim_survived(summary: str, variants: list) -> bool:
    low = summary.lower()
    return any(v.lower() in low for v in variants)


def answer_match(predicted: str, accept: list) -> bool:
    """Correct if any accepted substring appears in the prediction. Phrase matches
    use raw substring; short/numeric tokens require whole-token match (so "5" does
    not match inside "25")."""
    pred_norm = _norm(predicted)
    pred_tokens = set(pred_norm.split())
    for a in accept:
        a_low = a.lower().strip()
        if not a_low:
            continue
        if " " in a_low or "-" in a_low:
            if a_low in predicted.lower():
                return True
        elif len(a_low) <= 3 or a_low.replace(",", "").isdigit():
            if a_low in pred_tokens or a_low.replace(",", "") in pred_tokens:
                return True
        else:
            if a_low in pred_norm:
                return True
    return False


_ROUGE = None
def _rouge_l(prediction: str, reference: str):
    global _ROUGE
    if _ROUGE is None:
        try:
            from rouge_score import rouge_scorer
            _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        except Exception:
            _ROUGE = False
    if not _ROUGE:
        return None
    sc = _ROUGE.score(reference, prediction)
    return {k: round(v.fmeasure, 4) for k, v in sc.items()}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


# ── TASK 1: CONTEXT RETENTION UNDER LOAD ─────────────────────────────────────

def _fact_check(response_text: str, fact_variants: list) -> bool:
    """Same matching style as answer_match — normalized substring/token check."""
    if not fact_variants:
        return False
    norm = _norm(response_text)
    tokens = set(norm.split())
    for v in fact_variants:
        v_low = v.lower().strip()
        if " " in v_low or "-" in v_low:
            if v_low in response_text.lower():
                return True
        elif len(v_low) <= 3 or v_low.replace(",", "").isdigit():
            if v_low in tokens:
                return True
        else:
            if v_low in norm:
                return True
    return False


def _score_probe(response_text: str, facts: dict, fact_ids: list) -> dict:
    """Score a single probe response against a set of facts.
    Returns per-fact hit/miss/stale_contamination plus aggregate score."""
    results = {}
    for fid in fact_ids:
        fact = facts[fid]
        current_hit = _fact_check(response_text, fact["current"])
        stale_hit = _fact_check(response_text, fact.get("stale", []))
        if current_hit:
            status = "correct"
        elif stale_hit:
            status = "stale_contamination"
        else:
            status = "missing"
        results[fid] = status
    correct = sum(1 for s in results.values() if s == "correct")
    stale = sum(1 for s in results.values() if s == "stale_contamination")
    score = round(correct / len(fact_ids), 3) if fact_ids else 0.0
    return {"per_fact": results, "score": score, "stale_count": stale, "total_facts": len(fact_ids)}


def _run_retention_script(model: str, script_obj: dict, max_tokens: int) -> dict:
    messages = []
    turn_log = []
    mid_probe_result = None
    synthesis_probe_result = None

    for i, turn in enumerate(script_obj["turns"]):
        messages.append({"role": "user", "content": turn["content"]})
        resp = ollama_chat(model, messages, max_tokens=max_tokens, task="context_retention")
        response_text = resp.get("text", "")
        messages.append({"role": "assistant", "content": response_text})
        turn_log.append({
            "turn": i, "user": turn["content"], "response": response_text,
            "latency_s": round(resp.get("latency_s", 0), 2),
            "is_probe": turn.get("is_probe"),
        })

        if turn.get("is_probe") == "mid":
            mid_probe_result = _score_probe(response_text, script_obj["facts"],
                                            script_obj["facts_by_mid_probe"])
        elif turn.get("is_probe") == "synthesis":
            all_fact_ids = list(script_obj["facts"].keys())
            synthesis_probe_result = _score_probe(response_text, script_obj["facts"], all_fact_ids)

    weighted_score = round(
        0.3 * (mid_probe_result["score"] if mid_probe_result else 0)
        + 0.7 * (synthesis_probe_result["score"] if synthesis_probe_result else 0), 3)

    return {
        "id": script_obj["id"],
        "character": script_obj["character"],
        "mid_probe": mid_probe_result,
        "synthesis_probe": synthesis_probe_result,
        "weighted_score": weighted_score,
        "turns": turn_log,
    }


def task_context_retention(model: str, smoke: bool = False) -> dict:
    max_tokens = 80 if smoke else 500
    scripts = RETENTION_SCRIPTS[:2] if smoke else RETENTION_SCRIPTS
    mem_before = get_memory_usage()
    results = [_run_retention_script(model, s, max_tokens) for s in scripts]
    mem_after = get_memory_usage()

    overall_score = _mean([r["weighted_score"] for r in results])
    total_stale = sum(r["synthesis_probe"]["stale_count"] for r in results if r["synthesis_probe"])
    total_facts_checked = sum(r["synthesis_probe"]["total_facts"] for r in results if r["synthesis_probe"])
    stale_contamination_rate = round(total_stale / total_facts_checked, 3) if total_facts_checked else 0.0

    return {
        "task": "context_retention",
        "score": overall_score,
        "stale_contamination_rate": stale_contamination_rate,
        "scripts": results,
        "mem_before": mem_before,
        "mem_after": mem_after,
    }


# ── TASK 2: CROSS-DOMAIN SUMMARIZATION ───────────────────────────────────────

def task_summarization(model: str, smoke: bool = False) -> dict:
    max_tokens = 200 if smoke else 350
    items = SUMMARIZATION_PASSAGES[:3] if smoke else SUMMARIZATION_PASSAGES
    results = []
    for content in items:
        prompt = ("Summarize the following text in 3-5 sentences. Preserve the key facts.\n\n"
                  + content["text"])
        resp = ollama_generate(model, prompt, max_tokens=max_tokens, task="summarization")
        summary = resp.get("text", "")
        survived = [claim_survived(summary, c) for c in content["key_claims"]]
        results.append({
            "id": content["id"], "domain": content["domain"], "source": content["source"],
            "summary": summary, "rouge": _rouge_l(summary, content["reference_summary"]),
            "claims_survived": sum(survived), "claims_total": len(content["key_claims"]),
            "claim_survival_rate": round(sum(survived) / len(content["key_claims"]), 3),
            "latency_s": round(resp.get("latency_s", 0), 2),
            "tokens_per_sec": round(resp.get("tokens_per_sec", 0), 1),
            "reasoning_truncated": resp.get("reasoning_truncated", False),
        })
    by_domain = {}
    for dom in ("prose", "mathematical", "scientific"):
        sub = [r["claim_survival_rate"] for r in results if r["domain"] == dom]
        if sub:
            by_domain[dom] = _mean(sub)
    drop = None
    if all(d in by_domain for d in ("prose", "mathematical", "scientific")):
        drop = round(by_domain["prose"] - (by_domain["mathematical"] + by_domain["scientific"]) / 2, 3)
    return {"task": "summarization",
            "claim_survival_by_domain": by_domain,
            "faithfulness_drop_prose_minus_technical": drop,
            "mean_rouge_l": _mean([r["rouge"]["rougeL"] for r in results if r.get("rouge")]),
            "results": results}


# ── TASK 3: OPEN-BOOK COMPREHENSION ──────────────────────────────────────────

def task_open_book_qa(model: str, smoke: bool = False) -> dict:
    max_tokens = 48 if smoke else 160
    items = OPEN_BOOK_ITEMS[:3] if smoke else OPEN_BOOK_ITEMS
    results = []
    for item in items:
        qrs = []
        for qa in item["questions"]:
            prompt = ("Read the following passage and answer the question using ONLY information "
                      "from the passage. Answer concisely.\n\nPassage:\n"
                      f"{item['passage']}\n\nQuestion: {qa['q']}\n\nAnswer:")
            resp = ollama_generate(model, prompt, max_tokens=max_tokens, task="open_book_qa")
            predicted = resp.get("text", "").strip()
            qrs.append({"difficulty": qa["difficulty"], "question": qa["q"], "gold": qa["gold"],
                        "predicted": predicted, "correct": answer_match(predicted, qa["accept"]),
                        "latency_s": round(resp.get("latency_s", 0), 2),
                        "reasoning_truncated": resp.get("reasoning_truncated", False)})
        results.append({"id": item["id"], "domain": item["domain"], "source": item["source"],
                        "questions": qrs,
                        "accuracy": round(sum(q["correct"] for q in qrs) / len(qrs), 3)})
    flat = [q for d in results for q in d["questions"]]
    by_tier = {}
    for tier in ("easy", "medium", "hard"):
        tq = [q for q in flat if q["difficulty"] == tier]
        if tq:
            by_tier[tier] = round(sum(q["correct"] for q in tq) / len(tq), 3)
    return {"task": "open_book_qa",
            "overall_accuracy": round(sum(q["correct"] for q in flat) / len(flat), 3) if flat else 0.0,
            "accuracy_by_tier": by_tier, "results": results}


# ── TASK 4: STRUCTURED OUTPUT COMPLIANCE ─────────────────────────────────────

def task_structured_output(model: str, smoke: bool = False) -> dict:
    max_tokens = 64 if smoke else 256
    items = STRUCTURED_TASKS[:3] if smoke else STRUCTURED_TASKS
    results = []
    for st in items:
        resp = ollama_generate(model, st["prompt"], max_tokens=max_tokens, task="structured_output")
        text = resp.get("text", "").strip()
        strict, lenient, details = VALIDATORS[st["validator"]](text, **st["args"])
        results.append({"name": st["name"], "validator": st["validator"],
                        "strict_compliant": strict, "lenient_compliant": lenient,
                        "details": details, "response": text,
                        "latency_s": round(resp.get("latency_s", 0), 2),
                        "tokens_per_sec": round(resp.get("tokens_per_sec", 0), 1),
                        "reasoning_truncated": resp.get("reasoning_truncated", False)})
    return {"task": "structured_output",
            "strict_compliance_rate": round(sum(r["strict_compliant"] for r in results) / len(results), 3),
            "lenient_compliance_rate": round(sum(r["lenient_compliant"] for r in results) / len(results), 3),
            "results": results}


def _v_json(text, required):
    strict_text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", strict_text, flags=re.S)
    strict = False
    if not fenced:
        try:
            obj = json.loads(strict_text)
            strict = isinstance(obj, dict) and all(k in obj for k in required)
        except Exception:
            strict = False
    lenient = False
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            lenient = isinstance(obj, dict) and all(k in obj for k in required)
        except Exception:
            lenient = False
    return strict, lenient, {"required": required, "had_fence": "```" in text}


def _v_markdown(text):
    has_header = any(l.lstrip().startswith("#") for l in text.split("\n"))
    has_bold = "**" in text or "__" in text
    has_bullet = any(l.strip().startswith(("-", "*", "+")) for l in text.split("\n"))
    lenient = has_header and has_bold and has_bullet
    lines = [l for l in text.split("\n") if l.strip()]
    non_md_lines = [l for l in lines if not (l.lstrip().startswith(("#", "-", "*", "+")) or "**" in l or "__" in l)]
    strict = lenient and len(non_md_lines) <= 2
    return strict, lenient, {"has_header": has_header, "has_bold": has_bold, "has_bullet": has_bullet, "non_md_lines": len(non_md_lines)}


def _v_code_only(text):
    has_def = "def " in text
    has_bt = "```" in text
    return (has_def and not has_bt), has_def, {"has_function": has_def, "no_backticks": not has_bt}


def _v_numbered_list(text, expected_count):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    numbered = [l for l in lines if l and l[0].isdigit()]
    strict = len(numbered) == expected_count and len(lines) == expected_count
    return strict, len(numbered) == expected_count, {
        "numbered": len(numbered), "total_lines": len(lines), "expected": expected_count}


def _v_csv_line(text, fields):
    lines = [l for l in text.split("\n") if l.strip()]
    strict = len(lines) == 1 and len(lines[0].split(",")) == fields
    lenient = any(len(l.split(",")) == fields for l in lines)
    return strict, lenient, {"lines": len(lines), "expected_fields": fields}


VALIDATORS = {
    "json": _v_json, "markdown": _v_markdown, "code_only": _v_code_only,
    "numbered_list": _v_numbered_list, "csv_line": _v_csv_line,
}


# ── TASK 5: CREATIVE GENERATION ──────────────────────────────────────────────

def task_creative_generation(model: str, smoke: bool = False) -> dict:
    items = CREATIVE_TASKS[:3] if smoke else CREATIVE_TASKS
    results = []
    for item in items:
        max_tokens = min(item["max_tokens"], 150) if smoke else item["max_tokens"]
        resp = ollama_generate(model, item["prompt"], max_tokens=max_tokens, task="creative_generation")
        text = resp.get("text", "").strip()
        words = len(text.split())
        lines = [l for l in text.split("\n") if l.strip()]
        checks = {}
        if "target_words" in item:
            tw = item["target_words"]; checks["word_count_within_30pct"] = abs(words - tw) <= 0.3 * tw
        if "exact_words" in item:
            checks["exact_word_count"] = words == item["exact_words"]
        if "target_lines" in item:
            checks["line_count_exact"] = len(lines) == item["target_lines"]
        if "must_start" in item:
            checks["starts_with_required"] = text.lower().lstrip("\"' ").startswith(item["must_start"])
        if "acrostic" in item:
            initials = "".join(l.strip()[0:1].upper() for l in lines)
            checks["acrostic_ok"] = initials.startswith(item["acrostic"])
        if item.get("line_prefix_colon"):
            checks["all_lines_have_name_colon"] = all(":" in l.split(" ")[0] or re.match(r"^\w+\s*:", l) for l in lines) and len(lines) > 0
        if item.get("question_word_lines"):
            _qwords = {"who", "what", "where", "when", "why", "how"}
            checks["line_count_exact"] = len(lines) == 12
            checks["all_lines_start_question_word"] = (
                len(lines) > 0
                and all(l.split()[0].lower() in _qwords for l in lines)
            )
        results.append({"type": item["type"], "text": text, "word_count": words,
                        "line_count": len(lines), "constraints": item["constraints"],
                        "constraint_checks": checks,
                        "constraint_pass_rate": round(sum(bool(v) for v in checks.values()) / len(checks), 3) if checks else None,
                        "latency_s": round(resp.get("latency_s", 0), 2),
                        "tokens_per_sec": round(resp.get("tokens_per_sec", 0), 1),
                        "human_score": None, "judge_score": None})
    return {"task": "creative_generation",
            "mean_constraint_pass_rate": _mean([r["constraint_pass_rate"] for r in results]),
            "results": results}


# ── TASK 6: CODE GENERATION ──────────────────────────────────────────────────

def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```[a-zA-Z0-9#+]*\s*\n?(.*?)\n?```", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    for lang in ("python", "ruby", "perl", "rb", "pl"):
        if t.lower().startswith(lang):
            t = t[len(lang):].lstrip(":\n ").strip()
            break
    return t


def _run_subprocess(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERR:{e}"


def _exec_python(code: str, test: str):
    harness = code + "\n\n# ---- EdgeLM test harness ----\n" + test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(harness); fname = f.name
    out = _run_subprocess([sys.executable, fname])
    try:
        os.unlink(fname)
    except OSError:
        pass
    return ("EDGELM_PASS" in out), out[-800:]


_PERL_FALLBACKS = [
    r"C:\Program Files\Git\usr\bin\perl.exe",
    r"C:\Strawberry\perl\bin\perl.exe",
    r"C:\Perl64\bin\perl.exe",
    r"C:\Perl\bin\perl.exe",
    "/opt/homebrew/bin/perl",
    "/usr/local/bin/perl",
    "/usr/bin/perl",
]


def _exec_perl(code: str, fixture: str, expect_substrings: list, ordered: bool = False):
    perl_path = shutil.which("perl")
    if not perl_path:
        for p in _PERL_FALLBACKS:
            if os.path.isfile(p):
                perl_path = p
                break
    if not perl_path:
        return None, "perl-not-installed"
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "s.pl"); fp = os.path.join(d, "f.txt")
    open(sp, "w", encoding="utf-8").write(code)
    open(fp, "w", encoding="utf-8").write(fixture)
    out = _run_subprocess([perl_path, sp, fp])
    shutil.rmtree(d, ignore_errors=True)
    if ordered:
        indices = [out.find(s) for s in expect_substrings]
        passed = all(i >= 0 for i in indices) and indices == sorted(indices)
    else:
        passed = all(s in out for s in expect_substrings)
    return passed, out[-400:]


def _exec_ruby(code: str, fixture: str, expect_substrings: list):
    ruby_path = shutil.which("ruby")
    if not ruby_path:
        fallback = r"C:\Ruby33-x64\bin\ruby.exe"
        if os.path.exists(fallback):
            ruby_path = fallback
    if not ruby_path:
        return None, "ruby-not-installed"
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "s.rb"); fp = os.path.join(d, "f.txt")
    open(sp, "w", encoding="utf-8").write(code)
    open(fp, "w", encoding="utf-8").write(fixture)
    out = _run_subprocess([ruby_path, sp, fp])
    shutil.rmtree(d, ignore_errors=True)
    return all(s in out for s in expect_substrings), out[-400:]


def task_code_generation(model: str, smoke: bool = False) -> dict:
    items = CODE_TASKS[:3] if smoke else CODE_TASKS
    results = []
    for task in items:
        max_tokens = min(task["max_tokens"], 150) if smoke else task["max_tokens"]
        resp = ollama_generate(model, task["prompt"], max_tokens=max_tokens, task="code_generation")
        code = _strip_code_fence(resp.get("text", ""))
        passed, exec_output = None, None
        if task["language"] == "perl":
            if "exec_perl" in task:
                passed, exec_output = _exec_perl(
                    code, task["exec_perl"]["fixture"],
                    task["exec_perl"]["expect_substrings"],
                    ordered=task["exec_perl"].get("ordered", False),
                )
            if passed is None and "static" in task:
                passed = bool(task["static"](code)); exec_output = "static-fallback"
        elif task["language"] == "ruby":
            if "exec_ruby" in task:
                passed, exec_output = _exec_ruby(code, task["exec_ruby"]["fixture"],
                                                 task["exec_ruby"]["expect_substrings"])
            if passed is None and "static" in task:
                passed = bool(task["static"](code)); exec_output = "static-fallback"
        results.append({"name": task["name"], "language": task["language"], "code": code,
                        "passed": passed, "exec_output": exec_output,
                        "latency_s": round(resp.get("latency_s", 0), 2),
                        "tokens_per_sec": round(resp.get("tokens_per_sec", 0), 1),
                        "reasoning_truncated": resp.get("reasoning_truncated", False)})
    scored = [r for r in results if r["passed"] is not None]
    exec_only = [r for r in results if r["exec_output"] not in (None, "static-fallback", "perl-not-installed", "ruby-not-installed")]
    return {"task": "code_generation",
            "pass_at_1": round(sum(1 for r in scored if r["passed"]) / len(scored), 3) if scored else 0.0,
            "executed_pass_at_1": round(sum(1 for r in exec_only if r["passed"]) / len(exec_only), 3) if exec_only else None,
            "results": results}


# ── OPTIONAL LLM-JUDGE PASS (Task 5 quality + Task 2 faithfulness) ────────────

def judge_pass(model_results: dict, judge_model: str = JUDGE_MODEL) -> dict:
    """Calibrated-ish judge pass run AFTER the benchmark. Off by default (--judge).
    For creative writing CONTEXT.md §5 warns small open judges are unreliable --
    treat these as scaffolding; point EDGELM_JUDGE_MODEL at a stronger judge."""
    def score_1to5(instruction, content):
        prompt = (instruction + "\n\nRespond with ONLY a single integer from 1 to 5.\n\n"
                  "TEXT:\n" + content[:2000] + "\n\nScore (1-5):")
        resp = ollama_generate(judge_model, prompt, max_tokens=8)
        m = re.search(r"[1-5]", resp.get("text", ""))
        return int(m.group(0)) if m else None

    out = {"judge_model": judge_model, "creative": [], "summarization": []}
    cg = model_results.get("tasks", {}).get("creative_generation", {})
    for r in cg.get("results", []):
        s = score_1to5("Rate the overall quality (creativity, coherence, fluency) of this piece.", r.get("text", ""))
        r["judge_score"] = s
        out["creative"].append({"type": r.get("type"), "judge_score": s})
    sm = model_results.get("tasks", {}).get("summarization", {})
    for r in sm.get("results", []):
        s = score_1to5("Rate how faithfully this summary preserves the key facts (1=poor, 5=excellent).", r.get("summary", ""))
        r["judge_faithfulness"] = s
        out["summarization"].append({"id": r.get("id"), "judge_faithfulness": s})
    return out


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

TASKS = [
    task_context_retention,
    task_summarization,
    task_open_book_qa,
    task_structured_output,
    task_creative_generation,
    task_code_generation,
]


def _headline(model_results: dict) -> dict:
    t = model_results.get("tasks", {})
    return {
        "context_retention": t.get("context_retention", {}).get("score"),
        "context_retention_stale_rate": t.get("context_retention", {}).get("stale_contamination_rate"),
        "summarization_claim_survival": t.get("summarization", {}).get("claim_survival_by_domain"),
        "summarization_faithfulness_drop": t.get("summarization", {}).get("faithfulness_drop_prose_minus_technical"),
        "summarization_mean_rouge_l": t.get("summarization", {}).get("mean_rouge_l"),
        "open_book_accuracy": t.get("open_book_qa", {}).get("overall_accuracy"),
        "open_book_by_tier": t.get("open_book_qa", {}).get("accuracy_by_tier"),
        "structured_strict": t.get("structured_output", {}).get("strict_compliance_rate"),
        "structured_lenient": t.get("structured_output", {}).get("lenient_compliance_rate"),
        "creative_constraints": t.get("creative_generation", {}).get("mean_constraint_pass_rate"),
        "code_pass_at_1": t.get("code_generation", {}).get("pass_at_1"),
    }


def _fmt_elapsed(seconds) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _print_timing_summary(all_model_results: list):
    completed = [r for r in all_model_results if not r.get("skipped")]
    if not completed:
        return
    task_names = list(dict.fromkeys(
        name for r in completed for name in r.get("tasks", {})
    ))
    label_w, col_w = 24, 12
    header = f"{'Model':<{label_w}}" + "".join(f"{n[:col_w]:>{col_w}}" for n in task_names) + f"{'TOTAL':>{col_w}}"
    bar = "-" * len(header)
    print(f"\n{bar}\n  TIMING SUMMARY\n{bar}")
    print(header)
    print(bar)
    for r in completed:
        row = f"{r['model'][:label_w]:<{label_w}}"
        for name in task_names:
            t = r.get("tasks", {}).get(name, {}).get("elapsed_s")
            row += f"{_fmt_elapsed(t):>{col_w}}"
        row += f"{_fmt_elapsed(r.get('total_elapsed_s')):>{col_w}}"
        print(row)
    print(bar)


def run_model(model_name: str, smoke: bool = False, only_tasks=None, judge=False) -> dict:
    print(f"\n{'='*60}\n  MODEL: {model_name}{'  [SMOKE]' if smoke else ''}\n{'='*60}")
    if not load_model(model_name):
        return {"model": model_name, "skipped": True, "reason": "not available"}

    model_t0 = time.perf_counter()
    model_results = {"model": model_name, "timestamp": datetime.now().isoformat(),
                     "decode_spec": decode_spec(model_name), "smoke": smoke, "tasks": {}}
    tasks = TASKS if not only_tasks else [TASKS[i - 1] for i in only_tasks]
    for task_fn in tqdm(tasks, desc=model_name):
        name = task_fn.__name__.replace("task_", "")
        print(f"\n  Running: {name}")
        task_t0 = time.perf_counter()
        try:
            model_results["tasks"][name] = task_fn(model_name, smoke=smoke)
            elapsed = round(time.perf_counter() - task_t0, 1)
            model_results["tasks"][name]["elapsed_s"] = elapsed
            print(f"  + {name} done  ({_fmt_elapsed(elapsed)})")
        except Exception as e:
            elapsed = round(time.perf_counter() - task_t0, 1)
            print(f"  X {name} failed: {e}  ({_fmt_elapsed(elapsed)})")
            model_results["tasks"][name] = {"error": str(e), "elapsed_s": elapsed}

    if judge:
        print("\n  Running: judge pass")
        try:
            model_results["judge"] = judge_pass(model_results)
        except Exception as e:
            model_results["judge"] = {"error": str(e)}

    model_results["total_elapsed_s"] = round(time.perf_counter() - model_t0, 1)
    model_results["headline"] = _headline(model_results)
    safe_name = model_name.replace(":", "_")
    if smoke:
        smoke_dir = RESULTS_DIR / "smoke"
        smoke_dir.mkdir(exist_ok=True)
        out_path = smoke_dir / f"smoke_{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"\n  [smoke] Saved: {out_path}  (leaderboard not updated)")
    elif only_tasks:
        ts = datetime.now().strftime("%H%M%S%d%m%Y")
        out_path = RESULTS_DIR / f"{safe_name}_partial_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"\n  [partial] Saved: {out_path}  (leaderboard not updated)")
    else:
        out_path = RESULTS_DIR / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")
        rebuild_leaderboard()
    _print_timing_summary([model_results])
    return model_results


def rebuild_leaderboard():
    """Scan all per-model result files and regenerate all_results.json + leaderboard.json.
    Called after every run_model() so adding models one-by-one stays consistent."""
    skip = {"all_results.json", "leaderboard.json"}
    all_results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name in skip or "_partial_" in p.name or "_qualitative" in p.name:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                all_results.append(json.load(f))
        except Exception:
            pass
    with open(RESULTS_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    board = {r["model"]: r.get("headline") for r in all_results if not r.get("skipped")}
    with open(RESULTS_DIR / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(board, f, indent=2, ensure_ascii=False)
    print(f"  Leaderboard updated ({len(all_results)} model(s))")
    return board


def run_all(smoke=False, only_tasks=None, judge=False):
    all_model_results = []
    for m in tqdm(MODELS, desc="All models"):
        all_model_results.append(run_model(m, smoke=smoke, only_tasks=only_tasks, judge=judge))
    board = rebuild_leaderboard()
    _print_timing_summary(all_model_results)
    print(f"\n\nSaved {RESULTS_DIR/'all_results.json'} and leaderboard.json")
    print(json.dumps(board, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="single model id (default: all in MODELS)")
    ap.add_argument("--smoke", action="store_true", help="tiny token caps + fewer items, fast check")
    ap.add_argument("--tasks", help="comma-separated task numbers 1-6 (e.g. 4,6)")
    ap.add_argument("--judge", action="store_true", help="also run the LLM-judge pass")
    ap.add_argument("--rebuild", action="store_true", help="rebuild all_results.json + leaderboard.json from existing result files, no model runs")
    args = ap.parse_args()
    if args.rebuild:
        board = rebuild_leaderboard()
        print(json.dumps(board, indent=2, ensure_ascii=False))
        return
    only = [int(x) for x in args.tasks.split(",")] if args.tasks else None
    if args.model:
        run_model(args.model, smoke=args.smoke, only_tasks=only, judge=args.judge)
    else:
        run_all(smoke=args.smoke, only_tasks=only, judge=args.judge)


if __name__ == "__main__":
    main()
