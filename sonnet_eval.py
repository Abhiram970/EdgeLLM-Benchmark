"""
EdgeLM Benchmark — Claude Sonnet 4.6 frontier baseline (Tasks 2–6, Lava API).
Task 1 (context retention) is skipped — no multi-turn retention scripts.

Usage:
    python sonnet_eval.py

Writes: results/sonnet-claude-4-6.json  (same format as other canonicals)
"""

import json
import os
import re
import sys
import time
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai python-dotenv")

# ── Lava client (same setup as judge.py) ─────────────────────────────────────
API_KEY  = os.environ.get("LAVA_API_KEY", "")
BASE_URL = os.environ.get("LAVA_BASE_URL", "https://api.lava.so/v1")
LAVA_MODEL = os.environ.get("LAVA_MODEL", "anthropic/claude-sonnet-4-6")

if not API_KEY or API_KEY == "paste_your_key_here":
    sys.exit("Set LAVA_API_KEY in .env before running.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── Imports from benchmark ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from stimuli import (
    SUMMARIZATION_PASSAGES, OPEN_BOOK_ITEMS, STRUCTURED_TASKS,
    CREATIVE_TASKS, CODE_TASKS,
)
from pipeline import (
    _rouge_l, answer_match, claim_survived, _mean,
    _strip_code_fence, _exec_perl, _exec_ruby, VALIDATORS,
)

RESULTS_DIR = Path("results")
MODEL_ID    = "sonnet-claude-4-6"


# ── Sonnet call ───────────────────────────────────────────────────────────────
def call_sonnet(prompt: str, max_tokens: int = 512, retries: int = 3) -> dict:
    t0 = time.perf_counter()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LAVA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            return {
                "text": text.strip(), "raw": text,
                "latency_s": round(time.perf_counter() - t0, 2),
                "reasoning_truncated": False,
            }
        except Exception as e:
            print(f"  [warn] API error attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {
        "text": "", "raw": "", "error": "all retries failed",
        "latency_s": round(time.perf_counter() - t0, 2),
        "reasoning_truncated": False,
    }


# ── Task 2: Cross-domain summarisation ───────────────────────────────────────
def task_summarization() -> dict:
    print("\n  Running: summarization")
    t0 = time.perf_counter()
    results = []
    for content in SUMMARIZATION_PASSAGES:
        prompt = ("Summarize the following text in 3-5 sentences. Preserve the key facts.\n\n"
                  + content["text"])
        resp = call_sonnet(prompt, max_tokens=350)
        summary = resp.get("text", "")
        survived = [claim_survived(summary, c) for c in content["key_claims"]]
        results.append({
            "id": content["id"], "domain": content["domain"], "source": content["source"],
            "summary": summary,
            "rouge": _rouge_l(summary, content["reference_summary"]),
            "claims_survived": sum(survived), "claims_total": len(content["key_claims"]),
            "claim_survival_rate": round(sum(survived) / len(content["key_claims"]), 3),
            "latency_s": resp.get("latency_s"),
            "reasoning_truncated": False,
        })
        print(f"    {content['id']:<30} claim_survival={results[-1]['claim_survival_rate']:.3f}  rouge_l={results[-1]['rouge']['rougeL'] if results[-1]['rouge'] else 'N/A'}")

    by_domain = {}
    for dom in ("prose", "mathematical", "scientific"):
        sub = [r["claim_survival_rate"] for r in results if r["domain"] == dom]
        if sub:
            by_domain[dom] = _mean(sub)
    drop = None
    if all(d in by_domain for d in ("prose", "mathematical", "scientific")):
        drop = round(by_domain["prose"] - (by_domain["mathematical"] + by_domain["scientific"]) / 2, 3)

    out = {
        "task": "summarization",
        "claim_survival_by_domain": by_domain,
        "faithfulness_drop_prose_minus_technical": drop,
        "mean_rouge_l": _mean([r["rouge"]["rougeL"] for r in results if r.get("rouge")]),
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(f"  + summarization done  mean_rouge_l={out['mean_rouge_l']}  claim_survival={by_domain}")
    return out


# ── Task 3: Open-book QA ──────────────────────────────────────────────────────
def task_open_book_qa() -> dict:
    print("\n  Running: open_book_qa")
    t0 = time.perf_counter()
    results = []
    for item in OPEN_BOOK_ITEMS:
        qrs = []
        for qa in item["questions"]:
            prompt = (
                "Read the following passage and answer the question using ONLY information "
                "from the passage. Answer concisely.\n\n"
                f"Passage:\n{item['passage']}\n\nQuestion: {qa['q']}\n\nAnswer:"
            )
            resp = call_sonnet(prompt, max_tokens=160)
            predicted = resp.get("text", "").strip()
            correct = answer_match(predicted, qa["accept"])
            qrs.append({
                "difficulty": qa["difficulty"], "question": qa["q"],
                "gold": qa["gold"], "predicted": predicted, "correct": correct,
                "latency_s": resp.get("latency_s"),
                "reasoning_truncated": False,
            })
        acc = round(sum(q["correct"] for q in qrs) / len(qrs), 3)
        results.append({
            "id": item["id"], "domain": item["domain"], "source": item["source"],
            "questions": qrs, "accuracy": acc,
        })
        print(f"    {item['id']:<30} accuracy={acc:.3f}")

    flat = [q for d in results for q in d["questions"]]
    by_tier = {}
    for tier in ("easy", "medium", "hard"):
        tq = [q for q in flat if q["difficulty"] == tier]
        if tq:
            by_tier[tier] = round(sum(q["correct"] for q in tq) / len(tq), 3)

    overall = round(sum(q["correct"] for q in flat) / len(flat), 3) if flat else 0.0
    out = {
        "task": "open_book_qa",
        "overall_accuracy": overall,
        "accuracy_by_tier": by_tier,
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(f"  + open_book_qa done  overall_accuracy={overall}  by_tier={by_tier}")
    return out


# ── Task 4: Structured output ─────────────────────────────────────────────────
def task_structured_output() -> dict:
    print("\n  Running: structured_output")
    t0 = time.perf_counter()
    results = []
    for st in STRUCTURED_TASKS:
        resp = call_sonnet(st["prompt"], max_tokens=256)
        text = resp.get("text", "").strip()
        strict, lenient, details = VALIDATORS[st["validator"]](text, **st["args"])
        results.append({
            "name": st["name"], "validator": st["validator"],
            "strict_compliant": strict, "lenient_compliant": lenient,
            "details": details, "response": text,
            "latency_s": resp.get("latency_s"),
            "reasoning_truncated": False,
        })
        print(f"    {st['name']:<30} strict={strict}  lenient={lenient}")

    n = len(results)
    out = {
        "task": "structured_output",
        "strict_compliance_rate":  round(sum(r["strict_compliant"]  for r in results) / n, 3),
        "lenient_compliance_rate": round(sum(r["lenient_compliant"] for r in results) / n, 3),
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(f"  + structured_output done  strict={out['strict_compliance_rate']}  lenient={out['lenient_compliance_rate']}")
    return out


# ── Task 5: Creative generation ───────────────────────────────────────────────
def task_creative_generation() -> dict:
    print("\n  Running: creative_generation")
    t0 = time.perf_counter()
    results = []
    for item in CREATIVE_TASKS:
        resp = call_sonnet(item["prompt"], max_tokens=item["max_tokens"])
        text = resp.get("text", "").strip()
        words = len(text.split())
        lines = [l for l in text.split("\n") if l.strip()]

        checks = {}
        if "target_words" in item:
            tw = item["target_words"]
            checks["word_count_within_30pct"] = abs(words - tw) <= 0.3 * tw
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
            checks["all_lines_have_name_colon"] = (
                all(":" in l.split(" ")[0] or re.match(r"^\w+\s*:", l) for l in lines)
                and len(lines) > 0
            )
        if item.get("question_word_lines"):
            _qwords = {"who", "what", "where", "when", "why", "how"}
            checks["line_count_exact"] = len(lines) == 12
            checks["all_lines_start_question_word"] = (
                len(lines) > 0
                and all(l.split()[0].lower() in _qwords for l in lines)
            )

        cpr = round(sum(bool(v) for v in checks.values()) / len(checks), 3) if checks else None
        results.append({
            "type": item["type"], "text": text, "word_count": words,
            "line_count": len(lines), "constraints": item["constraints"],
            "constraint_checks": checks,
            "constraint_pass_rate": cpr,
            "latency_s": resp.get("latency_s"),
            "reasoning_truncated": False,
            "human_score": None, "judge_score": None,
        })
        print(f"    {item['type']:<25} cpr={cpr}  words={words}")

    out = {
        "task": "creative_generation",
        "mean_constraint_pass_rate": _mean([r["constraint_pass_rate"] for r in results]),
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(f"  + creative_generation done  mean_cpr={out['mean_constraint_pass_rate']}")
    return out


# ── Task 6: Code generation ───────────────────────────────────────────────────
def task_code_generation() -> dict:
    print("\n  Running: code_generation")
    t0 = time.perf_counter()
    results = []
    for task in CODE_TASKS:
        resp = call_sonnet(task["prompt"], max_tokens=task["max_tokens"])
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
                passed = bool(task["static"](code))
                exec_output = "static-fallback"
        elif task["language"] == "ruby":
            if "exec_ruby" in task:
                passed, exec_output = _exec_ruby(
                    code, task["exec_ruby"]["fixture"],
                    task["exec_ruby"]["expect_substrings"],
                )
            if passed is None and "static" in task:
                passed = bool(task["static"](code))
                exec_output = "static-fallback"

        results.append({
            "name": task["name"], "language": task["language"],
            "code": code, "passed": passed, "exec_output": exec_output,
            "latency_s": resp.get("latency_s"),
            "reasoning_truncated": False,
        })
        print(f"    {task['name']:<30} passed={passed}")

    scored   = [r for r in results if r["passed"] is not None]
    exec_only = [r for r in results if r["exec_output"] not in
                 (None, "static-fallback", "perl-not-installed", "ruby-not-installed")]
    out = {
        "task": "code_generation",
        "pass_at_1": round(sum(1 for r in scored if r["passed"]) / len(scored), 3) if scored else 0.0,
        "executed_pass_at_1": (
            round(sum(1 for r in exec_only if r["passed"]) / len(exec_only), 3)
            if exec_only else None
        ),
        "results": results,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(f"  + code_generation done  pass_at_1={out['pass_at_1']}")
    return out


# ── Headline ──────────────────────────────────────────────────────────────────
def _headline(tasks: dict) -> dict:
    return {
        "context_retention":              None,
        "context_retention_stale_rate":   None,
        "summarization_claim_survival":   tasks.get("summarization", {}).get("claim_survival_by_domain"),
        "summarization_faithfulness_drop":tasks.get("summarization", {}).get("faithfulness_drop_prose_minus_technical"),
        "summarization_mean_rouge_l":     tasks.get("summarization", {}).get("mean_rouge_l"),
        "open_book_accuracy":             tasks.get("open_book_qa",  {}).get("overall_accuracy"),
        "open_book_by_tier":              tasks.get("open_book_qa",  {}).get("accuracy_by_tier"),
        "structured_strict":              tasks.get("structured_output", {}).get("strict_compliance_rate"),
        "structured_lenient":             tasks.get("structured_output", {}).get("lenient_compliance_rate"),
        "creative_constraints":           tasks.get("creative_generation", {}).get("mean_constraint_pass_rate"),
        "code_pass_at_1":                 tasks.get("code_generation", {}).get("pass_at_1"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}\n  MODEL: {MODEL_ID}  [frontier baseline, Lava API]\n{'='*60}")
    print(f"  API model: {LAVA_MODEL}\n  Task 1 (context_retention): SKIPPED\n")

    t0 = time.perf_counter()
    tasks = {}
    tasks["summarization"]      = task_summarization()
    tasks["open_book_qa"]       = task_open_book_qa()
    tasks["structured_output"]  = task_structured_output()
    tasks["creative_generation"]= task_creative_generation()
    tasks["code_generation"]    = task_code_generation()

    total = round(time.perf_counter() - t0, 1)

    result = {
        "model":       MODEL_ID,
        "api_model":   LAVA_MODEL,
        "timestamp":   datetime.now().isoformat(),
        "frontier":    True,
        "tasks":       tasks,
        "total_elapsed_s": total,
        "headline":    _headline(tasks),
    }

    out_path = RESULTS_DIR / f"{MODEL_ID}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {out_path}")

    # ── Score summary ─────────────────────────────────────────────────────────
    h = result["headline"]
    def f(v): return f"{v:.3f}" if v is not None else "  N/A"
    cs = h.get("summarization_claim_survival") or {}

    print(f"\n{'='*60}")
    print(f"  SCORE SUMMARY — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Task 1  Context Retention      {'SKIPPED':>8}")
    print(f"  Task 2  Summ. ROUGE-L          {f(h.get('summarization_mean_rouge_l')):>8}")
    print(f"          Claim survival prose    {f(cs.get('prose')):>8}")
    print(f"          Claim survival math     {f(cs.get('mathematical')):>8}")
    print(f"          Claim survival sci      {f(cs.get('scientific')):>8}")
    print(f"          Faithfulness drop       {f(h.get('summarization_faithfulness_drop')):>8}")
    print(f"  Task 3  Open-Book Accuracy     {f(h.get('open_book_accuracy')):>8}")
    bt = h.get("open_book_by_tier") or {}
    print(f"          Easy                    {f(bt.get('easy')):>8}")
    print(f"          Medium                  {f(bt.get('medium')):>8}")
    print(f"          Hard                    {f(bt.get('hard')):>8}")
    print(f"  Task 4  Structured Strict      {f(h.get('structured_strict')):>8}")
    print(f"          Structured Lenient      {f(h.get('structured_lenient')):>8}")
    print(f"  Task 5  Creative Constraints   {f(h.get('creative_constraints')):>8}")
    print(f"  Task 6  Code Pass@1            {f(h.get('code_pass_at_1')):>8}")
    print(f"{'='*60}")
    print(f"  Total elapsed: {total:.1f}s")


if __name__ == "__main__":
    main()
