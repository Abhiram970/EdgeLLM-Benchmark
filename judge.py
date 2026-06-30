"""
EdgeLM qualitative judge — Tasks 2 (summarization faithfulness) and 5 (creative quality).

Reads results/<model>.json (READ-ONLY). Writes scores to results/<model>_qualitative.json.
Uses Claude Sonnet via Lava's OpenAI-compatible endpoint.

Usage:
    python judge.py gemma3-4b
    python judge.py --all

Requires:
    pip install openai python-dotenv
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai python-dotenv")

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("./results")

API_KEY   = os.environ.get("LAVA_API_KEY", "")
BASE_URL  = os.environ.get("LAVA_BASE_URL", "https://api.lava.so/v1")
MODEL     = os.environ.get("LAVA_MODEL",    "anthropic/claude-sonnet-4-6")

if not API_KEY or API_KEY == "paste_your_key_here":
    sys.exit("Set LAVA_API_KEY in .env before running.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Pull stimulus data (passages + creative prompts) — read-only reference
sys.path.insert(0, str(Path(__file__).parent))
from stimuli import SUMMARIZATION_PASSAGES, CREATIVE_TASKS

PASSAGE_BY_ID = {p["id"]: p for p in SUMMARIZATION_PASSAGES}
CREATIVE_BY_TYPE = {t["type"]: t for t in CREATIVE_TASKS}

# ── Judge helpers ─────────────────────────────────────────────────────────────

TASK2_RUBRIC = """Rate how faithfully this summary preserves the key facts of the source passage. Use the full 1-10 range.

9-10 — All key facts preserved accurately; no hallucinations; domain-specific details correct.
7-8  — Most key facts preserved; minor omissions or slight inaccuracies.
5-6  — Core idea captured but significant facts missing or distorted.
3-4  — Partially captures the source with major errors or omissions.
1-2  — Largely unfaithful; hallucinated content or fundamentally misrepresents the source.

Respond with ONLY a JSON object: {"score": <1-10>, "rationale": "<one sentence>"}"""

TASK5_RUBRIC = """Rate the overall quality of this creative piece (creativity, coherence, voice, fluency).
Ignore whether it satisfies formal constraints — score quality only. Use the full 1-10 range.

9-10 — Excellent: vivid, distinctive voice, engaging and well-crafted throughout.
7-8  — Good: mostly engaging with minor weaknesses in voice or execution.
5-6  — Adequate: meets basic expectations but unremarkable; flat or generic.
3-4  — Weak: lacks creativity, coherence, or feels mechanical.
1-2  — Poor: incoherent, unengaging, or fails to constitute a meaningful piece.

Respond with ONLY a JSON object: {"score": <1-10>, "rationale": "<one sentence>"}"""


def call_judge(prompt: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            # strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            # try to extract score with a fallback regex
            import re
            m = re.search(r'"score"\s*:\s*(10|[1-9])', raw if 'raw' in dir() else "")
            if m:
                return {"score": int(m.group(1)), "rationale": ""}
            print(f"  [warn] JSON parse failed on attempt {attempt+1}: {raw!r:.120}")
        except Exception as e:
            print(f"  [warn] API error on attempt {attempt+1}: {e}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def judge_task2(results: list) -> dict:
    scores = {}
    for r in results:
        item_id = r.get("id")
        passage_entry = PASSAGE_BY_ID.get(item_id)
        if not passage_entry:
            print(f"  [skip] no passage found for id={item_id}")
            continue
        prompt = (
            f"SOURCE PASSAGE:\n{passage_entry['text']}\n\n"
            f"SUMMARY TO RATE:\n{r['summary']}\n\n"
            f"{TASK2_RUBRIC}"
        )
        print(f"  judging {item_id} (domain={r.get('domain')})...", end=" ", flush=True)
        result = call_judge(prompt)
        if result:
            scores[item_id] = result
            print(f"score={result['score']}")
        else:
            scores[item_id] = {"score": None, "rationale": "judge call failed"}
            print("FAILED")
    return scores


def judge_task5(results: list) -> dict:
    scores = {}
    for r in results:
        item_type = r.get("type")
        creative_entry = CREATIVE_BY_TYPE.get(item_type)
        if not creative_entry:
            print(f"  [skip] no prompt found for type={item_type}")
            continue
        constraints_str = ", ".join(r.get("constraints", []))
        prompt = (
            f"ORIGINAL PROMPT:\n{creative_entry['prompt']}\n\n"
            f"CONSTRAINTS (for context only — do NOT score constraint satisfaction):\n{constraints_str}\n\n"
            f"PIECE TO RATE:\n{r['text']}\n\n"
            f"{TASK5_RUBRIC}"
        )
        print(f"  judging {item_type}...", end=" ", flush=True)
        result = call_judge(prompt)
        if result:
            scores[item_type] = result
            print(f"score={result['score']}")
        else:
            scores[item_type] = {"score": None, "rationale": "judge call failed"}
            print("FAILED")
    return scores


# ── Per-model runner ──────────────────────────────────────────────────────────

def judge_model(model_name: str):
    src = RESULTS_DIR / f"{model_name}.json"
    dst = RESULTS_DIR / f"{model_name}_qualitative.json"

    if not src.exists():
        print(f"[skip] {src} not found")
        return

    data = json.loads(src.read_text(encoding="utf-8"))
    tasks = data.get("tasks", {})

    print(f"\n{'='*60}\n  JUDGING: {model_name}\n{'='*60}")

    output = {
        "model": model_name,
        "judge_model": MODEL,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "task2_faithfulness": {},
        "task5_quality": {},
    }

    # Task 2
    sum_results = tasks.get("summarization", {}).get("results", [])
    if sum_results:
        print("\n  Task 2 — Summarization faithfulness:")
        output["task2_faithfulness"] = judge_task2(sum_results)
    else:
        print("  [skip] no summarization results found")

    # Task 5
    cg_results = tasks.get("creative_generation", {}).get("results", [])
    if cg_results:
        print("\n  Task 5 — Creative quality:")
        output["task5_quality"] = judge_task5(cg_results)
    else:
        print("  [skip] no creative_generation results found")

    dst.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {dst}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EdgeLM qualitative judge (Tasks 2 + 5)")
    ap.add_argument("model", nargs="?", help="model id (e.g. gemma3-4b)")
    ap.add_argument("--all", action="store_true", help="judge all result files in results/")
    args = ap.parse_args()

    if args.all:
        models = sorted(
            p.stem for p in RESULTS_DIR.glob("*.json")
            if "_qualitative" not in p.stem
            and "leaderboard" not in p.stem
            and "all_results" not in p.stem
        )
        if not models:
            sys.exit("No result files found in results/")
        print(f"Judging {len(models)} model(s): {', '.join(models)}")
        for m in models:
            judge_model(m)
    elif args.model:
        judge_model(args.model)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
