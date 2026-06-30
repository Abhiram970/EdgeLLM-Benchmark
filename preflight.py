"""
EdgeLM preflight check -- run before any pipeline.py invocation.
Checks: Python packages, Ollama connectivity, model availability, Perl, Ruby.

Usage:
    python preflight.py              # check everything
    python preflight.py --models     # also check all 15 models are pulled
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"

MODELS = [
    "qwen25-1b5",
    "deepseek-r1-1b5",
    "deepscaler-1b5",
    "nemotron-1b5",
    "llama3.2:3b",
    "qwen2.5-coder:3b",
    "ministral-3:3b",
    "phi4-mini:latest",
    "vibethinker-3b",
    "hermes3-llama32-3b",
    "smollm3-3b",
    "gemma3-4b",
    "glm-edge-4b",
    "crow-4b",
    "qwen35-4b",
]

RUBY_FALLBACK_PATHS = [
    r"C:\Ruby33-x64\bin\ruby.exe",
    r"C:\Ruby32-x64\bin\ruby.exe",
    r"C:\Ruby31-x64\bin\ruby.exe",
    "/opt/homebrew/bin/ruby",
    "/usr/local/bin/ruby",
    "/usr/bin/ruby",
]

PASS  = "\033[32m[PASS]\033[0m"
FAIL  = "\033[31m[FAIL]\033[0m"
WARN  = "\033[33m[WARN]\033[0m"
INFO  = "\033[36m[INFO]\033[0m"

errors   = []
warnings = []


def ok(label):
    print(f"  {PASS} {label}")


def fail(label, fix=None):
    print(f"  {FAIL} {label}")
    if fix:
        print(f"         -> {fix}")
    errors.append(label)


def warn(label, fix=None):
    print(f"  {WARN} {label}")
    if fix:
        print(f"         -> {fix}")
    warnings.append(label)


def info(label):
    print(f"  {INFO} {label}")


# -- 1. PYTHON PACKAGES --------------------------------------------------------

def check_packages():
    print("\n-- Python packages --------------------------------------------------")

    required = {
        "requests":    "pip install requests",
        "psutil":      "pip install psutil",
        "tqdm":        "pip install tqdm",
        "rouge_score": "pip install rouge-score==0.1.2",
        "nltk":        "pip install nltk==3.9.4",
    }

    for pkg, install_cmd in required.items():
        try:
            importlib.import_module(pkg)
            ok(pkg)
        except ImportError:
            severity = warn if pkg == "nltk" else fail
            severity(
                f"{pkg} not installed -- Task 2 ROUGE scores will be null" if pkg in ("rouge_score", "nltk")
                else f"{pkg} not installed",
                fix=install_cmd,
            )

    # nltk punkt tokenizer (needed by rouge-score)
    try:
        import nltk
        tokenizer_ok = False
        for tok in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(tok)
                ok(f"nltk {tok.split('/')[1]} tokenizer data")
                tokenizer_ok = True
                break
            except LookupError:
                continue
        if not tokenizer_ok:
            warn(
                "nltk tokenizer data missing -- ROUGE-L may error",
                fix="python -c \"import nltk; nltk.download('punkt_tab')\"",
            )
    except ImportError:
        pass  # already reported above


# -- 2. OLLAMA -----------------------------------------------------------------

def check_ollama(check_models=False):
    print("\n-- Ollama -----------------------------------------------------------")

    try:
        import requests as req
        r = req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        ok(f"Ollama reachable at {OLLAMA_URL}")
    except Exception as e:
        fail(
            f"Ollama not reachable at {OLLAMA_URL} ({e})",
            fix="Start Ollama: `ollama serve`  (or open the Ollama desktop app)",
        )
        if check_models:
            warn("Skipping model checks -- Ollama is not running")
        return

    if not check_models:
        info("Skipping model availability check (pass --models to enable)")
        return

    print("\n-- Model availability -----------------------------------------------")
    try:
        import requests as req
        r = req.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        existing = {m["name"] for m in r.json().get("models", [])}
        # also match bare names against name:latest
        existing_bare = {n.split(":")[0] for n in existing}
    except Exception as e:
        warn(f"Could not fetch model list: {e}")
        return

    models_dir = Path("../models")
    for model in MODELS:
        bare = model.split(":")[0]
        if model in existing or f"{model}:latest" in existing or bare in existing_bare:
            ok(model)
        else:
            gguf = models_dir / f"{model}.gguf"
            if gguf.exists():
                info(f"{model} -- not in Ollama yet but GGUF exists at {gguf} (pipeline will auto-create)")
            else:
                fail(
                    f"{model} -- not in Ollama and no GGUF found",
                    fix=f"ollama pull {model}   OR   place {model}.gguf in ../models/",
                )


# -- 3. PERL -------------------------------------------------------------------

def check_perl():
    print("\n-- Perl -------------------------------------------------------------")

    perl = shutil.which("perl")
    if not perl:
        fail(
            "perl not found in PATH -- all 3 Perl code tasks will be skipped",
            fix="Windows: https://strawberryperl.com/   Mac: brew install perl   Linux: apt install perl",
        )
        return

    try:
        result = subprocess.run([perl, "-e", "print $^V"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip() or "unknown"
        ok(f"perl found at {perl}  ({version})")
    except Exception as e:
        warn(f"perl found at {perl} but version check failed: {e}")

    # quick sanity: run a one-liner
    try:
        result = subprocess.run(
            [perl, "-e", 'use strict; use warnings; print "ok\n"'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            ok("perl smoke test (use strict; use warnings)")
        else:
            warn(f"perl smoke test failed: {result.stderr.strip()}")
    except Exception as e:
        warn(f"perl smoke test error: {e}")


# -- 4. RUBY -------------------------------------------------------------------

def check_ruby():
    print("\n-- Ruby -------------------------------------------------------------")

    ruby = shutil.which("ruby")
    if not ruby:
        for path in RUBY_FALLBACK_PATHS:
            if os.path.isfile(path):
                ruby = path
                break

    if not ruby:
        fail(
            "ruby not found -- all 3 Ruby code tasks will be skipped",
            fix="Windows: https://rubyinstaller.org/   Mac: brew install ruby   Linux: apt install ruby",
        )
        return

    try:
        result = subprocess.run([ruby, "--version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip().split("\n")[0]
        ok(f"ruby found at {ruby}  ({version})")
    except Exception as e:
        warn(f"ruby found at {ruby} but version check failed: {e}")

    # warn if only found via hardcoded fallback (not in PATH)
    if not shutil.which("ruby") and ruby:
        warn(
            f"ruby found via fallback path ({ruby}) but not in PATH -- "
            "pipeline uses the same fallback so this will work, but adding Ruby to PATH is cleaner",
        )

    # quick sanity: run a one-liner
    try:
        result = subprocess.run(
            [ruby, "-e", 'puts "ok"'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            ok("ruby smoke test")
        else:
            warn(f"ruby smoke test failed: {result.stderr.strip()}")
    except Exception as e:
        warn(f"ruby smoke test error: {e}")


# -- 5. WRITE PERMISSIONS ------------------------------------------------------

def check_write_permissions():
    print("\n-- Write permissions ------------------------------------------------")

    # results/ dir
    results_dir = Path("results")
    try:
        results_dir.mkdir(exist_ok=True)
        test_file = results_dir / ".preflight_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        ok("results/ directory writable")
    except Exception as e:
        fail(f"results/ directory not writable: {e}")

    # temp dir (used by exec_perl / exec_ruby)
    try:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "test.txt").write_text("ok")
        ok("temp directory writable")
    except Exception as e:
        fail(f"temp directory not writable: {e}")


# -- SUMMARY -------------------------------------------------------------------

def summary():
    print("\n" + "-" * 70)
    if not errors and not warnings:
        print(f"  {PASS} All checks passed -- pipeline is ready to run.")
    else:
        if errors:
            print(f"  {FAIL} {len(errors)} error(s) -- pipeline will crash or produce incorrect results:")
            for e in errors:
                print(f"         * {e}")
        if warnings:
            print(f"  {WARN} {len(warnings)} warning(s) -- some tasks may produce null/incomplete data:")
            for w in warnings:
                print(f"         * {w}")
    print()
    return len(errors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EdgeLM preflight check")
    ap.add_argument("--models", action="store_true",
                    help="Also check that all 15 models are pulled in Ollama")
    args = ap.parse_args()

    print("EdgeLM preflight check")
    print("=" * 70)

    check_packages()
    check_ollama(check_models=args.models)
    check_perl()
    check_ruby()
    check_write_permissions()

    sys.exit(summary())
