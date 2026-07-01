"""
model_info.py - Display metadata for EdgeLM benchmark models via Ollama.

Queries the local Ollama instance (no model loading required).
Shows quantization, parameter size, family, context window, and decode config.

    python model_info.py                     # all models in MODELS list
    python model_info.py qwen25-1b5          # one model
    python model_info.py --available         # everything loaded in Ollama
    python model_info.py --json              # machine-readable output
    python model_info.py --summary           # compact one-line table
"""
import argparse
import json
import sys
import requests

OLLAMA_URL = "http://localhost:11434"

# Imported from pipeline.py so we stay in sync
from pipeline import MODELS, DECODE


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def ollama_tags() -> list[dict]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json().get("models", [])
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        sys.exit(1)


def ollama_show(model: str) -> dict:
    try:
        r = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model}, timeout=10)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ── Formatting ─────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BLUE   = "\033[94m"

def _c(text, *codes):
    return "".join(codes) + str(text) + _RESET

def _row(label: str, value, width: int = 22):
    return f"  {_c(f'{label:<{width}}', _DIM)}  {value}"


# ── Collect metadata ───────────────────────────────────────────────────────────

def collect(model_name: str, tags_list: list[dict]) -> dict:
    loaded_names = [m["name"] for m in tags_list]
    canonical = model_name if model_name in loaded_names else f"{model_name}:latest"
    is_loaded = canonical in loaded_names or model_name in loaded_names

    # find the tag entry for disk size / modified date
    tag_entry = next(
        (m for m in tags_list if m["name"] in (model_name, canonical)),
        None
    )

    show = ollama_show(model_name) if is_loaded else {}
    details = show.get("details", {})
    model_info = show.get("model_info", {})

    # context window: look in model_info keys
    ctx = None
    for k, v in model_info.items():
        if "context_length" in k and isinstance(v, int):
            ctx = v
            break

    decode = DECODE.get(model_name, DECODE["default"])
    think = decode.get("think", False)

    return {
        "name": model_name,
        "loaded": is_loaded,
        # Ollama details
        "family":       details.get("family", "-"),
        "families":     details.get("families", []),
        "format":       details.get("format", "-"),
        "param_size":   details.get("parameter_size", "-"),
        "quant":        details.get("quantization_level", "-"),
        "context":      ctx,
        "disk_size":    _fmt_bytes(tag_entry["size"]) if tag_entry and "size" in tag_entry else "-",
        "modified":     (tag_entry.get("modified_at", "") or "")[:10] if tag_entry else "-",
        # decode spec from pipeline.py
        "think":        think,
        "temperature":  decode.get("temperature", 0.0),
        "top_p":        decode.get("top_p", 1.0),
        "max_factor":   decode.get("max_factor", 1.0),
        "task_factors": decode.get("task_max_factors", {}),
        # raw for --json
        "_show": show,
    }


# ── Display ────────────────────────────────────────────────────────────────────

def print_model(info: dict):
    status = _c("LOADED", _GREEN, _BOLD) if info["loaded"] else _c("NOT IN OLLAMA", _RED, _BOLD)
    think_tag = _c(" [reasoning/<think>]", _YELLOW) if info["think"] else ""
    print()
    print(_c(f"  {info['name']}{think_tag}", _CYAN, _BOLD))
    print(_c(f"  {'-' * 58}", _DIM))
    print(_row("Status", status))

    if info["loaded"]:
        print(_row("Family", info["family"]))
        print(_row("Format", info["format"]))
        print(_row("Parameters", _c(info["param_size"], _BOLD)))
        quant_color = _YELLOW if info["quant"] not in ("-", "F16", "BF16", "F32") else _DIM
        print(_row("Quantization", _c(info["quant"], quant_color, _BOLD)))
        ctx_str = f"{info['context']:,}" if isinstance(info["context"], int) else "-"
        print(_row("Context window", ctx_str))
        print(_row("Disk size", _c(info["disk_size"], _BOLD)))
        print(_row("Last modified", info["modified"]))

    print()
    temp = info["temperature"]
    temp_str = f"{temp}  {'(greedy)' if temp == 0.0 else '(sampling)'}"
    print(_row("Temperature", temp_str))
    print(_row("top_p", info["top_p"]))
    print(_row("max_factor", f"{info['max_factor']}x token budget"))
    if info["task_factors"]:
        for task, factor in info["task_factors"].items():
            print(_row(f"  {task}", f"{factor}x"))
    print()


def print_summary_table(infos: list[dict]):
    name_w = max(len(i["name"]) for i in infos)
    cols = [name_w, 7, 9, 9, 10, 6, 8, 5]
    header = (
        f"  {'Model':<{cols[0]}}  "
        f"{'Loaded':<{cols[1]}}  "
        f"{'Params':<{cols[2]}}  "
        f"{'Quant':<{cols[3]}}  "
        f"{'Context':<{cols[4]}}  "
        f"{'Disk':<{cols[5]}}  "
        f"{'Temp':<{cols[6]}}  "
        f"{'Think':<{cols[7]}}"
    )
    sep = "  " + "-" * (sum(cols) + len(cols) * 2 + 2)
    print()
    print(_c(header, _BOLD))
    print(_c(sep, _DIM))
    for i in infos:
        loaded = _c("YES", _GREEN) if i["loaded"] else _c("NO ", _RED)
        ctx = f"{i['context']:,}" if isinstance(i["context"], int) else "-"
        quant = i["quant"] if i["loaded"] else "-"
        params = i["param_size"] if i["loaded"] else "-"
        disk = i["disk_size"] if i["loaded"] else "-"
        think = _c("yes", _YELLOW) if i["think"] else "no"
        print(
            f"  {i['name']:<{cols[0]}}  "
            f"{loaded:<{cols[1]}}  "
            f"{params:<{cols[2]}}  "
            f"{quant:<{cols[3]}}  "
            f"{ctx:<{cols[4]}}  "
            f"{disk:<{cols[5]}}  "
            f"{str(i['temperature']):<{cols[6]}}  "
            f"{think:<{cols[7]}}"
        )
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Show Ollama model metadata for the EdgeLM benchmark.")
    ap.add_argument("model", nargs="?", help="Single model name (default: all in pipeline.MODELS)")
    ap.add_argument("--available", action="store_true", help="List everything loaded in Ollama (not just benchmark models)")
    ap.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    ap.add_argument("--summary", action="store_true", help="Compact one-line-per-model table")
    args = ap.parse_args()

    tags = ollama_tags()

    if args.available:
        print("\nModels currently loaded in Ollama:\n")
        for m in tags:
            size = _fmt_bytes(m.get("size", 0))
            d = m.get("details", {})
            print(f"  {m['name']:<40}  {d.get('parameter_size', '?'):<8}  {d.get('quantization_level', '?'):<10}  {size}")
        print()
        return

    if args.model:
        targets = [args.model]
    else:
        targets = MODELS

    infos = [collect(name, tags) for name in targets]

    if args.json:
        clean = [{k: v for k, v in i.items() if k != "_show"} for i in infos]
        print(json.dumps(clean, indent=2, default=str))
        return

    if args.summary:
        print_summary_table(infos)
        return

    print()
    print(_c("  EdgeLM - Model Metadata  (via Ollama)", _BOLD, _BLUE))
    print(_c(f"  {'=' * 58}", _BLUE))

    for info in infos:
        print_model(info)

    if len(infos) > 1:
        print(_c("  Summary", _BOLD))
        print(_c(f"  {'-' * 58}", _DIM))
        print_summary_table(infos)


if __name__ == "__main__":
    main()
