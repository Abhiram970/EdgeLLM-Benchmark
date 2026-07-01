"""
EdgeLM benchmark results visualisation.
Generates: figures/benchmark_results.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ── Model groups ──────────────────────────────────────────────────────────────
MODELS_4B = ["crow-4b", "qwen35-4b", "gemma3-4b", "glm-edge-4b", "phi4-mini"]
MODELS_1B = ["qwen25-1b5", "granite4-1b", "llama32-1b", "deepseek-r1-1b5", "deepscaler-1b5"]
MODELS_ALL = MODELS_4B + MODELS_1B

DISPLAY = {
    "crow-4b":         "Crow-4B",
    "qwen35-4b":       "Qwen3.5-4B",
    "gemma3-4b":       "Gemma3-4B",
    "glm-edge-4b":     "GLM-Edge-4B",
    "phi4-mini":       "Phi4-mini",
    "qwen25-1b5":      "Qwen2.5-1.5B",
    "granite4-1b":     "Granite4-1B",
    "llama32-1b":      "Llama3.2-1B",
    "deepseek-r1-1b5": "DeepSeek-R1-1.5B",
    "deepscaler-1b5":  "DeepScaleR-1.5B",
}

# ── Metric metadata ───────────────────────────────────────────────────────────
CANON_METRICS = [
    ("context_retention",  "Context\nRetention",    "#4C72B0"),
    ("summarization_rougeL","Summarization\nROUGE-L","#DD8452"),
    ("open_book_qa",       "Open-Book\nQA",         "#55A868"),
    ("structured_output",  "Structured\nOutput",    "#C44E52"),
    ("creative_constraints","Creative\nConstraints","#8172B2"),
    ("code_pass_at_1",     "Code\nPass@1",          "#937860"),
]

QUAL_METRICS = [
    ("t2_faithfulness", "Summ.\nFaithfulness\n(/10)", "#DD8452"),
    ("t5_quality",      "Creative\nQuality\n(/10)",   "#8172B2"),
]

# ── Load data ─────────────────────────────────────────────────────────────────
def load_canonical(model):
    data = json.loads((RESULTS / f"{model}.json").read_text(encoding="utf-8"))
    t = data["tasks"]
    return {
        "context_retention":   t.get("context_retention",  {}).get("score"),
        "summarization_rougeL":t.get("summarization",      {}).get("mean_rouge_l"),
        "open_book_qa":        t.get("open_book_qa",        {}).get("overall_accuracy"),
        "structured_output":   t.get("structured_output",  {}).get("strict_compliance_rate"),
        "creative_constraints":t.get("creative_generation",{}).get("mean_constraint_pass_rate"),
        "code_pass_at_1":      t.get("code_generation",    {}).get("pass_at_1"),
    }

def load_qualitative(model):
    p = RESULTS / f"{model}_qualitative.json"
    if not p.exists():
        return {"t2_faithfulness": None, "t5_quality": None}
    data = json.loads(p.read_text(encoding="utf-8"))
    t2_scores = [v["score"] for v in data.get("task2_faithfulness", {}).values()
                 if isinstance(v, dict) and v.get("score") is not None]
    t5_scores = [v["score"] for v in data.get("task5_quality", {}).values()
                 if isinstance(v, dict) and v.get("score") is not None]
    return {
        "t2_faithfulness": sum(t2_scores) / len(t2_scores) if t2_scores else None,
        "t5_quality":      sum(t5_scores) / len(t5_scores) if t5_scores else None,
    }

canon = {m: load_canonical(m)  for m in MODELS_ALL}
qual  = {m: load_qualitative(m) for m in MODELS_ALL}

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#FAFAFA")
gs = GridSpec(2, 2, figure=fig,
              left=0.06, right=0.98, top=0.92, bottom=0.08,
              wspace=0.32, hspace=0.52)

ax_canon  = fig.add_subplot(gs[0, 0])   # canonical heatmap
ax_qual   = fig.add_subplot(gs[0, 1])   # qualitative heatmap
ax_radar  = fig.add_subplot(gs[1, 0], polar=True)  # radar
ax_bar    = fig.add_subplot(gs[1, 1])   # per-task ranked bars

fig.suptitle("EdgeLM Benchmark — Results Overview (10 Models)",
             fontsize=16, fontweight="bold", y=0.97, color="#1a1a1a")

LABEL_NAMES = [DISPLAY[m] for m in MODELS_ALL]

# ── Panel 1: Canonical heatmap ────────────────────────────────────────────────
canon_keys  = [k for k, _, _ in CANON_METRICS]
canon_labels = [l for _, l, _ in CANON_METRICS]
data_matrix = np.array([[canon[m].get(k) if canon[m].get(k) is not None else np.nan
                          for k in canon_keys] for m in MODELS_ALL])

im1 = ax_canon.imshow(data_matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, interpolation="none")

ax_canon.set_xticks(range(len(canon_keys)))
ax_canon.set_xticklabels(canon_labels, fontsize=7.5, ha="center")
ax_canon.set_yticks(range(len(MODELS_ALL)))
ax_canon.set_yticklabels(LABEL_NAMES, fontsize=8.5)
ax_canon.set_title("Canonical Scores (0–1)", fontsize=10, fontweight="bold", pad=8)

# Divider between 4B and 1B groups
ax_canon.axhline(len(MODELS_4B) - 0.5, color="white", linewidth=2.5)
ax_canon.text(len(canon_keys) - 0.45, len(MODELS_4B) / 2 - 0.5,
              "4B", fontsize=8, color="white", fontweight="bold",
              ha="right", va="center", rotation=90)
ax_canon.text(len(canon_keys) - 0.45, len(MODELS_4B) + len(MODELS_1B) / 2 - 0.5,
              "1B", fontsize=8, color="white", fontweight="bold",
              ha="right", va="center", rotation=90)

for i, m in enumerate(MODELS_ALL):
    for j, k in enumerate(canon_keys):
        v = canon[m].get(k)
        if v is not None:
            ax_canon.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=7, color="black" if 0.3 < v < 0.85 else "white",
                          fontweight="bold")
        else:
            ax_canon.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888")

plt.colorbar(im1, ax=ax_canon, fraction=0.025, pad=0.02)

# ── Panel 2: Qualitative heatmap ─────────────────────────────────────────────
qual_keys   = [k for k, _, _ in QUAL_METRICS]
qual_labels = [l for _, l, _ in QUAL_METRICS]
qual_matrix = np.array([[qual[m].get(k) if qual[m].get(k) is not None else np.nan
                          for k in qual_keys] for m in MODELS_ALL])

im2 = ax_qual.imshow(qual_matrix, aspect="auto", cmap="RdYlGn",
                      vmin=1, vmax=10, interpolation="none")

ax_qual.set_xticks(range(len(qual_keys)))
ax_qual.set_xticklabels(qual_labels, fontsize=8.5, ha="center")
ax_qual.set_yticks(range(len(MODELS_ALL)))
ax_qual.set_yticklabels(LABEL_NAMES, fontsize=8.5)
ax_qual.set_title("LLM Judge Scores (1–10)", fontsize=10, fontweight="bold", pad=8)
ax_qual.axhline(len(MODELS_4B) - 0.5, color="white", linewidth=2.5)

for i, m in enumerate(MODELS_ALL):
    for j, k in enumerate(qual_keys):
        v = qual[m].get(k)
        if v is not None:
            normed = (v - 1) / 9
            ax_qual.text(j, i, f"{v:.1f}", ha="center", va="center",
                          fontsize=9, color="black" if 0.25 < normed < 0.85 else "white",
                          fontweight="bold")
        else:
            ax_qual.text(j, i, "—", ha="center", va="center", fontsize=9, color="#888")

plt.colorbar(im2, ax=ax_qual, fraction=0.08, pad=0.04)

# ── Panel 3: Radar chart ──────────────────────────────────────────────────────
radar_labels = ["Context\nRetention", "Summ.\nROUGE-L", "Open-Book\nQA",
                "Structured\nOutput", "Creative\nConstraints", "Code\nPass@1"]
N = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

COLORS_4B = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
COLORS_1B = ["#937860", "#64B5CD", "#B2912F", "#DA8BC3", "#8C8C8C"]

for idx, model in enumerate(MODELS_4B):
    vals = [canon[model].get(k, 0) or 0 for k in canon_keys]
    vals += vals[:1]
    ax_radar.plot(angles, vals, color=COLORS_4B[idx], linewidth=1.8, linestyle="-")
    ax_radar.fill(angles, vals, color=COLORS_4B[idx], alpha=0.08)

for idx, model in enumerate(MODELS_1B):
    vals = [canon[model].get(k, 0) or 0 for k in canon_keys]
    vals += vals[:1]
    ax_radar.plot(angles, vals, color=COLORS_1B[idx], linewidth=1.8, linestyle="--")
    ax_radar.fill(angles, vals, color=COLORS_1B[idx], alpha=0.06)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_labels, fontsize=7.5)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6, color="#888")
ax_radar.set_title("Task Profile — All Models", fontsize=10, fontweight="bold", pad=18)
ax_radar.grid(color="#cccccc", linewidth=0.6)

legend_patches = (
    [mpatches.Patch(color=COLORS_4B[i], label=DISPLAY[m]) for i, m in enumerate(MODELS_4B)] +
    [mpatches.Patch(color=COLORS_1B[i], label=DISPLAY[m]) for i, m in enumerate(MODELS_1B)]
)
ax_radar.legend(handles=legend_patches, loc="upper right",
                bbox_to_anchor=(1.55, 1.12), fontsize=7.5, framealpha=0.85,
                title="Model  (-- = 1B)", title_fontsize=7.5)

# ── Panel 4: Per-task ranked bar chart ───────────────────────────────────────
task_short = ["T1\nContext", "T2\nSumm.", "T3\nQA", "T4\nStruct.", "T5\nCreative", "T6\nCode"]
x = np.arange(len(canon_keys))
bar_width = 0.075
offsets = np.linspace(-(len(MODELS_ALL) - 1) / 2 * bar_width,
                       (len(MODELS_ALL) - 1) / 2 * bar_width, len(MODELS_ALL))

all_colors = COLORS_4B + COLORS_1B
for idx, model in enumerate(MODELS_ALL):
    vals = [canon[model].get(k) or 0 for k in canon_keys]
    ls = "-" if model in MODELS_4B else "--"
    ax_bar.bar(x + offsets[idx], vals, width=bar_width,
               color=all_colors[idx], alpha=0.82,
               edgecolor="white", linewidth=0.4,
               label=DISPLAY[model])

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(task_short, fontsize=8.5)
ax_bar.set_ylim(0, 1.08)
ax_bar.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_bar.set_ylabel("Score", fontsize=8.5)
ax_bar.set_title("Per-Task Score by Model", fontsize=10, fontweight="bold")
ax_bar.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle=":")
ax_bar.set_facecolor("#F8F8F8")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

# Divider annotation
ax_bar.annotate("— 4B   -- 1B", xy=(0.98, 0.97), xycoords="axes fraction",
                fontsize=7.5, ha="right", va="top", color="#555",
                style="italic")

out_path = FIGURES / "benchmark_results.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
