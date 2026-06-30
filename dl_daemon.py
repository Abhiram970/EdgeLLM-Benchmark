from huggingface_hub import hf_hub_download, list_repo_files
import os

os.makedirs("./models", exist_ok=True)

# Format: (repo_id, local_name)
MODELS = [
    # ("mradermacher/VibeThinker-3B-GGUF",                         "vibethinker-3b"),       # DONE
    #("bartowski/microsoft_Phi-4-mini-instruct-GGUF",               "phi4-mini"),             # FIXED
    #("mradermacher/Ministral-3b-instruct-GGUF",                    "ministral-3b"),          # FIXED (lowercase b)
    # ("mradermacher/SmolLM3-3B-GGUF",                             "smollm3-3b"),            # DONE
    # ("mradermacher/Qwen2.5-Coder-3B-Instruct-GGUF",              "qwen25-coder-3b"),       # DONE
    # ("mradermacher/gemma-3-4b-it-GGUF",                          "gemma3-4b"),             # DONE
    # ("mradermacher/Llama-3.2-3B-Instruct-GGUF",                  "llama32-3b"),            # DONE
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF",                          "qwen25-1b5"),            # DONE
    #("bartowski/Qwen_Qwen3.5-4B-GGUF",                            "qwen35-4b"),             # FIXED
    ("mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",          "deepseek-r1-1b5"),       # DONE
    # ("mradermacher/glm-edge-4b-chat-GGUF",                       "glm-edge-4b"),           # DONE
    ("mradermacher/DeepScaleR-1.5B-Preview-GGUF",                  "deepscaler-1b5"),        # retry - may exist now
    ("mradermacher/OpenReasoning-Nemotron-1.5B-GGUF",            "nemotron-1b5"),           # DONE
    # ("Crownelius/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5",      "crow-4b"),               # DONE
    # Grok distill - dropping, no Q4 available and only 100 training examples
    ("bartowski/agentica-org_DeepScaleR-1.5B-Preview-GGUF", "deepscaler-1b5"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF",                 "llama32-1b"),            # DONE
]

def find_q4km_file(repo_id):
    """Auto-find Q4_K_M gguf filename in a repo."""
    try:
        files = list(list_repo_files(repo_id))
        for f in files:
            if "Q4_K_M" in f and f.endswith(".gguf"):
                return f
        for f in files:
            if "q4_k_m" in f.lower() and f.endswith(".gguf"):
                return f
        # fallback Q5_K_M
        for f in files:
            if "Q5_K_M" in f and f.endswith(".gguf"):
                print(f"  No Q4_K_M, falling back to Q5_K_M")
                return f
    except Exception as e:
        print(f"  Could not list files: {e}")
    return None

for repo_id, local_name in MODELS:
    print(f"\nLooking for Q4_K_M in {repo_id}...")
    filename = find_q4km_file(repo_id)
    if not filename:
        print(f"  ✗ No Q4_K_M found in {repo_id} - SKIP")
        continue
    print(f"  Found: {filename}")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="./models",
            local_dir_use_symlinks=False,
        )
        final_path = f"./models/{local_name}.gguf"
        os.rename(path, final_path)
        print(f"  ✓ Saved as {local_name}.gguf")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")

print("\nDone.")