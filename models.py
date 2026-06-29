"""
EdgeLM -- 1.5B track model registry.

Per-model configuration. The important field is `is_reasoning`: three of the
four models emit long <think>...</think> traces and need (a) large token
budgets, (b) sampling rather than greedy decoding (greedy degenerates into
repetition loops on the R1-distill family), and (c) reasoning/answer splitting.

Verify repo IDs and any recommended generation settings against the model cards
before a full run; settings below are sensible defaults, not gospel.
"""
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str            # short id used throughout the benchmark
    repo: str            # Hugging Face repo id
    is_reasoning: bool   # emits <think>...</think> traces
    max_new_tokens: int  # generation budget (reasoning models need a lot)
    temperature: float   # 0.0 => greedy/deterministic; >0 => sampling
    top_p: float
    use_system_prompt: bool   # R1-distill recommends putting instructions in the user turn
    force_think_prefix: bool  # prepend "<think>\n" so the model can't skip reasoning
    notes: str = ""


# NOTE: 3 of these 4 share a Qwen2.5-1.5B backbone -> the 1.5B track is close to
# a controlled study of post-training recipes on one base.
# Clean base->finetune pair inside this track:
#   DeepSeek-R1-Distill-1.5B  --(RL)-->  DeepScaleR-1.5B
MODELS = {
    "qwen2.5-1.5b": ModelSpec(
        name="qwen2.5-1.5b",
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        is_reasoning=False,
        max_new_tokens=2048,
        temperature=0.0,          # standard instruct model -> greedy is fine & deterministic
        top_p=1.0,
        use_system_prompt=True,
        force_think_prefix=False,
        notes="General instruct. The non-reasoning control in the 1.5B track.",
    ),
    "r1-distill-1.5b": ModelSpec(
        name="r1-distill-1.5b",
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        is_reasoning=True,
        max_new_tokens=32768,
        temperature=0.6,          # DeepSeek: temp 0.5-0.7 (0.6), top_p 0.95; greedy degenerates
        top_p=0.95,
        use_system_prompt=False,  # DeepSeek: avoid system prompt, put instructions in user turn
        force_think_prefix=True,  # DeepSeek: enforce starting with <think>
        notes="Reasoning distill (Qwen2.5-Math-1.5B base). Base of the RL pair.",
    ),
    "deepscaler-1.5b": ModelSpec(
        name="deepscaler-1.5b",
        repo="agentica-org/DeepScaleR-1.5B-Preview",
        is_reasoning=True,
        max_new_tokens=32768,     # trained with context scaling up to 24K
        temperature=0.6,
        top_p=0.95,
        use_system_prompt=False,
        force_think_prefix=True,
        notes="RL fine-tune of r1-distill-1.5b. Same base -> isolates the RL delta.",
    ),
    "openreasoning-nemotron-1.5b": ModelSpec(
        name="openreasoning-nemotron-1.5b",
        repo="nvidia/OpenReasoning-Nemotron-1.5B",
        is_reasoning=True,
        max_new_tokens=32768,     # NVIDIA evaluated with up to 64K output tokens
        temperature=0.6,
        top_p=0.95,
        use_system_prompt=True,   # ChatML; a short system prompt is fine
        force_think_prefix=False,
        notes="Qwen2.5-1.5B base, distilled from DeepSeek-R1-0528 traces (math/code/science).",
    ),
}


def get(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. known: {list(MODELS)}")
    return MODELS[name]
