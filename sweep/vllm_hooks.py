"""Where the steering fork of vLLM (RhizoNymph/vllm) can carry a neuron push for a given model.

From the fork's docs/features/steering.md (branch feat/integration, Oct 2026): `pre_attn`, `post_attn` and `post_block`
(the residual stream) are wired on every steerable model; `mlp_in` and `mlp_out` (the MLP branch) only on gemma3,
gemma4 and the qwen3 family, and are silent no-ops elsewhere, so an `mlp_out` push on Qwen2.5 or Llama would leave the
model untouched without an error. Adding a * down[:, n] to the residual stream right after the MLP is the same push
as adding it to the MLP output, so `post_block` stands in for `mlp_out` on the other models.

On models that normalise the MLP output before the residual add (Gemma 2/3/4, OLMo 2) no fork hook is a neuron push:
the fork's `mlp_out` sits after that norm, while a neuron push passes through it. Those models sweep with
`steer_sweep.py --backend hf`. sweep/check_equivalence.py compares the fork with transformers on any model.
"""
STEERABLE = {
    "llama", "llama4", "arcee", "nemotron", "nemotron_nas", "granite", "solar", "minicpm", "minicpm3", "mistral",
    "apertus", "qwen2", "qwen3", "qwen2_moe", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_moe", "gemma", "gemma2",
    "gemma3", "gemma3_text", "gemma3n", "gemma4", "gemma4_text", "mixtral", "glm4", "internlm2", "olmo", "olmo2",
    "exaone", "exaone4", "phi", "falcon", "baichuan", "cohere", "commandr", "stablelm", "gpt_neox", "opt", "orion",
    "persimmon", "seed_oss", "starcoder2",
}
MLP_HOOKS = {"qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_moe", "gemma3", "gemma3_text", "gemma4",
             "gemma4_text"}


def push_hook(model_type, post_mlp_norm):
    """(hook name, None) for the fork hook where adding a * down[:, n] equals adding a to neuron n, or (None, reason)."""
    if model_type not in STEERABLE:
        return None, f"the vLLM fork has no steering hooks for model type {model_type!r}"
    if post_mlp_norm:
        return None, (f"{model_type} normalises the MLP output before the residual add, and the fork's hooks sit after "
                      "that norm, so none of them is a neuron push")
    return ("mlp_out" if model_type in MLP_HOOKS else "post_block"), None


def can_capture_mlp_in(model_type):
    """Whether the fork's `mlp_in` capture is wired for this model type (elsewhere sweep/capture_prefill.py recomputes
    the same captures with transformers after the sweep)."""
    return model_type in MLP_HOOKS
