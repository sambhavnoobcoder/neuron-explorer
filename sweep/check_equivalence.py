"""GPU check of the two equivalences the steering sweep rests on. Run once on a new box before a full sweep.

1. Steering: the fork adding v * W_down[:, n] at its push hook (mlp_out, or post_block where the fork leaves mlp_out
   unwired; sweep/vllm_hooks.py) gives the same greedy rollout as HF adding v to neuron n's activation (a
   forward-pre-hook on the MLP output projection). HF and vLLM kernels differ in bf16 rounding, so the reference is how
   well an unsteered rollout agrees across the two engines; steered agreement should look the same.
2. Capture: neuron activations recomputed from a captured mlp_in, act(W_gate x) * (W_up x), match HF's activations
   at the same tokens (all prompt positions; capture positions must exist when a request is admitted). Only on models
   where the fork captures mlp_in (sweep/vllm_hooks.py); elsewhere sweep/capture_prefill.py does the capturing.

    uv run python sweep/check_equivalence.py --stats stats/q35-4b_L22.npz --neurons 2742,17,5000
    uv run python sweep/check_equivalence.py --stats stats/Qwen--Qwen2.5-0.5B-Instruct_L12.npz --neurons 1,2,3
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as A  # noqa: E402
import vllm_hooks  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.v1.capture.consumers.filesystem.reader import read_request  # noqa: E402


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to the model the stats file was made for")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--hook", default=None, choices=["mlp_out", "post_block"], help="override the fork hook")
    ap.add_argument("--neurons", default="2742,17,5000")
    ap.add_argument("--doses", default="10,-2")
    ap.add_argument("--prompt", default="Who are you?")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--capture-gen", type=int, default=8)
    ap.add_argument("--capture-root", default="captures-check")
    a = ap.parse_args()

    st = np.load(a.stats)
    model_name = a.model or (str(st["model"]) if "model" in st.files else A.DEFAULT_MODEL)
    layer, abs_max, w_down = int(st["layer"]), st["abs_max"], st["w_down"]
    raw = A.raw_config(model_name)
    post = bool(st["post_mlp_norm"]) if "post_mlp_norm" in st.files else False
    hook, why = vllm_hooks.push_hook(raw.get("model_type", ""), post)
    hook = a.hook or hook
    if hook is None:
        raise SystemExit(f"nothing to compare: {why}")
    capture = vllm_hooks.can_capture_mlp_in(raw.get("model_type", ""))
    neurons = [int(x) for x in a.neurons.split(",")]
    doses = [float(x) for x in a.doses.split(",")]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=A._remote_code())
    ids = A.chat_ids(tok, [{"role": "user", "content": a.prompt}])
    plen = len(ids)
    cases = [(None, 0.0)] + [(n, d) for n in neurons for d in doses]
    print(f"{model_name} L{layer}: fork hook {hook}, mlp_in capture {'on' if capture else 'not wired for this model'}")

    # vLLM fork
    shutil.rmtree(a.capture_root, ignore_errors=True)
    llm = LLM(model=model_name, dtype="bfloat16", enable_steering=True, max_steering_configs=64, max_num_seqs=16,
              gpu_memory_utilization=0.6, max_model_len=plen + a.new + 16,
              capture_consumers=[{"name": "filesystem", "params": {"root": a.capture_root}}] if capture else None,
              **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in raw else {}))
    reqs = []
    for n, d in cases:
        kw = {"temperature": 0.0, "max_tokens": a.new}
        if n is not None:
            kw["steering_vectors"] = {hook: {layer: {"vector": w_down[n].astype(np.float32).tolist(),
                                                     "scale": float(d * abs_max[n])}}}
        elif capture:
            kw["capture"] = {"filesystem": {
                "request_id": "baseline", "tag": "check", "layout": "per_file", "hooks": {"mlp_in": [layer]},
                "positions": "all_prompt"}}
        reqs.append(SamplingParams(**kw))
    v_out = llm.generate([TokensPrompt(prompt_token_ids=ids)] * len(cases), reqs, use_tqdm=False)
    v_ids = [list(o.outputs[0].token_ids) for o in v_out]
    if capture:
        print("capture result:", v_out[0].capture_results)

    # HF reference: A.load takes the class from config.json, since importing vllm re-registers some config classes
    model, _ = A.load(model_name, "cuda", torch.bfloat16)
    ar = A.Arch(model, tok, name=model_name)
    mlp = ar.mlps[layer]
    push = {"n": None, "v": 0.0}
    seen = {}

    def hook_fn(m, inp):
        h = inp[0]
        seen["act"] = h.detach()
        if push["n"] is not None:
            h = h.clone()
            h[..., push["n"]] += push["v"]
            return (h,)

    mlp.down.register_forward_pre_hook(hook_fn)
    x = torch.tensor([ids], device="cuda")
    print(f"\nL{layer}, prompt {a.prompt!r}, {a.new} greedy tokens. agree = shared prefix length HF vs vLLM")
    for (n, d), vi in zip(cases, v_ids):
        push.update(n=n, v=float(d * abs_max[n]) if n is not None else 0.0)
        with torch.inference_mode():
            hi = model.generate(x, max_new_tokens=a.new, do_sample=False)[0, plen:].tolist()
        label = "baseline" if n is None else f"n{n} {d:+g}"
        print(f"  {label:14s} agree {common_prefix(hi, vi):3d}/{min(len(hi), len(vi))}   "
              f"vLLM: {tok.decode(vi[:24])!r}")
        if n is not None and common_prefix(hi, vi) < 8:
            print(f"  {'':14s}             HF:   {tok.decode(hi[:24])!r}")

    # capture round-trip on the baseline rollout
    push.update(n=None, v=0.0)
    del llm
    if not capture:
        return
    entries = read_request(os.path.join(a.capture_root, "check", "baseline"))
    e = entries[(layer, "mlp_in")]
    arr = e.array
    if arr.dtype == np.uint16:
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    print(f"\ncapture: requested all {plen} prompt positions; "
          f"got {e.positions}")
    xs = torch.tensor(arr, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        recomputed = mlp.acts(xs).float()
    full = torch.tensor([ids + v_ids[0]], device="cuda")
    with torch.inference_mode():
        ar.lm(input_ids=full, use_cache=False)
    ref = seen["act"][0].float()[torch.tensor(e.positions, device="cuda").clamp(max=full.shape[1] - 1)]
    rel = (recomputed - ref).norm(dim=-1) / ref.norm(dim=-1)
    same_top = (recomputed.argmax(-1) == ref.argmax(-1)).float().mean()
    print(f"recomputed vs HF neuron activations: relative error per position {rel.cpu().numpy().round(4)}; "
          f"same top neuron at {same_top:.0%} of positions")


if __name__ == "__main__":
    main()
