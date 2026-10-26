"""Throughput of the steering fork on this GPU under three conditions, same prompt and fixed-length rollouts:
unsteered, a distinct steering config per request, and steering plus a last_prompt mlp_in capture.

    uv run python sweep/bench.py --stats stats/q35-4b_L22.npz --n 1280 --max-num-seqs 320
"""
import argparse
import os
import sys
import time

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as A  # noqa: E402
import vllm_hooks  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to the model the stats file was made for")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--n", type=int, default=1280)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=320)
    ap.add_argument("--max-configs", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--only", default="", help="comma-separated condition names")
    a = ap.parse_args()

    st = np.load(a.stats)
    model = a.model or (str(st["model"]) if "model" in st.files else A.DEFAULT_MODEL)
    layer, abs_max, w_down = int(st["layer"]), st["abs_max"], st["w_down"]
    raw = A.raw_config(model)
    hook, why = vllm_hooks.push_hook(raw.get("model_type", ""), bool(st["post_mlp_norm"]) if "post_mlp_norm" in st.files else False)
    if hook is None:
        raise SystemExit(f"--backend vllm can't steer {model}: {why}")
    capture = vllm_hooks.can_capture_mlp_in(raw.get("model_type", ""))
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=A._remote_code())
    ids = A.chat_ids(tok, [{"role": "user", "content": "Describe the color orange."}])
    llm = LLM(model=model, dtype="bfloat16", enable_steering=True, max_steering_configs=a.max_configs,
              max_num_seqs=a.max_num_seqs, max_model_len=len(ids) + a.tokens + 16, gpu_memory_utilization=a.gpu_mem,
              capture_consumers=[{"name": "filesystem", "params": {"root": "/tmp/bench-captures"}}] if capture else None,
              **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in raw else {}))
    prompts = [TokensPrompt(prompt_token_ids=ids)] * a.n
    base = {"temperature": 0.0, "max_tokens": a.tokens, "ignore_eos": True}

    def steer(i):
        n = i % len(abs_max)
        return {"steering_vectors": {hook: {layer: {"vector": w_down[n].astype(np.float32).tolist(),
                                                    "scale": float(0.5 * abs_max[n])}}}}

    def cap(i):
        if not capture:  # the fork can't capture mlp_in on this model; capture conditions run without it
            return {}
        return {"capture": {"filesystem": {"request_id": f"b{i}", "tag": "bench", "layout": "packed",
                                           "hooks": {"mlp_in": [layer, layer + 1, layer + 4, layer + 8]}, "positions": "last_prompt"}}}

    conditions = {
        "unsteered": lambda i: SamplingParams(**base),
        "steered": lambda i: SamplingParams(**base, **steer(i)),
        "steered+capture": lambda i: SamplingParams(**base, **steer(i), **cap(i)),
        "steered+logprobs5": lambda i: SamplingParams(**base, **steer(i), logprobs=5),
        # the sweep's real configuration, with natural stopping vs fixed length
        "sweep-varlen": lambda i: SamplingParams(**{**base, "ignore_eos": False}, **steer(i), **cap(i), logprobs=5),
        "sweep-fixed": lambda i: SamplingParams(**base, **steer(i), **cap(i), logprobs=5),
    }
    llm.generate(prompts[:32], [SamplingParams(**base)] * 32, use_tqdm=False)  # warm-up
    for name, make in conditions.items():
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        params = [make(i) for i in range(a.n)]
        t1 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.time() - t1
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        stop = A.stop_ids(model, tok)
        kept = sum(next((k for k, t in enumerate(o.outputs[0].token_ids) if t in stop), len(o.outputs[0].token_ids))
                   for o in outs)
        print(f"{name:18s} {n_tok / dt:8,.0f} gen tok/s  {kept / dt:8,.0f} kept tok/s  {a.n / dt:6.1f} rollouts/s  "
              f"(building params {t1 - t0:.1f}s)",
              flush=True)
    llm.llm_engine.engine_core.shutdown(timeout=60)


if __name__ == "__main__":
    main()
