"""Steer every neuron of one MLP layer, one rollout per (prompt, dose), and keep everything a rollout produces.

Steering neuron n by dose d adds d * abs_max[n] to its activation at every position (prompt and generated). That is
the same as adding d * abs_max[n] * W_down[:, n] to the layer's MLP output projection's output, which is what both
backends push. abs_max is the neuron's largest |activation| over the stats corpus, so doses are in units of the
neuron's own range.

Design "main", per neuron, 10 rollouts: {who, snake, orange} x {-2, +2, +10} and {free} x {+10}. Design "calib" is a
dose-response ladder, {who, orange, free} x ±{1, 2, 4, 8, 16, 32}, for finding where effects begin and where text
breaks. Design "main_rel" (with --dose-unit rel) gives every neuron the same push sizes, as fractions of the median
norm of the stream the push lands in (the residual stream; the MLP output on models that normalise it, Gemma 2/3 and
OLMo 2): {who, snake, orange} x {-0.7, +0.5, +0.8} and {free} x {+3}. One unsteered baseline per prompt. Each rollout
also records rel_push, the size of its push relative to that median norm.
Random-direction controls (a neuron's down column rotated to a random direction, same norm and dose scale) get the
same rollouts as the neurons and give the null for "how often does a theme appear by chance".

Each rollout keeps:
    text, generated token ids, the chosen token's logprob and the top-k alternatives at every step   (jsonl)
    mlp_in activations at the capture layers at the last prompt token, the state just before the model answers
    (vLLM filesystem capture under --capture-root/<run>/<capture_id>, on models where the fork can capture mlp_in;
    elsewhere sweep/capture_prefill.py computes the same after the run). Every neuron of a captured layer is
    recoverable from these as act(W_gate x) * (W_up x). Activations at generated tokens come from replaying the saved
    token ids through the model with the same steering, one teacher-forced forward pass.

Backends:
    vllm  the steering fork of vLLM (GPU, fast). The push goes in at the fork's `mlp_out` hook where the fork wires it
          (Qwen3 family, Qwen3.5, Gemma 3/4) and at `post_block` elsewhere, the same push (sweep/vllm_hooks.py). Not
          for models that normalise the MLP output (Gemma 2/3, OLMo 2): no fork hook is a neuron push there.
    hf    transformers, batched greedy decoding with the push added at the MLP output projection's output, where the
          explorer pushes too. Exact on every model the explorer runs (CUDA, MPS or CPU); slower than vLLM.

    uv run python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --neurons 2742 --run smoke-2742
    uv run python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design calib --neurons random:128 --random-controls 64 --run calib-L22
    uv run python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --neurons all --run q35-4b-L22
    uv run python sweep/steer_sweep.py --stats stats/google--gemma-3-1b-it_L13.npz --backend hf --design main_rel \
        --dose-unit rel --neurons random:256 --random-controls 64 --run google--gemma-3-1b-it-L13-r256

The model is the one the stats file was made for. Rerunning the same command resumes: rollouts already in the jsonl
are skipped.
"""
import argparse
import functools
import inspect
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as A  # noqa: E402
import vllm_hooks  # noqa: E402

PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
LADDER = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DESIGNS = {
    "main": [(p, d) for p in ("who", "snake", "orange") for d in (-2.0, 2.0, 10.0)] + [("free", 10.0)],
    # dose-response calibration: where effects begin and where text breaks
    "calib": [(p, s * d) for p in ("who", "orange", "free") for d in LADDER for s in (-1.0, 1.0)],
    # doses as fractions of the median residual norm (use with --dose-unit rel): the coherent window is ~0.5-0.8x,
    # and +3x reads the direction out as wreckage tokens
    "main_rel": [(p, d) for p in ("who", "snake", "orange") for d in (-0.7, 0.5, 0.8)] + [("free", 3.0)],
}


def parse_neurons(spec, n, seed):
    if spec == "all":
        return list(range(n))
    if spec.startswith("random:"):
        return sorted(np.random.default_rng(seed + 1).choice(n, int(spec.split(":")[1]), replace=False).tolist())
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def record(job, layer, ids, chosen, top_ids, top_lps, text, finish):
    return {
        "layer": layer, "kind": job["kind"], "neuron": job["neuron"], "ref_neuron": job["ref_neuron"],
        "prompt": job["prompt"], "dose": job["dose"], "add": job["add"], "rel_push": job["rel_push"],
        "text": text, "finish": finish, "n_gen": len(ids), "token_ids": ids,
        "logprob": chosen, "top_ids": top_ids, "top_lps": top_lps, "capture_id": job["capture_id"],
    }


def rollout_record(out, job, layer, top_k):
    o = out.outputs[0]
    ids = list(o.token_ids)
    chosen, top_ids, top_lps = [], [], []
    for tid, lp in zip(ids, o.logprobs or []):
        chosen.append(round(lp[tid].logprob, 4))
        ranked = sorted((v.rank, k, v.logprob) for k, v in lp.items() if v.rank is not None and v.rank <= top_k)
        top_ids.append([k for _, k, _ in ranked])
        top_lps.append([round(x, 4) for _, _, x in ranked])
    return record(job, layer, ids, chosen, top_ids, top_lps, o.text, o.finish_reason)


class HFRunner:
    """Batched greedy rollouts in transformers, each row with its own push: a vector added to the output of the steered
    layer's MLP output projection at every position (add * W_down[:, n] for a neuron, add * v for a control)."""

    def __init__(self, model_name, layer, device, batch):
        import torch
        self.torch = torch
        self.model, self.tok = A.load(model_name, device, A.dtype_from_env())
        ar = A.Arch(self.model, self.tok, name=model_name)
        self.stop = A.stop_ids(self.model, self.tok)
        self.dev, self.batch, self.vec = device, batch, None
        self.last = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(self.model.forward).parameters else {}
        ar.mlps[layer].down.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        return out + self.vec[:, None, :].to(out.dtype) if self.vec is not None else None

    def generate(self, prompts, vecs, max_tokens, top_k):
        """prompts: token-id lists; vecs: one push vector per prompt (zeros for a baseline). Greedy, like the vLLM path
        at temperature 0; a reply's stop token is kept in its token ids (finish "stop"), as vLLM keeps an EOS."""
        out = []
        for s in range(0, len(prompts), self.batch):
            out += self._batch(prompts[s:s + self.batch], vecs[s:s + self.batch], max_tokens, top_k)
        return out

    def _batch(self, prompts, vecs, max_tokens, top_k):
        torch = self.torch
        B, T = len(prompts), max(map(len, prompts))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([[pad] * (T - len(p)) + p for p in prompts], device=self.dev)
        mask = torch.tensor([[0] * (T - len(p)) + [1] * len(p) for p in prompts], device=self.dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        rows = [{"ids": [], "lp": [], "top_ids": [], "top_lps": [], "finish": "length"} for _ in range(B)]
        done = [False] * B
        self.vec = torch.tensor(np.stack(vecs), dtype=torch.float32, device=self.dev)
        try:
            with torch.inference_mode():
                out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, **self.last)
                for step in range(max_tokens):
                    lp = torch.log_softmax(out.logits[:, -1].float(), -1)
                    nxt = lp.argmax(-1)
                    top = lp.topk(top_k)
                    chosen = lp.gather(1, nxt[:, None])[:, 0].tolist()
                    ti, tv = top.indices.tolist(), top.values.tolist()
                    for r, t in enumerate(nxt.tolist()):
                        if done[r]:
                            continue
                        row = rows[r]
                        row["ids"].append(t)
                        row["lp"].append(round(chosen[r], 4))
                        row["top_ids"].append(ti[r])
                        row["top_lps"].append([round(x, 4) for x in tv[r]])
                        if t in self.stop:
                            done[r], row["finish"] = True, "stop"
                    if all(done) or step == max_tokens - 1:
                        break
                    mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                    pos = pos[:, -1:] + 1
                    out = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                     past_key_values=out.past_key_values, use_cache=True, **self.last)
        finally:
            self.vec = None
        for row in rows:
            row["text"] = self.tok.decode(row["ids"], skip_special_tokens=True)
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to the model the stats file was made for")
    ap.add_argument("--stats", required=True, help="npz from neuron_stats.py for the layer to steer")
    ap.add_argument("--neurons", default="all", help='"all", "random:128", "2742", "0-99", or "1,5,9-12"')
    ap.add_argument("--design", default="main", choices=sorted(DESIGNS))
    ap.add_argument("--dose-unit", default="absmax", choices=["absmax", "rel"], help="absmax: dose x the neuron's "
                    "abs_max; rel: dose x the median push-stream norm, the same push size for every neuron")
    ap.add_argument("--run", required=True, help="run name: output file and capture tag")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--max-tokens", type=int, default=256, help="steered rollouts often loop; this bounds them")
    ap.add_argument("--top-k", type=int, default=5, help="alternative tokens kept per generated step")
    ap.add_argument("--capture-layers", default="+0,+1,+4,+8", help="mlp_in layers to capture, relative to the "
                    "steered layer; empty string turns capture off")
    ap.add_argument("--capture-root", default="captures")
    ap.add_argument("--capture-layout", default="packed", choices=["packed", "sharded"], help="packed publishes each "
                    "rollout when it finishes; sharded publishes only when a shard seals")
    ap.add_argument("--random-controls", type=int, default=256, help="random-direction controls spread through the run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4000, help="requests per generate call (one jsonl flush each)")
    ap.add_argument("--max-configs", type=int, default=512, help="distinct steering configs resident at once")
    ap.add_argument("--max-num-seqs", type=int, default=256, help="concurrent sequences; capped by the per-sequence "
                    "linear-attention state memory (about 350 on a 24 GB card at --gpu-mem 0.85)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--vllm-hook", default=None, choices=["mlp_out", "post_block"], help="override the fork hook "
                    "(sweep/vllm_hooks.py picks it from the model type); check with sweep/check_equivalence.py")
    ap.add_argument("--batch", type=int, default=32, help="hf backend: rollouts decoded together")
    ap.add_argument("--device", default=None, help="hf backend: cuda, mps or cpu (default: the best available)")
    a = ap.parse_args()

    st = np.load(a.stats)
    stats_model = str(st["model"]) if "model" in st.files else A.DEFAULT_MODEL
    if a.model and a.model != stats_model:
        raise SystemExit(f"{a.stats} was made for {stats_model}, not {a.model}")
    model = a.model or stats_model
    layer, abs_max, w_down = int(st["layer"]), st["abs_max"], st["w_down"]
    neurons = parse_neurons(a.neurons, len(abs_max), a.seed)
    design = DESIGNS[a.design]
    resid_p50 = float(st["resid_norm_q10_50_90"][1])
    # the scale of rel doses: the median norm of the stream the push lands in (the residual, or the MLP output on
    # models that normalise it); stats files from before push_norm was recorded are residual-stream models
    push_p50 = float(st["push_norm_q10_50_90"][1]) if "push_norm_q10_50_90" in st.files else resid_p50
    push_space = str(st["push_space"]) if "push_space" in st.files else "residual"
    post_mlp_norm = bool(st["post_mlp_norm"]) if "post_mlp_norm" in st.files else False
    down_norm = np.linalg.norm(w_down.astype(np.float32), axis=1)

    from transformers import AutoConfig, AutoTokenizer  # before vllm, which re-registers some config classes
    rc = A._remote_code()
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=rc)
    text_cfg = A.text_config(cfg)
    model_type = A.raw_config(model).get("model_type", "")
    n_layers = A.n_layers_of(cfg)
    cap_layers = sorted({layer + int(x) for x in a.capture_layers.split(",") if x} & set(range(n_layers)))
    hook = None
    if a.backend == "vllm":
        assert not getattr(text_cfg, "layer_scale", False), "mlp_out sits after an FFN layer scale on this model, so " \
            "adding W_down[:, n] there is no longer a pure push on neuron n"
        hook, why = vllm_hooks.push_hook(model_type, post_mlp_norm)
        hook = a.vllm_hook or hook
        if hook is None:
            raise SystemExit(f"--backend vllm can't sweep {model}: {why}. Use --backend hf.")
    captured = bool(cap_layers) and a.backend == "vllm" and vllm_hooks.can_capture_mlp_in(model_type)
    if cap_layers and not captured:
        print(f"capture: not in this run ({a.backend} backend, {model_type}); sweep/capture_prefill.py on the finished "
              f"run computes the same mlp_in captures for layers {cap_layers}", flush=True)

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=rc)
    prompt_ids = {k: A.chat_ids(tok, [{"role": "user", "content": v}]) for k, v in PROMPTS.items()}

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{a.run}.jsonl")
    with open(os.path.join(a.out_dir, f"{a.run}.meta.json"), "w") as f:
        json.dump({"model": model, "layer": layer, "stats": os.path.abspath(a.stats), "prompts": PROMPTS,
                   "design_name": a.design, "design": design, "resid_norm_p50": resid_p50,
                   "push_norm_p50": push_p50, "push_space": push_space, "dose_unit": a.dose_unit,
                   "backend": a.backend, "hook": hook or "down_proj_out", "max_tokens": a.max_tokens, "top_k": a.top_k,
                   "capture_layers": cap_layers, "captured": captured, "capture_hook": "mlp_in",
                   "capture_positions": "last_prompt", "capture_root": os.path.abspath(a.capture_root) if captured else None,
                   "capture_tag": a.run, "prompt_ids": prompt_ids}, f, indent=1)

    # Random-direction controls: neuron j's down column rotated to a random direction (same norm, same abs_max).
    # They answer "how often does a theme appear when the steered direction means nothing?"
    rng = np.random.default_rng(a.seed)
    ctrl_ref = rng.integers(len(abs_max), size=a.random_controls)
    ctrl_vecs = rng.standard_normal((a.random_controls, w_down.shape[1])).astype(np.float32)
    ctrl_vecs *= (np.linalg.norm(w_down[ctrl_ref].astype(np.float32), axis=1)
                  / np.linalg.norm(ctrl_vecs, axis=1))[:, None]

    units = [("neuron", n) for n in neurons]
    for k in range(a.random_controls):  # spread evenly so any prefix of the run has its share of controls
        units.insert(round((k + 1) * len(units) / (a.random_controls + 1)), ("random", k))

    key = lambda j: (j["kind"], j["neuron"], j["prompt"], j["dose"])
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {key(json.loads(line)) for line in f}

    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in PROMPTS]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": d} for kind, n in units for p, d in design]
    jobs = [j for j in jobs if key(j) not in done]
    print(f"{model} L{layer}: {len(neurons)} neurons + {a.random_controls} random controls, {len(jobs)} rollouts to "
          f"run ({len(done)} already done), {a.backend} backend" + (f", hook {hook}" if hook else ""), flush=True)
    if not jobs:
        return

    @functools.lru_cache(maxsize=256)
    def direction(kind, n):
        return (w_down[n] if kind == "neuron" else ctrl_vecs[n]).astype(np.float32)

    def prepare(job):
        kind, n, p, d = job["kind"], job["neuron"], job["prompt"], job["dose"]
        ref = n if kind == "neuron" else int(ctrl_ref[n]) if kind == "random" else None
        job["ref_neuron"] = ref
        if ref is None:
            job["add"] = 0.0
        elif a.dose_unit == "rel":
            job["add"] = float(d * push_p50 / down_norm[ref])
        else:
            job["add"] = float(d * abs_max[ref])
        job["rel_push"] = abs(job["add"]) * float(down_norm[ref]) / push_p50 if ref is not None else 0.0
        job["capture_id"] = f"L{layer}_{kind}{n}_{p}_{d:+g}".replace("+", "p").replace("-", "m")
        return job

    if a.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        llm = LLM(
            model=model, dtype="bfloat16", enable_steering=True, max_steering_configs=a.max_configs,
            max_num_seqs=a.max_num_seqs, max_model_len=max(map(len, prompt_ids.values())) + a.max_tokens + 16,
            gpu_memory_utilization=a.gpu_mem, shutdown_timeout=120,
            capture_consumers=[{"name": "filesystem", "params": {"root": a.capture_root}}] if captured else None,
            **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in A.raw_config(model) else {}),
        )

        def params(job):
            kw = {"temperature": 0.0, "max_tokens": a.max_tokens, "logprobs": a.top_k}
            if job["ref_neuron"] is not None:
                kw["steering_vectors"] = {hook: {layer: {"vector": direction(job["kind"], job["neuron"]).tolist(),
                                                         "scale": job["add"]}}}
            if captured:
                kw["capture"] = {"filesystem": {
                    "request_id": job["capture_id"], "tag": a.run, "layout": a.capture_layout,
                    "hooks": {"mlp_in": cap_layers},
                    "positions": "last_prompt",
                }}
            return SamplingParams(**kw)

        def run(chunk):
            outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[j["prompt"]]) for j in chunk],
                                [params(j) for j in chunk], use_tqdm=False)
            return [rollout_record(out, job, layer, a.top_k) for job, out in zip(chunk, outs)]
    else:
        hf = HFRunner(model, layer, a.device or A.default_device(), a.batch)

        def run(chunk):
            vecs = [direction(j["kind"], j["neuron"]) * j["add"] if j["ref_neuron"] is not None
                    else np.zeros(w_down.shape[1], np.float32) for j in chunk]
            rows = hf.generate([prompt_ids[j["prompt"]] for j in chunk], vecs, a.max_tokens, a.top_k)
            return [record(j, layer, r["ids"], r["lp"], r["top_ids"], r["top_lps"], r["text"], r["finish"])
                    for j, r in zip(chunk, rows)]

    with open(out_path, "a") as f:
        for c in range(0, len(jobs), a.chunk):
            chunk = [prepare(j) for j in jobs[c : c + a.chunk]]
            t0 = time.time()
            n_tok = 0
            for rec in run(chunk):
                n_tok += rec["n_gen"]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            dt = time.time() - t0
            print(f"{min(c + a.chunk, len(jobs))}/{len(jobs)} rollouts, {n_tok / dt:,.0f} gen tok/s, "
                  f"{len(chunk) / dt:.1f} rollouts/s", flush=True)

    if a.backend == "vllm":
        llm.llm_engine.engine_core.shutdown(timeout=300)  # a clean engine shutdown seals the sharded captures


if __name__ == "__main__":
    main()
