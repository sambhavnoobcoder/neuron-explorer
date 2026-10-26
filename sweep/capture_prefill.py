"""Downstream state of every rollout in a sweep run: the MLP input (the pre-MLP-norm hidden state, the vector the
gate and up projections read) at the last prompt token, at the run's capture layers, with the rollout's steering push
applied at every prompt position. One prefill per rollout, batched by prompt. Works for any model the explorer runs
(the run's meta.json names it), so it also fills in captures for runs the vLLM fork could not capture.

    uv run python sweep/capture_prefill.py runs/q35-4b-L22-full.jsonl runs/q35-4b-L13-r384.jsonl ...

Writes captures/<run>.mlp_in.npy, float16 [rows, capture layers, hidden], row i = line i of the jsonl, and
captures/<run>.mlp_in.json with the capture layers. Neuron activations follow as act(x @ W_gate.T) * (x @ W_up.T)
(explorer/arch.py: Arch.mlps[layer].acts(x)).
"""
import argparse
import json
import os
import time

import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import arch as A  # noqa: E402


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default=None, help="defaults to the model in each run's meta.json")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="the sweep's --seed (random-control directions)")
    ap.add_argument("--out-dir", default="captures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    loaded = {}  # model name -> Arch, one model at a time; consecutive runs of one model share it
    state = {"push_layer": None, "vec": None, "caps": {}, "last": None}

    def push_hook(i):
        def hook(mod, inp, out):
            if state["push_layer"] == i:
                return out + state["vec"][:, None, :].to(out.dtype)
        return hook

    def cap_hook(i):
        def hook(mod, inp):
            if i in state["caps"]:
                state["caps"][i] = inp[0][:, -1].float().cpu()
                if i == state["last"]:
                    raise Stop
        return hook

    def get(model_name):
        if model_name not in loaded:
            loaded.clear()
            model, _ = A.load(model_name, a.device, torch.bfloat16)
            ar = A.Arch(model, name=model_name)
            # the push at the MLP output projection's output (the sweep's push point), the capture at the MLP's input
            for i in range(ar.n_layers):
                ar.mlps[i].down.register_forward_hook(push_hook(i))
                ar.mlps[i].module.register_forward_pre_hook(cap_hook(i))
            loaded[model_name] = ar
        return loaded[model_name]

    for run in a.runs:
        meta = json.load(open(run.replace(".jsonl", ".meta.json")))
        layer, cap_layers = meta["layer"], meta["capture_layers"]
        ar = get(a.model or meta.get("model", A.DEFAULT_MODEL))
        lm, dev = ar.lm, a.device
        name = os.path.basename(run).replace(".jsonl", "")
        out_path = os.path.join(a.out_dir, f"{name}.mlp_in.npy")
        if os.path.exists(out_path):
            print(f"{name}: exists, skipping", flush=True)
            continue
        w_down = ar.mlps[layer].down_w().detach().T.float().cpu().numpy()  # [N, hidden]
        rows = [json.loads(l) for l in open(run)]
        n_ctrl = max([r["neuron"] for r in rows if r["kind"] == "random"], default=-1) + 1
        rng = np.random.default_rng(a.seed)
        ctrl_ref = rng.integers(len(w_down), size=n_ctrl)
        ctrl_vecs = rng.standard_normal((n_ctrl, w_down.shape[1])).astype(np.float32)
        ctrl_vecs *= (np.linalg.norm(w_down[ctrl_ref], axis=1) / np.linalg.norm(ctrl_vecs, axis=1))[:, None]
        for r in rows:  # the sweep's own record of which neuron a control borrowed its norm from
            if r["kind"] == "random":
                assert r["ref_neuron"] == int(ctrl_ref[r["neuron"]]), "control directions don't match the sweep's seed"

        def vec(r):
            if r["kind"] == "neuron":
                return w_down[r["neuron"]] * r["add"]
            if r["kind"] == "random":
                return ctrl_vecs[r["neuron"]] * r["add"]
            return np.zeros(w_down.shape[1], np.float32)

        out = np.lib.format.open_memmap(out_path + ".tmp", mode="w+", dtype=np.float16,
                                        shape=(len(rows), len(cap_layers), w_down.shape[1]))
        t0 = time.time()
        by_prompt = {}
        for i, r in enumerate(rows):
            by_prompt.setdefault(r["prompt"], []).append(i)
        for p, idx in by_prompt.items():
            ids = torch.tensor(meta["prompt_ids"][p], device=dev)[None]
            for s in range(0, len(idx), a.batch):
                chunk = idx[s:s + a.batch]
                state.update(push_layer=layer, vec=torch.tensor(np.stack([vec(rows[i]) for i in chunk]), device=dev),
                             caps={c: None for c in cap_layers}, last=max(cap_layers))
                with torch.inference_mode():
                    try:
                        lm(input_ids=ids.expand(len(chunk), -1), use_cache=False)
                    except Stop:
                        pass
                out[chunk] = torch.stack([state["caps"][c] for c in cap_layers], 1).numpy().astype(np.float16)
        out.flush()
        del out
        os.replace(out_path + ".tmp", out_path)
        json.dump({"run": name, "capture_layers": cap_layers, "hook": "mlp_in", "position": "last_prompt",
                   "rows": len(rows)}, open(out_path.replace(".npy", ".json"), "w"))
        print(f"{name}: {len(rows)} rows, layers {cap_layers}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
