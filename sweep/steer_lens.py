"""Steer lens: what pushing each neuron does to the model's next-token prediction, through every layer after it.

For each neuron and sign, adds ±dose x (median push-stream norm / ||W_down[:, n]||) to the neuron at every position of
a few short contexts, runs one forward pass, and compares the last position's log-probabilities with the unsteered run.
The push-stream norm is that of the residual stream leaving the layer, or of the MLP output on models that normalise it
(Gemma 2/3, OLMo 2), as in the explorer. Keeps, per neuron and sign, the 12 tokens whose log-probability rises most
(averaged over contexts) and the KL divergence from the unsteered prediction.

    uv run python sweep/steer_lens.py --layers 0-7 --out stats/steer_lens
    uv run python sweep/steer_lens.py --model Qwen/Qwen2.5-0.5B-Instruct --index    # every layer, then the explorer cache

Writes <out>_L{l}.npz per layer (--out defaults to stats/steer_lens for Qwen3.5-4B, stats/<org>--<name>_steer_lens for
other models): up_tok[N, 12], down_tok[N, 12] (token ids for the + push and - push), up_kl[N], down_kl[N]. With
--index it skips layers whose file is already there, then joins every layer's file into the explorer's cache as
steer_index.npz: up, down [L, N, 12], up_kl, down_kl [L, N].
"""
import argparse
import inspect
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import arch as A  # noqa: E402

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


def layer_file(out, l, model, n):
    """The layer's file when it exists and was made for this model with n neurons, else None."""
    path = f"{out}_L{l}.npz"
    if not os.path.exists(path):
        return None
    z = np.load(path)
    same = str(z["model"]) == model if "model" in z.files else model == A.DEFAULT_MODEL
    return z if same and z["up_tok"].shape[0] == n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=A.DEFAULT_MODEL)
    ap.add_argument("--layers", default="", help='"0-31", "22"; default every layer')
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--neurons-per-batch", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--index", action="store_true", help="write the explorer's steer_index.npz from every layer's file")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out_prefix = a.out or ("stats/steer_lens" if a.model == A.DEFAULT_MODEL
                           else os.path.join("stats", f"{A.slug(a.model)}_steer_lens"))
    dev = a.device or A.default_device()

    model, tok = A.load(a.model, dev, A.dtype_from_env())
    ar = A.Arch(model, tok, name=a.model)
    L, N = ar.n_layers, ar.n_neurons
    lo, _, hi = (a.layers or f"0-{L - 1}").partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))
    if a.index:
        layers = [l for l in layers if layer_file(out_prefix, l, a.model, N) is None]
    last = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}

    ctxs = [A.chat_ids(tok, [{"role": "user", "content": c}]) for c in CHAT] + [tok(p).input_ids for p in PLAIN]
    C, T = len(ctxs), max(map(len, ctxs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([[pad] * (T - len(c)) + c for c in ctxs], device=dev)
    mask = torch.tensor([[0] * (T - len(c)) + [1] * len(c) for c in ctxs], device=dev)
    posn = (mask.cumsum(-1) - 1).clamp(min=0)

    push = {"layer": None, "cols": None, "vals": None}
    stream = {}

    def make_hook(i):
        def hook(mod, inp):
            if push["layer"] != i:
                return
            h = inp[0].clone()
            rows = torch.arange(h.shape[0], device=h.device)
            h[rows, :, push["cols"]] += push["vals"][:, None].to(h.dtype)
            return (h,)
        return hook

    for i in range(L):
        ar.mlps[i].down.register_forward_pre_hook(make_hook(i))
        # the stream the push lands in: the layer's output, or the MLP output when a norm follows it
        src = ar.layers[i] if ar.post_norms[i] is None else ar.mlps[i].down
        src.register_forward_hook(lambda m, inp, out, i=i: stream.__setitem__(i, (out[0] if isinstance(out, tuple) else out)))

    def last_logprobs(input_ids, attn, pos):
        logits = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False, **last).logits
        return torch.log_softmax(logits[:, -1, : ar.n_vocab].float(), -1)

    with torch.inference_mode():
        base = last_logprobs(ids, mask, posn)  # [C, V]
        # median norm of the push stream of each layer over these contexts' real tokens
        norms = {i: r[mask.bool()].float().norm(dim=-1).median().item() for i, r in stream.items()}

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    B = a.neurons_per_batch
    for l in layers:
        W = ar.mlps[l].down_w()  # [hidden, inter]
        unit = norms[l] / (W.float().norm(dim=0) * ar.push_scale(l))  # [N]
        out = {k: [] for k in ("up_tok", "down_tok", "up_kl", "down_kl")}
        t0 = time.time()
        for s in range(0, N, B):
            n = torch.arange(s, min(s + B, N), device=dev)
            b = len(n)
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                # rows: neuron-major, contexts repeated
                cols = n.repeat_interleave(C)
                vals = sign * a.dose * unit[cols]
                push.update(layer=l, cols=cols, vals=vals)
                with torch.inference_mode():
                    lp = last_logprobs(ids.repeat(b, 1), mask.repeat(b, 1), posn.repeat(b, 1)).view(b, C, -1)
                push["layer"] = None
                delta = (lp - base[None]).mean(1)  # [b, V]
                kl = (lp.exp() * (lp - base[None])).sum(-1).mean(1)  # KL(steered || base), averaged over contexts
                out[f"{key}_tok"].append(delta.topk(a.k).indices.int().cpu())
                out[f"{key}_kl"].append(kl.cpu())
        arrs = {k: torch.cat(v).numpy() for k, v in out.items()}
        np.savez(f"{out_prefix}_L{l}.npz", **arrs, dose=a.dose, contexts=np.array(CHAT + PLAIN), model=a.model)
        print(f"L{l}: {N} neurons in {time.time() - t0:.0f}s, KL+ median {np.median(arrs['up_kl']):.3f}", flush=True)

    if a.index:
        files = [layer_file(out_prefix, l, a.model, N) for l in range(L)]
        missing = [l for l, z in enumerate(files) if z is None]
        if missing:
            raise SystemExit(f"no steer lens yet for layers {missing}; run them first (--layers)")
        path = os.path.join(A.cache_dir(a.model), "steer_index.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, up=np.stack([z["up_tok"] for z in files]).astype(np.int32),
                 down=np.stack([z["down_tok"] for z in files]).astype(np.int32),
                 up_kl=np.stack([z["up_kl"] for z in files]).astype(np.float32),
                 down_kl=np.stack([z["down_kl"] for z in files]).astype(np.float32), dose=a.dose)
        print(f"-> {path}")


if __name__ == "__main__":
    main()
