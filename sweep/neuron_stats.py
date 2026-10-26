"""Per-neuron activation ranges for MLP layers of a model, plus their down-projection columns.

A neuron's activation is act(gate) * up, the input to the MLP's output projection (down_proj on most models; silu for
Qwen and Llama, gelu for Gemma). Positive activations come from two sign regimes (gate>0, up>0) and (gate<0, up<0);
negative ones from (gate>0, up<0) and (gate<0, up>0). This records the extreme of each regime per neuron over a mixed
web + chat corpus, and saves W_down[:, n] for every neuron, which is the direction a steering push on neuron n adds to
the MLP output. A plain (ungated) MLP's activation is act(fc_in); its regimes are the sign of the pre-activation alone,
recorded as if up were +1.

    uv run python sweep/neuron_stats.py --model Qwen/Qwen3.5-4B --layers 22 --tokens 2000000 --out stats/q35-4b
    uv run python sweep/neuron_stats.py --model Qwen/Qwen2.5-0.5B-Instruct --layers 12 --tokens 200000

Writes <out>_L22.npz (--out defaults to stats/q35-4b for Qwen3.5-4B, stats/<org>--<name> for other models) with
    pp_max, nn_max   largest positive activation in the (+,+) and (-,-) regimes          [I]
    pn_min, np_min   most negative activation in the (gate+, up-) and (gate-, up+) regimes [I]
    abs_max          max |activation| over everything (the unit for steering doses)      [I]
    w_down           down_proj columns, one row per neuron                                [I, H] float16
    resid_norm_q10_50_90   quantiles of the per-token residual norm leaving the layer
    push_norm_q10_50_90    the same for the stream an add push lands in (the scale for push sizes): the residual
                     leaving the layer, or the MLP output on models that normalise it (push_space says which)
    push_scale       what a push on an activation reaches the push stream with, per unit of its down column (the
                     block's residual scale: Granite, MiniCPM; 1 on most models)
    post_mlp_norm, model_type, model, layer, tokens
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import arch as A  # noqa: E402


class Stop(Exception):
    pass


def text_stream(tok, seed):
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    for w, c in zip(web, chat):
        yield w["text"]
        yield A.chat_text(tok, c["messages"])


def windows(tok, seq_len, batch, seed):
    """Packs documents (EOS-separated) into [batch, seq_len] token windows."""
    buf = []
    sep = [tok.eos_token_id] if tok.eos_token_id is not None else []
    for text in text_stream(tok, seed):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + sep)
        while len(buf) >= seq_len * batch:
            chunk, buf = buf[: seq_len * batch], buf[seq_len * batch :]
            yield torch.tensor(chunk).view(batch, seq_len)


class RegimeStats:
    def __init__(self, n, device):
        f = lambda v: torch.full((n,), v, device=device, dtype=torch.float32)
        self.pp_max, self.nn_max = f(0.0), f(0.0)
        self.pn_min, self.np_min = f(0.0), f(0.0)
        self.gate = self.up = None
        self.resid_norms = []  # per-token residual norm leaving the layer, positions >= 1 (position 0 is a sink)
        self.push_norms = []  # per-token MLP output norm, on models that normalise the MLP output

    def update_resid(self, out):
        h = out[0] if isinstance(out, tuple) else out
        self.resid_norms.append(h[:, 1:].float().norm(dim=-1).flatten().cpu())

    def update_push(self, out):
        self.push_norms.append(out[:, 1:].float().norm(dim=-1).flatten().cpu())

    def update(self, act):
        # position 0 of each window is an attention sink with outsized activations; it would inflate abs_max
        g, a = (t[:, 1:].flatten(0, 1).float() for t in (self.gate, act))
        gp = g > 0
        up_ = self.up[:, 1:].flatten(0, 1) > 0 if self.up is not None else torch.ones_like(gp)
        ninf = torch.tensor(float("-inf"), device=a.device)
        pinf = torch.tensor(float("inf"), device=a.device)
        self.pp_max = torch.maximum(self.pp_max, torch.where(gp & up_, a, ninf).amax(0))
        self.nn_max = torch.maximum(self.nn_max, torch.where(~gp & ~up_, a, ninf).amax(0))
        self.pn_min = torch.minimum(self.pn_min, torch.where(gp & ~up_, a, pinf).amin(0))
        self.np_min = torch.minimum(self.np_min, torch.where(~gp & up_, a, pinf).amin(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=A.DEFAULT_MODEL)
    ap.add_argument("--layers", default="22", help="comma-separated decoder layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output prefix (default stats/<model short name>)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    out_prefix = a.out or os.path.join("stats", A.short_name(a.model))

    model, tok = A.load(a.model, a.device, torch.bfloat16)
    ar = A.Arch(model, tok, name=a.model)

    stats = {}
    for l in layers:
        mlp = ar.mlps[l]
        s = stats[l] = RegimeStats(mlp.width, a.device)
        if mlp.kind == "fused":  # gate is the first half of the fused projection's output
            mlp.gate.register_forward_hook(lambda m, i, o, s=s: (setattr(s, "gate", o.chunk(2, -1)[0]),
                                                                 setattr(s, "up", o.chunk(2, -1)[1])))
        else:
            mlp.gate.register_forward_hook(lambda m, i, o, s=s: setattr(s, "gate", o))
            if mlp.up is not None:
                mlp.up.register_forward_hook(lambda m, i, o, s=s: setattr(s, "up", o))
        mlp.down.register_forward_pre_hook(lambda m, i, s=s: s.update(i[0]))
        if ar.post_norms[l] is not None:
            mlp.down.register_forward_hook(lambda m, i, o, s=s: s.update_push(o))
        ar.layers[l].register_forward_hook(lambda m, i, o, s=s: s.update_resid(o))

    def stop(m, i, o):  # nothing past the deepest target layer is needed
        raise Stop
    ar.layers[max(layers)].register_forward_hook(stop)

    n_steps = -(-a.tokens // (a.seq_len * a.batch))
    t0 = time.time()
    with torch.inference_mode():
        for step, ids in enumerate(itertools.islice(windows(tok, a.seq_len, a.batch, a.seed), n_steps)):
            try:
                ar.lm(input_ids=ids.to(a.device), use_cache=False)
            except Stop:
                pass
            if step % 10 == 0 or step == n_steps - 1:
                done = (step + 1) * a.seq_len * a.batch
                print(f"{step + 1}/{n_steps} steps, {done:,} tokens, {done / (time.time() - t0):,.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    model_type = A.raw_config(a.model).get("model_type", "")
    for l, s in stats.items():
        pp, nn, pn, np_ = (x.cpu().numpy() for x in (s.pp_max, s.nn_max, s.pn_min, s.np_min))
        abs_max = np.maximum.reduce([pp, nn, -pn, -np_])
        w_down = ar.mlps[l].down_w().detach().float().T.cpu().numpy().astype(np.float16)
        resid_q = np.quantile(torch.cat(s.resid_norms).numpy(), [0.1, 0.5, 0.9])
        post = ar.post_norms[l] is not None
        push_q = np.quantile(torch.cat(s.push_norms).numpy(), [0.1, 0.5, 0.9]) if post else resid_q
        path = f"{out_prefix}_L{l}.npz"
        np.savez(path, pp_max=pp, nn_max=nn, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down,
                 resid_norm_q10_50_90=resid_q, push_norm_q10_50_90=push_q, push_space="mlp_out" if post else "residual",
                 post_mlp_norm=post, push_scale=ar.push_scale(l), model_type=model_type, model=a.model, layer=l,
                 tokens=n_steps * a.seq_len * a.batch)
        print(f"L{l}: residual norm leaving the layer p10/p50/p90 = {resid_q.round(1)}"
              + (f"; MLP output, where pushes land before the post-MLP norm, {push_q.round(1)}" if post else ""))
        dn_top = (nn > pp).mean()
        print(f"L{l}: abs_max median {np.median(abs_max):.3f} [{abs_max.min():.3f}, {abs_max.max():.3f}]; "
              f"(-,-) regime holds the positive max for {dn_top:.1%} of neurons -> {path}")


if __name__ == "__main__":
    main()
