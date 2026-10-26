"""Direct-path connections for every neuron in every layer, for the explorer's details tab.

For layers i < j, P[B, A] = (W_gate_j[B] . v_A) * (W_up_j[B] . v_A) with v_A = s_j * w_A, where w_A is neuron A's write
column (W_down_i[:, A], times the post-MLP norm's scale on models that have one) and s_j the per-channel scale of layer
j's pre-MLP norm (1 + w_j on Qwen3.5; explorer/arch.py reads it off the module). A plain (ungated) MLP has no up rows:
P[B, A] = W_in_j[B] . v_A. Neuron A's downstream links in layer j are column A of P, |P| z-scored over B; neuron B's
upstream links in layer i are row B of P, |P| z-scored over A. Each neuron keeps its k strongest links per side across
all layers, with sign(P).

    uv run python explorer/build_connections.py                                    # CUDA, about 5 min on a 4090
    uv run python explorer/build_connections.py --model Qwen/Qwen2.5-0.5B-Instruct

Writes connections.npz to the model's cache folder (explorer/cache/ for Qwen3.5-4B, explorer/cache/<org>--<name>/
otherwise): {down,up}_layer [L, N, k] int8, {down,up}_n int16 (int32 past 32767 neurons), {down,up}_z float16,
{down,up}_sign int8.
"""
import argparse
import os
import time

import numpy as np
import torch

import arch as A

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=A.DEFAULT_MODEL)
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", default=None)
ap.add_argument("--device", default=None)
a = ap.parse_args()
dev = a.device or ("cuda" if torch.cuda.is_available() else A.default_device())
out_path = a.out or os.path.join(A.cache_dir(a.model), "connections.npz")

model, tok = A.load(a.model, dev, torch.bfloat16)
ar = A.Arch(model, tok, name=a.model)
L, N, K = ar.n_layers, ar.n_neurons, a.k
n_dtype = torch.int16 if N <= np.iinfo(np.int16).max else torch.int32

with torch.inference_mode():
    down = [ar.write_w(l) for l in range(L)]  # [hidden, N]
    reads = [ar.mlps[l].read_w() for l in range(L)]  # (gate [N, hidden], up [N, hidden] or None)
    norm = [ar.pre_scale(l).to(torch.bfloat16) for l in range(L)]

# running top-k per neuron and side: score, source layer, source neuron, sign
best = {s: (torch.full((L, N, K), -1e9, device=dev), torch.zeros((L, N, K), dtype=torch.int16, device=dev),
            torch.zeros((L, N, K), dtype=n_dtype, device=dev), torch.zeros((L, N, K), dtype=torch.int8, device=dev))
        for s in ("down", "up")}


def merge(side, l, z, idx, other_layer, sign):
    bz, bl, bn, bs = best[side]
    cz = torch.cat([bz[l], z], 1)
    cl = torch.cat([bl[l], torch.full_like(idx, other_layer, dtype=torch.int16)], 1)
    cn = torch.cat([bn[l], idx.to(n_dtype)], 1)
    cs = torch.cat([bs[l], sign], 1)
    top = cz.topk(K, dim=1)
    bz[l], bl[l], bn[l], bs[l] = top.values, cl.gather(1, top.indices), cn.gather(1, top.indices), cs.gather(1, top.indices)


t0 = time.time()
with torch.inference_mode():
    for j in range(1, L):
        gate, up = reads[j]
        for i in range(j):
            V = down[i] * norm[j][:, None]  # [hidden, N_i]
            P = (gate @ V).float()  # [N_j (B), N_i (A)]
            if up is not None:
                P *= (up @ V).float()
            M = P.abs()
            # downstream of A (layer i): column A over B
            zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-8)
            top = zc.topk(K, dim=0)  # [K, N_i]
            sgn = torch.where(P.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
            merge("down", i, top.values.T, top.indices.T, j, sgn.T)
            # upstream of B (layer j): row B over A
            zr = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
            top = zr.topk(K, dim=1)  # [N_j, K]
            sgn = torch.where(P.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
            merge("up", j, top.values, top.indices, i, sgn)
            del P, M, zc, zr
        print(f"L{j} done ({time.time() - t0:.0f}s)", flush=True)

out = {}
for side, (bz, bl, bn, bs) in best.items():
    out[f"{side}_z"] = bz.clamp(min=-60000).cpu().numpy().astype(np.float16)
    out[f"{side}_layer"] = bl.cpu().numpy().astype(np.int8 if L <= 127 else np.int16)
    out[f"{side}_n"] = bn.cpu().numpy()
    out[f"{side}_sign"] = bs.cpu().numpy()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(out_path, **out, k=K)
print("->", out_path, f"{os.path.getsize(out_path) / 1e6:.0f} MB")
