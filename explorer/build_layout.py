"""Seriated sheet layouts: places each layer's neurons on its square grid (96x96 for Qwen3.5-4B's 9216; ceil(sqrt(N))
on a side for other widths, the last row partly empty) so that neurons with similar weights sit together. Embeds each
neuron by its write column ("writes": the direction it adds to the residual stream) or its gate and up rows ("reads"),
cosine-normalised, PCA to 50 dims, UMAP to 2D, then snaps to the grid: sort by y into rows of the side length, then
by x within each row.

    uv run python explorer/build_layout.py --kind writes
    uv run python explorer/build_layout.py --kind writes --model Qwen/Qwen2.5-0.5B-Instruct

Writes layout_<kind>.npz to the model's cache folder (explorer/cache/ for Qwen3.5-4B, explorer/cache/<org>--<name>/
otherwise): cell[L, N] (int16, int32 for grids past 32768 cells; row * side + col for neuron n of layer L).
"""
import argparse
import os

import numpy as np
import torch
import umap

import arch as A

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=A.DEFAULT_MODEL)
ap.add_argument("--kind", default="writes", choices=["writes", "reads"])
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--neighbors", type=int, default=30)
a = ap.parse_args()

model, tok = A.load(a.model, "cpu", torch.bfloat16)  # each layer's weights go to float32 one at a time
ar = A.Arch(model, tok, name=a.model)
L, N = ar.n_layers, ar.n_neurons
side = int(np.ceil(np.sqrt(N)))

cells = np.zeros((L, N), np.int16 if side * side <= 32768 else np.int32)
for l in range(L):
    with torch.inference_mode():
        if a.kind == "writes":
            X = ar.write_w(l).T.float().numpy()  # [N, hidden]
        else:
            g, u = ar.mlps[l].read_w()
            X = torch.cat([g, u], 1).float().numpy() if u is not None else g.float().numpy()  # [N, 2*hidden]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, : a.pca] * S[: a.pca]
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for r in range(side):
        band = by_y[r * side:(r + 1) * side]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cells[l, band] = r * side + np.arange(len(band))
    print(f"L{l} done", flush=True)

out = os.path.join(A.cache_dir(a.model), f"layout_{a.kind}.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, cell=cells)
print("->", out)
