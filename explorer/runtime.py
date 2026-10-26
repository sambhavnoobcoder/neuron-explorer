"""Live model for the explorer (EXPLORER_MODEL, Qwen3.5-4B by default): steered generation (add or multiply one neuron,
per batch row), per-token neuron activations on arbitrary text, and logit-lens tokens for a neuron's down column.

The hook sits on the input of a layer's MLP output projection (down_proj on most models), where the neuron activations
act(gate) * up live, so add, multiply, and reading are all exact. Add doses are in units of the median norm of the
stream the push lands in, the same unit as the sweep's `--dose-unit rel`: the residual stream leaving the layer, or on
models that normalise the MLP output before the residual add (Gemma 2/3, OLMo 2) the MLP output itself, since that norm
caps how far any push can move the residual stream.
"""
import json
import os
import sqlite3
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import arch as A

MODEL = os.environ.get("EXPLORER_MODEL", A.DEFAULT_MODEL)
NO_MODEL = bool(os.environ.get("EXPLORER_NO_MODEL"))  # cache-only: never download or load weights
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


CACHE = A.cache_dir(MODEL)


_warned = set()


def fits(arr, shape, what):
    """Whether a cached [layers, neurons, ...] array was built for a model of this shape; says so (once) when it wasn't."""
    if shape is None or tuple(arr.shape[:2]) == tuple(shape):
        return True
    if what not in _warned:
        _warned.add(what)
        print(f"[explorer] ignoring {what}: built for {arr.shape[0]} layers x {arr.shape[1]} neurons, "
              f"{MODEL} has {shape[0]} x {shape[1]}", flush=True)
    return False


def _locked(f):
    """Runs a method entirely under the model lock, so its GPU temporaries are created and freed while no other
    thread is using the GPU (MPS fails when two threads' work interleaves)."""
    def g(self, *a, **k):
        with self.lock:
            return f(self, *a, **k)
    g.__doc__, g.__name__ = f.__doc__, f.__name__
    return g


class Runtime:
    def __init__(self, stats_dir, shape=None):
        self.stats_dir = stats_dir
        self.shape = shape  # (layers, neurons) of MODEL; caches of any other shape are ignored
        self.arch = None
        os.makedirs(CACHE, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(CACHE, "connections.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS con (key TEXT PRIMARY KEY, val TEXT)")
        self._db_lock = threading.Lock()
        self._lens_idx = None
        self._con_idx = None
        self.lock = threading.RLock()  # re-entrant: a locked method can call another
        self.model = self.tok = None
        self.status = "idle"
        self.push_p50 = {}
        self._push = None  # [(layer, neuron, mode, per-row values tensor)], any number of neurons at once (neuron None = all)
        self._heads = None  # [(layer, head, scale)]: attention heads scaled (0 = ablated) during a read; head None = all
        self._read = None  # (layer, neuron) while reading
        self._seen = {}
        self._con_cache = {}
        self._capture_all = None
        self._capture_from = 0
        self._capture_full = None
        self._capture_heads = None  # while set: each layer's query/key side and each head's write size, per token
        self.heads = None  # the last text's captures: {"T", "q", "writes"}, see full_activations(heads=True)
        self._lens_cache = {}

    # ---------- loading ----------
    def ensure_loaded(self):
        if self.model is not None:
            return
        if NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.model is not None:
                return
            self.status = "loading"
            dev = os.environ.get("EXPLORER_DEVICE") or A.default_device()
            m, tok = A.load(MODEL, dev, A.dtype_from_env())
            a = A.Arch(m, tok, name=MODEL)
            self.dev, self.lm, self.arch, self.tok = dev, a.lm, a, tok
            self.stop = A.stop_ids(m, tok)
            # read every norm's scale and shape now: a probe runs the norm, which must happen before the hooks below
            # are on it (a hook probing its own norm would recurse)
            for i in range(a.n_layers):
                a.pre_scale(i), a.attn_pre_scale(i), a.attn_post_scale(i)
                if a.pre_norms[i] is not None:
                    a.norm_centred(a.pre_norms[i])
            # logits for the last position only, where the model's forward takes it
            import inspect
            self._last = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(m.forward).parameters else {}
            for i, layer in enumerate(a.layers):
                down = a.mlps[i].down
                down.register_forward_pre_hook(self._hook(i))
                # the push unit's stream: the residual leaving the layer, or the MLP output when a norm follows it
                if a.post_norms[i] is None:
                    layer.register_forward_hook(self._resid_hook(i))
                else:
                    down.register_forward_hook(self._resid_hook(i))
                # attention output projection: its input is the heads side by side (Qwen3.5 full attention: o_proj,
                # 16 x 256 after the output gate; linear attention: out_proj, 32 value heads x 128 after the gated norm)
                _, out_proj, n_heads = a.attn[i]
                if out_proj is not None and n_heads:
                    out_proj.register_forward_pre_hook(self._head_hook(i, n_heads))
                # head captures (full_activations(heads=True)): the MLP's input and its norm's divisor, the attention
                # output norm's divisor, and each head's queries, keys and values (softmax) or DeltaNet state
                if a.pre_norms[i] is not None:
                    a.pre_norms[i].register_forward_hook(self._mlp_in_hook(i))
                else:
                    a.mlps[i].module.register_forward_pre_hook(self._mlp_in_hook(i))
                if a.attn_post_norms[i] is not None:
                    a.attn_post_norms[i].register_forward_hook(self._attn_post_hook(i))
                p = a.attn_parts(i)
                if p is not None:
                    attn = getattr(layer, a.attn[i][0])
                    attn.register_forward_pre_hook(self._qk_hook(i), with_kwargs=True)
                    if p.kind == "softmax":
                        attn.register_forward_hook(self._qk_done_hook(i))
                    else:
                        attn.norm.register_forward_pre_hook(self._gnorm_hook(i, n_heads))
            A.install_capture(m)
            self.model = m  # last: ensure_loaded takes a set model to mean everything above is in place
            self.status = "ready"

    def _hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            if self._capture_full is not None:
                self._capture_full[i] = h[0].float().cpu()  # [tokens, inter]
            if self._capture_all is not None:
                a = h[:, self._capture_from:].float().flatten(0, 1)  # [tokens, inter]
                mx, amx = a.max(0)
                mn, amn = a.min(0)
                self._capture_all[i] = (mx.cpu(), amx.cpu(), mn.cpu(), amn.cpu())
            if self._read is not None and self._read[0] == i:
                n = self._read[1]
                self._seen["act"] = h[0, :, n].float().cpu()
            mine = [p for p in self._push if p[0] == i] if self._push else []
            if mine:
                h = h.clone()
                for _, n, mode, vals in mine:
                    if n is None:  # the whole layer's MLP (multiply only)
                        h = h * vals[:, None, None].to(h.dtype)
                    elif mode == "mul":
                        h[:, :, n] = h[:, :, n] * vals[:, None].to(h.dtype)
                    else:
                        h[:, :, n] = h[:, :, n] + vals[:, None].to(h.dtype)
                return (h,)
        return hook

    def _mlp_in_hook(self, i):
        """On the norm in front of the MLP (or the MLP itself when it reads the raw residual): the MLP's input, and the
        divisor the norm applied at each token (its rms, or standard deviation for a LayerNorm; 1 without a norm)."""
        a = self.arch
        norm = a.pre_norms[i]

        def hook(mod, args, out=None):
            if self._capture_heads is None:
                return
            if norm is None:  # forward pre-hook on the MLP: the residual is the input
                x = args[0][0]
                self._capture_heads["x"][i] = x.half()
                self._capture_heads["rms"][i] = torch.ones(x.shape[0], device=x.device)
                return
            r = args[0][0].float()
            if a.norm_centred(norm):
                r = r - r.mean(-1, keepdim=True)
            self._capture_heads["x"][i] = (out[0] if isinstance(out, tuple) else out)[0].half()
            self._capture_heads["rms"][i] = (r.pow(2).mean(-1) + A.norm_eps(norm)).sqrt()
        return hook

    def _attn_post_hook(self, i):
        """On a norm applied to the attention output (Gemma 2/3, OLMo 2): its divisor per token, which also scales each
        head's write size."""
        norm = self.arch.attn_post_norms[i]

        def hook(mod, args, out):
            if self._capture_heads is None:
                return
            o = args[0][0].float()
            rms = (o.pow(2).mean(-1) + A.norm_eps(norm)).sqrt()
            self._capture_heads["orms"][i] = rms
            if i in self._capture_heads["writes"]:
                self._capture_heads["writes"][i] /= rms.cpu()[:, None]
        return hook

    def _gnorm_hook(self, i, n_heads):
        def hook(mod, args):
            if self._capture_heads is not None:  # linear attention: each head's output = scale * (sum of A_ij v_j)
                o, z = args[0].float(), args[1].float()
                sc = mod.weight.float() * F.silu(z) * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + mod.variance_epsilon)
                self._capture_heads["s"][i] = sc.view(-1, n_heads, sc.shape[-1]).transpose(0, 1).half()  # [H, T, D]
        return hook

    def _head_hook(self, i, n_heads):
        post = self.arch.attn_post_norms[i] is not None

        def hook(mod, inp):
            if self._capture_heads is not None:  # each head's write into the residual: |W_O[:, head] . out_head|
                x = inp[0][0]
                xv = x.view(x.shape[0], n_heads, -1)
                W = A._weight(mod)
                if post:  # through the post-attention norm's scale; its per-token divisor follows (_attn_post_hook)
                    W = W * self.arch.attn_post_scale(i).to(W.dtype)[:, None]
                W = W.view(W.shape[0], n_heads, -1)
                self._capture_heads["writes"][i] = torch.einsum("thd,ohd->tho", xv, W).float().norm(dim=-1).cpu()  # [T, H]
                self._capture_heads["y"][i] = xv.transpose(0, 1).half()  # [H, T, D]: each head's output
            mine = [x for x in self._heads if x[0] == i] if self._heads else []
            if not mine:
                return None
            h = inp[0].clone()
            v = h.view(*h.shape[:-1], n_heads, h.shape[-1] // n_heads)
            for _, head, scale in mine:
                if head is None:
                    v.mul_(scale)
                else:
                    v[..., head, :] *= scale
            return (h,)
        return hook

    def _qk_hook(self, i):
        """While capturing heads: what decides each head's attention pattern. Softmax attention (full or sliding-window):
        the queries, keys and values the model's own attention call receives (after norms and rotary; A.CAPTURE), with
        its scale, softcap and window, plus the output gate where the query projection carries one (Qwen3.5). Linear
        attention (Gated DeltaNet): queries and keys after the short conv and l2 norm, plus each token's write strength
        (beta) and the per-step log decay; head_row() turns these into the exact effective attention row, the weight
        each earlier token's value gets in a query's output."""
        p = self.arch.attn_parts(i)

        def hook(mod, args, kwargs):
            if self._capture_heads is None:
                return None
            x = kwargs.get("hidden_states", args[0] if args else None)
            with torch.inference_mode():
                T = x.shape[1]
                if p.kind == "softmax":
                    c = self._capture_heads["q"][i] = {"kind": "full"}
                    A.CAPTURE[id(mod)] = c
                    if p.gated:
                        hd = p.q.shape[1]
                        gate = p.q_proj(x).view(T, p.n, 2 * hd)[..., hd:]
                        c["s"] = gate.float().sigmoid().transpose(0, 1).half()  # [H, T, D]
                else:
                    l2norm = sys.modules[type(mod).__module__].l2norm
                    qkv = mod.in_proj_qkv(x).transpose(1, 2)
                    w = mod.conv1d.weight
                    qkv = F.silu(F.conv1d(F.pad(qkv, (w.shape[-1] - 1, 0)), w, groups=mod.conv_dim)).transpose(1, 2)[0]
                    q, k, v = torch.split(qkv, [mod.key_dim, mod.key_dim, mod.value_dim], -1)
                    q = l2norm(q.float().view(T, -1, mod.head_k_dim)) / mod.head_k_dim ** 0.5
                    k = l2norm(k.float().view(T, -1, mod.head_k_dim))
                    beta = mod.in_proj_b(x)[0].float().sigmoid()  # [T, Hv]
                    logd = -mod.A_log.float().exp() * F.softplus(mod.in_proj_a(x)[0].float() + mod.dt_bias)  # [T, Hv]
                    self._capture_heads["q"][i] = {"kind": "linear", "q": q.transpose(0, 1).half(), "k": k.transpose(0, 1).half(),
                                                   "beta": beta.T.contiguous(), "logd": logd.T.contiguous(),
                                                   "rep": mod.num_v_heads // mod.num_k_heads,
                                                   "v": v.view(T, -1, mod.head_v_dim).transpose(0, 1).half()}
            return None
        return hook

    def _qk_done_hook(self, i):
        """After a softmax layer's attention ran during a capture: its captured queries, keys and values as float32
        [heads, T, dim] (float16 would round large logits enough to move the softmax: Pythia, Qwen2.5's first layer),
        and the scale (1 / sqrt(dim) when the call passed none)."""
        def hook(mod, args, out):
            c = A.CAPTURE.pop(id(mod), None)
            if c is None or "q" not in c:
                if c is not None:  # the attention call never came through the capture: no pattern for this layer
                    self._capture_heads["q"].pop(i, None)
                return
            c.update(q=c["q"].float(), k=c["k"].float(), v=c["v"].float())
            c["scale"] = c.pop("scaling") or c["q"].shape[-1] ** -0.5
        return hook

    @staticmethod
    def _soft_scores(c, s, qpos, kpos):
        """Softmax logits s (already scaled) with the layer's softcap, causal mask and sliding window: positions are
        query indices [Q] and key indices [K]."""
        if c.get("softcap"):
            s = torch.tanh(s / c["softcap"]) * c["softcap"]
        allowed = kpos[None, :] <= qpos[:, None]
        if c.get("window"):
            allowed &= qpos[:, None] - kpos[None, :] < c["window"]
        return s.masked_fill(~allowed, float("-inf"))

    def head_row(self, layer, head, pos):
        with self.lock:
            return self._head_row(layer, head, pos)

    def _head_row(self, layer, head, pos):
        """Attention row of one head for query token `pos` of the last captured text: the weight of every token
        0..pos in that head's output there. Softmax attention: the softmax pattern. Linear attention: the exact
        effective weight (signed; queries carried back through each later token's decay and delta-rule write), scaled
        so its absolute values sum to 1."""
        c = self.heads["q"][layer]
        row = self.raw_row(layer, head, pos)
        if c["kind"] == "full":
            return row.cpu().tolist()
        return (row / row.abs().sum().clamp(min=1e-12)).cpu().tolist()

    def head_pattern(self, layer, head):
        with self.lock:
            return self._head_pattern(layer, head)

    def _head_pattern(self, layer, head):
        """The whole attention pattern of one head on the last captured text: [T, T], row = query token, column = the
        earlier token it reads (zero above the diagonal). Linear layers give the exact effective weights, each row
        scaled so its absolute values sum to 1, as in head_row()."""
        return self._layer_patterns(layer, None, [head])[0]

    def raw_row(self, layer, head, pos, caps=None):
        """(Called with the model lock held.) The attention row of one head at token `pos`, unscaled: softmax weights (softmax attention) or the exact
        effective weights (linear attention), so that the head's output there is scale * sum_j row[j] * v_j."""
        c = (caps or self.heads)["q"][layer]
        with torch.inference_mode():
            if c["kind"] == "full":
                kv = head // (c["q"].shape[0] // c["k"].shape[0])
                s = (c["k"][kv, :pos + 1].float() @ c["q"][head, pos].float()) * c["scale"]
                dev = s.device
                s = self._soft_scores(c, s[None], torch.tensor([pos], device=dev), torch.arange(pos + 1, device=dev))[0]
                return s.softmax(-1)
            kh = head // c["rep"]
            q, k = c["q"][kh].float(), c["k"][kh].float()
            beta, d = c["beta"][head], c["logd"][head].exp()
            u, row = q[pos].clone(), torch.zeros(pos + 1, device=q.device)
            for j in range(pos, -1, -1):
                dot = u @ k[j]
                row[j] = beta[j] * dot
                u = d[j] * (u - beta[j] * dot * k[j])
            return row

    def attn_drive(self, layer, n, pos, caps=None, top=6):
        with self.lock:
            return self._attn_drive(layer, n, pos, caps, top)

    def _attn_drive(self, layer, n, pos, caps=None, top=6):
        """How attention drives neuron n of `layer` at token `pos` of a captured text (full_activations(heads=True)).
        The activation act(gate) * up is linearised in the residual at this token, through the pre-MLP norm with its
        divisor held at this token's value (and, on models that normalise the attention output, through that norm the
        same way). Returns act (the activation), total (the linearised drive of the whole residual), heads
        ([(layer, head, contribution)] for every head at this layer or earlier whose output the MLP reads: its output
        read by the linearisation) and src ({(layer, head): [contribution per token 0..pos]}, for the `top` heads by
        size, softmax and linear heads: attention weight times what that token's value carries)."""
        c = caps or self.heads
        a = self.arch
        mlp = a.mlps[layer]
        with torch.inference_mode():
            x, rms = c["x"][layer][pos].float(), c["rms"][layer][pos].float()
            G, U = mlp.read_w()
            bg, bu = mlp.read_b()
            wg = G[n].float()
            g = float(wg @ x) + (float(bg[n]) if bg is not None else 0.0)
            wu = U[n].float() if U is not None else None
            u = float(wu @ x) + (float(bu[n]) if bu is not None else 0.0) if U is not None else None
        with torch.inference_mode(False), torch.enable_grad():  # the activation's slope at g, from the model's own act
            gt = torch.tensor(float(g), requires_grad=True)
            av = mlp.act(gt)
            av.backward()
            act_g, slope = float(av.detach()), float(gt.grad)
        with torch.inference_mode():
            Jx = slope * u * wg + act_g * wu if U is not None else slope * wg  # d act / d (MLP input)
            norm = a.pre_norms[layer]
            b = getattr(norm, "bias", None) if norm is not None else None
            total = float(Jx @ (x - b.float() if b is not None else x))
            J = Jx * a.pre_scale(layer) / rms  # d act / d residual, the norm's divisor held
            if norm is not None and a.norm_centred(norm):
                J = J - J.mean()
            out, T = [], {}
            for l in range(layer + (0 if a.parallel[layer] else 1)):
                if l not in c["y"]:
                    continue
                Jl = J * a.attn_post_scale(l) / c["orms"][l][pos].float() if l in c["orms"] else J
                Jl = Jl * a.attn_scale[l]  # the block's residual scale (Granite, MiniCPM)
                W = A._weight(a.attn[l][1]).float()
                y = c["y"][l][:, pos].float()  # [H, D]
                t = (W.T @ Jl).view(y.shape)
                T[l] = t
                out += [(l, h, float(v)) for h, v in enumerate((t * y).sum(-1).tolist())]
            src = {}
            for l, h, _ in sorted(out, key=lambda r: -abs(r[2]))[:top]:
                q = c["q"].get(l)
                if q is None:
                    continue
                if q["kind"] == "full":
                    v = q["v"][h // (q["q"].shape[0] // q["k"].shape[0]), :pos + 1].float()
                    sc = q["s"][h, pos].float() if "s" in q else 1.0
                else:
                    v, sc = q["v"][h, :pos + 1].float(), c["s"][l][h, pos].float()
                src[(l, h)] = (self.raw_row(l, h, pos, c) * (v @ (sc * T[l][h]))).tolist()
            return {"act": act_g * u if U is not None else act_g, "total": total, "heads": out, "src": src}

    def layer_patterns(self, layer, caps=None):
        with self.lock:
            return self._layer_patterns(layer, caps)

    def _layer_patterns(self, layer, caps=None, heads=None):
        """head_pattern() for every head of a layer at once (or those in `heads`): [H, T, T], from `caps` (a
        self.heads capture) or the last captured text."""
        c = (caps or self.heads)["q"][layer]
        with torch.inference_mode():
            if c["kind"] == "full":
                q = c["q"].float() if heads is None else c["q"][heads].float()
                rep = c["q"].shape[0] // c["k"].shape[0]
                k = c["k"].float().repeat_interleave(rep, 0)
                k = k if heads is None else k[heads]
                s = (q @ k.transpose(1, 2)) * c["scale"]
                T = s.shape[-1]
                pos = torch.arange(T, device=s.device)
                return self._soft_scores(c, s, pos, pos).softmax(-1).cpu().numpy()
            q = c["q"].float().repeat_interleave(c["rep"], 0)
            k = c["k"].float().repeat_interleave(c["rep"], 0)
            beta, d = c["beta"].float(), c["logd"].float().exp()  # [H, T]
            if heads is not None:
                q, k, beta, d = q[heads], k[heads], beta[heads], d[heads]
            H, T, D = q.shape
            A_ = torch.zeros(H, T, T, device=q.device)
            U = torch.zeros(H, T, D, device=q.device)
            for j in range(T - 1, -1, -1):
                U[:, j] = q[:, j]
                dot = (U @ k[:, j, :, None])[..., 0]  # [H, T]
                A_[:, :, j] = beta[:, j, None] * dot
                U = d[:, j, None, None] * (U - (beta[:, j, None] * dot)[..., None] * k[:, j, None, :])
            A_ = A_ / A_.abs().sum(-1, keepdim=True).clamp(min=1e-12)
            return A_.cpu().numpy()

    def _head_slices(self, l, h):
        """One head's weight slices after the norms around it: value rows and query / key rows [dim, hidden] after the
        pre-attention norm's scale, output columns [hidden, dim] through the gated norm (DeltaNet) and the
        post-attention norm's scale (Gemma); None when the layer's attention isn't one this code can cut into heads."""
        a = self.arch
        p = a.attn_parts(l)
        if p is None:
            return None
        g = a.attn_pre_scale(l).float()
        WO = p.out[:, h].float()
        if p.o_scale is not None:
            WO = WO * p.o_scale.float()
        WO = WO * a.attn_post_scale(l).float()[:, None]
        return {"v": p.v[p.v_of(h)].float() * g, "o": WO, "q": p.q[p.q_of(h)].float() * g, "k": p.k[p.k_of(h)].float() * g}

    def _mlp_reads(self, l):
        """Layer l's MLP read rows after its pre-MLP norm's scale: (gate [N, hidden], up [N, hidden] or None)."""
        a = self.arch
        w = a.pre_scale(l).float()
        G, U = a.mlps[l].read_w()
        return G.float() * w, (U.float() * w if U is not None else None)

    @_locked
    def head_links(self, layer, head, k=40):
        """Weight-level reads and writes of one attention head. reads: earlier neurons whose write column lands in the
        head's value read space, as the share of the column's length the head picks up (after the pre-attention norm
        weight); writes: neurons whose MLP reads this head's output (this layer or later; later only when attention
        and MLP run side by side) and whose gate and up rows read the head's output space, the same share for each
        row, multiplied (the gate row alone for an ungated MLP). Shares make large-norm neurons that line up with every
        head (hubs) sink. Each list is the top k by that share, z-scored over all candidate neurons. q and k: earlier
        neurons whose write column lands in the head's query or key read space, the same way as reads."""
        key = f"{layer}:{head}:{k}:qk"
        with self._db_lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS headcon (key TEXT PRIMARY KEY, val TEXT)")
            row = self._db.execute("SELECT val FROM headcon WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        self.ensure_loaded()
        a = self.arch
        with self.lock, torch.inference_mode():
            hs = self._head_slices(layer, head)
            if hs is None:
                return {"reads": [], "writes": [], "q": [], "k": [],
                        "error": f"layer {layer}'s attention can't be cut into heads for {MODEL}"}
            # orthonormal bases of the read and write spaces, so a share is a plain projection length
            QV = torch.linalg.qr(hs["v"].T.cpu())[0].to(self.dev)  # [hidden, hd]
            QO = torch.linalg.qr(hs["o"].cpu())[0].to(self.dev)
            QQ = torch.linalg.qr(hs["q"].T.cpu())[0].to(self.dev)
            QK = torch.linalg.qr(hs["k"].T.cpu())[0].to(self.dev)

            def top(scores):
                s = torch.cat(scores)
                z = (s - s.mean()) / (s.std() + 1e-8)
                t = z.topk(min(k, z.numel()))
                N = scores[0].numel()
                return t.values.tolist(), t.indices.tolist(), N
            reads, rq, rk, wr = [], [], [], []
            for l in range(layer):
                D = a.write_w(l).float()  # [hidden, N]
                dn = D.norm(dim=0).clamp(min=1e-8)
                reads.append(((QV.T @ D).norm(dim=0) / dn).cpu())
                rq.append(((QQ.T @ D).norm(dim=0) / dn).cpu())
                rk.append(((QK.T @ D).norm(dim=0) / dn).cpu())
            first = layer + 1 if a.parallel[layer] else layer
            share = lambda R: (R @ QO).norm(dim=1) / R.norm(dim=1).clamp(min=1e-8)
            for l in range(first, a.n_layers):
                G, U = self._mlp_reads(l)
                wr.append((share(G) * share(U) if U is not None else share(G)).cpu())
            out = {}
            for side, scores, l0 in (("reads", reads, 0), ("q", rq, 0), ("k", rk, 0), ("writes", wr, first)):
                if not scores:
                    out[side] = []
                    continue
                zs, idx, N = top(scores)
                out[side] = [{"layer": l0 + i // N, "n": i % N, "z": round(z, 2)} for z, i in zip(zs, idx)]
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO headcon VALUES (?, ?)", (key, json.dumps(out)))
            self._db.commit()
        return out

    @_locked
    def _head_bases(self):
        """Orthonormal bases of every head's read space (value rows after the pre-attention norm weight) and write
        space (output columns, after the gated norm weight on linear layers and the post-attention norm weight where
        there is one), stacked: (QV [hidden, sum hd], QO [hidden, sum hd], [(layer, head, start, end)])."""
        if getattr(self, "_hb", None) is not None:
            return self._hb
        self.ensure_loaded()
        qv, qo, spans, at = [], [], [], 0
        with self.lock, torch.inference_mode():
            for l in range(self.arch.n_layers):
                p = self.arch.attn_parts(l)
                for h in range(p.n if p is not None else 0):
                    hs = self._head_slices(l, h)
                    qv.append(torch.linalg.qr(hs["v"].T.cpu())[0]); qo.append(torch.linalg.qr(hs["o"].cpu())[0])
                    spans.append((l, h, at, at + hs["v"].shape[0])); at += hs["v"].shape[0]
        self._hb = (torch.cat(qv, 1).to(self.dev), torch.cat(qo, 1).to(self.dev), spans) if spans else (None, None, [])
        return self._hb

    @_locked
    def neuron_heads(self, layer, n):
        """Which attention heads carry a neuron's output on, and which feed it, from weights. Later heads: the share
        of the neuron's write column inside each head's read space. Heads whose output its MLP reads (this layer or
        earlier; earlier only when attention and MLP run side by side): the share of its gate row inside the head's
        write space times the same for its up row (after the pre-MLP norm weight; the gate row alone for an ungated
        MLP). Each side is z-scored over its heads; returns {layer: [z per head]}."""
        QV, QO, spans = self._head_bases()
        if not spans:
            return {}
        a = self.arch
        last_writer = layer - 1 if a.parallel[layer] else layer
        with self.lock, torch.inference_mode():
            d = a.write_w(layer)[:, n].float()
            G, U = self._mlp_reads(layer)
            gr, ur = G[n], (U[n] if U is not None else None)
            pr = (QV.T @ d).cpu()  # projections of the write column on every head's read basis
            pg, pu = (QO.T @ gr).cpu(), ((QO.T @ ur).cpu() if ur is not None else None)
        dn, gn, hid = float(d.norm()), float(gr.norm()), d.shape[0]
        un = float(ur.norm()) if ur is not None else None
        reads, writes = [], []
        for l, h, s, e in spans:
            base = ((e - s) / hid) ** 0.5  # a random direction's expected share of a (e - s)-dim space
            if l > layer:
                reads.append((l, h, float(pr[s:e].norm()) / dn / base))
            elif l <= last_writer:
                w = float(pg[s:e].norm()) / gn / base
                writes.append((l, h, w * float(pu[s:e].norm()) / un / base if pu is not None else w))
        out = {}
        for side in (reads, writes):
            if len(side) < 2:
                continue
            v = np.array([x[2] for x in side])
            z = (v - v.mean()) / (v.std() + 1e-12)
            for (l, h, _), zz in zip(side, z):
                out.setdefault(l, {})[h] = round(float(zz), 2)
        return {l: [hs[h] for h in sorted(hs)] for l, hs in out.items()}

    @_locked
    def _qk_bases(self):
        """Orthonormal bases of every attention head's query and key read spaces (input rows, after the pre-attention
        norm weight), stacked as float16 [hidden, sum dims] with spans [(layer, heads, start, end)]; `heads` lists the
        output heads that share the space (grouped keys in softmax attention, shared q/k heads in linear attention)."""
        if getattr(self, "_qkb", None) is not None:
            return self._qkb
        self.ensure_loaded()
        out = {}
        a = self.arch
        with self.lock, torch.inference_mode():
            for role in ("q", "k"):
                cols, spans, at = [], [], 0
                for l in range(a.n_layers):
                    p = a.attn_parts(l)
                    if p is None:
                        continue
                    g = a.attn_pre_scale(l).float()
                    W = p.q if role == "q" else p.k
                    for u in range(W.shape[0]):
                        cols.append(torch.linalg.qr((W[u].float() * g).T.cpu())[0].half())
                        spans.append((l, p.heads_of(role, u), at, at + W.shape[1])); at += W.shape[1]
                out[role] = (torch.cat(cols, 1).to(self.dev) if cols else None, spans)
        self._qkb = out
        return out

    @_locked
    def neuron_head_roles(self, layer, n, k=8):
        """The attention heads a neuron meets, by role, from weights. q / k / v: later heads whose query, key or value
        read space holds a large share of the neuron's write column (it shapes what they look for from its token,
        makes its token findable, or is what they carry on). o: heads whose output its MLP reads (neuron_heads). Each
        role z-scored over its heads; the top k each, {layer, head, z}."""
        qk = self._qk_bases()
        zs = self.neuron_heads(layer, n)
        a = self.arch
        with self.lock, torch.inference_mode():
            d = a.write_w(layer)[:, n].float()
            dn = float(d.norm())
            proj = {r: (B.T @ d.half()).float().cpu() for r, (B, _) in qk.items() if B is not None}
        roles = {}
        for r, (_, spans) in qk.items():  # shares relative to a random direction's, as in neuron_heads
            side = [(l, hs, float(proj[r][s:e].norm()) / dn / ((e - s) / d.shape[0]) ** 0.5)
                    for l, hs, s, e in spans if l > layer] if r in proj else []
            if len(side) < 2:
                roles[r] = []
                continue
            v = np.array([x[2] for x in side]); z = (v - v.mean()) / (v.std() + 1e-12)
            rows = sorted(((zz, l, hs) for (l, hs, _), zz in zip(side, z)), key=lambda x: -x[0])[:k]
            roles[r] = [{"layer": l, "head": hs[0], "heads": hs, "z": round(float(zz), 1)} for zz, l, hs in rows]
        flat = [(z, int(l), h) for l, row in zs.items() for h, z in enumerate(row)]
        for r, keep in (("v", lambda l: l > layer), ("o", lambda l: l <= layer)):
            rows = []  # heads sharing one value space (grouped in softmax attention) score alike: one entry each
            for z, l, h in sorted((x for x in flat if keep(x[1])), reverse=True):
                if rows and rows[-1]["layer"] == l and rows[-1]["z"] == round(z, 1):
                    rows[-1]["heads"].append(h)
                elif len(rows) < k:
                    rows.append({"layer": l, "head": h, "heads": [h], "z": round(z, 1)})
                else:
                    break
            for x in rows:
                x["heads"].sort(); x["head"] = x["heads"][0]
            roles[r] = rows
        return roles

    def n_heads(self, layer):
        self.ensure_loaded()
        return self.arch.attn[layer][2]

    def _resid_hook(self, i):
        def hook(mod, inp, out):
            if self._seen.get("want_resid") == i:
                h = out[0] if isinstance(out, tuple) else out
                self._seen["resid"] = h[0, 1:].float().norm(dim=-1).cpu()
        return hook

    def stats_file(self, layer):
        """sweep/neuron_stats.py's file for this layer of MODEL, or None."""
        path = os.path.join(self.stats_dir, f"{A.short_name(MODEL)}_L{layer}.npz")
        if not os.path.exists(path):
            return None
        z = np.load(path)
        return z if "model" not in z.files or str(z["model"]) == MODEL else None

    def push_norm(self, layer):
        """Median norm of the stream an add push on this layer lands in (the residual leaving the layer, or the MLP
        output on models with a post-MLP norm), over the stats corpus when there is a stats file, else over a short
        reference text."""
        if layer in self.push_p50:
            return self.push_p50[layer]
        z = self.stats_file(layer)
        if z is not None:
            v = float((z["push_norm_q10_50_90"] if "push_norm_q10_50_90" in z.files else z["resid_norm_q10_50_90"])[1])
        else:
            ids = self.tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.dev)
            self._seen["want_resid"] = layer
            with torch.inference_mode():
                self.lm(input_ids=ids, use_cache=False)
            self._seen.pop("want_resid")
            v = float(self._seen.pop("resid").median())
        self.push_p50[layer] = v
        return v

    def down_col(self, layer, n):
        return self.arch.mlps[layer].down_w()[:, n].detach().float()

    # ---------- steering ----------
    def chat_ids(self, prompt):
        return self.messages_ids([{"role": "user", "content": prompt}])

    def messages_ids(self, messages):
        return A.chat_ids(self.tok, messages)

    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=200, cancel=None):
        """Greedy-decodes three conversations in one batch: row 0 pushed up, row 1 unsteered, row 2 pushed down.
        rows_messages holds each row's own message list (the rows can differ in length; they are left-padded).
        Yields (row, text_delta), then (None, None)."""
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel)

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None):
        """steer_stream for any number of neurons at once: pushes = [(layer, n, dose, mode)], all applied together.
        Each is reversed in row 2 the same way a single neuron is (add: -dose; mul: x -dose)."""
        self.ensure_loaded()
        with self.lock:
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            try:
                push = [self._push_entry(layer, n, dose, mode, (1, 0, -1)) for layer, n, dose, mode in pushes]
                seqs = [self.messages_ids(m) for m in rows_messages]
                T = max(map(len, seqs))
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=self.dev)
                mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=self.dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                done, out_ids, sent = [False] * 3, [[] for _ in range(3)], [""] * 3
                self._push = push
                with torch.inference_mode():
                    out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                                     **self._last)
                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            break
                        nxt = out.logits[:, -1].argmax(-1)
                        for r in range(3):
                            if done[r]:
                                continue
                            t = int(nxt[r])
                            if t in self.stop:
                                done[r] = True
                                continue
                            out_ids[r].append(t)
                            text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                yield r, text[len(sent[r]):]
                                sent[r] = text
                        if all(done):
                            break
                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                        pos = pos[:, -1:] + 1
                        out = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                         past_key_values=out.past_key_values, use_cache=True, **self._last)
            finally:
                self._push = None
        yield None, None

    def token_units(self, ids):
        """Groups token ids into displayable units: byte-level pieces of one multi-byte character (emoji, CJK split
        across tokens) join the token that completes it. Returns [(text, [token indices])]."""
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        # keep emoji sequences whole: joiners, variation selectors and skin tones attach to the unit before them
        glue = {"\u200d", "\ufe0f", "\ufe0e", "\u20e3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
        merged = []
        for text, idx in units:
            if merged and text and (text[0] in glue or merged[-1][0].endswith("\u200d")):
                merged[-1] = (merged[-1][0] + text, merged[-1][1] + idx)
            else:
                merged.append((text, idx))
        return merged

    # ---------- reading ----------
    def _push_entry(self, layer, n, dose, mode, rows):
        """One neuron's push for the down_proj hook, per batch row: rows holds +1 (pushed), 0 (left alone) or -1
        (pushed the other way). add doses are in units of the layer's median push-stream norm (push_norm) along the
        neuron's down column; mul doses multiply the activation (left alone = x1, the other way = x -dose)."""
        if n is None and mode != "mul":
            raise ValueError("a whole-layer MLP push must be mode 'mul' (e.g. dose 0 to ablate)")
        if mode == "mul":
            vals = [dose if r > 0 else 1.0 if r == 0 else -dose for r in rows]
        else:
            unit = self.push_norm(layer) / (float(self.down_col(layer, n).norm()) * self.arch.push_scale(layer))
            vals = [r * dose * unit for r in rows]
        return (layer, n, mode, torch.tensor(vals, device=self.dev))

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        """Per-token activation of neuron n at `layer` on `text` (as a user turn when chat=True), optionally with other
        neurons pushed at the same time: pushes = [(layer, n, dose, mode)], the same units as steering (n None = the
        layer's whole MLP, mul only), and attention heads scaled: heads = [(layer, head, scale)], head None = all heads,
        scale 0 = ablated. Only earlier layers can move the read (a same-layer MLP push lands after the capture; a
        same-layer head change lands before it)."""
        self.ensure_loaded()
        with self.lock:
            if chat:
                ids = self.chat_ids(text)
            else:
                ids = self.tok(text).input_ids
            self._read = (layer, n)
            self._push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
            self._heads = list(heads) or None if heads else None
            try:
                with torch.inference_mode():
                    self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
            finally:
                self._read = None
                self._push = None
                self._heads = None
            act = self._seen.pop("act").tolist()
        units = []
        for text, idx in self.token_units(ids):
            vals = [act[i] for i in idx]
            units.append({"text": text, "act": max(vals, key=abs), "pieces": vals})
        return {"units": units}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        """Log-probabilities of the next token after `text` (as a user turn when chat=True, so the next token opens the
        reply) followed by `prefix`, as a [vocab] tensor. With pushes / heads (same units as read()) it also returns
        the same distribution under that intervention: (base, intervened or None)."""
        self.ensure_loaded()
        ids = self.chat_ids(text) if chat else self.tok(text).input_ids
        if prefix:
            ids = ids + self.tok(prefix, add_special_tokens=False).input_ids

        def run(push, hd):
            with self.lock:
                self._push, self._heads = push, hd
                try:
                    with torch.inference_mode():
                        out = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, **self._last)
                finally:
                    self._push = self._heads = None
                return torch.log_softmax(out.logits[0, -1].float(), -1).cpu()

        base = run(None, None)
        if not pushes and not heads:
            return base, None
        push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
        return base, run(push, list(heads) if heads else None)

    def con_index(self):
        """Precomputed connections (build_connections.py), loaded once."""
        if self._con_idx is None:
            path = os.path.join(CACHE, "connections.npz")
            self._con_idx = False
            if os.path.exists(path):
                z = np.load(path)
                if fits(z["down_z"], self.shape, path):
                    self._con_idx = {key: z[key] for key in z.files}
                    self._con_idx["k"] = int(z["k"])
        return self._con_idx or None

    # ---------- logit lens ----------
    def lens_index(self):
        if self._lens_idx is None:
            path = os.path.join(CACHE, "lens_index.npz")
            if os.path.exists(path):
                z = np.load(path)
                if fits(z["up"], self.shape, path):
                    self._lens_idx = {"up": z["up"], "down": z["down"]}
        return self._lens_idx

    def steer_index(self):
        if getattr(self, "_steer_idx", None) is None:
            path = os.path.join(CACHE, "steer_index.npz")
            self._steer_idx = False
            if os.path.exists(path):
                z = dict(np.load(path))
                self._steer_idx = z if fits(z["up"], self.shape, path) else False
        return self._steer_idx or None

    @_locked
    def lens(self, layer, n, k=16):
        idx = self.lens_index()
        if idx is not None and k <= idx["up"].shape[-1]:
            v = self.vocab_strings()
            out = {"up": [v[t] for t in idx["up"][layer, n, :k]], "down": [v[t] for t in idx["down"][layer, n, :k]]}
            st = self.steer_index()
            if st is not None:
                out.update(steer_up=[v[t] for t in st["up"][layer, n, :k]], steer_down=[v[t] for t in st["down"][layer, n, :k]],
                           kl_up=round(float(st["up_kl"][layer, n]), 3), kl_down=round(float(st["down_kl"][layer, n]), 3))
            return out
        self.ensure_loaded()
        if (layer, n, k) in self._lens_cache:
            return self._lens_cache[(layer, n, k)]
        with self.lock, torch.inference_mode():
            d = self.arch.write_w(layer)[:, n][None, None]
            out = {}
            for sign, name in ((1, "up"), (-1, "down")):
                logits = self.arch.logits(sign * d)[0, 0].float()
                top = logits.topk(k).indices.tolist()
                out[name] = [self.tok.decode([t]) for t in top]
        self._lens_cache[(layer, n, k)] = out
        return out

    # ---------- connections ----------
    @_locked
    def connections(self, layer, n, k=24, lens_k=5):
        """Direct-path weights between neuron (layer, n) and neurons in other layers, ignoring everything between.
        Downstream B in layer j > layer: g = W_gate_j[B] . (s_j * d_n), u = W_up_j[B] . (...), where d_n is this
        neuron's write column (Arch.write_w) and s_j the per-channel scale of layer j's pre-MLP norm (1 + w_j on
        Qwen3.5). Upstream A in layer i < layer: the same with A's write column against this neuron's gate and up rows.
        Score = |g * u| z-scored within each layer; sign = sign(g * u). A plain (ungated) MLP has no up rows: score
        |g|, sign sign(g). Each returned neuron carries its own logit-lens tokens."""
        key = (layer, n, k, lens_k)
        if key in self._con_cache:
            return self._con_cache[key]
        ci = self.con_index()
        if ci is not None and k <= ci["k"]:
            out = {}
            for side in ("down", "up"):
                z = ci[f"{side}_z"][layer, n]
                out[side] = [{"layer": int(ci[f"{side}_layer"][layer, n, i]), "n": int(ci[f"{side}_n"][layer, n, i]),
                              "z": round(float(z[i]), 2), "sign": int(ci[f"{side}_sign"][layer, n, i])}
                             for i in range(k) if z[i] > -1e4]
            idx = self.steer_index() or self.lens_index()  # linked neurons show their steer lens when there is one
            if lens_k and idx is not None:
                v = self.vocab_strings()
                lg = self.lens_index()
                for c in out["down"] + out["up"]:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
                    if lg is not None and lg is not idx:  # the logit lens too, shown faintly under the steer lens
                        c["logit"] = [v[t] for t in lg["up"][c["layer"], c["n"], :lens_k]]
                        c["logit_neg"] = [v[t] for t in lg["down"][c["layer"], c["n"], :lens_k]]
            return out
        dkey = f"{layer}:{n}:{k}:{lens_k}"
        with self._db_lock:
            row = self._db.execute("SELECT val FROM con WHERE key = ?", (dkey,)).fetchone()
        if row:
            out = json.loads(row[0])
            self._con_cache[key] = out
            return out
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            a = self.arch
            out = {"down": [], "up": []}
            d = a.write_w(layer)[:, n]
            dt = d.dtype
            scale_me = a.pre_scale(layer).to(dt)
            G_me, U_me = a.mlps[layer].read_w()
            r_g = G_me[n] * scale_me
            r_u = U_me[n] * scale_me if U_me is not None else None
            for j in range(a.n_layers):
                if j == layer:
                    continue
                if j > layer:
                    v = d * a.pre_scale(j).to(dt)
                    G, U = a.mlps[j].read_w()
                    g, u = (G @ v).float(), (U @ v).float() if U is not None else None
                    side = "down"
                else:
                    W = a.write_w(j)
                    g, u = (r_g @ W).float(), (r_u @ W).float() if r_u is not None else None
                    side = "up"
                prod = g * u if u is not None else g
                mag = prod.abs()
                z = (mag - mag.mean()) / (mag.std() + 1e-8)
                top = z.topk(min(k, z.numel()))
                out[side] += [{"layer": j, "n": int(i), "z": round(float(sc), 2), "sign": 1 if prod[i] > 0 else -1}
                              for sc, i in zip(top.values.tolist(), top.indices.tolist())]
            for side in out:
                out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
            every = out["down"] + out["up"]
            idx = self.steer_index() or self.lens_index()
            if every and lens_k and idx is not None and lens_k <= idx["up"].shape[-1]:
                v = self.vocab_strings()
                for c in every:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
            elif every and lens_k:
                D = torch.stack([a.write_w(c["layer"])[:, c["n"]] for c in every])
                logits = a.logits(D[:, None])[:, 0].float()
                ids, ids_neg = logits.topk(lens_k).indices.tolist(), (-logits).topk(lens_k).indices.tolist()
                for c, row, row_neg in zip(every, ids, ids_neg):
                    c["lens"] = [self.tok.decode([t]) for t in row]
                    c["lens_neg"] = [self.tok.decode([t]) for t in row_neg]
        if len(self._con_cache) > 2000:
            self._con_cache.clear()
        self._con_cache[key] = out
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO con VALUES (?, ?)", (dkey, json.dumps(out)))
            self._db.commit()
        return out

    # ---------- logit-lens index over every neuron ----------
    def build_lens_index(self, path, k=12, chunk=1024):
        """Top-k tokens each neuron's down column pushes up and down, for every layer. Saved as int32 arrays
        up[L, N, k], down[L, N, k]."""
        self.ensure_loaded()
        L, N = self.arch.n_layers, self.arch.n_neurons
        up = np.zeros((L, N, k), np.int32)
        down = np.zeros((L, N, k), np.int32)
        for li in range(L):
            self.status = f"indexing lens {li}/{L}"
            with self.lock, torch.inference_mode():
                W = self.arch.write_w(li)  # [hidden, inter]
                for s in range(0, N, chunk):
                    D = W[:, s:s + chunk].T[:, None]  # [c, 1, hidden]
                    logits = self.arch.logits(D)[:, 0]
                    up[li, s:s + chunk] = logits.topk(k).indices.cpu().numpy()
                    down[li, s:s + chunk] = (-logits).topk(k).indices.cpu().numpy()
        np.savez(path, up=up, down=down)
        self.status = "ready"
        return path

    def vocab_strings(self):
        if getattr(self, "_vocab", None) is None:
            path = os.path.join(CACHE, "vocab.json")
            vocab = json.load(open(path)) if os.path.exists(path) else None
            if vocab is None or (self.tok is not None and len(vocab) != len(self.tok)):
                if self.tok is None:
                    self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=A._remote_code())
                vocab = [self.tok.decode([i]) for i in range(len(self.tok))]
                json.dump(vocab, open(path, "w"))
            self._vocab = vocab
        return self._vocab

    # ---------- which neurons fire on a text ----------
    def all_activations(self, ids, skip_first=False):
        """Per layer: (max, argmax, min, argmin) over tokens for every neuron, on token ids [T] or [B, T]. With
        skip_first, position 0 (the attention sink) is left out and argmax/argmin count from position 1."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_all, self._capture_from = {}, int(skip_first)
            try:
                x = torch.tensor(ids, device=self.dev)
                self.lm(input_ids=x if x.dim() == 2 else x[None], use_cache=False)
                return self._capture_all
            finally:
                self._capture_all, self._capture_from = None, 0

    def build_act_scale(self, path, tokens=10000, seq=512, seed=0):
        """Each neuron's largest positive and most negative activation over web text (positions >= 1)."""
        from datasets import load_dataset
        self.ensure_loaded()
        L, N = self.arch.n_layers, self.arch.n_neurons
        pos, neg = np.zeros((L, N), np.float32), np.zeros((L, N), np.float32)
        buf, seen = [], 0
        sep = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []
        for d in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed):
            buf += self.tok(d["text"], add_special_tokens=False).input_ids + sep
            while len(buf) >= seq and seen < tokens:
                chunk, buf = buf[:seq], buf[seq:]
                self.status = f"measuring neuron ranges {seen // 1000}k/{tokens // 1000}k tokens"
                acts = self.all_activations(chunk, skip_first=True)
                for l, (mx, _, mn, _) in acts.items():
                    pos[l] = np.maximum(pos[l], mx.numpy()); neg[l] = np.minimum(neg[l], mn.numpy())
                seen += seq
            if seen >= tokens:
                break
        np.savez(path, pos=pos, neg=neg, tokens=seen)
        self.status = "ready"
        return path

    def full_activations(self, ids, heads=False):
        """Every neuron's activation at every token: float32 array [layers, tokens, inter]. With heads, the same pass
        also keeps what head_row() needs and every head's write size per token (self.heads)."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_full = {}
            if heads:
                self._capture_heads = {"q": {}, "writes": {}, "y": {}, "s": {}, "x": {}, "rms": {}, "orms": {}}
            try:
                self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
                if heads:
                    self.heads = {"T": len(ids), "ids": list(ids), **self._capture_heads}
                return torch.stack([self._capture_full[l] for l in sorted(self._capture_full)]).numpy()
            finally:
                self._capture_full = None
                self._capture_heads = None
                A.CAPTURE.clear()
