"""Checks a model against everything the explorer and the sweep assume about it (explorer/arch.py), on a few layers,
and prints what it found. Run it once before exploring or sweeping a new model; it exits non-zero on a failure.

    uv run python scripts/check_model.py --model Qwen/Qwen2.5-0.5B-Instruct
    uv run python scripts/check_model.py --model google/gemma-3-1b-it --device cpu --dtype float32

Checks, each on the first, a middle and the last layer:
    acts      neuron activations recomputed from the MLP input (act(gate) * up from the gate and up weights) match the
              values at the output projection's input, where the explorer reads and pushes
    push      adding d to neuron n at that input gives the same logits as adding d * down[:, n] at the projection's
              output (the sweep's push), and moves them (the hook is live)
    norm      the pre-MLP norm's per-channel scale, read off the module, reproduces its output on real hidden states
              (exact for RMSNorm; LayerNorm also centres, which the weight-only links ignore)
    lens      the logit lens (final norm, then unembedding) on the last layer's output reproduces the model's logits
    heads     scaling the attention output projection's input to zero changes the logits (head ablation is live)
    stop      the chat template's end-of-turn token (the one after an assistant message) is among the stop tokens
    chat      a greedy reply to a chat prompt, for a look (base models ramble; that is not a failure)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import arch as A  # noqa: E402

TEXT = "The quick brown fox jumps over the lazy dog. In 1969, astronauts first walked on the Moon, and"
failed = []


def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:6s} {detail}", flush=True)
    if not ok:
        failed.append(name)


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=A.DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", help="float32 makes the equalities sharp; bfloat16 is what runs")
    ap.add_argument("--tol", type=float, default=None, help="relative-error bound (default 1e-3 float32, 3e-2 else)")
    a = ap.parse_args()
    dev = a.device or A.default_device()
    dtype = getattr(torch, a.dtype)
    tol = a.tol or (1e-3 if dtype == torch.float32 else 3e-2)

    model, tok = A.load(a.model, dev, dtype)
    ar = A.Arch(model, tok, name=a.model)
    d = ar.describe()
    print(f"{a.model}: {d['n_layers']} layers, {d['inter']} {d['mlp']} neurons per layer ({ar.mlps[0].kind}: "
          f"{type(ar.mlps[0].down).__name__} {[n for n, m in ar.mlps[0].module.named_children() if m is ar.mlps[0].down][0]}), "
          f"hidden {d['hidden']}, vocab {d['vocab']} (unembedding rows {A._weight(ar.head).shape[0]})")
    print(f"  text model {type(ar.lm).__name__}; pre-MLP norm {type(ar.pre_norms[0]).__name__ if ar.pre_norms[0] is not None else 'none'}; "
          f"post-MLP norm {type(ar.post_norms[0]).__name__ if ar.post_norms[0] is not None else 'none'}; "
          f"final norm {type(ar.final_norm).__name__ if ar.final_norm is not None else 'none'}")
    kinds = {}
    for t, h in zip(d["layer_types"], d["heads"]):
        kinds[(t, h)] = kinds.get((t, h), 0) + 1
    print("  token mixers: " + ", ".join(f"{c} x {t} ({h} heads)" for (t, h), c in kinds.items()))
    print(f"  chat template: {'yes' if getattr(tok, 'chat_template', None) else 'no (plain User:/Assistant: transcript)'}; "
          f"stop tokens: {sorted(tok.decode([i]) for i in A.stop_ids(model, tok))}")
    if len({id(n) for n in ar.pre_norms}) != ar.n_layers and ar.pre_norms[0] is not None:
        check("trace", False, "layers share a pre-MLP norm module")
    if any((n is None) != (ar.pre_norms[0] is None) for n in ar.pre_norms):
        check("trace", False, "some layers have a pre-MLP norm and some don't")

    ids = torch.tensor([tok(TEXT).input_ids], device=dev)
    layers = sorted({0, ar.n_layers // 2, ar.n_layers - 1})
    rec, hooks = {}, []
    for l in layers:
        hooks.append(ar.mlps[l].module.register_forward_pre_hook(
            lambda m, args, kw, l=l: rec.__setitem__(("x", l), A._first_tensor(args, kw)), with_kwargs=True))
        hooks.append(ar.mlps[l].down.register_forward_pre_hook(lambda m, inp, l=l: rec.__setitem__(("a", l), inp[0])))
        if ar.pre_norms[l] is not None:
            hooks.append(ar.pre_norms[l].register_forward_hook(
                lambda m, args, kw, out, l=l: rec.__setitem__(("n", l), (A._first_tensor(args, kw), out)), with_kwargs=True))
    if ar.final_norm is not None:
        hooks.append(ar.final_norm.register_forward_pre_hook(
            lambda m, args, kw: rec.__setitem__("final_in", A._first_tensor(args, kw)), with_kwargs=True))
    with torch.inference_mode():
        base = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    for h in hooks:
        h.remove()

    for l in layers:
        print(f" L{l}")
        mlp = ar.mlps[l]
        with torch.inference_mode():
            check("acts", (e := rel(mlp.acts(rec[("x", l)]), rec[("a", l)])) < tol,
                  f"recomputed vs read activations, relative error {e:.2e}")
        # push: one neuron at the projection's input vs its column at the projection's output
        a_l = rec[("a", l)][0, 1:]
        n = int(a_l.abs().float().mean(0).argmax())
        delta = 4.0 * float(a_l[:, n].abs().max()) + 1.0
        col = mlp.down_w()[:, n]

        def run(hook, pre):
            h = (mlp.down.register_forward_pre_hook(hook) if pre else mlp.down.register_forward_hook(hook))
            try:
                with torch.inference_mode():
                    return model(input_ids=ids, use_cache=False).logits[0, -1].float()
            finally:
                h.remove()

        def at_input(m, inp):
            x = inp[0].clone()
            x[..., n] += delta
            return (x,)

        at_in = run(at_input, True)
        at_out = run(lambda m, inp, out: out + delta * col.to(out.dtype), False)
        moved, agree = rel(at_in, base), rel(at_in, at_out)
        check("push", agree < max(tol, 0.05 * moved) and moved > 1e-3,
              f"neuron {n} +{delta:.3g}: input-side vs output-side push differ by {agree:.2e}; the push moved the logits by {moved:.2e}")
        if ar.pre_norms[l] is not None:
            h_in, h_out = (t.detach() for t in rec[("n", l)])
            s = ar.pre_scale(l)
            hf = h_in.float()
            centred = "layernorm" in type(ar.pre_norms[l]).__name__.lower()
            if centred:
                hf = hf - hf.mean(-1, keepdim=True)
            norm = ar.pre_norms[l]
            eps = next((float(v) for k in ("variance_epsilon", "eps", "epsilon") if isinstance(v := getattr(norm, k, None), (int, float))), 0.0)
            pred = hf / (hf.pow(2).mean(-1, keepdim=True) + eps).sqrt() * s
            if centred and getattr(ar.pre_norms[l], "bias", None) is not None:
                pred = pred + ar.pre_norms[l].bias.detach().float()
            e = rel(pred, h_out)
            check("norm", e < max(tol, 1e-2), f"{type(ar.pre_norms[l]).__name__} = x / sqrt(mean(x^2) + eps) * scale"
                  f"{' (after centring, plus bias)' if centred else ''}, relative error {e:.2e}; scale mean {float(s.mean()):.3f}")
        else:
            print("  info   norm   the MLP reads the residual stream unnormalised")
        if ar.post_norms[l] is not None:
            print(f"  info   post   {type(ar.post_norms[l]).__name__} normalises the MLP output: pushes saturate, and "
                  f"add doses are fractions of the MLP output's median norm")
        _, out_proj, n_heads = ar.attn[l]
        if n_heads:
            def ablate(m, inp):
                return (inp[0] * 0,)
            hook = out_proj.register_forward_pre_hook(ablate)
            try:
                with torch.inference_mode():
                    ab = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            finally:
                hook.remove()
            check("heads", rel(ab, base) > 1e-4, f"{n_heads} heads on {type(out_proj).__name__}; zeroing all moves the "
                  f"logits by {rel(ab, base):.2e}")
        else:
            print(f"  info   heads  no head layout found ({ar.attn[l][0]}); head scaling is off for this layer")

    print(" model")
    if "final_in" in rec:
        with torch.inference_mode():
            lens = ar.logits(rec["final_in"])[0, -1].float()
        cap = getattr(ar.config, "final_logit_softcapping", None)
        if cap:  # Gemma 2 squashes its logits; monotonic, so the lens's rankings are unaffected
            lens = torch.tanh(lens / cap) * cap
        e = rel(lens, base[: ar.n_vocab])
        check("lens", e < max(tol, 1e-2), f"final norm + unembedding vs the model's own logits, relative error {e:.2e}"
              + (f" (after the model's logit softcap of {cap})" if cap else ""))
    else:
        print("  info   lens   no final norm found by tracing; the lens uses the unembedding directly")
    with torch.inference_mode():
        x = torch.tensor([A.chat_ids(tok, [{"role": "user", "content": "Name three colours."}])], device=dev)
        stop, out, past, done = A.stop_ids(model, tok), [], None, False
        for _ in range(60):
            r = model(input_ids=x if past is None else x[:, -1:], past_key_values=past, use_cache=True)
            past, t = r.past_key_values, int(r.logits[0, -1].argmax())
            if t in stop:
                done = True
                break
            out.append(t)
            x = torch.cat([x, torch.tensor([[t]], device=dev)], 1)
    reply = tok.decode(out, skip_special_tokens=True).strip().replace("\n", " ")
    if getattr(tok, "chat_template", None):
        mark = "Zebra crossing."
        text = tok.apply_chat_template([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": mark}],
                                       tokenize=False)
        after = tok(text[text.rindex(mark) + len(mark):], add_special_tokens=False).input_ids
        check("stop", bool(after) and after[0] in stop, f"the template ends a reply with {tok.decode(after[:1])!r}"
              + ("" if after and after[0] in stop else f", not one of the stop tokens"))
    else:
        print("  info   stop   no chat template: replies end on eos or at the length cap")
    print(f"  info   chat   {'stopped' if done else 'no stop within 60 tokens'}: {reply[:120]!r}")

    print(f"\n{'all checks passed' if not failed else 'FAILED: ' + ', '.join(sorted(set(failed)))}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
