"""Depth curve from per-layer random-subset sweeps: how many neurons steer the model's text further from baseline
than random directions do, per layer, plus coherence numbers per dose and each layer's strongest neurons.

    uv run python sweep/depth_report.py --out runs/depth_report.json
    uv run python sweep/depth_report.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/qwen25-depth.json

Ranks runs/<prefix>-L{l}-r384.jsonl (prefix q35-4b for Qwen3.5-4B, <org>--<name> for other models, as run_depth.sh
names them), writing <run>.units.jsonl. Layers with only a full run (-full.jsonl) are subsampled to the same pool size
(384 neurons, 64 controls) so word-cluster mass is comparable across layers.
"""
import argparse
import json
import os
import random

import sys

import numpy as np

from rank import compute_units

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
from arch import DEFAULT_MODEL, short_name  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL, help="names the runs: runs/<short name>-L{l}-r{neurons}.jsonl")
ap.add_argument("--layers", default="0-127", help="layers without a run are skipped")
ap.add_argument("--neurons", type=int, default=384)
ap.add_argument("--controls", type=int, default=64)
ap.add_argument("--out", default="runs/depth_report.json")
a = ap.parse_args()
lo, _, hi = a.layers.partition("-")

report = []
for l in range(int(lo), int(hi or lo) + 1):
    prefix = short_name(a.model)
    path = f"runs/{prefix}-L{l}-r{a.neurons}.jsonl"
    sub = False
    if not os.path.exists(path):
        path, sub = f"runs/{prefix}-L{l}-full.jsonl", True
        if not os.path.exists(path):
            continue
    rows = [json.loads(x) for x in open(path)]
    if sub:
        rng = random.Random(l)
        nn = sorted({r["neuron"] for r in rows if r["kind"] == "neuron"})
        cc = sorted({r["neuron"] for r in rows if r["kind"] == "random"})
        keep = {("neuron", n) for n in rng.sample(nn, a.neurons)} | {("random", n) for n in rng.sample(cc, a.controls)}
        rows = [r for r in rows if r["kind"] == "baseline" or (r["kind"], r["neuron"]) in keep]
    recs = compute_units(rows)
    if not sub:
        with open(path.replace(".jsonl", ".units.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    null = np.array([r["mass"] for r in recs if r["kind"] == "random"])
    neu = [r for r in recs if r["kind"] == "neuron"]
    m = np.array([r["mass"] for r in neu])
    p95 = np.percentile(null, 95)

    def coh(kind, dose):
        v = [r["dose"].get(dose) for r in recs if r["kind"] == kind]
        v = [x for x in v if x]
        return {k: round(float(np.mean([x[k] for x in v])), 3) for k in ("distinct4", "capped", "logprob")}

    report.append({
        "layer": l, "subsampled": sub, "n_neurons": len(neu), "n_controls": len(null),
        "null_p50": round(float(np.median(null)), 2), "null_p95": round(float(p95), 2), "null_max": round(float(null.max()), 2),
        "neuron_p50": round(float(np.median(m)), 2), "neuron_p90": round(float(np.percentile(m, 90)), 2),
        "above_p95": round(float((m > p95).mean()), 3), "above_max": round(float((m > null.max()).mean()), 3),
        "coh_neuron_+0.8": coh("neuron", "+0.8"), "coh_random_+0.8": coh("random", "+0.8"),
        "coh_neuron_-0.7": coh("neuron", "-0.7"),
        "top": [{"n": r["neuron"], "mass": r["mass"], "words": [w for w, _, _ in r["words"][:6]]} for r in neu[:5]],
    })
    x = report[-1]
    print(f"L{l:<2}{'*' if sub else ' '} null p50/p95 {x['null_p50']:5.1f}/{x['null_p95']:5.1f}  neuron p50/p90 "
          f"{x['neuron_p50']:5.1f}/{x['neuron_p90']:5.1f}  >p95 {x['above_p95']:.0%}  >max {x['above_max']:.0%}  "
          f"d4(+.8) n {x['coh_neuron_+0.8']['distinct4']:.2f} r {x['coh_random_+0.8']['distinct4']:.2f}  | "
          + " ; ".join(f"{t['n']}:{' '.join(t['words'][:4])}" for t in x["top"][:3]), flush=True)
json.dump(report, open(a.out, "w"), indent=1)
