#!/usr/bin/env bash
# Depth survey for a list of layers: one short stats pass covering them all, then a random-subset sweep per layer.
#     setsid nohup sweep/run_depth.sh "0 4 9 13" 384 64 </dev/null >/dev/null 2>&1 &
# MODEL picks the model (default Qwen/Qwen3.5-4B); BACKEND=hf sweeps with transformers instead of the vLLM fork
# (required for models that normalise the MLP output: Gemma 2/3, OLMo 2).
set -uo pipefail
LAYERS="$1"; NNEUR="${2:-384}"; NCTRL="${3:-64}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"; BACKEND="${BACKEND:-vllm}"
cd /workspace/neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p stats runs
P=$(python -c "import sys; sys.path.insert(0, 'explorer'); import arch; print(arch.short_name('$MODEL'))")
need=""; for l in $LAYERS; do [ -f "stats/${P}_L$l.npz" ] || need="$need,$l"; done
if [ -n "$need" ]; then
  python sweep/neuron_stats.py --model "$MODEL" --layers "${need#,}" --tokens 200000 --out "stats/$P" > depth_stats.log 2>&1
fi
for l in $LAYERS; do
  echo "L$l start $(date)" >> depth.log
  python sweep/steer_sweep.py --stats "stats/${P}_L$l.npz" --backend "$BACKEND" --design main_rel --dose-unit rel \
    --neurons "random:$NNEUR" --random-controls "$NCTRL" --max-num-seqs 320 --run "$P-L$l-r$NNEUR" \
    > "sweep_L$l-r$NNEUR.log" 2>&1 && echo "L$l done $(date)" >> depth.log || echo "L$l FAILED $(date)" >> depth.log
done
echo ALL_DONE >> depth.log
