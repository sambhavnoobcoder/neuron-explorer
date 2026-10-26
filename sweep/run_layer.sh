#!/usr/bin/env bash
# Stats pass then a full-layer sweep for one layer. Logs to stats_L<layer>.log and sweep_<run>.log; detach with
#     setsid nohup sweep/run_layer.sh 8 q35-4b-L8 </dev/null >/dev/null 2>&1 &
# Extra arguments pass through to steer_sweep.py. MODEL picks the model (default Qwen/Qwen3.5-4B); add --backend hf
# for models that normalise the MLP output (Gemma 2/3, OLMo 2).
set -euo pipefail
LAYER="$1"; RUN="$2"; shift 2
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
cd /workspace/neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p stats runs
P=$(python -c "import sys; sys.path.insert(0, 'explorer'); import arch; print(arch.short_name('$MODEL'))")
[ -f "stats/${P}_L${LAYER}.npz" ] || python sweep/neuron_stats.py --model "$MODEL" --layers "$LAYER" --tokens 2000000 \
    --out "stats/$P" > "stats_L${LAYER}.log" 2>&1
python sweep/steer_sweep.py --stats "stats/${P}_L${LAYER}.npz" --design main_rel --dose-unit rel --neurons all \
    --random-controls 256 --max-num-seqs 320 --run "$RUN" "$@" > "sweep_${RUN}.log" 2>&1
echo SWEEP_DONE >> "sweep_${RUN}.log"
