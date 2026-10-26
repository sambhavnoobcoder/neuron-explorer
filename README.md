# neuron explorer

Steering-first neuron explorer for Qwen3.5-4B, and for other dense Hugging Face decoders (Qwen 2/2.5/3/3.5, Llama,
Mistral, Gemma 1/2/3, SmolLM, OLMo, Phi, GPT-2, Pythia, ...; see [Other models](#other-models)). Every MLP neuron in a
layer is pushed up and down on a few prompts, every rollout is kept, and the explorer lays the results out over a 3D
view of the model.

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Loads Qwen3.5-4B locally (MPS, CUDA or CPU) for live
steered chat, lens, connections and live inference over your own text. `EXPLORER_MODEL` picks another model.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

The scripts take `--model` (the stats file names the model for everything after `neuron_stats.py`), and
`run_layer.sh` / `run_depth.sh` read `MODEL`. Two backends:

- `--backend vllm` (default): the steering fork, fast. It pushes at the fork's `mlp_out` hook on the models where the
  fork wires it (Qwen3, Qwen3.5, Gemma 3/4) and at `post_block` on the others (Llama, Qwen2/2.5, Mistral, ...), where
  `mlp_out` is a silent no-op in the fork; the two are the same push (`sweep/vllm_hooks.py`). It captures `mlp_in`
  only where the fork wires it; elsewhere run `sweep/capture_prefill.py` on the finished run. Run
  `sweep/check_equivalence.py` on a new model before a full sweep.
- `--backend hf`: plain transformers on CUDA, MPS or CPU, pushing at the MLP output projection exactly as the explorer
  does. Slower, but exact on every model the explorer runs, and the only choice for models that normalise the MLP
  output (Gemma 2/3, OLMo 2), where no fork hook is a neuron push.

        python sweep/neuron_stats.py --model Qwen/Qwen2.5-0.5B-Instruct --layers 12 --tokens 2000000
        python sweep/steer_sweep.py --stats stats/Qwen--Qwen2.5-0.5B-Instruct_L12.npz --backend hf --design main_rel \
            --dose-unit rel --neurons random:384 --random-controls 64 --run Qwen--Qwen2.5-0.5B-Instruct-L12-r384

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron; `--index` joins them into the explorer's `steer_index.npz` |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |
| `explorer/check_model.py` | checks a model against everything the explorer assumes about it |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.

## Other models

    uv run python explorer/check_model.py --model Qwen/Qwen2.5-0.5B-Instruct    # once per new model
    EXPLORER_MODEL=Qwen/Qwen2.5-0.5B-Instruct uv run python explorer/server.py

Nothing about the model is written down per architecture: `explorer/arch.py` finds the decoder layers, the MLP (gated
`gate_proj`/`up_proj`/`down_proj`, a fused `gate_up_proj`, or a plain `fc_in`/`fc_out` such as GPT-2's `c_fc`/`c_proj`),
the norms in front of and behind it, the attention heads, the final norm and the unembedding on the loaded model, with
one traced forward pass, and reads each norm's per-channel scale off the module. `check_model.py` then confirms, on the
first, a middle and the last layer, that the neuron activations recompute from the MLP input, that a push on a neuron
equals adding its column to the MLP output and moves the logits, that the norm scales reproduce the norms, that the
logit lens reproduces the model's own logits, that head ablation is live, that the chat template's end-of-turn token
stops generation, and that the attention-head tools (patterns, drive, links) agree with the model.

All checks pass on Qwen3.5-0.8B, Qwen3-0.6B, Qwen2.5-0.5B-Instruct, SmolLM2-135M-Instruct (Llama), gemma-3-270m-it,
gemma-3-1b-it, OLMo-2-0425-1B-Instruct, GPT-2 and Pythia-70m. Mixture-of-experts MLPs and layers of different MLP widths
are refused with an error. Models that need custom code load with `EXPLORER_TRUST_REMOTE_CODE=1`; `EXPLORER_DEVICE`,
`EXPLORER_DTYPE` (default `bfloat16`) and `EXPLORER_ATTN` (transformers' attention implementation, e.g. `eager`) override
the device, precision and attention.

- **Caches**: `explorer/cache/` holds Qwen3.5-4B's; any other model gets `explorer/cache/<org>--<name>/`. At startup
  the server builds the logit-lens index and the activation scale (10k tokens of FineWeb-Edu) there, and lists the
  optional caches still missing. Without them the explorer works with less (no web or links, no `writes` layout, logit
  lens in place of the steer lens):

        uv run python explorer/build_connections.py --model <id>     # links, the map's web, the cone
        uv run python explorer/build_layout.py --model <id>          # the `writes` sheet layout
        uv run python sweep/steer_lens.py --model <id> --index       # the steer lens

  A cache of the wrong shape (another model's) is ignored with a message, never read.
- **Runs and names**: the explorer shows only runs whose `meta.json` names its model; stats files are
  `stats/<org>--<name>_L{l}.npz` (`q35-4b` for Qwen3.5-4B). Names are kept per model, in `explorer/names.json` for
  Qwen3.5-4B and `explorer/names.<org>--<name>.json` for others.
- **Chat**: the tokenizer's chat template, with thinking off where it has the switch; a model without one (GPT-2,
  Pythia, many base models) gets a plain `User: / Assistant:` transcript. Replies stop on generation_config's eos ids,
  the tokenizer's eos and the template's end-of-turn token.
- **Residual scales**: a block whose output enters the residual stream multiplied by a constant (Granite's
  `residual_multiplier`, MiniCPM's `scale_depth`) has that constant measured in the traced pass and folded into push
  sizes, write directions and attention drive; `check_model.py` checks that a push moves the residual by exactly that.
- **Post-MLP norms** (Gemma 2/3, OLMo 2): these models normalise the MLP output before adding it to the residual
  stream. A push on a neuron passes through that norm, so it saturates: at most it turns the layer's MLP write into
  the neuron's direction. Add doses on these models are fractions of the MLP output's median norm instead of the
  residual's: on gemma-3-270m layer 9 the residual's median norm is about 3,400 times the MLP output's, so in residual
  units every dose from 0.05 to 3 gives the same saturated text, while in MLP-output units the same doses run from
  subtle to broken. A neuron's write direction (logit lens, connections) includes the norm's per-channel scale.
- **Attention heads**: patterns, attention drive and q / k / v / o roles work on softmax attention (full or
  sliding-window) of any layout the adapter can cut into heads (separate, gated or fused q/k/v projections, GPT-2's
  `c_attn`, GPT-NeoX's `query_key_value`) and on Qwen3.5's Gated DeltaNet layers. Queries, keys and values come from
  the model's own attention call (`arch.install_capture`), so rotary embeddings, query/key norms, softcaps and sliding
  windows are the model's, as is the attention implementation's handling of them (sdpa, transformers' default for
  Gemma 2, leaves out its attention-logit softcap; `EXPLORER_ATTN=eager` keeps it); `check_model.py` compares the
  patterns with the attention weights the model returns itself, and checks that attention drive splits add up. A layer whose attention can't be cut into heads has the head
  tools off, and says so.
- **Claude**: the agent's prompt gets the model's name, shape, head layout and dose unit filled in; the method's
  example neurons stay Qwen3.5-4B's and are labelled as such.
