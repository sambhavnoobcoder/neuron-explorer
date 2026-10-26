"""Model anatomy for the explorer and the sweep: where a Hugging Face causal LM keeps the parts they read and push.
Found on the model itself (module names, shapes, and one short traced forward pass), not written down per architecture.

A neuron is one input channel of an MLP's output projection (down_proj, c_proj, fc2, dense_4h_to_h, ...). Its activation
is the value there: act(gate) * up for a gated MLP (SwiGLU, GeGLU), act(fc_in) for a plain one. Its write direction is
that column of the output projection, times the post-MLP norm's per-channel scale on models that normalise the MLP
output before adding it to the residual stream (Gemma 2/3, OLMo 2).

Covered: dense decoder-only models with a gated MLP (Llama, Mistral, Qwen 2/2.5/3/3.5, Gemma 1/2/3, SmolLM, OLMo,
Granite, Phi-3/4 with a fused gate_up_proj, ...) or a plain one (GPT-2, GPT-NeoX/Pythia, Phi-1/2, ...). Mixture-of-
experts MLPs are refused. `python explorer/check_model.py --model <id>` checks a model against every assumption here.
"""
import os

import torch
import torch.nn.functional as F

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

GATED = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))  # (gate, up, down)
FUSED = (("gate_up_proj", "down_proj"),)  # gate is the first half of the fused output (Phi-3, GLM)
PLAIN = (("fc1", "fc2"), ("c_fc", "c_proj"), ("dense_h_to_4h", "dense_4h_to_h"), ("fc_in", "fc_out"))
MLP_NAMES = ("mlp", "feed_forward", "ffn")
ACT_NAMES = ("act_fn", "activation_fn", "act", "activation", "gelu_impl")
ATTN_NAMES = ("self_attn", "linear_attn", "attn", "attention", "self_attention")
ATTN_OUT_NAMES = ("o_proj", "out_proj", "c_proj", "dense", "wo")
# reply-ending tokens of common chat templates, used as stops when the tokenizer has them (generation_config's eos
# list covers most models already)
END_OF_TURN = ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|end|>", "<|endofturn|>")


def slug(model):
    """A model id or path as one folder name: "Qwen/Qwen2.5-0.5B" -> "Qwen--Qwen2.5-0.5B"."""
    return model.strip().rstrip("/\\").lstrip("./\\").replace("\\", "/").replace("/", "--") or "model"


def short_name(model):
    """File-name prefix for a model's stats and runs. Qwen3.5-4B keeps its original "q35-4b"."""
    return "q35-4b" if model == DEFAULT_MODEL else slug(model)


def cache_dir(model):
    """The caches shipped in explorer/cache/ are Qwen3.5-4B's; every other model gets its own folder under it."""
    return CACHE_ROOT if model == DEFAULT_MODEL else os.path.join(CACHE_ROOT, slug(model))


def _remote_code():
    return bool(os.environ.get("EXPLORER_TRUST_REMOTE_CODE"))


def model_class(model_name):
    """The class the checkpoint was saved from (config.json "architectures", e.g. Qwen3_5ForConditionalGeneration,
    Qwen2ForCausalLM), read from the raw config so that a library that re-registers config classes (vllm does) can't
    redirect it. Falls back to the Auto class: image-text for vision-language configs, causal LM otherwise."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, PretrainedConfig
    raw, _ = PretrainedConfig.get_config_dict(model_name)
    for name in raw.get("architectures") or []:
        if hasattr(transformers, name):
            return getattr(transformers, name)
    return AutoModelForImageTextToText if "vision_config" in raw else AutoModelForCausalLM


def raw_config(model_name):
    """config.json as a dict (model_type, architectures, ...), without building a config class."""
    from transformers import PretrainedConfig
    return PretrainedConfig.get_config_dict(model_name)[0]


def load(model_name, device, dtype=torch.bfloat16):
    """(model, tokenizer), the model in eval mode on `device`."""
    from transformers import AutoTokenizer
    rc = _remote_code()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=rc)
    model = model_class(model_name).from_pretrained(model_name, dtype=dtype, trust_remote_code=rc).to(device).eval()
    return model, tok


def default_device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def dtype_from_env(default=torch.bfloat16):
    name = os.environ.get("EXPLORER_DTYPE")
    return getattr(torch, name) if name else default


def describe(model_name, tok=None):
    """The architecture summary (Arch.describe) without loading weights: the model is built on the meta device."""
    from transformers import AutoConfig
    rc = _remote_code()
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=rc)
    cls = model_class(model_name)
    with torch.device("meta"):
        model = cls.from_config(cfg, trust_remote_code=rc) if cls.__name__.startswith("Auto") else cls._from_config(cfg)
    return Arch(model, tok, name=model_name, trace=False).describe()


def _weight(mod):
    """A projection's weight as [out, in], whatever the layer type (GPT-2's Conv1D stores [in, out])."""
    return mod.weight.T if type(mod).__name__ == "Conv1D" else mod.weight


def norm_eps(norm):
    """The epsilon a norm adds under its square root (RMSNorm's variance_epsilon or eps, LayerNorm's eps)."""
    return next((float(v) for k in ("variance_epsilon", "eps", "epsilon") if isinstance(v := getattr(norm, k, None), (int, float))), 1e-6)


def _first_tensor(args, kwargs):
    for x in list(args) + list(kwargs.values()):
        if torch.is_tensor(x):
            return x
    return None


def _same(a, b):
    if a is None or b is None:
        return False
    if a is b:
        return True
    try:
        return a.shape == b.shape and torch.equal(a, b)
    except RuntimeError:  # meta tensors have no values
        return False


class Mlp:
    """One layer's MLP: `down` is the output projection whose input channels are the neurons; the read side is gate and
    up (gated), one fused gate_up projection (fused), or a single input projection (plain)."""

    def __init__(self, module, kind, down, gate=None, up=None, act=None):
        self.module, self.kind, self.down, self.gate, self.up, self.act = module, kind, down, gate, up, act

    @property
    def width(self):
        return _weight(self.down).shape[1]

    def down_w(self):
        """[hidden, N]: column n is what neuron n's activation multiplies on its way out of the MLP."""
        return _weight(self.down)

    def read_w(self):
        """(gate rows [N, hidden], up rows [N, hidden]); a plain MLP's single input projection comes back as the gate,
        with up None."""
        if self.kind == "fused":
            w = _weight(self.gate)
            return w[: w.shape[0] // 2], w[w.shape[0] // 2:]
        return _weight(self.gate), (_weight(self.up) if self.up is not None else None)

    def read_b(self):
        b = getattr(self.gate, "bias", None)
        if self.kind == "fused":
            return (b[: b.shape[0] // 2], b[b.shape[0] // 2:]) if b is not None else (None, None)
        return b, (getattr(self.up, "bias", None) if self.up is not None else None)

    def preacts(self, x):
        """(gate, up) pre-activations for MLP input x; up is None for a plain MLP."""
        (g, u), (bg, bu) = self.read_w(), self.read_b()
        return F.linear(x, g, bg), (F.linear(x, u, bu) if u is not None else None)

    def acts(self, x):
        """Neuron activations for MLP input x, as the model computes them."""
        if self.act is None:
            raise RuntimeError(f"no activation function found on {type(self.module).__name__}")
        g, u = self.preacts(x)
        return self.act(g) * u if u is not None else self.act(g)


def find_mlp(layer):
    """The layer's MLP as an Mlp, or None when it has none this code recognises. Raises on a mixture of experts."""
    for name in MLP_NAMES:
        m = getattr(layer, name, None)
        if isinstance(m, torch.nn.Module):
            break
    else:
        return None
    if hasattr(m, "experts"):
        raise NotImplementedError(f"{type(m).__name__} is a mixture of experts; only dense MLPs are supported")
    act = next((a for a in (getattr(m, n, None) for n in ACT_NAMES) if callable(a)), None)
    for g, u, d in GATED:
        if all(isinstance(getattr(m, x, None), torch.nn.Module) for x in (g, u, d)):
            return Mlp(m, "gated", getattr(m, d), getattr(m, g), getattr(m, u), act)
    for gu, d in FUSED:
        if all(isinstance(getattr(m, x, None), torch.nn.Module) for x in (gu, d)):
            return Mlp(m, "fused", getattr(m, d), getattr(m, gu), None, act)
    for i, o in PLAIN:
        if all(isinstance(getattr(m, x, None), torch.nn.Module) for x in (i, o)):
            return Mlp(m, "plain", getattr(m, o), getattr(m, i), None, act)
    return None


def find_layers(model, n_layers):
    """(text model, its decoder-layer ModuleList): the list of n_layers blocks that each have a recognised MLP. Vision
    and audio towers are skipped; between text candidates, one under `language_model` wins."""
    found = []
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            if not (isinstance(child, torch.nn.ModuleList) and len(child) == n_layers):
                continue
            full = f"{name}.{cname}".lower()
            if any(w in full for w in ("vision", "visual", "audio", "image", "encoder")):
                continue
            try:
                ok = all(find_mlp(x) is not None for x in child)
            except NotImplementedError:
                raise
            if ok:
                found.append((full, mod, child))
    if not found:
        raise NotImplementedError(f"no list of {n_layers} decoder layers with a recognised MLP in {type(model).__name__}")
    if len(found) > 1:
        lm = [f for f in found if "language_model" in f[0] or "text_model" in f[0]]
        if len(lm) != 1:
            raise NotImplementedError(f"several candidate layer lists: {[f[0] for f in found]}")
        found = lm
    return found[0][1], found[0][2]


def find_attention(layer, cfg):
    """(module name, output projection, head count) for the layer's token mixer, or (name, None, 0) when the heads
    can't be read off it. The output projection's input is the heads side by side."""
    for name in ATTN_NAMES:
        a = getattr(layer, name, None)
        if not isinstance(a, torch.nn.Module):
            continue
        out = next((o for o in (getattr(a, n, None) for n in ATTN_OUT_NAMES) if isinstance(o, torch.nn.Module)
                    and hasattr(o, "weight")), None)
        if out is None:
            return name, None, 0
        width = _weight(out).shape[1]
        n = getattr(a, "num_v_heads", None)  # linear attention (Gated DeltaNet): value heads
        if not isinstance(n, int):
            hd = next((h for h in (getattr(a, k, None) for k in ("head_dim", "head_size")) if isinstance(h, int)), None)
            n = width // hd if hd else None
        if not isinstance(n, int):
            n = next((h for h in (getattr(a, k, None) for k in ("num_heads", "num_attention_heads", "n_head"))
                      if isinstance(h, int)), getattr(cfg, "num_attention_heads", None))
        if not isinstance(n, int) or n <= 0 or width % n:
            return name, out, 0
        return name, out, n
    return None, None, 0


def text_config(cfg):
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def n_layers_of(cfg):
    c = text_config(cfg)
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        if isinstance(getattr(c, k, None), int):
            return getattr(c, k)
    raise NotImplementedError("the config gives no layer count")


def chat_ids(tok, messages):
    """Token ids for a conversation ending in a user turn, ready for the assistant's reply (thinking off where the
    template has a switch). A tokenizer without a chat template gets a plain "User: / Assistant:" transcript."""
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)
    return tok(plain_transcript(messages) + "Assistant:").input_ids


def chat_text(tok, messages):
    """A whole conversation as text, as the model would see it (for corpus statistics)."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    return plain_transcript(messages)


def plain_transcript(messages):
    role = {"user": "User", "assistant": "Assistant", "system": "System"}
    return "".join(f"{role.get(m['role'], m['role'])}: {m['content']}\n\n" for m in messages)


def stop_ids(model, tok):
    """Token ids that end a reply: generation_config's eos ids, the tokenizer's eos, and the template's end-of-turn
    token. `model` is a loaded model, or a model name whose generation_config.json is read."""
    ids = set()
    if isinstance(model, str):
        from transformers import GenerationConfig
        try:
            gc = GenerationConfig.from_pretrained(model)
        except OSError:  # no generation_config.json
            gc = None
    else:
        gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    ids.update(eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else [])
    if tok is not None:
        if tok.eos_token_id is not None:
            ids.add(tok.eos_token_id)
        vocab = tok.get_vocab()
        ids.update(vocab[t] for t in END_OF_TURN if t in vocab)
    return {int(i) for i in ids if i is not None}


class Arch:
    """Where the explorer's reads and pushes go in one loaded model.

    layers[l], mlps[l]   decoder block and its MLP (Mlp)
    pre_norms[l]         the norm whose output is the MLP's input (None: the MLP reads the raw residual, as in OLMo 2)
    post_norms[l]        the norm applied to the MLP's output before the residual add (None on most models)
    attn[l]              (name, output projection, head count) of the token mixer
    attn_pre_norms[l]    the norm whose output is the attention's input; attn_post_norms[l]: the norm on its output
                         before the residual add (Gemma 2/3, OLMo 2)
    parallel[l]          attention and MLP read the same residual (GPT-NeoX, Phi-2): the MLP doesn't see this layer's heads
    final_norm, head     the norm in front of the unembedding, and the unembedding
    """

    def __init__(self, model, tok=None, name=None, trace=True):
        self.model, self.tok, self.name = model, tok, name or getattr(model, "name_or_path", "")
        self.config = text_config(model.config)
        self.n_layers = n_layers_of(model.config)
        self.lm, self.layers = find_layers(model, self.n_layers)
        self.mlps = [find_mlp(x) for x in self.layers]
        widths = {m.width for m in self.mlps}
        if len(widths) != 1:
            raise NotImplementedError(f"MLP widths differ across layers ({sorted(widths)}); one width is supported")
        self.n_neurons = widths.pop()
        self.hidden = self.mlps[0].down_w().shape[0]
        self.attn = [find_attention(x, self.config) for x in self.layers]
        self.head = model.get_output_embeddings()
        if self.head is None:
            raise NotImplementedError(f"{type(model).__name__} has no output embeddings (unembedding)")
        self.n_vocab = len(tok) if tok is not None else _weight(self.head).shape[0]
        self.pre_norms = [None] * self.n_layers
        self.post_norms = [None] * self.n_layers
        self.attn_pre_norms = [None] * self.n_layers
        self.attn_post_norms = [None] * self.n_layers
        self.parallel = [False] * self.n_layers
        self.final_norm = None
        self.traced = False
        if trace:
            self._trace()
        self._scales = {}
        self._parts = {}

    # ---------- discovery by tracing ----------
    def _trace(self):
        """One short forward pass that finds, per layer, the norm whose output is the MLP's input and the norm whose
        input is the MLP's output, and the norm in front of the unembedding."""
        rec, hooks = {}, []

        def norm_hook(key, name):
            def hook(mod, args, kwargs, out):
                rec.setdefault(key, []).append((name, mod, _first_tensor(args, kwargs),
                                                out[0] if isinstance(out, tuple) else out))
            return hook

        def mlp_hook(l):
            def hook(mod, args, kwargs, out):
                rec[("mlp", l)] = (_first_tensor(args, kwargs), out[0] if isinstance(out, tuple) else out)
            return hook

        in_layers = set()
        for l, (layer, mlp) in enumerate(zip(self.layers, self.mlps)):
            inside = {id(m) for m in mlp.module.modules()}
            for name, m in layer.named_modules():
                in_layers.add(id(m))
                if id(m) not in inside and "norm" in type(m).__name__.lower():
                    hooks.append(m.register_forward_hook(norm_hook(("norm", l), name), with_kwargs=True))
            hooks.append(mlp.module.register_forward_hook(mlp_hook(l), with_kwargs=True))
            if self.attn[l][0] is not None:
                hooks.append(getattr(layer, self.attn[l][0]).register_forward_hook(
                    lambda mod, args, kwargs, out, l=l: rec.__setitem__(
                        ("attn", l), (_first_tensor(args, kwargs), out[0] if isinstance(out, tuple) else out)),
                    with_kwargs=True))
        for name, m in self.lm.named_modules():
            if id(m) not in in_layers and "norm" in type(m).__name__.lower():
                hooks.append(m.register_forward_hook(norm_hook("final", name), with_kwargs=True))
        hooks.append(self.head.register_forward_pre_hook(
            lambda mod, args, kwargs: rec.__setitem__("head_in", _first_tensor(args, kwargs)), with_kwargs=True))
        dev = self.head.weight.device
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=dev).clamp(max=self.n_vocab - 1)
        try:
            with torch.inference_mode():
                self.model(input_ids=ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for l in range(self.n_layers):
            x_in, x_out = rec.get(("mlp", l), (None, None))
            norms = rec.get(("norm", l), [])
            self.pre_norms[l] = next((m for _, m, _, o in norms if _same(o, x_in)), None)
            self.post_norms[l] = next((m for _, m, i, _ in norms if _same(i, x_out)), None)
            a_in, a_out = rec.get(("attn", l), (None, None))
            self.attn_pre_norms[l] = next((m for _, m, _, o in norms if _same(o, a_in)), None)
            self.attn_post_norms[l] = next((m for _, m, i, _ in norms if _same(i, a_out)), None)
            # what each block reads before its norm: the same tensor means attention and MLP run side by side
            src = lambda norm, x: next((i for _, m, i, _ in norms if m is norm), None) if norm is not None else x
            self.parallel[l] = a_in is not None and _same(src(self.pre_norms[l], x_in), src(self.attn_pre_norms[l], a_in))
        self.final_norm = next((m for _, m, _, o in rec.get("final", []) if _same(o, rec.get("head_in"))), None)
        if self.final_norm is None:  # something between the norm and the head (a scale); the lens ignores it
            self.final_norm = next((getattr(self.lm, n) for n in ("norm", "final_layernorm", "final_layer_norm",
                                                                    "ln_f", "norm_f") if hasattr(self.lm, n)), None)
        self.traced = True

    # ---------- weights in residual-stream terms ----------
    def norm_scale(self, norm):
        """The per-channel scale a norm applies, read off the module: norm(e) - norm(-e) = 2 * scale * e for a probe e
        of alternating +-1 (zero mean, unit rms), whatever its form (w, 1 + w, a LayerNorm with a bias). Ones for None."""
        dev = self.head.weight.device
        if norm is None:
            return torch.ones(self.hidden, device=dev)
        if id(norm) not in self._scales:
            p = next((t for t in norm.parameters()), self.head.weight)
            # a float32 probe keeps the scale exact on bfloat16 models (RMSNorms compute in float32 anyway); a norm
            # that insists on its own dtype (torch's LayerNorm) gets the probe in that dtype
            for dt in (torch.float32, p.dtype):
                e = torch.ones(self.hidden, device=dev, dtype=dt)
                e[1::2] = -1
                try:
                    with torch.inference_mode():
                        d = norm(e[None, None]).float() - norm(-e[None, None]).float()
                    break
                except RuntimeError:
                    continue
            self._scales[id(norm)] = (d[0, 0] / (2 * e.float())).clone()
        return self._scales[id(norm)]

    def norm_centred(self, norm):
        """Whether a norm subtracts the mean first (LayerNorm) rather than only dividing by the rms (RMSNorm), read off
        the module: a centring norm maps a constant vector and its negative to the same output."""
        key = ("centred", id(norm))
        if key not in self._scales:
            p = next((t for t in norm.parameters()), self.head.weight)
            e = torch.ones(1, 1, self.hidden, device=self.head.weight.device, dtype=p.dtype)
            with torch.inference_mode():
                self._scales[key] = bool(torch.allclose(norm(e).float(), norm(-e).float(), atol=1e-4))
        return self._scales[key]

    def pre_scale(self, l):
        """[hidden] float: what the MLP of layer l multiplies a residual direction by before its gate and up rows."""
        return self.norm_scale(self.pre_norms[l])

    def write_w(self, l):
        """[hidden, N] in the model's dtype: neuron n's write into the residual stream, column n (the down column, scaled
        by the post-MLP norm when there is one; that norm also divides by the whole MLP output's rms, which no
        per-neuron direction can show)."""
        w = self.mlps[l].down_w()
        if self.post_norms[l] is None:
            return w
        return w * self.norm_scale(self.post_norms[l]).to(w.dtype)[:, None]

    # ---------- attention heads ----------
    def attn_kind(self, l):
        """"softmax" (full or sliding-window attention whose queries, keys and values the capture sees), "deltanet"
        (Gated DeltaNet linear attention, Qwen3.5 / Qwen3-Next), or None: head patterns and links are off there."""
        p = self.attn_parts(l)
        return p.kind if p is not None else None

    def attn_parts(self, l):
        """Layer l's attention cut into heads (AttnParts), or None for a layout this code doesn't know."""
        if l not in self._parts:
            name, out, n = self.attn[l]
            mod = getattr(self.layers[l], name) if name else None
            try:
                self._parts[l] = AttnParts.of(mod, out, n) if mod is not None and out is not None and n else None
            except (RuntimeError, ValueError, AttributeError):  # weights that don't split the way a known layout does
                self._parts[l] = None
        return self._parts[l]

    def attn_pre_scale(self, l):
        """[hidden] float: the per-channel scale of the norm in front of layer l's attention."""
        return self.norm_scale(self.attn_pre_norms[l])

    def attn_post_scale(self, l):
        """[hidden] float: the per-channel scale of the norm on layer l's attention output (ones when there is none)."""
        return self.norm_scale(self.attn_post_norms[l])

    def logits(self, x):
        """Unembed residual-stream vectors x [..., hidden] through the final norm, as the logit lens reads them. Cut to
        the tokenizer's vocabulary: rows past it pad the unembedding and decode to nothing."""
        h = self.final_norm(x) if self.final_norm is not None else x
        return self.head(h)[..., : self.n_vocab]

    @property
    def post_mlp_norm(self):
        return any(n is not None for n in self.post_norms)

    # ---------- summary ----------
    def layer_types(self):
        lt = getattr(self.config, "layer_types", None)
        if isinstance(lt, (list, tuple)) and len(lt) == self.n_layers:
            return [str(t) for t in lt]
        return ["linear_attention" if a[0] == "linear_attn" else "full_attention" for a in self.attn]

    def mlp_name(self):
        kind = self.mlps[0].kind
        act = type(self.mlps[0].act).__name__.lower() if self.mlps[0].act is not None else ""
        if kind != "plain":
            return "SwiGLU" if "silu" in act or "swish" in act else "GeGLU" if "gelu" in act else "gated"
        return "GELU" if "gelu" in act else "ReLU" if "relu" in act else "plain"

    def describe(self):
        """What the frontend and the Claude agent are told about the model."""
        c = self.config
        lt = self.layer_types()
        heads = [a[2] for a in self.attn]
        full = [h for h, t in zip(heads, lt) if t != "linear_attention"]
        lin = [h for h, t in zip(heads, lt) if t == "linear_attention"]
        return {
            "model": self.name, "n_layers": self.n_layers, "hidden": self.hidden, "inter": self.n_neurons,
            "layer_types": lt, "heads": heads, "mlp": self.mlp_name(),
            "attn_heads": full[0] if full else getattr(c, "num_attention_heads", None),
            "kv_heads": getattr(c, "num_key_value_heads", None) or getattr(c, "num_attention_heads", None),
            "linear_k_heads": getattr(c, "linear_num_key_heads", None),
            "linear_v_heads": getattr(c, "linear_num_value_heads", None) or (lin[0] if lin else None),
            # untraced (weights on the meta device): read off the name every such model uses so far
            "post_mlp_norm": self.post_mlp_norm if self.traced else any(
                "post_feedforward" in name or "post_mlp" in name for name, _ in self.layers[0].named_modules()),
            "vocab": self.n_vocab,
            "attn_kinds": [self.attn_kind(l) for l in range(self.n_layers)],  # softmax / deltanet / None (no head tools)
        }


class AttnParts:
    """One layer's attention cut into heads, from its weights (each [rows, hidden] or [hidden, cols], before any norm
    scale). Output heads h = 0..n-1 each have a value slice and an output slice; queries and keys come in groups that
    several heads share (grouped-query attention; Gated DeltaNet's key heads).

    kind      "softmax" or "deltanet"
    q, k, v   [units, dim, hidden]: query and key units, value units
    out       [hidden, n, dim]: the output projection's columns per head
    o_scale   [dim] or None: a per-channel weight between the head and the output projection (DeltaNet's gated norm)
    gated     softmax heads whose output is multiplied by a sigmoid gate from the query projection (Qwen3.5)
    """

    def __init__(self, kind, n, q, k, v, out, q_of, k_of, v_of, gated=False, o_scale=None, q_proj=None):
        self.kind, self.n, self.q, self.k, self.v, self.out = kind, n, q, k, v, out
        self.q_of, self.k_of, self.v_of = q_of, k_of, v_of  # output head -> its query / key / value unit
        self.gated, self.o_scale, self.q_proj = gated, o_scale, q_proj

    def heads_of(self, role, unit):
        """The output heads that share query / key / value unit `unit`."""
        f = {"q": self.q_of, "k": self.k_of, "v": self.v_of}[role]
        return [h for h in range(self.n) if f(h) == unit]

    @classmethod
    def of(cls, mod, out_proj, n):
        W_out = _weight(out_proj)  # [hidden, n * dim]
        H = W_out.shape[0]
        if all(hasattr(mod, x) for x in ("in_proj_qkv", "A_log", "conv1d")):  # Gated DeltaNet
            nk, kd, vd = mod.num_k_heads, mod.head_k_dim, mod.head_v_dim
            W = _weight(mod.in_proj_qkv)
            rep = n // nk
            return cls("deltanet", n, W[: nk * kd].view(nk, kd, H), W[nk * kd: 2 * nk * kd].view(nk, kd, H),
                       W[2 * nk * kd:].view(n, vd, H), W_out.view(H, n, vd), q_of=lambda h: h // rep,
                       k_of=lambda h: h // rep, v_of=lambda h: h, o_scale=mod.norm.weight)
        vd = W_out.shape[1] // n
        hd = next((d for d in (getattr(mod, "head_dim", None), getattr(mod, "head_size", None)) if isinstance(d, int)), vd)
        gated, q_proj = False, None
        if all(isinstance(getattr(mod, x, None), torch.nn.Module) for x in ("q_proj", "k_proj", "v_proj")):
            Wq, Wk, Wv = (_weight(getattr(mod, x)) for x in ("q_proj", "k_proj", "v_proj"))
            nkv = Wk.shape[0] // hd
            if Wq.shape[0] == 2 * n * hd:  # each head's query followed by its output gate (Qwen3.5)
                q, gated, q_proj = Wq.view(n, 2 * hd, H)[:, :hd], True, mod.q_proj
            elif Wq.shape[0] == n * hd:
                q = Wq.view(n, hd, H)
            else:
                return None
            k, v = Wk.view(nkv, hd, H), Wv.view(nkv, Wv.shape[0] // nkv, H)
        elif isinstance(getattr(mod, "qkv_proj", None), torch.nn.Module):  # Phi-3: q, then k, then v
            W = _weight(mod.qkv_proj)
            nkv = (W.shape[0] - n * hd) // (2 * hd)
            q, k, v = W[: n * hd].view(n, hd, H), W[n * hd: (n + nkv) * hd].view(nkv, hd, H), W[(n + nkv) * hd:].view(nkv, hd, H)
        elif isinstance(getattr(mod, "c_attn", None), torch.nn.Module):  # GPT-2: q, k, v thirds
            W = _weight(mod.c_attn)
            nkv = n
            q, k, v = (x.reshape(n, hd, H) for x in W.chunk(3, 0))
        elif isinstance(getattr(mod, "query_key_value", None), torch.nn.Module) and "neox" in type(mod).__name__.lower():
            W = _weight(mod.query_key_value).view(n, 3, hd, H)  # GPT-NeoX: per head, q k v
            nkv = n
            q, k, v = W[:, 0], W[:, 1], W[:, 2]
        else:
            return None
        if n % nkv or k.shape[1] != q.shape[1]:
            return None
        rep = n // nkv
        return cls("softmax", n, q, k, v, W_out.view(H, n, vd), q_of=lambda h: h, k_of=lambda h: h // rep,
                   v_of=lambda h: h // rep, gated=gated, q_proj=q_proj)


# id(attention module) -> the dict its attention call fills while a head capture runs (install_capture)
CAPTURE = {}


def _capturing(fn):
    def attn(module, query, key, value, attention_mask, *args, **kwargs):
        c = CAPTURE.get(id(module))
        if c is not None:  # [batch, heads, T, dim]: the queries, keys and values the softmax uses, after rotary
            c.update(q=query[0], k=key[0], v=value[0], scaling=kwargs.get("scaling"), softcap=kwargs.get("softcap"),
                     window=kwargs.get("sliding_window"))
        return fn(module, query, key, value, attention_mask, *args, **kwargs)
    attn._explorer = True
    return attn


def install_capture(model):
    """Wraps the attention functions transformers dispatches to (the registered ones, and each model file's own eager
    function, which is used when "eager" is picked) so a capture can see the queries, keys and values a softmax head
    attends with: exactly the model's, with its norms, rotary embedding, scale, softcap and sliding window. Calls pass
    through unchanged; idempotent."""
    import sys
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as REG
    for key in REG.valid_keys():
        if not getattr(REG[key], "_explorer", False):
            REG[key] = _capturing(REG[key])
    for m in model.modules():
        mod = sys.modules.get(type(m).__module__)
        fn = getattr(mod, "eager_attention_forward", None)
        if fn is not None and not getattr(fn, "_explorer", False):
            mod.eager_attention_forward = _capturing(fn)
