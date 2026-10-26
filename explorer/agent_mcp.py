"""Stdio MCP server that gives the explorer's Claude chat its tools: read any neuron data through the explorer's HTTP API,
run the live model (steered rollouts, activations) when it is loaded, and drive the map (select, highlight) in the browser that started the chat. Standard library only; started by
`claude -p --mcp-config ...` from server.py with EXPLORER_URL and EXPLORER_CHAT set.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

URL = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
CHAT = os.environ.get("EXPLORER_CHAT", "")


def _ranges():
    """The loaded model's layer and neuron ranges, from the explorer (which started this server and is up)."""
    try:
        with urllib.request.urlopen(f"{URL}/api/index", timeout=10) as r:
            a = json.load(r)["arch"]
        return f"layer, 0-{a['n_layers'] - 1}", f"neuron, 0-{a['inter'] - 1}"
    except Exception:  # noqa: BLE001  the ranges are a hint; the tools still work without them
        return "layer index", "neuron index within the layer"


LN = dict(zip(("layer", "n"), ({"type": "integer", "description": d} for d in _ranges())))
TOOLS = [
    {"name": "overview", "description": "Model architecture and which layers have sweep runs (rollouts and rankings).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "neuron", "description": "Everything cached for one neuron: name, logit lens and steer lens (with KL) for "
     "both poles, activation range, sweep rank if any, and its 24 strongest direct-path links each way (writes_into, "
     "reads_from) with each linked neuron's steer lens, top words and name.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "rollouts", "description": "Sweep rollouts for a neuron: steered text per prompt and dose beside the "
     "unsteered baselines, distinctive words and stats. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {**LN, "max_chars": {"type": "integer", "default": 500}},
                     "required": ["layer", "n"]}},
    {"name": "cone", "description": "The neuron's connection cone: links chained forward (downstream) and backward "
     "(upstream), first hop all links, later hops each node's top k. Returns the neurons reached with hop, best z and "
     "steer lens. Multi-hop paths are weight-only guesses: leads to check, not circuits.",
     "inputSchema": {"type": "object", "properties": {**LN, "k": {"type": "integer", "default": 2},
                                                       "limit": {"type": "integer", "default": 120}},
                     "required": ["layer", "n"]}},
    {"name": "web_summary", "description": "The token summary of the neuron's web, the same one the person sees in the "
     "left panel: steer-lens tokens pooled over its cone, each connected neuron read on the pole the chain drives, ranked "
     "by lift over that layer's base rate. `toward` / `away`: what its downstream neurons push the model toward and away "
     "from when this neuron fires (indirect effect). `fed_by` / `held_by`: what its upstream writers say when they "
     "excite / inhibit it (its inputs). Each token has n (neurons carrying it) and a score.",
     "inputSchema": {"type": "object", "properties": {**LN, "first": {"type": "integer", "default": 5},
                                                       "k": {"type": "integer", "default": 2}},
                     "required": ["layer", "n"]}},
    {"name": "search_lens", "description": "Neurons whose logit lens or steer lens (top 12, either pole) contains a token "
     "matching the query (case-insensitive substring). Best matches first.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 40}},
                     "required": ["query"]}},
    {"name": "top_units", "description": "A layer's neurons ranked by sweep word-cluster mass against random-direction "
     "controls, with top words and names. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {"layer": LN["layer"], "limit": {"type": "integer", "default": 30}},
                     "required": ["layer"]}},
    {"name": "set_name", "description": "Name a neuron in the explorer (the name field in its panel, shown in lists and on "
     "the map), or clear its name with an empty string. Only when the person asks, or after they agree to a name you "
     "suggested.",
     "inputSchema": {"type": "object", "properties": {**LN, "name": {"type": "string"}}, "required": ["layer", "n", "name"]}},
    {"name": "names", "description": "Every neuron the person has named so far, as {\"L14/N3262\": name}.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "steer", "description": "Live: generate from a chat prompt three ways in one batch, pushed up, unsteered, "
     "and pushed down (greedy). Steer one neuron (layer, n) or several at once (neurons: [{layer, n, dose?, mode?}]; all "
     "are applied together, each reversed in the down row; a negative per-neuron dose pushes that one the other way). "
     "dose in units of the layer's median residual norm (MLP-output norm on models that normalise the MLP output) for "
     "mode 'add' (0.8 is the steer-lens push; 2-3 is strong), or "
     "a multiplier for mode 'mul'. Needs the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "prompt": {"type": "string"},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer", "n"]}},
                                                       "dose": {"type": "number", "default": 0.8},
                                                       "mode": {"type": "string", "enum": ["add", "mul"], "default": "add"},
                                                       "max_new": {"type": "integer", "default": 150}},
                     "required": ["prompt"]}},
    {"name": "activations", "description": "Live: the neuron's raw activation on every token of a text (chat=true wraps "
     "it as a user turn). Compare against act_range from `neuron`. Optional `neurons` [{layer, n, dose?, mode?}] are "
     "pushed during the same pass (steering units), to test causally whether they drive this neuron: compare with the "
     "same call without them. {layer, all: true, dose: 0} zeroes that layer's whole MLP. Optional `heads` [{layer, "
     "head, dose}] scale attention heads (dose 0 = ablate; head 'all' = every head; each layer's head count and type are "
     "in `overview`, arch.heads and arch.layer_types). Only earlier layers can move it. Needs "
     "the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "text": {"type": "string"},
                                                       "chat": {"type": "boolean", "default": False},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["layer", "n", "text"]}},
    {"name": "compare", "description": "Live: read many neurons across several texts in one call (one forward pass per "
     "text). Per neuron and text: its strongest positive and negative activation with the token, and peak_score (the "
     "activation over the neuron's usual extreme; 1 = as strong as its usual max). For screening candidates against "
     "target texts and controls; `activations` gives one neuron's full per-token trace and takes pushes for causal "
     "tests. Up to 300 neurons and 24 texts.",
     "inputSchema": {"type": "object", "properties": {
         "neurons": {"type": "array", "items": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
         "texts": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {
             "label": {"type": "string"}, "text": {"type": "string"}}, "required": ["text"]}]}},
         "chat": {"type": "boolean", "default": False}},
                     "required": ["neurons", "texts"]}},
    {"name": "logprobs", "description": "Live: the next token's log-probabilities after a text (chat=true by default: "
     "the text is a user turn and the next token opens the reply; `prefix` continues from the start of a reply or any "
     "text). Returns the top k tokens and the logprob of each `targets` string's first token. With `neurons` and/or "
     "`heads` (same as `activations`: pushes, whole-layer zeroing, head scaling) it runs with and without them and "
     "returns both, the KL from baseline to intervened, and the tokens that gained and lost the most probability "
     "across the whole vocabulary: a direct read of what an intervention does to the output.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "chat": {"type": "boolean", "default": True},
                                                       "prefix": {"type": "string"}, "k": {"type": "integer", "default": 15},
                                                       "targets": {"type": "array", "items": {"type": "string"}},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["text"]}},
    {"name": "top_firing", "description": "Live: run text through the model and list the neurons firing hardest at one "
     "token (the first token containing `token`; omit it for the token with the strongest firing). Each neuron has its "
     "pole, score (activation over its usual extreme; 1 = as strong as its usual max), raw activation and steer lens. "
     "`layers` [lo, hi] restricts the layers. Doesn't change the person's screen; use show_text to show them.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 30},
                                                       "layers": {"type": "array", "items": {"type": "integer"}},
                                                       "chat": {"type": "boolean", "default": False}},
                     "required": ["text"]}},
    {"name": "camera", "description": "Move the person's 3D view. mode 'home': the whole model. 'layer': look at one "
     "layer's neuron sheet face-on (give layer). 'neuron': fly close to one neuron (give layer and n; closeness 0.1 = very "
     "close, 0.5 = wide).",
     "inputSchema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["home", "layer", "neuron"]},
                                                       **LN, "closeness": {"type": "number", "default": 0.22}},
                     "required": ["mode"]}},
    {"name": "annotate_neurons", "description": "Pin short text labels to neurons on the 3D map (they follow the camera). "
     "Replaces the previous labels; an empty list clears them. Keep each label a few words.",
     "inputSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "text": {"type": "string"}}, "required": ["layer", "n", "text"]}}},
                     "required": ["items"]}},
    {"name": "show_text", "description": "Put text in the explorer's paste box and run it through the live model, so the "
     "person sees its tokens; pin the first token containing `token` to show the neurons firing most there and the "
     "direct links among them on the map. Needs the live model.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "show_search", "description": "Run a lens search in the explorer's left panel, so the person sees the "
     "matching neurons listed and can click them.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "open_tab", "description": "Switch the open neuron's panel to a tab: examples (sweep rollouts), chat (the "
     "person's own steered chat) or details (lenses and connections).",
     "inputSchema": {"type": "object", "properties": {"tab": {"type": "string", "enum": ["examples", "chat", "details"]}},
                     "required": ["tab"]}},
    {"name": "select_neuron", "description": "Open a neuron in the explorer: its panel, and its connection web on the map.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "highlight_neurons", "description": "Mark a set of neurons on the 3D map (replaces the previous marks; an "
     "empty list clears them). sign '-' draws a neuron in the negative-pole colour.",
     "inputSchema": {"type": "object", "properties": {"neurons": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "sign": {"type": "string", "enum": ["+", "-"]}},
         "required": ["layer", "n"]}}}, "required": ["neurons"]}},
]


def get(path, **q):
    with urllib.request.urlopen(f"{URL}{path}?{urllib.parse.urlencode(q)}", timeout=60) as r:
        return json.load(r)


def post(path, body, timeout=60):
    req = urllib.request.Request(f"{URL}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def call(name, a):
    if name == "overview":
        return get("/api/agent/overview")
    if name == "neuron":
        return get("/api/agent/neuron", layer=a["layer"], n=a["n"])
    if name == "rollouts":
        u = get("/api/unit", layer=a["layer"], n=a["n"])
        cut = int(a.get("max_chars", 500))
        for key in ("rollouts", "baselines"):
            for r in u.get(key, []):
                r["text"] = r["text"][:cut]
        return u
    if name == "cone":
        return get("/api/agent/cone", layer=a["layer"], n=a["n"], k=a.get("k", 2), limit=a.get("limit", 120))
    if name == "web_summary":
        r = get("/api/web/summary", layer=a["layer"], n=a["n"], first=a.get("first", 5), k=a.get("k", 2), top=20)
        slim = lambda lst: [f"{t['tok']} ({t['n']})" for t in lst]
        return {"lens": r["src"], "links": r["links"],
                "downstream_neurons": r["down"]["neurons"], "toward": slim(r["down"]["toward"]), "away": slim(r["down"]["away"]),
                "upstream_neurons": r["up"]["neurons"], "fed_by": slim(r["up"]["fed_by"]), "held_by": slim(r["up"]["held_by"])}
    if name == "search_lens":
        r = get("/api/search", q=a["query"], limit=a.get("limit", 40))
        return {"tokens_matched": r.get("tokens"), "total": r.get("total"), "results": r.get("results", [])}
    if name == "top_units":
        us = sorted((u for u in get("/api/units", layer=a["layer"]) if u["kind"] == "neuron"), key=lambda u: -u["z"])
        return us[: int(a.get("limit", 30))]
    if name == "set_name":
        post("/api/name", {"layer": a["layer"], "n": a["n"], "name": a["name"]})
        post("/api/agent/ui", {"chat": CHAT, "action": "rename", "layer": a["layer"], "n": a["n"], "name": a["name"]})
        return {"ok": True, "neuron": f"L{a['layer']} N{a['n']}", "name": a["name"].strip() or None}
    if name == "names":
        return get("/api/agent/names")
    if name == "steer":
        return post("/api/agent/steer", a, timeout=900)  # the first call may wait for the model to load
    if name == "top_firing":
        return post("/api/agent/fires", a, timeout=900)
    if name == "logprobs":
        return post("/api/agent/logprobs", a, timeout=900)
    if name == "compare":
        return post("/api/agent/compare", a, timeout=900)
    if name == "activations":
        return post("/api/agent/activations", a, timeout=900)
    ui = {"camera": "camera", "annotate_neurons": "annotate", "show_text": "text", "show_search": "search", "open_tab": "tab"}
    if name in ui:
        return post("/api/agent/ui", {"chat": CHAT, "action": ui[name], **a})
    if name == "select_neuron":
        return post("/api/agent/ui", {"chat": CHAT, "action": "select", "layer": a["layer"], "n": a["n"]})
    if name == "highlight_neurons":
        return post("/api/agent/ui", {"chat": CHAT, "action": "highlight", "neurons": a["neurons"]})
    raise ValueError(f"unknown tool {name}")


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    if not line.strip():
        continue
    m = json.loads(line)
    method, id_ = m.get("method"), m.get("id")
    if id_ is None:
        continue  # notifications
    if method == "initialize":
        reply(id_, {"protocolVersion": m["params"].get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {}}, "serverInfo": {"name": "explorer", "version": "0.1"}})
    elif method == "tools/list":
        reply(id_, {"tools": TOOLS})
    elif method == "tools/call":
        p = m["params"]
        try:
            out = call(p["name"], p.get("arguments") or {})
            reply(id_, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]})
        except Exception as e:  # tool errors go back to the model, not the protocol
            reply(id_, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
    elif method == "ping":
        reply(id_, {})
    else:
        reply(id_, error={"code": -32601, "message": f"method not found: {method}"})
