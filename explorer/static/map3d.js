import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

// geometry: layers run left to right along x. Each layer is the residual stream passing through a token-mixer slab
// (linear attention thin, softmax attention, full or sliding-window, thick) and then the MLP, drawn as a vertical sheet
// of its neurons.
const SHEET = 4.2, GAP = 1.9, SHEET_Y0 = 0.7, ATTN_DX = -0.62, MLP_DX = 0.18;
const PAL = {
  bg: 0x000000,
  idle: new THREE.Color(0.036, 0.046, 0.052), swept: new THREE.Color(0.2, 0.24, 0.26),
  lo: new THREE.Color(0.42, 0.30, 0.10), hi: new THREE.Color(1.0, 0.71, 0.28), named: new THREE.Color(1, 1, 1),
  link: new THREE.Color(0.22, 0.78, 1.0),
  wire: 0x1c2a30, wireData: 0x3a5a66, slabEdge: 0x3c4a52, slabEdgeFull: 0x6a8592, head: 0x2f4a56, headFull: 0x5f8fa3,
};
// overall brightness of everything drawn in the map (points, lines, slabs); 1 = original
const BRIGHT = Number(new URLSearchParams(location.search).get("bright")) || 2.5;
// the boost lifts the structure (sheets, wires, idle points); data colours keep their hue
for (const k in PAL) if (!["bg", "swept", "lo", "hi", "named"].includes(k)) PAL[k] = new THREE.Color(PAL[k]).multiplyScalar(BRIGHT);

function glowPoints(sizePx, maxScale = 3) {
  return new THREE.RawShaderMaterial({
    uniforms: { size: { value: sizePx * Math.min(devicePixelRatio, 2) }, ref: { value: 1 }, maxScale: { value: maxScale },
      focusX: { value: 0 }, fadeW: { value: 1 }, fadeAmt: { value: 0 }, camSide: { value: 1 }, tdist: { value: 1 } },
    vertexShader: `precision highp float;
      uniform mat4 modelViewMatrix, projectionMatrix; uniform float size, ref, maxScale, focusX, fadeW, fadeAmt, camSide, tdist;
      attribute vec3 position; attribute vec3 color; varying vec3 vC;
      void main() { float d = (position.x - focusX) / fadeW;
        vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        // up close nothing overlaps, so the scene brightens as the camera nears its focus; one factor for every point,
        // so layers nearer the camera than the focus don't outshine it
        float near = clamp(sqrt(ref / tdist), 1.0, 3.5);
        // layers away from the focus fade as you zoom in (never below a quarter); layers between the camera and the
        // focus fade further, to 40% up close (x-ray)
        float front = step(0.45, (position.x - focusX) * camSide / ${GAP.toFixed(3)});
        float keep = mix(1.0, exp(-d * d), fadeAmt * 0.75) * mix(1.0, 1.0 - 0.6 * fadeAmt, front);
        // points nearer than the focus are drawn bigger; dim them by the extra size so they don't outshine it
        float sc = clamp(sqrt(ref / -mv.z), 0.6, maxScale), sc0 = clamp(sqrt(ref / tdist), 0.6, maxScale);
        vC = color * keep * near * min(1.0, sc0 / sc);
        gl_PointSize = size * sc; }`,
    fragmentShader: `precision highp float; varying vec3 vC;
      void main() { float r = length(gl_PointCoord - 0.5); if (r > 0.5) discard;
        float edge = 1.0 - smoothstep(0.38, 0.5, r); gl_FragColor = vec4(vC * edge, 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}

export function createMap(canvas, labelsDiv, arch, cb) {
  const L = arch.n_layers, N = arch.inter;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(PAL.bg);
  const scene = new THREE.Scene();
  const layerX = l => l * GAP;
  const X0 = ATTN_DX - 1.6, X1 = (L - 1) * GAP + MLP_DX + 1.4;

  const insets = { left: 410, right: 0 };
  const TARGET_X = Number(new URLSearchParams(location.search).get("targetx")) || 0;
  let fitPerPx = null;
  const camera = new THREE.PerspectiveCamera(32, 1, 0.02, 2000);
  let homeDist = 60;
  const home = { target: new THREE.Vector3((L - 1) * GAP / 2, SHEET_Y0 + SHEET / 2 - 0.4, 0), dir: new THREE.Vector3(0.62, 0.4, 1).normalize() };
  camera.position.copy(home.target).addScaledVector(home.dir, homeDist);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(home.target);
  controls.enableDamping = false;
  controls.zoomToCursor = true;
  controls.minDistance = 1.6; controls.maxDistance = 400;
  controls.zoomSpeed = 0.6;
  controls.update();

  const gridSide = Math.ceil(Math.sqrt(N));
  let layoutCells = null;  // Int16Array (Int32Array for big sheets) [L * N] of grid cells, or null for neuron-index order
  const neuronPos = (layer, n) => {
    const c = layoutCells ? layoutCells[layer * N + n] : n;
    const i = c % gridSide, j = Math.floor(c / gridSide);
    return new THREE.Vector3(layerX(layer) + MLP_DX, SHEET_Y0 + (j / (gridSide - 1)) * SHEET, (i / (gridSide - 1) - 0.5) * SHEET);
  };

  // every neuron of every layer, one draw call
  const pos = new Float32Array(L * N * 3), col = new Float32Array(L * N * 3);
  for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
    const p = neuronPos(l, n), k = (l * N + n) * 3;
    pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z;
    PAL.idle.toArray(col, k);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const pointMat = glowPoints(2.0, 5);
  const points = new THREE.Points(geo, pointMat);
  scene.add(points);

  // structure
  const wires = [];
  const seg = (a, b) => wires.push(a.x, a.y, a.z, b.x, b.y, b.z);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const sheetCorners = x => { const h = SHEET / 2; return [V(x, SHEET_Y0, -h), V(x, SHEET_Y0, h), V(x, SHEET_Y0 + SHEET, h), V(x, SHEET_Y0 + SHEET, -h)]; };
  seg(V(X0, 0, 0), V(X1, 0, 0));
  // heads are horizontal lines stacked down the slab's face, head 0 at the top
  const HEAD_SPAN = SHEET * 0.82, HEAD_W = SHEET * 0.41;
  const headY = (h, H) => SHEET_Y0 + SHEET / 2 + (0.5 - (h + 0.5) / H) * HEAD_SPAN;
  const slabs = [], headPos = [], headBase = [], headStart = [];  // headStart[l]: first line of layer l's heads
  for (let l = 0; l < L; l++) {
    const x = layerX(l), mx = x + MLP_DX, ax = x + ATTN_DX, cs = sheetCorners(mx);
    for (let i = 0; i < 4; i++) seg(cs[i], cs[(i + 1) % 4]);
    seg(V(ax, 0, 0), V(ax, SHEET_Y0, 0)); seg(V(mx, 0, 0), V(mx, SHEET_Y0, 0));
    const full = arch.layer_types[l] !== "linear_attention", t = full ? 0.3 : 0.1;
    const slab = new THREE.Mesh(new THREE.BoxGeometry(t, SHEET * 0.92, SHEET * 0.92), new THREE.MeshBasicMaterial({
      color: full ? 0x8fa6b2 : 0x6f8590, transparent: true, opacity: (full ? 0.075 : 0.045) * BRIGHT, depthWrite: false }));
    slab.position.set(ax, SHEET_Y0 + SHEET / 2, 0);
    slab.userData = { layer: l, full };
    scene.add(slab); slabs.push(slab);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry),
      new THREE.LineBasicMaterial({ color: full ? PAL.slabEdgeFull : PAL.slabEdge, transparent: true }));
    edges.position.copy(slab.position); scene.add(edges);
    slab.userData.edges = edges; slab.userData.baseOpacity = slab.material.opacity;
    const heads = arch.heads?.[l] ?? (full ? arch.attn_heads : arch.linear_v_heads);
    slab.userData.heads = heads;
    headStart.push(headPos.length / 6);
    for (let h = 0; h < heads; h++) {
      const y = headY(h, heads);
      headPos.push(ax + t / 2 + 0.003, y, -HEAD_W, ax + t / 2 + 0.003, y, HEAD_W);
      headBase.push(full ? PAL.headFull : PAL.head);
    }
  }
  const fadeLines = [];  // whole-scene line work that dims as the camera closes in on one layer
  // the attention heads: one line per head on its layer's slab, coloured one by one
  const headGeo = new THREE.BufferGeometry(); headGeo.setAttribute("position", new THREE.Float32BufferAttribute(headPos, 3));
  const headCol = new Float32Array(headPos.length);
  headGeo.setAttribute("color", new THREE.BufferAttribute(headCol, 3));
  const headMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }); headMat.userData.base = 0.55;
  scene.add(new THREE.LineSegments(headGeo, headMat)); fadeLines.push(headMat);
  let headLevels = null, headSel = null, headHover = null;
  const band = op => { const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({
    color: PAL.named, transparent: true, opacity: op, depthWrite: false })); m.visible = false; m.renderOrder = 1; scene.add(m); return m; };
  const hoverBand = band(0.1), selBand = band(0.18);
  function placeBand(m, at) {
    m.visible = !!at;
    if (!at) return;
    const s = slabs[at.layer], H = s.userData.heads;
    m.position.set(s.position.x, headY(at.head, H), 0);
    m.scale.set(s.geometry.parameters.width + 0.02, HEAD_SPAN / H * 0.9, HEAD_W * 2 + 0.1);
  }
  function paintHeads() {
    placeBand(hoverBand, headHover && !(headSel && headSel.layer === headHover.layer && headSel.head === headHover.head) ? headHover : null);
    placeBand(selBand, headSel);
    const c = new THREE.Color();
    for (let l = 0; l < L; l++) {
      const H = (headStart[l + 1] ?? headBase.length) - headStart[l];
      for (let h = 0; h < H; h++) {
        const i = headStart[l] + h, lv = headLevels?.[l]?.[h] || 0;
        c.copy(headBase[i]);
        if (headLevels) c.multiplyScalar(0.3).lerp(PAL.hi, Math.min(1, lv));
        if (headHover && headHover.layer === l && headHover.head === h) c.lerp(PAL.named, 0.5);
        if (headSel && headSel.layer === l && headSel.head === h) c.copy(PAL.named);
        c.toArray(headCol, i * 6); c.toArray(headCol, i * 6 + 3);
      }
    }
    headGeo.attributes.color.needsUpdate = true;
  }
  paintHeads();
  const endSlabs = [];
  for (const x of [X0, X1]) {
    const m = new THREE.LineBasicMaterial({ color: PAL.slabEdgeFull, transparent: true }); m.userData.base = 1;
    const slab = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.14, 1.4, 2.2)), m);
    slab.position.set(x, 0, 0); scene.add(slab); endSlabs.push(m);
  }
  const wg = new THREE.BufferGeometry(); wg.setAttribute("position", new THREE.Float32BufferAttribute(wires, 3));
  const wireMat = new THREE.LineBasicMaterial({ color: PAL.wire, transparent: true }); wireMat.userData.base = 1;
  scene.add(new THREE.LineSegments(wg, wireMat)); fadeLines.push(wireMat, ...endSlabs);
  const dataOutline = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: PAL.wireData }));
  dataOutline.frustumCulled = false;
  scene.add(dataOutline);
  const bounds = [];
  for (const x of [X0, X1]) for (const y of [-0.8, SHEET_Y0 + SHEET]) for (const z of [-SHEET / 2, SHEET / 2]) bounds.push(V(x, y, z));

  // highlights and connections (their geometry changes every update, so culling is off)
  const hiGeo = new THREE.BufferGeometry();
  const hiMat = glowPoints(5, 2);
  const hiPts = new THREE.Points(hiGeo, hiMat);
  hiPts.frustumCulled = false; hiPts.renderOrder = 3;
  scene.add(hiPts);
  const conGeo = new THREE.BufferGeometry();
  const conLines = new THREE.LineSegments(conGeo, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  conLines.frustumCulled = false; conLines.renderOrder = 2;
  scene.add(conLines);
  // connection web of the selected neuron: upstream and downstream cones, coloured by link sign, fading per hop.
  const qs = new URLSearchParams(location.search);
  const WEB_ALPHA = Number(qs.get("webalpha")) || 0.5, WEB_DECAY = Number(qs.get("webdecay")) || 0.85;
  // by the sign of the link: yellow when the writer pushes the reader up, magenta when it pushes it down
  const WEB_UP = new THREE.Color(1.0, 0.82, 0.2), WEB_DOWN = new THREE.Color(1.0, 0.25, 0.85), HEAD_LINK = new THREE.Color(0.4, 0.85, 1.0);
  // a head's links by role: q and k (where it looks) apart from v and o (what it moves)
  const HEAD_ROLE = { q: new THREE.Color(0.45, 1.0, 0.55), k: new THREE.Color(0.7, 0.6, 1.0), v: HEAD_LINK, o: HEAD_LINK };
  // fat lines (WebGL draws plain lines 1px wide), additive with the intensity folded into the colour, and outside the
  // zoom fade, so the web stays bright up close
  const webMat = new LineMaterial({ vertexColors: true, linewidth: Number(qs.get("webwidth")) || 2, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending });
  const webLines = new LineSegments2(new LineSegmentsGeometry(), webMat);
  webLines.frustumCulled = false; webLines.renderOrder = 1; webLines.visible = false;
  scene.add(webLines);
  const webPtGeo = new THREE.BufferGeometry(), webPts = new THREE.Points(webPtGeo, glowPoints(4.5, 2.5));
  webPts.frustumCulled = false; webPts.renderOrder = 2; webPts.visible = false;
  scene.add(webPts);
  let webEdges = null;
  const webFed = new Set(), webFeeds = new Set();  // neurons the drawn web reads into / out of (layer * N + n)
  const DEAD = 0.08;  // brightness of lines the web doesn't continue through
  const DIRECT = 0.6;  // the open neuron's direct links, relative to full brightness
  function buildWeb() {
    if (!webEdges || !webEdges.length) { webFed.clear(); webFeeds.clear(); webLines.visible = false; webPts.visible = false; return; }
    const m = webEdges.length / 6, p = new Float32Array(m * 6), c = new Float32Array(m * 6);
    let zhi = 1;
    for (let i = 0; i < m; i++) zhi = Math.max(zhi, Math.abs(webEdges[i * 6 + 4]));
    // a line is "through" when the web continues past both ends: its writer is read into by another drawn line (or is
    // the open neuron) and its reader writes on into another (or is the open neuron); dead ends draw dimmer
    const fed = webFed, feeds = webFeeds, nk = (l, n) => l * N + n;
    fed.clear(); feeds.clear();
    for (let i = 0; i < m; i++) { const e = webEdges.subarray(i * 6, i * 6 + 6); feeds.add(nk(e[0], e[1])); fed.add(nk(e[2], e[3])); }
    const root = selected ? nk(selected.layer, selected.n) : -1;
    const touched = new Map();
    for (let i = 0; i < m; i++) {
      const e = webEdges.subarray(i * 6, i * 6 + 6), a = neuronPos(e[0], e[1]), b = neuronPos(e[2], e[3]);
      const k = e[4] > 0 ? WEB_UP : WEB_DOWN;
      const s0 = nk(e[0], e[1]), d0 = nk(e[2], e[3]);
      const through = (s0 === root || fed.has(s0)) && (d0 === root || feeds.has(d0));
      const al = WEB_ALPHA * (0.6 + 0.4 * Math.abs(e[4]) / zhi) * Math.max(0.35, WEB_DECAY ** (Math.abs(e[5]) - 1)) * (through ? 1 : DEAD);
      p.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6); c.set([k.r * al, k.g * al, k.b * al, k.r * al, k.g * al, k.b * al], i * 6);
      // each touched neuron takes the colour of the hop that first reached it (edges arrive in hop order)
      const key = e[5] > 0 ? e[2] * N + e[3] : e[0] * N + e[1];
      if (!touched.has(key)) touched.set(key, [e[5] > 0 ? b : a, k]);
    }
    // dots too: a neuron the web runs through (read into and out of, or next to the open neuron) at full brightness
    const tp = [], tc = [];
    for (const [id, [v, k]] of touched) {
      const f = (fed.has(id) || id === root) && (feeds.has(id) || id === root) ? 1 : DEAD * 3;
      tp.push(v.x, v.y, v.z); tc.push(k.r * f, k.g * f, k.b * f);
    }
    webPtGeo.setAttribute("position", new THREE.Float32BufferAttribute(tp, 3));
    webPtGeo.setAttribute("color", new THREE.Float32BufferAttribute(tc, 3));
    webPts.visible = true;
    webLines.geometry.dispose();
    webLines.geometry = new LineSegmentsGeometry().setPositions(p).setColors(c);
    webLines.visible = true;
  }
  let selected = null, hovered = null, connections = [], hoverCon = null, headSrc = null;  // headSrc: an open head's line middle
  let neuronHeads = [];  // the open neuron's strongest linked heads: {layer, head, z}
  const headMid = (layer, head) => { const i = (headStart[layer] + head) * 6;
    return new THREE.Vector3((headPos[i] + headPos[i + 3]) / 2, (headPos[i + 1] + headPos[i + 4]) / 2, (headPos[i + 2] + headPos[i + 5]) / 2); };
  const markGeo = new THREE.BufferGeometry();
  const markPts = new THREE.Points(markGeo, glowPoints(4, 2));
  markPts.frustumCulled = false; markPts.renderOrder = 1;
  scene.add(markPts);

  function refreshHighlights() {
    const p = [], c = [];
    const add = (l, n, color) => { const v = neuronPos(l, n); p.push(v.x, v.y, v.z); c.push(color.r, color.g, color.b); };
    const src = hoverCon ? hoverCon.src : selected, shown = hoverCon ? hoverCon.list : connections;
    const zmax = Math.max(...shown.map(k => k.z), 1);
    // direct links coloured like the web: yellow feeds (pushes the reader up), magenta holds back
    // head links carry a role instead of a sign
    const linkCol = k => (k.role ? HEAD_ROLE[k.role] : k.sign > 0 ? WEB_UP : k.sign < 0 ? WEB_DOWN : HEAD_LINK).clone();
    for (const k of shown) add(k.layer, k.n, linkCol(k).multiplyScalar(0.35 + 0.65 * k.z / zmax));
    if (hovered) add(hovered.layer, hovered.n, PAL.named.clone().multiplyScalar(0.7));
    if (selected) add(selected.layer, selected.n, PAL.named);
    if (headSrc && !selected) { p.push(headSrc.x, headSrc.y, headSrc.z); c.push(PAL.named.r, PAL.named.g, PAL.named.b); }
    hiGeo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    hiGeo.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
    const lp = [], lc = [];
    const a = src ? neuronPos(src.layer, src.n) : hoverCon ? hoverCon.from : headSrc;
    if (a) {
      for (const k of shown) {
        // with the web on, a direct link stays bright only if the web goes on past its far end
        const far = k.layer * N + k.n, on = !webEdges || !webEdges.length || hoverCon
          || (k.side === "up" ? webFed.has(far) : webFeeds.has(far));
        const b = neuronPos(k.layer, k.n), s = DIRECT * (0.2 + 0.8 * k.z / zmax) * (on ? 1 : DEAD), cl = linkCol(k).multiplyScalar(s);
        lp.push(a.x, a.y, a.z, b.x, b.y, b.z); lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    if (selected && !hoverCon && neuronHeads.length) {
      const o = neuronPos(selected.layer, selected.n), hz = Math.max(...neuronHeads.map(k => k.z));
      for (const k of neuronHeads) {
        const b = headMid(k.layer, k.head), cl = HEAD_LINK.clone().multiplyScalar(DIRECT * (0.2 + 0.8 * k.z / hz));
        lp.push(o.x, o.y, o.z, b.x, b.y, b.z); lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    conGeo.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
    conGeo.setAttribute("color", new THREE.Float32BufferAttribute(lc, 3));
    render();
  }

  // layer labels: text drawn in each layer's plane, just above the top-right corner of its neuron sheet
  const LABEL_H = Number(new URLSearchParams(location.search).get("labelsize")) || 0.2;
  const labels = [];
  for (let l = 0; l < L; l++) {
    const cv = document.createElement("canvas"), ctx = cv.getContext("2d"), font = "600 64px ui-monospace, Menlo, monospace";
    ctx.font = font; const tw = Math.ceil(ctx.measureText(`L${l}`).width) + 8;
    cv.width = tw; cv.height = 80;
    ctx.font = font; ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(`L${l}`, 4, 42);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
    const w = LABEL_H * tw / 80;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide, color: 0xc8d4da });
    mat.userData.base = 0.85;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, LABEL_H), mat);
    // the plane's +x runs toward -z after this turn, so the text reads toward the sheet's right edge seen from +x
    m.rotation.y = Math.PI / 2;
    m.position.set(layerX(l) + MLP_DX, SHEET_Y0 + SHEET + 0.08 + LABEL_H / 2, -SHEET / 2 + w / 2);
    m.userData = { layer: l };
    scene.add(m); labels.push(m);
  }
  const dataLayers = new Set();
  let unitsLayer = null;

  // annotations: short HTML labels pinned to neurons, re-projected every frame (Claude points things out with these)
  let notes = [];
  function placeNotes() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    for (const a of notes) {
      const v = neuronPos(a.layer, a.n).project(camera);
      a.el.style.display = v.z > 1 || layerMode !== null && a.layer !== layerMode ? "none" : "";
      a.el.style.left = `${(v.x + 1) / 2 * w}px`; a.el.style.top = `${(1 - v.y) / 2 * h}px`;
    }
  }
  function setNotes(items) {
    for (const a of notes) a.el.remove();
    notes = (items || []).map(it => {
      // a zero-size anchor at the neuron: the ring is centred on it, the text sits to its right
      const el = document.createElement("div"); el.className = "note";
      const t = document.createElement("span"); t.textContent = it.text || `L${it.layer} N${it.n}`; el.appendChild(t);
      labelsDiv.appendChild(el); return { layer: it.layer, n: it.n, el };
    });
    render();
  }

  let pending = false;
  function render() {
    if (pending) return; pending = true;
    requestAnimationFrame(() => {
      pending = false;
      for (const m of [pointMat, hiMat, markPts.material, webPts.material]) { m.uniforms.ref.value = homeDist; m.uniforms.tdist.value = curDist(); }
      applyFade();
      renderer.render(scene, camera); placeNotes();
    });
  }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    webMat.resolution.set(w, h);
    const narrow = w <= 760, li = narrow ? 0 : insets.left, ri = narrow ? 0 : insets.right;
    const availW = Math.max(200, w - li - ri);
    camera.aspect = w / h;
    if (fitPerPx === null) {
      // distance at which the model's bounding sphere fits the visible gap, then place the camera there once
      const center = home.target, r = Math.max(...bounds.map(b => b.distanceTo(center)));
      const vHalf = THREE.MathUtils.degToRad(camera.fov / 2), hHalf = Math.atan(Math.tan(vHalf) * availW / h);
      homeDist = r / Math.sin(Math.min(vHalf, hHalf)) * 0.82;
      camera.position.copy(home.target).addScaledVector(home.dir, homeDist);
      controls.update();
      fitPerPx = homeDist;
    }
    // the orbit target (and vanishing point) sits in the middle of the visible gap between the list and the neuron
    // panel, or at ?targetx= of the canvas width
    const tx = TARGET_X ? w * TARGET_X : li + availW / 2;
    camera.setViewOffset(w, h, w / 2 - tx, 0, w, h);
    camera.updateProjectionMatrix(); render();
  }
  new ResizeObserver(resize).observe(canvas);
  controls.addEventListener("change", render);

  // picking
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.05;
  const mouse = new THREE.Vector2();
  function pick(ev) {
    const r = canvas.getBoundingClientRect();
    mouse.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(mouse, camera);
    // hits come sorted by depth; on a sheet seen at an angle several neighbours fall within the threshold, so take
    // the frontmost layer's hits and pick the point closest to the ray
    // a head is under the pointer when its line is within a few pixels on screen (on the nearest visible slab that
    // has one); it wins over a neuron only when its slab is in front of that neuron
    const px = v => { v.project(camera); return Math.hypot((v.x + 1) / 2 * r.width - (ev.clientX - r.left), (1 - v.y) / 2 * r.height - (ev.clientY - r.top)); };
    let headHit = null;
    // the open head's dot, where its links meet
    if (headSrc && headSel && px(headSrc.clone()) < 8) {
      const sl = slabs[headSel.layer].userData;
      headHit = { type: "head", layer: headSel.layer, full: sl.full, head: headSel.head, distance: ray.ray.origin.distanceTo(headSrc) };
    }
    for (const sh of headHit ? [] : ray.intersectObjects(slabs)) {
      if (sh.object.material.opacity <= 0.005) continue;  // faded slabs don't pick
      const { layer, full, heads } = sh.object.userData;
      // where the ray crosses the face the head lines are drawn on
      const lx = sh.object.position.x + sh.object.geometry.parameters.width / 2;
      if (Math.abs(ray.ray.direction.x) < 1e-6) continue;
      const on = ray.ray.origin.clone().addScaledVector(ray.ray.direction, (lx - ray.ray.origin.x) / ray.ray.direction.x);
      if (Math.abs(on.z) > HEAD_W) continue;  // past the ends of the lines
      const head = Math.floor((0.5 - (on.y - SHEET_Y0 - SHEET / 2) / HEAD_SPAN) * heads);
      if (head < 0 || head >= heads) continue;
      if (px(new THREE.Vector3(lx, headY(head, heads), on.z)) < 6) { headHit = { type: "head", layer, full, head, distance: sh.distance }; break; }
    }
    const hits = ray.intersectObject(points);
    if (hits.length && !(headHit && headHit.distance < hits[0].distance)) {
      // the frontmost layer that is actually visible at the current fade, then the point closest to the ray in it
      const layers = [...new Set(hits.map(h => Math.floor(h.index / N)))];
      const front = layers.find(l => layerVisibility(l) > 0.5) ?? layers.reduce((a, b) => layerVisibility(b) > layerVisibility(a) ? b : a);
      let best = null;
      for (const h of hits) if (Math.floor(h.index / N) === front && (!best || h.distanceToRay < best.distanceToRay)) best = h;
      return { type: "neuron", layer: front, n: best.index % N };
    }
    if (headHit) return headHit;
    const lb = ray.intersectObjects(labels)[0];
    if (lb) return { type: "block", layer: lb.object.userData.layer };
    const b = ray.intersectObjects(slabs).find(x => x.object.material.opacity > 0.005);
    if (b) return { type: "block", ...b.object.userData };
    return null;
  }
  let moveQueued = null, downAt = null;
  canvas.addEventListener("pointermove", ev => {
    if (moveQueued) { moveQueued = ev; return; }
    moveQueued = ev;
    requestAnimationFrame(() => {
      const e = moveQueued; moveQueued = null;
      const p = pick(e);
      const h = p && p.type === "neuron" ? { layer: p.layer, n: p.n } : null;
      if (JSON.stringify(h) !== JSON.stringify(hovered)) { hovered = h; refreshHighlights(); }
      const hh = p && p.type === "head" ? { layer: p.layer, head: p.head } : null;
      if (JSON.stringify(hh) !== JSON.stringify(headHover)) { headHover = hh; paintHeads(); render(); }
      cb.onHover(p, e.clientX, e.clientY);
    });
  });
  canvas.addEventListener("pointerleave", () => { hovered = null; refreshHighlights(); if (headHover) { headHover = null; paintHeads(); render(); } cb.onHover(null); });
  canvas.addEventListener("pointerdown", ev => { downAt = [ev.clientX, ev.clientY]; });
  canvas.addEventListener("pointerup", ev => {
    if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
    const p = pick(ev);
    if (p && p.type === "neuron") cb.onPick(p.layer, p.n);
    else if (p && p.type === "head") cb.onPickHead && cb.onPickHead(p.layer, p.head);
    else if (p && p.type === "block") cb.onPickLayer && cb.onPickLayer(p.layer);
  });

  // the closer the camera is to the orbit target, the more the layers away from the target's x fade out
  const fade = { fx: 0, fw: 1, amt: 0, side: 1 };
  const layerVisibility = l => {
    const dx = layerX(l) + MLP_DX - fade.fx, d = dx / fade.fw;
    if (dx * fade.side / GAP > 0.45) return 1 - fade.amt;
    return 1 - 0.75 * fade.amt * (1 - Math.exp(-d * d));
  };
  let layerMode = null, saved = null, lastMarks = [];
  function enterLayer(l) {
    if (layerMode === null) saved = { pos: camera.position.clone(), tgt: controls.target.clone() };
    layerMode = l;
    const c = new THREE.Vector3(layerX(l) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12;
    controls.target.copy(c); camera.position.copy(c).add(new THREE.Vector3(-dist, 0, 0));
    controls.enableRotate = false;
    controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.update(); render(); cb.onLayerMode && cb.onLayerMode(l);
  }
  function exitLayer() {
    if (layerMode === null) return;
    layerMode = null;
    controls.enableRotate = true;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    camera.position.copy(saved.pos); controls.target.copy(saved.tgt);
    controls.update(); render(); cb.onLayerMode && cb.onLayerMode(null);
  }

  function applyFade() {
    if (layerMode !== null) {
      const fx = layerX(layerMode) + MLP_DX;
      pointMat.uniforms.focusX.value = fx; pointMat.uniforms.fadeW.value = 0.05; pointMat.uniforms.fadeAmt.value = 1;
      fade.fx = fx; fade.fw = 0.05; fade.amt = 1;
      for (const slab of slabs) { slab.material.opacity = 0; slab.userData.edges.material.opacity = 0; }
      for (let l = 0; l < L; l++) labels[l].material.opacity = 0;
      for (const m of fadeLines) m.opacity = 0;
      dataOutline.visible = false;
      return;
    }
    dataOutline.visible = true;
    // continuous in zoom: eased on log-distance from the overview (no fade) to a sheet filling the view (full fade)
    const dist = camera.position.distanceTo(controls.target);
    const far = Math.log(homeDist * 0.7), near = Math.log(homeDist * 0.07);
    const amt = THREE.MathUtils.smoothstep(THREE.MathUtils.clamp((far - Math.log(dist)) / (far - near), 0, 1), 0, 1);
    // the focus is the layer whose sheet centre lands closest to the middle of the view on screen
    const tv = controls.target.clone().project(camera), pv = new THREE.Vector3();
    let fx = controls.target.x, best = Infinity;
    for (let l = 0; l < L; l++) {
      pv.set(layerX(l) + MLP_DX, SHEET_Y0 + SHEET / 2, 0).project(camera);
      const dd = Math.hypot((pv.x - tv.x) * camera.aspect, pv.y - tv.y);
      if (pv.z < 1 && dd < best) { best = dd; fx = layerX(l) + MLP_DX; }
    }
    const fw = GAP * (0.85 + 2.5 * (1 - amt));
    const side = Math.sign(camera.position.x - fx) || 1;
    pointMat.uniforms.focusX.value = fx; pointMat.uniforms.fadeW.value = fw; pointMat.uniforms.fadeAmt.value = amt;
    pointMat.uniforms.camSide.value = side;
    fade.fx = fx; fade.fw = fw; fade.amt = amt; fade.side = side;
    for (const slab of slabs) {
      const d = (slab.position.x - fx) / fw, f = 1 - amt * (1 - Math.exp(-d * d));
      slab.material.opacity = slab.userData.baseOpacity * f; slab.userData.edges.material.opacity = f;
    }
    for (let l = 0; l < L; l++) {
      const d = (layerX(l) + MLP_DX - fx) / fw;
      labels[l].material.opacity = labels[l].material.userData.base * (1 - 0.5 * amt * (1 - Math.exp(-d * d)));  // other layers keep half
    }
    for (const m of fadeLines) m.opacity = m.userData.base * (1 - 0.9 * amt);
  }

  function flyTo(target, dist) {
    const from = controls.target.clone(), to = target.clone();
    const dir = camera.position.clone().sub(controls.target), d0 = dir.length();
    dir.normalize();
    const t0 = performance.now();
    (function step(t) {
      const k = Math.min(1, (t - t0) / 450), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, to, e);
      camera.position.copy(controls.target).addScaledVector(dir, d0 + (dist - d0) * e);
      controls.update(); render();
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }
  const curDist = () => camera.position.distanceTo(controls.target);

  // face a layer's neuron sheet head-on, close enough that the neighbours fade away
  function viewLayer(layer) {
    const c = new THREE.Vector3(layerX(layer) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
    const from = controls.target.clone(), p0 = camera.position.clone();
    const p1 = c.clone().add(new THREE.Vector3(dist, 0.0001, 0)), t0 = performance.now();
    (function step(t) {
      const k = Math.min(1, (t - t0) / 600), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, c, e); camera.position.lerpVectors(p0, p1, e);
      controls.update(); render();
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }

  return {
    neuronPos,
    // heads lit by level (0..1 per head, per layer; null clears), and the selected head
    setHeadLevels(levels) { headLevels = levels; paintHeads(); render(); },
    selectHead(layer, head) {
      headSel = layer == null ? null : { layer, head };
      if (!headSel && headSrc) { headSrc = null; connections = []; refreshHighlights(); }
      paintHeads(); render();
    },
    // an open head's strongest neuron links, drawn from the middle of its line like a neuron's direct links
    setHeadLinks(layer, head, list) {
      const i = (headStart[layer] + head) * 6;
      headSrc = headMid(layer, head);
      connections = list.map(k => ({ ...k, sign: 0 }));
      refreshHighlights();
    },
    // a hovered head's links, previewed like a hovered neuron's (null clears)
    setHoverHead(layer, head, list) {
      hoverCon = layer == null ? null : { from: headMid(layer, head), list: list.map(k => ({ ...k, sign: 0 })) };
      refreshHighlights();
    },
    // mark a head as hovered from outside the map (null clears)
    hoverHead(layer, head) {
      const hh = layer == null ? null : { layer, head };
      if (JSON.stringify(hh) !== JSON.stringify(headHover)) { headHover = hh; paintHeads(); render(); }
    },
    // the open neuron's strongest linked heads, drawn from the neuron to each head's line
    setNeuronHeads(list) { neuronHeads = list || []; refreshHighlights(); },
    // the swept neurons' colours show on one layer at a time (the current one); the others go back to idle
    setUnits(layer, units) {
      if (unitsLayer !== null && unitsLayer !== layer) for (let n = 0; n < N; n++) PAL.idle.toArray(col, (unitsLayer * N + n) * 3);
      unitsLayer = layer;
      dataLayers.add(layer);
      labels[layer].material.userData.base = 1; labels[layer].material.color.set(0xffffff);
      const zs = units.filter(u => u.kind === "neuron").map(u => u.z);
      const rnd = units.filter(u => u.kind === "random").map(u => u.z);
      const floor = rnd.length ? Math.max(...rnd) : Math.min(...zs), top = Math.max(...zs, floor + 1e-6);
      for (const u of units) {
        if (u.kind !== "neuron") continue;
        let c = PAL.swept;
        if (u.name) c = PAL.named;
        else if (u.z > floor) c = PAL.lo.clone().lerp(PAL.hi, Math.min(1, (u.z - floor) / (top - floor)));
        c.toArray(col, (layer * N + u.n) * 3);
      }
      geo.attributes.color.needsUpdate = true;
      const dl = [];
      for (const l of dataLayers) {
        const cs = sheetCorners(layerX(l) + MLP_DX);
        for (let i = 0; i < 4; i++) dl.push(...cs[i].toArray(), ...cs[(i + 1) % 4].toArray());
      }
      dataOutline.geometry.setAttribute("position", new THREE.Float32BufferAttribute(dl, 3));
      render();
    },
    // fly: bring the neuron into view without ever zooming out from where the user is
    select(layer, n, fly = true) {
      selected = layer == null ? null : { layer, n };
      connections = []; headSrc = null; neuronHeads = [];
      refreshHighlights();
      if (!fly || layer == null || layerMode !== null) return;
      // off screen (or behind the camera): fly to it, never zooming out from where the user is
      const v = neuronPos(layer, n).project(camera);
      if (v.z > 1 || Math.abs(v.x) > 0.9 || Math.abs(v.y) > 0.9) flyTo(neuronPos(layer, n), Math.min(curDist(), homeDist * 0.35));
    },
    setConnections(con) {
      connections = [...con.down.map(c => ({ ...c, side: "down" })), ...con.up.map(c => ({ ...c, side: "up" }))];
      refreshHighlights();
    },
    setHoverConnections(src, con) {
      hoverCon = src ? { src, list: [...con.down, ...con.up] } : null;
      refreshHighlights();
    },
    focusLayer() {},
    setLayout(cells) {
      layoutCells = cells;
      this.setMarks(lastMarks);
      for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
        const p = neuronPos(l, n), k = (l * N + n) * 3;
        pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z;
      }
      geo.attributes.position.needsUpdate = true;
      geo.computeBoundingSphere();
      buildWeb();
      refreshHighlights();
    },
    // edges: Int16Array (Int32Array for wide models), 6 per link (src layer, src n, dst layer, dst n, 10 * z * sign,
    // hop), or null to hide
    setWeb(edges) { webEdges = edges; buildWeb(); refreshHighlights(); },
    enterLayer, exitLayer, viewLayer, setNotes,
    // centre the view on a neuron, keeping the zoom unless it is wider than a few layers' worth
    focusNeuron(layer, n) { if (layerMode === null) flyTo(neuronPos(layer, n), Math.min(curDist(), homeDist * 0.35)); },
    // fly close to one neuron, keeping the current viewing direction
    flyToNeuron(layer, n, closeness = 0.22) { flyTo(neuronPos(layer, n), homeDist * closeness); },
    get layerMode() { return layerMode; },
    home() { flyTo(home.target, homeDist); },
    setMarks(marks) {
      lastMarks = marks;
      const p = [], c = [], up = new THREE.Color(1.0, 0.71, 0.28), down = new THREE.Color(1.0, 0.31, 0.6);
      for (const m of marks) {
        const v = neuronPos(m.layer, m.n), k = (m.sign === "-" ? down : up).clone().multiplyScalar(m.w ?? 1);
        p.push(v.x, v.y, v.z); c.push(k.r, k.g, k.b);
      }
      markGeo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
      markGeo.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
      render();
    },
    setInsets(left, right) { insets.left = left; insets.right = right; resize(); },
    hover(layer, n) { hovered = layer == null ? null : { layer, n }; refreshHighlights(); },
    debug() { const tv = controls.target.clone().project(camera); return { tgtScreenX: +((tv.x + 1) / 2).toFixed(3), cam: camera.position.toArray().map(v => +v.toFixed(2)), tgt: controls.target.toArray().map(v => +v.toFixed(2)), selected, hovered, n: connections.length, hover: hoverCon ? hoverCon.list.length : 0, dist: curDist(), homeDist, w: canvas.clientWidth, h: canvas.clientHeight, insets }; },
    render,
  };
}
