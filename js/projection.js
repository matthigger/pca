// Projection tab: n samples whose covariance (with 1/n) is exactly the
// S the student types, a draggable unit direction w, the scores
// c_i = wᵀx_i, the rebuilds x̂_i = c_i w and the errors e_i = x_i − x̂_i.
//
// S can be edited as a matrix or as eigenvalues and an eigenvector angle,
// S = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ. The samples are fixed standard draws z_i,
// re-whitened to mean 0 and covariance exactly I, mapped by the symmetric
// square root: x_i = S^½ z_i. S^½ moves continuously with S, so editing
// either form reshapes the cloud smoothly. The data mean is 0 = x̄.

const PN = 60;
const LMAX = 16;

const PRESETS = [
  ["ica", "[[5, 2], [2, 2]]", { a: 5, b: 2, d: 2 }],
  ["round", "round", { a: 3, b: 0, d: 3 }],
  ["uncorr", "uncorrelated", { a: 6, b: 0, d: 1 }],
  ["neg", "[[4, −3], [−3, 4]]", { a: 4, b: -3, d: 4 }],
  ["line", "nearly a line", matFromEig({ l1: 9, l2: 0.05, th: 60 * DEG })],
];

const P = {
  S: { a: 5, b: 2, d: 2 },
  // What the eigen editor shows: {l1, l2, th (radians)}. Kept apart from
  // S so typing a λ₂ above λ₁ does not swap the fields mid-edit; it is
  // re-sorted from S when an edit is committed.
  eig: null,
  phi: 110 * DEG,
  seed: 4,
  z: null,
  show: { eig: false, err: true, sw: false },
  spin: null,
};

/** Seeded uniform [0, 1) generator (mulberry32). */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** S = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ with v₁ = [cos th, sin th], v₂ ⟂ v₁. */
function matFromEig(e) {
  const c = Math.cos(e.th), s = Math.sin(e.th);
  return { a: e.l1 * c * c + e.l2 * s * s, b: (e.l1 - e.l2) * c * s,
    d: e.l1 * s * s + e.l2 * c * c };
}

/** f(S) = R diag(f(λ)) Rᵀ for a symmetric 2 x 2 S = {a, b, d}. */
function matFn(M, f) {
  const e = eig2(M.a, M.b, M.d);
  return matFromEig({ l1: f(Math.max(e.l1, 0)), l2: f(Math.max(e.l2, 0)),
    th: isNaN(e.theta) ? 0 : e.theta });
}

/** Angle th folded into (−90°, 90°], so v₁ points rightward. */
function foldTh(th) {
  let t = ((th + Math.PI / 2) % Math.PI + Math.PI) % Math.PI - Math.PI / 2;
  if (t <= -Math.PI / 2 + 1e-12) t += Math.PI;
  return t;
}

/** Sorted eigen form of the current S, keeping the old angle if round. */
function eigOfS() {
  const e = eig2(P.S.a, P.S.b, P.S.d);
  const th = isNaN(e.theta) ? (P.eig ? foldTh(P.eig.th) : 0) : e.theta;
  return { l1: e.l1, l2: e.l2, th };
}

/** Draw PN standard normals and whiten them: mean 0, covariance I. */
function drawZ(seed) {
  const r = rng(seed), z = [];
  for (let i = 0; i < PN; i++) {
    const u = 1 - r(), v = r(), rad = Math.sqrt(-2 * Math.log(u));
    z.push([rad * Math.cos(2 * Math.PI * v), rad * Math.sin(2 * Math.PI * v)]);
  }
  const m = z.reduce((s, q) => add(s, q), [0, 0]).map(v => v / PN);
  const zc = z.map(q => [q[0] - m[0], q[1] - m[1]]);
  const C = { a: 0, b: 0, d: 0 };
  for (const q of zc) {
    C.a += q[0] * q[0] / PN;
    C.b += q[0] * q[1] / PN;
    C.d += q[1] * q[1] / PN;
  }
  const W = matFn(C, l => 1 / Math.sqrt(l));
  return zc.map(q => [W.a * q[0] + W.b * q[1], W.b * q[0] + W.d * q[1]]);
}

/** The samples x_i = S^½ z_i. */
function samples() {
  const R = matFn(P.S, Math.sqrt);
  return P.z.map(q => [R.a * q[0] + R.b * q[1], R.b * q[0] + R.d * q[1]]);
}

const Sx = v => [P.S.a * v[0] + P.S.b * v[1], P.S.b * v[0] + P.S.d * v[1]];

/** Why S = {a, b, d} cannot be used, or "" if it can. */
function sProblem(M) {
  if (![M.a, M.b, M.d].every(Number.isFinite)) return "Enter a number.";
  const e = eig2(M.a, M.b, M.d);
  if (e.l2 < -1e-9) {
    return `This <i>S</i> has eigenvalues ${fmt(e.l1)} and ${fmt(e.l2)}.
      A variance <b>w</b><sup>T</sup><i>S</i><b>w</b> cannot be negative,
      so no data has this covariance: it needs <i>S</i><sub>00</sub>,
      <i>S</i><sub>11</sub> &ge; 0 and |<i>S</i><sub>01</sub>| &le;
      &radic;(<i>S</i><sub>00</sub><i>S</i><sub>11</sub>).`;
  }
  if (e.l1 > LMAX + 1e-9) {
    return `Keep the eigenvalues at most ${LMAX} (this <i>S</i> has
      <i>&lambda;</i><sub>1</sub> = ${fmt(e.l1)}) so the cloud fits on
      the plot.`;
  }
  return "";
}

// ------------------------------------------------------------- drawing

const pPlane = document.getElementById("p-plane");
const pHist = document.getElementById("p-hist");
const pDial = document.getElementById("p-dial");
pPlane.setAttribute("viewBox", `0 0 ${PLANE.w} ${PLANE.h}`);

function planeRender() {
  const X = samples(), w = unit(P.phi), e = P.eig;
  let s = `<defs><clipPath id="p-clip"><rect x="${PLANE.x0}"
    y="${PLANE.y0}" width="${PLANE.x1 - PLANE.x0}"
    height="${PLANE.y1 - PLANE.y0}"/></clipPath></defs>${axesSvg()}
    <g clip-path="url(#p-clip)">`;
  if (P.show.eig) {
    const v1 = unit(e.th), v2 = perp(v1);
    const r1 = Math.sqrt(Math.max(e.l1, 0)), r2 = Math.sqrt(Math.max(e.l2, 0));
    let d = "";
    for (let t = 0; t <= 96; t++) {
      const a = 2 * Math.PI * t / 96;
      const q = add(mul(v1, 2 * r1 * Math.cos(a)), mul(v2, 2 * r2 * Math.sin(a)));
      d += `${t ? "L" : "M"}${px(q)}`;
    }
    s += `<path class="ellipse" d="${d}Z"/>`;
  }
  const L = 12;
  s += segSvg(mul(w, -L), mul(w, L), `class="wline"`);
  s += segSvg(mul(w, -L), mul(w, L), `class="hit" data-h="line"`);
  const xh = X.map(x => mul(w, dot(w, x)));
  if (P.show.err) {
    X.forEach((x, i) => { s += segSvg(x, xh[i], `class="err"`); });
  }
  for (const q of xh) {
    s += `<circle class="proj" cx="${S.sx(q[0])}" cy="${S.sy(q[1])}" r="3.6"/>`;
  }
  for (const q of X) {
    s += `<circle class="pt" cx="${S.sx(q[0])}" cy="${S.sy(q[1])}" r="5.5"/>`;
  }
  s += "</g>";
  if (P.show.eig) {
    const v1 = unit(e.th), v2 = perp(v1);
    const t1 = mul(v1, Math.sqrt(Math.max(e.l1, 0)));
    const t2 = mul(v2, Math.sqrt(Math.max(e.l2, 0)));
    s += arrowSvg([0, 0], t1, "eig") + arrowSvg([0, 0], t2, "eig light");
    for (const [h, t, lab] of [["e1", t1, "1"], ["e2", t2, "2"]]) {
      s += `<circle class="tip eig" data-h="${h}" cx="${S.sx(t[0])}"
        cy="${S.sy(t[1])}" r="10"/>`;
      const o = mul(t, 1 + 0.55 / Math.max(norm(t), 0.3));
      s += `<text class="lbl green" x="${S.sx(o[0]) - 16}"
        y="${S.sy(o[1]) + 6}">√<tspan class="sym">λ</tspan>${
        lab === "1" ? "₁" : "₂"}<tspan class="vec">v</tspan>${
        lab === "1" ? "₁" : "₂"}</text>`;
    }
  }
  if (P.show.sw) {
    const sw = Sx(w);
    s += arrowSvg([0, 0], sw, "sw");
    const o = mul(sw, 1 + 0.5 / Math.max(norm(sw), 0.3));
    s += `<text class="lbl amber" x="${S.sx(o[0]) - 12}"
      y="${S.sy(o[1]) + 6}"><tspan class="sym">S</tspan><tspan
      class="vec">w</tspan></text>`;
  }
  s += arrowSvg([0, 0], w, "w", 12);
  s += `<circle class="tip" data-h="w" cx="${S.sx(w[0])}"
    cy="${S.sy(w[1])}" r="11"/>`;
  const o = mul(w, 1.55);
  s += `<text class="lbl violet" x="${S.sx(o[0]) - 7}" y="${S.sy(o[1]) + 6}"
    ><tspan class="vec">w</tspan></text>`;
  s += `<text class="lbl small ink" x="${S.sx(0) + 8}" y="${S.sy(0) + 18}"
    >x̄</text>`;
  pPlane.innerHTML = s;
  return X;
}

const HIST = { w: 700, h: 184, top: 26, bot: 140 };
pHist.setAttribute("viewBox", `0 0 ${HIST.w} ${HIST.h}`);

/** Histogram of the scores c_i, binned on the plane's own x scale. */
function histRender(c, v) {
  const bw = 0.5, lo = PLANE.dom[0], nb = (PLANE.dom[1] - lo) / bw;
  const cnt = new Array(nb).fill(0);
  for (const ci of c) cnt[clamp(Math.floor((ci - lo) / bw), 0, nb - 1)]++;
  const pdf = t => v > 1e-9 ?
    PN * bw * Math.exp(-t * t / (2 * v)) / Math.sqrt(2 * Math.PI * v) : 0;
  const ymax = Math.max(10, ...cnt, Math.min(pdf(0), PN) * 1.05);
  const sy = n => HIST.bot - n / ymax * (HIST.bot - HIST.top);
  let s = `<line class="grid zero" x1="${PLANE.x0}" x2="${PLANE.x1}"
    y1="${HIST.bot}" y2="${HIST.bot}"/>`;
  cnt.forEach((n, b) => {
    if (!n) return;
    const x0 = S.sx(lo + b * bw);
    s += `<rect class="bar" x="${x0 + 0.5}" width="${bw * S.k - 1}"
      y="${sy(n)}" height="${HIST.bot - sy(n)}"/>`;
  });
  if (v > 1e-9) {
    let d = "";
    for (let t = lo; t <= PLANE.dom[1] + 1e-9; t += 0.1) {
      d += `${d ? "L" : "M"}${S.sx(t).toFixed(1)},${sy(Math.min(pdf(t),
        ymax)).toFixed(1)}`;
    }
    s += `<path class="curve" d="${d}"/>`;
  }
  for (let t = lo; t <= PLANE.dom[1]; t += 2) {
    s += `<text class="tick" x="${S.sx(t)}" y="${HIST.bot + 15}"
      text-anchor="middle">${fmt(t, 0)}</text>`;
  }
  s += `<text class="axis-label sm" x="${(PLANE.x0 + PLANE.x1) / 2}"
    y="${HIST.bot + 30}" text-anchor="middle">score <tspan class="sym"
    >c</tspan> = <tspan class="vec">w</tspan>ᵀ(<tspan class="vec">x</tspan>
    − <tspan class="vec">x̄</tspan>), the position along <tspan
    class="vec">w</tspan>'s line</text>
    <text class="axis-label sm" x="${PLANE.x0}" y="16">histogram of the
    ${PN} scores; curve: normal with variance ${fmt(v)}</text>`;
  pHist.innerHTML = s;
}

const DIAL = { w: 440, h: 236, x0: 44, x1: 428, y0: 14, y1: 196 };
pDial.setAttribute("viewBox", `0 0 ${DIAL.w} ${DIAL.h}`);
const dsx = a => DIAL.x0 + a / 180 * (DIAL.x1 - DIAL.x0);

/** wᵀSw (kept), tr S − wᵀSw (lost) and tr S against the angle of w. */
function dialRender() {
  const tr = P.S.a + P.S.d, top = Math.max(tr, 0.1) * 1.12;
  const sy = v => DIAL.y1 - v / top * (DIAL.y1 - DIAL.y0);
  const f = a => { const w = unit(a * DEG); return dot(w, Sx(w)); };
  let kept = "", lost = "";
  for (let a = 0; a <= 180; a += 2) {
    const k = f(a);
    kept += `${a ? "L" : "M"}${dsx(a).toFixed(1)},${sy(k).toFixed(1)}`;
    lost += `${a ? "L" : "M"}${dsx(a).toFixed(1)},${sy(tr - k).toFixed(1)}`;
  }
  let s = "";
  for (let a = 0; a <= 180; a += 45) {
    s += `<line class="grid" x1="${dsx(a)}" x2="${dsx(a)}" y1="${DIAL.y0}"
      y2="${DIAL.y1}"/><text class="tick" x="${dsx(a)}" y="${DIAL.y1 + 15}"
      text-anchor="middle">${a}°</text>`;
  }
  const step = top > 12 ? 4 : top > 6 ? 2 : top > 2.4 ? 1 : 0.5;
  for (let v = 0; v <= top; v += step) {
    s += `<line class="grid${v ? "" : " zero"}" x1="${DIAL.x0}"
      x2="${DIAL.x1}" y1="${sy(v)}" y2="${sy(v)}"/><text class="tick"
      x="${DIAL.x0 - 6}" y="${sy(v) + 4}" text-anchor="end">${fmt(v,
      step < 1 ? 1 : 0)}</text>`;
  }
  s += `<line class="total" x1="${DIAL.x0}" x2="${DIAL.x1}" y1="${sy(tr)}"
    y2="${sy(tr)}"/><text class="zl" x="${DIAL.x1}" y="${sy(tr) - 5}"
    text-anchor="end">tr S = ${fmt(tr)}</text>
    <path class="c-lost" d="${lost}"/><path class="c-kept" d="${kept}"/>`;
  const e = eigOfS();
  const a1 = ((e.th / DEG) % 180 + 180) % 180, a2 = (a1 + 90) % 180;
  if (Math.abs(e.l1 - e.l2) > 1e-6) {
    s += `<circle class="eigdot" cx="${dsx(a1)}" cy="${sy(e.l1)}" r="5"/>
      <text class="zl green" x="${dsx(a1)}" y="${sy(e.l1) - 9}"
      text-anchor="middle">λ₁ at v₁</text>
      <circle class="eigdot" cx="${dsx(a2)}" cy="${sy(e.l2)}" r="5"/>
      <text class="zl green" x="${dsx(a2)}" y="${sy(e.l2) - 9}"
      text-anchor="middle">λ₂ at v₂</text>`;
  }
  const a = ((P.phi / DEG) % 180 + 180) % 180, k = f(a);
  s += `<line class="c-now" x1="${dsx(a)}" x2="${dsx(a)}" y1="${DIAL.y0}"
    y2="${DIAL.y1}"/><circle class="kdot" cx="${dsx(a)}" cy="${sy(k)}"
    r="5"/><circle class="ldot" cx="${dsx(a)}" cy="${sy(tr - k)}" r="4"/>
    <text class="axis-label sm" x="${(DIAL.x0 + DIAL.x1) / 2}"
    y="${DIAL.y1 + 32}" text-anchor="middle">angle of <tspan class="vec"
    >w</tspan>; <tspan class="kept-t">kept wᵀSw</tspan>, <tspan
    class="lost-t">lost tr S − wᵀSw</tspan></text>`;
  pDial.innerHTML = s;
}

// ------------------------------------------------------------ readouts

const sIn = ["s00", "s01", "s10", "s11"].map(id => document.getElementById(id));
const eIn = {
  l1: document.getElementById("l1"), l1r: document.getElementById("l1r"),
  l2: document.getElementById("l2"), l2r: document.getElementById("l2r"),
  th: document.getElementById("th"), thr: document.getElementById("thr"),
};
const sWarn = document.getElementById("s-warn");
const eWarn = document.getElementById("eig-warn");

/** Input text for v: at most 2 decimals, no trailing zeros. */
const inTxt = v => String(Number(v.toFixed(2)));

/** Set an input's value unless the student is typing in it. */
function setIn(el, v) {
  if (document.activeElement !== el) el.value = v;
}

function fieldsRender() {
  const { a, b, d } = P.S;
  [a, b, b, d].forEach((v, i) => setIn(sIn[i], inTxt(v)));
  const e = P.eig;
  setIn(eIn.l1, inTxt(e.l1));
  setIn(eIn.l1r, e.l1);
  setIn(eIn.l2, inTxt(e.l2));
  setIn(eIn.l2r, e.l2);
  setIn(eIn.th, inTxt(e.th / DEG));
  setIn(eIn.thr, e.th / DEG);
  eWarn.hidden = !(e.l2 > e.l1 + 1e-9);
  eWarn.innerHTML = `<i>&lambda;</i><sub>2</sub> &gt;
    <i>&lambda;</i><sub>1</sub> for now: PCA sorts them, so the names
    swap when you let go.`;
  const v1 = unit(e.th), v2 = perp(v1);
  document.getElementById("eig-read").innerHTML =
    row("<b>v</b><sub>1</sub>, <b>v</b><sub>2</sub>",
      `${fvec(v1)}, ${fvec(v2)}`) +
    row("<i>S</i><b>v</b><sub>1</sub> = <i>&lambda;</i><sub>1</sub><b>v</b><sub>1</sub>",
      `${fvec(Sx(v1))} = ${fmt(e.l1)} <b>v</b><sub>1</sub>`) +
    row("<i>S</i><b>v</b><sub>2</sub> = <i>&lambda;</i><sub>2</sub><b>v</b><sub>2</sub>",
      `${fvec(Sx(v2))} = ${fmt(e.l2)} <b>v</b><sub>2</sub>`) +
    row("tr <i>S</i> = <i>&lambda;</i><sub>1</sub> + <i>&lambda;</i><sub>2</sub>",
      fmt(a + d)) +
    row("det <i>S</i> = <i>&lambda;</i><sub>1</sub><i>&lambda;</i><sub>2</sub>",
      fmt(a * d - b * b));
  for (const b2 of document.querySelectorAll("#p-presets button")) {
    const M = PRESETS.find(p => p[0] === b2.dataset.v)[2];
    b2.setAttribute("aria-checked", Math.abs(M.a - a) + Math.abs(M.b - b) +
      Math.abs(M.d - d) < 1e-6);
  }
}

function readRender(X) {
  const w = unit(P.phi), c = X.map(x => dot(w, x));
  const kept = c.reduce((s, v) => s + v * v, 0) / PN;
  const lost = X.reduce((s, x, i) => {
    const e = add(x, mul(w, -c[i]));
    return s + dot(e, e);
  }, 0) / PN;
  const tr = P.S.a + P.S.d, wsw = dot(w, Sx(w));
  let r = row("<b>w</b> (unit)", `${fvec(w)} at ${fmt(((P.phi / DEG) % 360 +
      360) % 360, 0)}°`) +
    row("kept: (1/<i>n</i>)&sum; <i>c<sub>i</sub></i><sup>2</sup>",
      `<span class="kept-t">${fmt(kept, 3)}</span>`) +
    row("&nbsp;&nbsp;= <b>w</b><sup>T</sup><i>S</i><b>w</b>",
      `<span class="kept-t">${fmt(wsw, 3)}</span>`) +
    row("lost: (1/<i>n</i>)&sum; ‖<b>e</b><sub><i>i</i></sub>‖<sup>2</sup>",
      `<span class="lost-t">${fmt(lost, 3)}</span>`) +
    row("kept + lost = tr <i>S</i>", `${fmt(kept + lost, 3)} = ${fmt(tr, 3)}`);
  if (P.show.sw) {
    const sw = Sx(w), ns = norm(sw);
    const ang = ns < 1e-9 ? 0 :
      Math.acos(clamp(dot(sw, w) / ns, -1, 1)) / DEG;
    r += row("<i>S</i><b>w</b>", fvec(sw)) +
      row("angle from <b>w</b> to <i>S</i><b>w</b>", `${fmt(ang, 1)}°${
        ang < 0.5 ? " (eigenvector!)" : ""}`);
  }
  document.getElementById("p-read").innerHTML = r;
  const f = tr > 0 ? kept / tr : 1;
  document.getElementById("p-bar").innerHTML =
    `<div class="bar2"><span class="k" style="width:${(100 * f).toFixed(1)}%"
    ></span><span class="l"></span></div><div class="muted">the line keeps
    ${(100 * f).toFixed(1)}% of tr <i>S</i></div>`;
  return c;
}

function projRender() {
  const X = planeRender();
  const c = readRender(X);
  const w = unit(P.phi);
  histRender(c, dot(w, Sx(w)));
  dialRender();
  fieldsRender();
  document.getElementById("p-key-eig").hidden = !P.show.eig;
  document.getElementById("p-key-sw").hidden = !P.show.sw;
}

// ------------------------------------------------------------- editing

/** Commit an edit: re-sort the eigen form from S. */
function commitEig() {
  P.eig = eigOfS();
  projRender();
}

/** Try S = M; warn and keep the old S if it is not a covariance. */
function trySetS(M) {
  const why = sProblem(M);
  sWarn.hidden = !why;
  sWarn.innerHTML = why;
  if (why) return false;
  P.S = M;
  P.eig = eigOfS();
  projRender();
  return true;
}

sIn.forEach((el, i) => {
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (!Number.isFinite(v)) return;
    const M = { ...P.S };
    if (i === 0) M.a = v;
    else if (i === 3) M.d = v;
    else M.b = v;
    trySetS(M);
  });
  el.addEventListener("change", () => {
    el.blur();
    projRender();
  });
});

/** Eigen form from the editor's fields after one of them moved. */
function eigInput(src) {
  const twin = { l1: "l1r", l1r: "l1", l2: "l2r", l2r: "l2", th: "thr",
    thr: "th" }[src];
  const v = parseFloat(eIn[src].value);
  if (!Number.isFinite(v)) return;
  eIn[twin].value = v;
  const e = { ...P.eig };
  if (src.startsWith("l1")) e.l1 = clamp(v, 0, LMAX);
  else if (src.startsWith("l2")) e.l2 = clamp(v, 0, LMAX);
  else e.th = clamp(v, -90, 90) * DEG;
  P.eig = e;
  P.S = matFromEig(e);
  sWarn.hidden = true;
  projRender();
}

for (const k of Object.keys(eIn)) {
  eIn[k].addEventListener("input", () => eigInput(k));
  eIn[k].addEventListener("change", () => {
    if (eIn[k].type === "number") eIn[k].blur();
    commitEig();
  });
}

segmented(document.getElementById("p-presets"),
  PRESETS.map(p => [p[0], p[1]]), v => {
    trySetS({ ...PRESETS.find(p => p[0] === v)[2] });
  });

document.getElementById("p-resample").onclick = () => {
  P.seed++;
  P.z = drawZ(P.seed);
  projRender();
};

for (const [id, key] of [["p-eig", "eig"], ["p-err", "err"], ["p-sw", "sw"]]) {
  const el = document.getElementById(id);
  el.checked = P.show[key];
  el.onchange = () => { P.show[key] = el.checked; projRender(); };
}

function stopSpin() {
  if (!P.spin) return;
  cancelAnimationFrame(P.spin);
  P.spin = null;
  document.getElementById("p-spin").textContent = "Sweep w";
}

document.getElementById("p-v1").onclick = () => {
  stopSpin();
  P.phi = eigOfS().th;
  projRender();
};
document.getElementById("p-v2").onclick = () => {
  stopSpin();
  P.phi = eigOfS().th + Math.PI / 2;
  projRender();
};
document.getElementById("p-spin").onclick = () => {
  if (P.spin) { stopSpin(); return; }
  document.getElementById("p-spin").textContent = "Stop";
  let last = performance.now();
  const step = now => {
    P.phi += (now - last) / 1000 * 40 * DEG;
    last = now;
    projRender();
    P.spin = requestAnimationFrame(step);
  };
  P.spin = requestAnimationFrame(step);
};

let pDrag = null;
dragOn(pPlane, {
  down(h, q) {
    stopSpin();
    pDrag = h === "e1" || h === "e2" ? h : "w";
    this.move(q);
    return true;
  },
  move(q) {
    const p = [S.ix(q.x), S.iy(q.y)];
    if (norm(p) < 0.15) return;
    const t = Math.atan2(p[1], p[0]);
    if (pDrag === "w") {
      P.phi = t;
    } else {
      const l = Math.min(dot(p, p), LMAX);
      P.eig = pDrag === "e1" ? { ...P.eig, l1: l, th: t } :
        { ...P.eig, l2: l, th: t - Math.PI / 2 };
      P.S = matFromEig(P.eig);
      sWarn.hidden = true;
    }
    projRender();
  },
  up() {
    if (pDrag !== "w") {
      P.eig.th = foldTh(P.eig.th);
      commitEig();
    }
    pDrag = null;
  },
});

dragOn(pDial, {
  down(h, q) { stopSpin(); this.move(q); return true; },
  move(q) {
    P.phi = clamp((q.x - DIAL.x0) / (DIAL.x1 - DIAL.x0), 0, 1) * 180 * DEG;
    projRender();
  },
});

P.z = drawZ(P.seed);
P.eig = eigOfS();

/** Show the projection tab (called by main.js). */
function projShow() { projRender(); }
