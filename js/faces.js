// Eigenfaces tab: pick a dataset, fit PCA in the browser (js/pca.js),
// rebuild a picked image from its top k components, see the eigenvalues
// and their running sum, the mean and top eigenfaces, walk along
// w_1..w_4 in the mixer, or play Guess who with a hidden image.

const SETS = [["guesswho", "Guess Who"], ["olivetti", "Olivetti"],
  ["emoji", "Emoji"], ["cats", "Cats"]];
const N_EIG = 8, N_MIX = 4;

const F = {
  key: "guesswho",
  held: false,
  m: null,
  fit: null,
  fits: {},
  token: 0,
  sel: 0,
  c: null,
  k: 5,
  mixFrom: "mean",
  mix: new Array(N_MIX).fill(0),
  game: null,
};

const el = id => document.getElementById(id);
const fStatus = el("f-status");
const fK = el("f-k");
const fCurve = el("f-curve");
const fGal = el("f-gallery");

/** Load dataset key (if needed) and fit it, ignoring stale requests. */
async function useSet(key) {
  const token = ++F.token;
  F.key = key;
  segSet(el("f-sets"), key);
  gameReset();
  fStatus.textContent = "loading images…";
  let m;
  try {
    m = await loadSet(key);
  } catch (err) {
    fStatus.textContent = String(err);
    return;
  }
  if (token !== F.token) return;
  F.m = m;
  F.sel = 0;
  F.mix.fill(0);
  galleryBuild();
  el("f-credit").textContent = m.credit;
  el("f-held-n").textContent = m.holdout.length;
  await refit(token);
}

/** Fit PCA for the current set and hold-out choice (cached). */
async function refit(token = ++F.token) {
  const m = F.m, id = `${m.key}:${F.held}`;
  if (!F.fits[id]) {
    const train = trainIdx(m, F.held);
    fStatus.textContent = `computing PCA on ${train.length} images…`;
    await afterPaint();
    F.fits[id] = await fitPca(m, train, f => {
      if (token === F.token) {
        fStatus.textContent = `computing PCA on ${train.length} images… ${
          Math.round(100 * f)}%`;
      }
    });
  }
  if (token !== F.token) return;
  F.fit = F.fits[id];
  fStatus.textContent = `${m.n} images of ${m.w} × ${m.h}${
    m.ch === 3 ? " × 3 colors" : ""} = ${m.d} pixel values each`;
  fK.max = F.fit.rank;
  F.k = Math.min(F.k, F.fit.rank);
  eigsRender();
  mixBuild();
  select(F.sel);
}

/** Pick image i: its scores on every direction, then redraw. */
function select(i) {
  F.sel = i;
  F.c = scores(F.fit, imageOf(F.m, i));
  facesRender();
}

const isOut = i => F.held && F.m.holdout.includes(i);

function facesRender() {
  const { m, fit, k } = F;
  if (!fit) return;
  const x = imageOf(m, F.sel), xh = rebuild(fit, F.c, k);
  const orig = el("f-orig");
  if (F.game && !F.game.done) {
    hiddenImage(orig, m);
    el("f-orig-cap").innerHTML = "hidden: who is it?";
  } else {
    drawImage(orig, m, x);
    el("f-orig-cap").innerHTML = `<b>x</b>: ${m.labels[F.sel]}${
      isOut(F.sel) ? ` <span class="out-t">(held out)</span>` : ""}`;
  }
  drawImage(el("f-rec"), m, xh);
  let err = 0;
  for (let j = 0; j < m.d; j++) err += (xh[j] - x[j]) ** 2;
  el("f-rec-cap").innerHTML = `<b>x&#770;</b> from <i>k</i> = ${k}
    number${k === 1 ? "" : "s"} (instead of ${m.d})<br><span class="muted"
    >mean squared error per pixel ${(err / m.d).toFixed(5)}</span>`;
  fK.value = k;
  const kept = cumKept(k);
  el("f-kval").innerHTML = `<i>k</i> = ${k} of ${fit.rank} &middot; keeps
    ${(100 * kept).toFixed(1)}% of the variance${k === 0 ?
    " (<i>k</i> = 0 is the mean <b>x&#772;</b>)" : ""}`;
  curveRender();
  for (const t of fGal.children) {
    const i = +t.dataset.i;
    t.classList.toggle("on", i === F.sel && !(F.game && !F.game.done));
    t.classList.toggle("out", isOut(i));
  }
  el("f-gal-note").textContent = F.game && !F.game.done ?
    "click the hidden image's match" : F.held ?
    "orange: held out of the fit" : "";
  mixRender();
}

/** (λ_1 + ... + λ_k) / Σλ. */
function cumKept(k) {
  let s = 0;
  for (let j = 0; j < k; j++) s += F.fit.lam[j];
  return s / F.fit.total;
}

/** A gray "?" card the size of an image of m. */
function hiddenImage(cv, m) {
  cv.width = m.w;
  cv.height = m.h;
  const cx = cv.getContext("2d");
  cx.fillStyle = "#d9dde5";
  cx.fillRect(0, 0, m.w, m.h);
  cx.fillStyle = "#5d6677";
  cx.font = `bold ${Math.round(m.h * 0.6)}px system-ui, sans-serif`;
  cx.textAlign = "center";
  cx.textBaseline = "middle";
  cx.fillText("?", m.w / 2, m.h / 2 + 2);
}

// ---------------------------------------------- eigenvalue curve and k

const CV = { w: 700, h: 300, x0: 52, x1: 684, b0: 24, b1: 84, c0: 112,
  c1: 264 };
fCurve.setAttribute("viewBox", `0 0 ${CV.w} ${CV.h}`);

/** Bars of λ_j / Σλ over their running sum, the kept fraction at k. */
function curveRender() {
  const { fit, k } = F, r = fit.rank;
  const sx = j => CV.x0 + j / r * (CV.x1 - CV.x0);
  const top = fit.lam[0] / fit.total;
  const by = v => CV.b1 - v / top * (CV.b1 - CV.b0);
  const cy = v => CV.c1 - v * (CV.c1 - CV.c0);
  const bw = Math.max((CV.x1 - CV.x0) / r - 1, 0.6);
  let s = "";
  for (const q of [0, 0.25, 0.5, 0.75, 1]) {
    s += `<line class="grid${q === 0 ? " zero" : ""}" x1="${CV.x0}"
      x2="${CV.x1}" y1="${cy(q)}" y2="${cy(q)}"/><text class="tick"
      x="${CV.x0 - 6}" y="${cy(q) + 4}" text-anchor="end">${
      Math.round(q * 100)}%</text>`;
  }
  s += `<line class="grid zero" x1="${CV.x0}" x2="${CV.x1}" y1="${CV.b1}"
    y2="${CV.b1}"/><text class="tick" x="${CV.x0 - 6}" y="${CV.b0 + 8}"
    text-anchor="end">${(100 * top).toFixed(0)}%</text>`;
  for (let j = 0; j < r; j++) {
    const v = fit.lam[j] / fit.total, x = sx(j + 1) - bw;
    s += `<rect class="${j < k ? "bar on" : "bar off"}" x="${x.toFixed(2)}"
      width="${bw.toFixed(2)}" y="${by(v).toFixed(2)}"
      height="${(CV.b1 - by(v)).toFixed(2)}"/>`;
  }
  let d = `M${sx(0)},${cy(0)}`, acc = 0;
  for (let j = 0; j < r; j++) {
    acc += fit.lam[j] / fit.total;
    d += `L${sx(j + 1).toFixed(2)},${cy(acc).toFixed(2)}`;
  }
  const kept = cumKept(k);
  s += `<path class="c-kept" d="${d}"/>
    <line class="c-now" x1="${sx(k)}" x2="${sx(k)}" y1="${CV.b0}"
    y2="${CV.c1}"/><circle class="kdot" cx="${sx(k)}" cy="${cy(kept)}"
    r="5.5"/>
    <text class="zl" x="${Math.min(sx(k) + 8, CV.x1 - 70)}"
    y="${cy(kept) + (kept > 0.85 ? 16 : -8)}">${(100 * kept).toFixed(1)}%</text>`;
  const ticks = niceTicks(r);
  for (const t of ticks) {
    s += `<text class="tick" x="${sx(t)}" y="${CV.c1 + 15}"
      text-anchor="middle">${t}</text>`;
  }
  s += `<text class="axis-label sm" x="${CV.x1}" y="${CV.b0 - 8}"
    text-anchor="end">eigenvalues <tspan class="sym">λ</tspan><tspan
    baseline-shift="sub" font-size="9">j</tspan> / Σ<tspan class="sym"
    >λ</tspan>, the first <tspan class="sym">k</tspan> in color</text>
    <text class="axis-label sm" x="${CV.x1}" y="${CV.c0 - 8}"
    text-anchor="end">their
    running sum: fraction of variance kept</text>
    <text class="axis-label sm" x="${(CV.x0 + CV.x1) / 2}" y="${CV.c1 + 32}"
    text-anchor="middle">components kept, <tspan class="sym">k</tspan></text>`;
  fCurve.innerHTML = s;
}

/** About six round tick values in 0..r, always including r. */
function niceTicks(r) {
  const raw = r / 6, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map(f => f * p).find(v => v >= raw);
  const t = [];
  for (let v = 0; v <= r - step * 0.4; v += step) t.push(v);
  t.push(r);
  return t;
}

fK.addEventListener("input", () => {
  F.k = +fK.value;
  facesRender();
});

dragOn(fCurve, {
  down(h, q) { if (!F.fit) return false; this.move(q); return true; },
  move(q) {
    const f = clamp((q.x - CV.x0) / (CV.x1 - CV.x0), 0, 1);
    F.k = Math.round(f * F.fit.rank);
    facesRender();
  },
});

// -------------------------------------------------------------- gallery

/** One thumbnail per image, all cut from the shared sprite by CSS. */
function galleryBuild() {
  const m = F.m, tw = m.w > m.h ? 44 : Math.round(44 * m.w / m.h);
  const th = Math.round(tw * m.h / m.w);
  fGal.style.setProperty("--sprite", `url("${m.sprite}")`);
  fGal.style.setProperty("--tw", `${tw}px`);
  fGal.style.setProperty("--th", `${th}px`);
  let s = "";
  for (let i = 0; i < m.n; i++) {
    const c = i % m.cols, r = Math.floor(i / m.cols);
    s += `<button class="thumb" data-i="${i}" title="${m.labels[i]}"
      style="background-position:${-c * tw}px ${-r * th}px"></button>`;
  }
  fGal.innerHTML = s;
  fGal.style.backgroundSize = "";
  for (const t of fGal.children) {
    t.style.backgroundSize = `${m.cols * tw}px auto`;
  }
}

fGal.addEventListener("click", e => {
  const t = e.target.closest(".thumb");
  if (!t || !F.fit) return;
  const i = +t.dataset.i;
  if (F.game && !F.game.done) guess(i);
  else select(i);
});

// --------------------------------------------------- eigenfaces, mixer

function eigsRender() {
  const { m, fit } = F;
  const box = el("f-eigs");
  let s = `<figure><canvas data-j="mean"></canvas><figcaption><b>x&#772;</b>
    mean</figcaption></figure>`;
  for (let j = 0; j < Math.min(N_EIG, fit.rank); j++) {
    s += `<figure><canvas data-j="${j}"></canvas><figcaption><b>w</b><sub>${
      j + 1}</sub> ${(100 * fit.lam[j] / fit.total).toFixed(1)}%</figcaption>
      </figure>`;
  }
  box.innerHTML = s;
  for (const cv of box.querySelectorAll("canvas")) {
    const j = cv.dataset.j;
    if (j === "mean") drawImage(cv, m, fit.mean);
    else drawImage(cv, m, direction(fit, +j), true);
  }
}

function mixBuild() {
  let s = "";
  for (let j = 0; j < N_MIX; j++) {
    s += `<label class="slider"><span><i>c</i><sub>${j + 1}</sub></span>
      <input type="range" min="-3" max="3" step="0.05" data-j="${j}"
      value="${F.mix[j]}"><span class="val" id="f-mix-v${j}"></span></label>`;
  }
  el("f-mix-sl").innerHTML = s;
  for (const r of el("f-mix-sl").querySelectorAll("input")) {
    r.addEventListener("input", () => {
      F.mix[+r.dataset.j] = +r.value;
      mixRender();
    });
  }
}

/** x̄ (or the picked image) plus Σ c_j w_j with c_j = mix_j √λ_j. */
function mixRender() {
  const { m, fit } = F;
  if (!fit) return;
  const playing = F.game && !F.game.done;
  const fromSel = F.mixFrom === "sel" && !playing;
  const x = Float64Array.from(fromSel ? imageOf(m, F.sel) : fit.mean);
  for (let j = 0; j < Math.min(N_MIX, fit.rank); j++) {
    const c = F.mix[j] * Math.sqrt(fit.lam[j]);
    el(`f-mix-v${j}`).innerHTML = `${fmt(F.mix[j], 2)} sd = ${fmt(c, 2)}`;
    if (c === 0) continue;
    const w = direction(fit, j);
    for (let q = 0; q < m.d; q++) x[q] += c * w[q];
  }
  drawImage(el("f-mix"), m, x);
  segSet(el("f-mix-from"), fromSel ? "sel" : "mean");
  el("f-mix-from").querySelector('[data-v="sel"]').disabled = playing;
}

for (const b of el("f-mix-from").querySelectorAll("button")) {
  b.onclick = () => { F.mixFrom = b.dataset.v; mixRender(); };
}
el("f-mix-reset").onclick = () => {
  F.mix.fill(0);
  for (const r of el("f-mix-sl").querySelectorAll("input")) r.value = 0;
  mixRender();
};

// ----------------------------------------------------------- the game

/** Images i and j count as the same answer (Olivetti: same person). */
function sameAnswer(i, j) {
  const L = F.m.labels;
  if (F.m.key === "olivetti") return L[i].split(",")[0] === L[j].split(",")[0];
  return i === j;
}

/** End any game and restore the card's opening text. */
function gameReset() {
  F.game = null;
  el("f-game-msg").textContent = GAME_INTRO;
  gameButtons();
}

const GAME_INTRO = el("f-game-msg").textContent.replace(/\s+/g, " ").trim();

function gameButtons() {
  const on = F.game && !F.game.done;
  el("f-game-go").hidden = on;
  el("f-game-go").textContent = F.game ? "Play again" : "Start";
  for (const id of ["f-game-p1", "f-game-p5", "f-game-quit"]) {
    el(id).hidden = !on;
  }
}

el("f-game-go").onclick = () => {
  if (!F.fit) return;
  const pool = F.held ? F.m.holdout : F.m.labels.map((_, i) => i);
  let t;
  do {
    t = pool[Math.floor(Math.random() * pool.length)];
  } while (pool.length > 1 && F.game && t === F.game.target);
  F.game = { target: t, tries: 0, done: false };
  F.k = 1;
  el("f-game-msg").innerHTML = `Hidden image rebuilt from <i>k</i> = 1
    component${F.held ? " (a held-out one, so PCA never saw it)" : ""}.
    Click its match in the gallery, or add components.`;
  gameButtons();
  select(t);
};

for (const [id, n] of [["f-game-p1", 1], ["f-game-p5", 5]]) {
  el(id).onclick = () => {
    F.k = Math.min(F.k + n, F.fit.rank);
    facesRender();
  };
}

el("f-game-quit").onclick = () => {
  F.game.done = true;
  el("f-game-msg").innerHTML = `It was ${F.m.labels[F.game.target]}.`;
  gameButtons();
  facesRender();
};

function guess(i) {
  const g = F.game, L = F.m.labels;
  g.tries++;
  if (sameAnswer(i, g.target)) {
    g.done = true;
    el("f-game-msg").innerHTML = `<b class="good">Yes!</b> ${L[g.target]},
      found with <i>k</i> = ${F.k} of ${F.fit.rank} components${
      g.tries > 1 ? ` after ${g.tries} guesses` : ""}.`;
    gameButtons();
    facesRender();
  } else {
    el("f-game-msg").innerHTML = `<b class="bad">Not ${L[i]}.</b> Add a
      component and look again (now <i>k</i> = ${F.k}).`;
  }
}

// ---------------------------------------------------------------- setup

segmented(el("f-sets"), SETS, v => { if (v !== F.key) useSet(v); });
el("f-held").onchange = () => {
  F.held = el("f-held").checked;
  gameReset();
  if (F.m) refit();
};

/** Show the eigenfaces tab (called by main.js); fits on first visit. */
function facesShow() {
  if (!F.m && !F.token) useSet(F.key);
  else facesRender();
}
