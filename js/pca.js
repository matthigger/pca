// Image sets and the PCA fit behind the eigenfaces tab.
//
// Each js/data/<key>.js calls PCA_DATA.add(meta) with a sprite of all its
// images (see build/make_data.py). load() injects that script on demand
// (works from file://) and decodes the sprite into pixel rows in [0, 1].
//
// PCA uses the Gram trick (Sirovich & Kirby 1987; Turk & Pentland 1991):
// with m centered training rows X (m x d) and d >> m, eigen-decompose
// G = X Xᵀ (m x m) = U diag(g) Uᵀ. Then w_j = Xᵀ u_j / √g_j are the unit
// principal directions, λ_j = g_j / (m − 1) the variances (sklearn's
// explained_variance_), and any rebuild x̄ + Σ c_j w_j equals
// x̄ + Xᵀ a with a = Σ c_j u_j / √g_j, so no d x d matrix is ever formed.

const PCA_DATA = {
  sets: {},
  /** Register one dataset; called by each js/data/<key>.js. */
  add(meta) { this.sets[meta.key] = meta; },
};

const DATA_KEYS = ["guesswho", "olivetti", "emoji", "cats"];

/** Run fn after the browser has painted (so a "working" note shows). */
function afterPaint() {
  return new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));
}

/**
 * Load dataset key: inject its script, decode the sprite.
 *
 * Returns:
 *   Promise of meta plus pix (Float32Array) (n * d), d = h * w * ch,
 *   row i image i with channels interleaved (RGB) and values in [0, 1]
 */
async function loadSet(key, base = "js/data/") {
  const have = PCA_DATA.sets[key];
  if (have && have.pix) return have;
  if (!have) {
    await new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = `${base}${key}.js`;
      s.onload = ok;
      s.onerror = () => fail(new Error(`could not load ${s.src}`));
      document.head.appendChild(s);
    });
  }
  const m = PCA_DATA.sets[key];
  const img = new Image();
  img.src = m.sprite;
  await img.decode();
  const cv = document.createElement("canvas");
  cv.width = img.naturalWidth;
  cv.height = img.naturalHeight;
  const cx = cv.getContext("2d", { willReadFrequently: true });
  cx.drawImage(img, 0, 0);
  const rgba = cx.getImageData(0, 0, cv.width, cv.height).data;
  const d = m.w * m.h * m.ch;
  m.d = d;
  m.pix = new Float32Array(m.n * d);
  for (let i = 0; i < m.n; i++) {
    const x0 = (i % m.cols) * m.w, y0 = Math.floor(i / m.cols) * m.h;
    let o = i * d;
    for (let y = 0; y < m.h; y++) {
      for (let x = 0; x < m.w; x++) {
        const q = ((y0 + y) * cv.width + x0 + x) * 4;
        for (let c = 0; c < m.ch; c++) m.pix[o++] = rgba[q + c] / 255;
      }
    }
  }
  return m;
}

/** Image i of dataset m as a Float32Array view (d,). */
function imageOf(m, i) { return m.pix.subarray(i * m.d, (i + 1) * m.d); }

/**
 * Fit PCA on rows train of dataset m.
 *
 * Args:
 *   m (object): a loaded dataset (loadSet)
 *   train (number[]): indices of the training images
 *   progress (Function | undefined): (fraction done) during the Gram
 *     product, the slow step
 *
 * Returns:
 *   Promise of a fit {m, train, mean (d,), X (mt * d) centered rows,
 *   g (mt,) Gram eigenvalues, U (mt * mt) column-major-by-j eigenvectors,
 *   lam (rank,) variances λ_j, rank, total Σλ_j}
 */
async function fitPca(m, train, progress) {
  const d = m.d, mt = train.length;
  const mean = new Float64Array(d);
  for (const i of train) {
    const x = imageOf(m, i);
    for (let k = 0; k < d; k++) mean[k] += x[k];
  }
  for (let k = 0; k < d; k++) mean[k] /= mt;
  const X = new Float32Array(mt * d);
  train.forEach((i, r) => {
    const x = imageOf(m, i);
    for (let k = 0; k < d; k++) X[r * d + k] = x[k] - mean[k];
  });

  const G = new Float64Array(mt * mt);
  let t = performance.now();
  for (let a = 0; a < mt; a++) {
    const xa = X.subarray(a * d, (a + 1) * d);
    for (let b = a; b < mt; b++) {
      const xb = X.subarray(b * d, (b + 1) * d);
      let s = 0;
      for (let k = 0; k < d; k++) s += xa[k] * xb[k];
      G[a * mt + b] = G[b * mt + a] = s;
    }
    if (progress && performance.now() - t > 60) {
      progress((a + 1) * (2 * mt - a) / (mt * (mt + 1)));
      await afterPaint();
      t = performance.now();
    }
  }

  const { val, vec } = eigSym(G, mt);
  let rank = 0;
  while (rank < mt && val[rank] > val[0] * 1e-10) rank++;
  const U = new Float64Array(mt * rank);
  for (let j = 0; j < rank; j++) {
    for (let i = 0; i < mt; i++) U[j * mt + i] = vec[i * mt + j];
  }
  const g = val.slice(0, rank);
  const lam = g.map(v => v / (mt - 1));
  const total = lam.reduce((s, v) => s + v, 0);
  return { m, train, mean, X, g, U, lam, rank, total, mt, d, w: {} };
}

/** Unit principal direction w_j (Float64Array (d,)), cached on the fit. */
function direction(fit, j) {
  if (fit.w[j]) return fit.w[j];
  const { X, U, g, mt, d } = fit, w = new Float64Array(d);
  for (let i = 0; i < mt; i++) {
    const u = U[j * mt + i];
    if (u === 0) continue;
    const xi = X.subarray(i * d, (i + 1) * d);
    for (let k = 0; k < d; k++) w[k] += u * xi[k];
  }
  const s = 1 / Math.sqrt(g[j]);
  for (let k = 0; k < d; k++) w[k] *= s;
  return (fit.w[j] = w);
}

/**
 * Scores c_j = w_jᵀ(x − x̄) of image x on every principal direction.
 *
 * Args:
 *   x (Float32Array): (d,) image
 *
 * Returns:
 *   c (Float64Array): (rank,)
 */
function scores(fit, x) {
  const { X, U, g, mt, d, mean, rank } = fit;
  const q = new Float64Array(mt);
  for (let i = 0; i < mt; i++) {
    const xi = X.subarray(i * d, (i + 1) * d);
    let s = 0;
    for (let k = 0; k < d; k++) s += xi[k] * (x[k] - mean[k]);
    q[i] = s;
  }
  const c = new Float64Array(rank);
  for (let j = 0; j < rank; j++) {
    let s = 0;
    for (let i = 0; i < mt; i++) s += U[j * mt + i] * q[i];
    c[j] = s / Math.sqrt(g[j]);
  }
  return c;
}

/**
 * Rebuild x̂ = x̄ + Σ_{j < k} c_j w_j.
 *
 * Args:
 *   c (Float64Array | number[]): scores, at least k of them
 *   k (number): components kept
 *
 * Returns:
 *   xhat (Float64Array): (d,)
 */
function rebuild(fit, c, k) {
  const { X, U, g, mt, d, mean } = fit;
  const a = new Float64Array(mt);
  for (let j = 0; j < k; j++) {
    const s = c[j] / Math.sqrt(g[j]);
    if (s === 0) continue;
    for (let i = 0; i < mt; i++) a[i] += s * U[j * mt + i];
  }
  const xh = Float64Array.from(mean);
  for (let i = 0; i < mt; i++) {
    if (a[i] === 0) continue;
    const xi = X.subarray(i * d, (i + 1) * d), ai = a[i];
    for (let k2 = 0; k2 < d; k2++) xh[k2] += ai * xi[k2];
  }
  return xh;
}

/** Indices 0..n-1 of dataset m, without the held-out ones if held. */
function trainIdx(m, held) {
  const out = new Set(held ? m.holdout : []);
  return Array.from({ length: m.n }, (_, i) => i).filter(i => !out.has(i));
}
