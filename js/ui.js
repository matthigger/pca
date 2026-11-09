// Shared helpers: plane geometry for the projection tab (the square
// [-8, 8]^2), 2-d vector math, SVG arrows, pointer dragging, number
// formatting, image drawing on canvases and the footer build stamp.

const PLANE = { w: 700, h: 700, x0: 46, x1: 686, y0: 14, y1: 654,
  dom: [-8, 8] };

/** Scales of plot box p: data to viewBox (sx, sy) and back (ix, iy). */
function scales(p) {
  const k = (p.x1 - p.x0) / (p.dom[1] - p.dom[0]);
  return {
    k,
    sx: v => p.x0 + (v - p.dom[0]) * k,
    sy: v => p.y1 - (v - p.dom[0]) * k,
    ix: px => p.dom[0] + (px - p.x0) / k,
    iy: py => p.dom[0] + (p.y1 - py) / k,
  };
}
const S = scales(PLANE);

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul = (a, s) => [a[0] * s, a[1] * s];
const norm = a => Math.hypot(a[0], a[1]);
const unit = t => [Math.cos(t), Math.sin(t)];
/** a turned 90 degrees counterclockwise. */
const perp = a => [-a[1], a[0]];
const DEG = Math.PI / 180;

/** Data point to "x,y" in viewBox units. */
function px(p) { return `${S.sx(p[0]).toFixed(1)},${S.sy(p[1]).toFixed(1)}`; }

/** SVG segment between data points a and c. */
function segSvg(a, c, attrs) {
  return `<line x1="${S.sx(a[0]).toFixed(1)}" y1="${S.sy(a[1]).toFixed(1)}"
    x2="${S.sx(c[0]).toFixed(1)}" y2="${S.sy(c[1]).toFixed(1)}" ${attrs}/>`;
}

/** SVG arrow from data point a to c; cls sets stroke and fill. */
function arrowSvg(a, c, cls, head = 14) {
  const x1 = S.sx(a[0]), y1 = S.sy(a[1]), x2 = S.sx(c[0]), y2 = S.sy(c[1]);
  const L = Math.hypot(x2 - x1, y2 - y1);
  if (L < 1) return "";
  const ux = (x2 - x1) / L, uy = (y2 - y1) / L, h = Math.min(head, L * .6);
  const bx = x2 - ux * h, by = y2 - uy * h;
  const pts = [[x2, y2], [bx - uy * h * .42, by + ux * h * .42],
    [bx + uy * h * .42, by - ux * h * .42]];
  return `<g class="arrow ${cls}"><line x1="${x1}" y1="${y1}" x2="${bx}"
    y2="${by}"/><polygon points="${pts.map(q => q.join(",")).join(" ")}"/>
    </g>`;
}

/** Grid lines (every unit), ticks (every 2) and axis labels. */
function axesSvg() {
  const p = PLANE;
  let a = "";
  for (let v = p.dom[0]; v <= p.dom[1]; v++) {
    const z = v === 0 ? " zero" : "";
    a += `<line class="grid${z}" x1="${S.sx(v)}" x2="${S.sx(v)}"
      y1="${p.y0}" y2="${p.y1}"/><line class="grid${z}" x1="${p.x0}"
      x2="${p.x1}" y1="${S.sy(v)}" y2="${S.sy(v)}"/>`;
    if (v % 2) continue;
    a += `<text class="tick" x="${S.sx(v)}" y="${p.y1 + 16}"
      text-anchor="middle">${fmt(v, 0)}</text><text class="tick"
      x="${p.x0 - 7}" y="${S.sy(v) + 4}" text-anchor="end">${fmt(v, 0)}</text>`;
  }
  a += `<text class="axis-label" x="${(p.x0 + p.x1) / 2}" y="${p.y1 + 38}"
    text-anchor="middle"><tspan class="sym">x</tspan>₀</text>
    <text class="axis-label" text-anchor="middle"
    transform="translate(16 ${(p.y0 + p.y1) / 2}) rotate(-90)"
    ><tspan class="sym">x</tspan>₁</text>`;
  return a;
}

/** Pointer position in the viewBox units of an SVG. */
function svgPoint(el, e) {
  const pt = el.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(el.getScreenCTM().inverse());
}

/**
 * Route pointer drags on svg to handlers.
 *
 * Args:
 *   h.down (Function): (handle, q, e) -> bool, handle = data-h of the
 *     element under the pointer (or null), q the pointer in viewBox
 *     units; return true to start a drag
 *   h.move (Function): (q, e) during the drag
 *   h.up (Function | undefined): (e) at the end
 */
function dragOn(svg, h) {
  svg.addEventListener("pointerdown", e => {
    const t = e.target.closest("[data-h]");
    if (!h.down(t ? t.dataset.h : null, svgPoint(svg, e), e)) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    const move = ev => h.move(svgPoint(svg, ev), ev);
    const up = ev => {
      svg.removeEventListener("pointermove", move);
      svg.removeEventListener("pointerup", up);
      svg.removeEventListener("pointercancel", up);
      if (h.up) h.up(ev);
    };
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerup", up);
    svg.addEventListener("pointercancel", up);
  });
}

/** Number with a true minus sign. */
function fmt(x, digits = 2) {
  const a = Math.abs(x).toFixed(digits);
  return (x < 0 && a !== (0).toFixed(digits) ? "−" : "") + a;
}

/** Vector [a, b] with true minus signs. */
function fvec(v, digits = 2) {
  return `[${fmt(v[0], digits)}, ${fmt(v[1], digits)}]`;
}

/** Readout row: label on the left, value on the right. */
function row(label, val) {
  return `<div class="row"><span>${label}</span><span class="val">${
    val}</span></div>`;
}

/** Single-choice segmented control; onPick(value) on click. */
function segmented(el, items, onPick) {
  el.innerHTML = items.map(([v, label]) =>
    `<button role="radio" data-v="${v}">${label}</button>`).join("");
  for (const b of el.querySelectorAll("button")) {
    b.onclick = () => onPick(b.dataset.v);
  }
}

/** Mark value v as the checked button of a segmented control. */
function segSet(el, v) {
  for (const b of el.querySelectorAll("button")) {
    b.setAttribute("aria-checked", b.dataset.v === String(v));
  }
}

/**
 * Draw an image vector on a canvas at its native size.
 *
 * Args:
 *   cv (HTMLCanvasElement): target
 *   m (object): dataset meta {w, h, ch}
 *   x (Float32Array | Float64Array): (h * w * ch,) values
 *   signed (bool): x is a direction: show 0 as mid-gray, scaled so the
 *     largest |x| reaches black or white; otherwise x is in [0, 1]
 */
function drawImage(cv, m, x, signed = false) {
  cv.width = m.w;
  cv.height = m.h;
  const cx = cv.getContext("2d");
  const im = cx.createImageData(m.w, m.h), out = im.data;
  let a = 1, b = 0;
  if (signed) {
    let mx = 1e-12;
    for (let k = 0; k < x.length; k++) mx = Math.max(mx, Math.abs(x[k]));
    a = 0.5 / mx;
    b = 0.5;
  }
  for (let p = 0; p < m.w * m.h; p++) {
    for (let c = 0; c < 3; c++) {
      const v = x[p * m.ch + (m.ch === 3 ? c : 0)] * a + b;
      out[p * 4 + c] = clamp(Math.round(v * 255), 0, 255);
    }
    out[p * 4 + 3] = 255;
  }
  cx.putImageData(im, 0, 0);
}

// Footer build stamp: the deployed commit (linked) and build time.
(function () {
  const el = document.getElementById("build");
  const repo = "https://github.com/matthigger/pca";
  if (!BUILD) { el.textContent = "local copy"; return; }
  const when = new Date(BUILD.time).toLocaleString("en-US", {
    dateStyle: "medium", timeStyle: "short" });
  el.innerHTML = `build <a href="${repo}/commit/${BUILD.sha}">${
    BUILD.sha.slice(0, 7)}</a>, ${when}`;
})();
