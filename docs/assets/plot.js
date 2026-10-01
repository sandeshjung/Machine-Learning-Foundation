/* Tiny canvas plotting + UI helpers shared by the explorables (no dependencies). */
(function (root) {
  "use strict";

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function hexToRgb(hex) {
    const h = hex.replace("#", "");
    const v = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }

  /** "Nice" tick positions covering [min, max]. */
  function niceTicks(min, max, count = 6) {
    const span = max - min;
    if (!(span > 0)) return [min];
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return ticks;
  }

  function fmt(v, digits = 3) {
    if (!isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
    return Number(v.toPrecision(digits)).toString();
  }

  class Plot {
    /**
     * opts: xmin, xmax, ymin, ymax, xlabel, ylabel, ylog (y values are plotted as log10),
     *       xlog, pad {l, r, t, b}
     */
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.o = Object.assign({ xmin: 0, xmax: 1, ymin: 0, ymax: 1, xlabel: "", ylabel: "", ylog: false, xlog: false }, opts);
      this.o.pad = Object.assign({ l: 52, r: 14, t: 12, b: 42 }, opts.pad || {});
      this._heat = null;
      this.resize();
    }

    setRange(xmin, xmax, ymin, ymax) {
      Object.assign(this.o, { xmin, xmax, ymin, ymax });
      this._heat = null;
    }

    resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.canvas.clientWidth || 400, h = this.canvas.clientHeight || 300;
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
        this._heat = null;
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = w; this.h = h;
      const p = this.o.pad;
      this.L = p.l; this.R = w - p.r; this.T = p.t; this.B = h - p.b;
    }

    // coordinate transforms (log axes take raw values)
    _tx(x) { return this.o.xlog ? Math.log10(x) : x; }
    _ty(y) { return this.o.ylog ? Math.log10(y) : y; }
    X(x) { const o = this.o; return this.L + ((this._tx(x) - o.xmin) / (o.xmax - o.xmin)) * (this.R - this.L); }
    Y(y) { const o = this.o; return this.B - ((this._ty(y) - o.ymin) / (o.ymax - o.ymin)) * (this.B - this.T); }
    invX(px) { const o = this.o; const v = o.xmin + ((px - this.L) / (this.R - this.L)) * (o.xmax - o.xmin); return o.xlog ? 10 ** v : v; }
    invY(py) { const o = this.o; const v = o.ymin + ((this.B - py) / (this.B - this.T)) * (o.ymax - o.ymin); return o.ylog ? 10 ** v : v; }
    inside(px, py) { return px >= this.L && px <= this.R && py >= this.T && py <= this.B; }

    clear() { this.ctx.clearRect(0, 0, this.w, this.h); }

    /** Grid, axes, tick labels and axis titles. */
    axes({ grid = true, zeroLines = false } = {}) {
      const c = this.ctx, o = this.o;
      c.save();
      c.font = "12px system-ui, sans-serif";
      c.fillStyle = cssVar("--muted");
      const intTicks = (a, b) => { const t = []; for (let v = Math.ceil(a); v <= Math.floor(b); v++) t.push(v); return t; };
      const xt = o.xlog ? intTicks(o.xmin, o.xmax) : niceTicks(o.xmin, o.xmax, Math.max(3, Math.floor((this.R - this.L) / 70)));
      const yt = o.ylog ? intTicks(o.ymin, o.ymax) : niceTicks(o.ymin, o.ymax, Math.max(3, Math.floor((this.B - this.T) / 45)));
      const lab = (v, log) => (log ? (Number.isInteger(v) ? "10" + superscript(v) : "") : fmt(v));
      c.strokeStyle = cssVar("--c-grid");
      c.lineWidth = 1;
      c.textAlign = "center"; c.textBaseline = "top";
      for (const t of xt) {
        const px = this.L + ((t - o.xmin) / (o.xmax - o.xmin)) * (this.R - this.L);
        if (grid) { c.beginPath(); c.moveTo(px, this.T); c.lineTo(px, this.B); c.stroke(); }
        c.fillText(lab(t, o.xlog), px, this.B + 6);
      }
      c.textAlign = "right"; c.textBaseline = "middle";
      for (const t of yt) {
        const py = this.B - ((t - o.ymin) / (o.ymax - o.ymin)) * (this.B - this.T);
        if (grid) { c.beginPath(); c.moveTo(this.L, py); c.lineTo(this.R, py); c.stroke(); }
        c.fillText(lab(t, o.ylog), this.L - 6, py);
      }
      if (zeroLines) {
        c.strokeStyle = cssVar("--c-axis");
        c.lineWidth = 1.2;
        if (o.xmin < 0 && o.xmax > 0) { const px = this.X(0); c.beginPath(); c.moveTo(px, this.T); c.lineTo(px, this.B); c.stroke(); }
        if (o.ymin < 0 && o.ymax > 0) { const py = this.Y(0); c.beginPath(); c.moveTo(this.L, py); c.lineTo(this.R, py); c.stroke(); }
      }
      c.strokeStyle = cssVar("--c-axis");
      c.lineWidth = 1;
      c.strokeRect(this.L + 0.5, this.T + 0.5, this.R - this.L - 1, this.B - this.T - 1);
      c.fillStyle = cssVar("--text");
      c.font = "13px system-ui, sans-serif";
      if (o.xlabel) { c.textAlign = "center"; c.textBaseline = "bottom"; c.fillText(o.xlabel, (this.L + this.R) / 2, this.h - 2); }
      if (o.ylabel) {
        c.save(); c.translate(13, (this.T + this.B) / 2); c.rotate(-Math.PI / 2);
        c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(o.ylabel, 0, 0); c.restore();
      }
      c.restore();
    }

    /** Run fn with drawing clipped to the plot area. */
    clipped(fn) {
      const c = this.ctx;
      c.save();
      c.beginPath(); c.rect(this.L, this.T, this.R - this.L, this.B - this.T); c.clip();
      fn(c);
      c.restore();
    }

    _style(s = {}) {
      const c = this.ctx;
      c.strokeStyle = s.color || cssVar("--c-fit");
      c.fillStyle = s.fill || s.color || cssVar("--c-fit");
      c.lineWidth = s.width || 2;
      c.globalAlpha = s.alpha == null ? 1 : s.alpha;
      c.setLineDash(s.dash || []);
      c.lineJoin = "round"; c.lineCap = "round";
    }

    /** Polyline through data points; breaks the line at non-finite values. */
    line(xs, ys, s) {
      const c = this.ctx;
      c.save(); this._style(s);
      c.beginPath();
      let pen = false;
      for (let i = 0; i < xs.length; i++) {
        const y = ys[i];
        if (!isFinite(y) || (this.o.ylog && y <= 0)) { pen = false; continue; }
        const px = this.X(xs[i]), py = Math.max(-1e4, Math.min(1e4, this.Y(y)));
        pen ? c.lineTo(px, py) : c.moveTo(px, py);
        pen = true;
      }
      c.stroke();
      c.restore();
    }

    fn(f, s, n = 400) {
      const xs = [], ys = [];
      for (let i = 0; i <= n; i++) {
        const t = this.o.xmin + ((this.o.xmax - this.o.xmin) * i) / n;
        const x = this.o.xlog ? 10 ** t : t;
        xs.push(x); ys.push(f(x));
      }
      this.line(xs, ys, s);
    }

    dots(xs, ys, s = {}) {
      const c = this.ctx, r = s.r || 4.5;
      c.save(); this._style(s);
      for (let i = 0; i < xs.length; i++) {
        c.beginPath(); c.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI);
        c.fill();
        if (s.stroke) { c.strokeStyle = s.stroke; c.lineWidth = 1.5; c.stroke(); }
      }
      c.restore();
    }

    segment(x1, y1, x2, y2, s) {
      const c = this.ctx;
      c.save(); this._style(s);
      c.beginPath(); c.moveTo(this.X(x1), this.Y(y1)); c.lineTo(this.X(x2), this.Y(y2)); c.stroke();
      c.restore();
    }

    rect(x1, y1, x2, y2, s) {
      const c = this.ctx;
      c.save(); this._style(s);
      const px = Math.min(this.X(x1), this.X(x2)), py = Math.min(this.Y(y1), this.Y(y2));
      const w = Math.abs(this.X(x2) - this.X(x1)), h = Math.abs(this.Y(y2) - this.Y(y1));
      if (s && s.fill) c.fillRect(px, py, w, h);
      if (!s || s.color) c.strokeRect(px, py, w, h);
      c.restore();
    }

    /** Filled polygon from data points. */
    polygon(pts, s) {
      const c = this.ctx;
      c.save(); this._style(s);
      c.beginPath();
      pts.forEach(([x, y], i) => (i ? c.lineTo(this.X(x), this.Y(y)) : c.moveTo(this.X(x), this.Y(y))));
      c.closePath();
      if (s && s.fill) c.fill();
      if (s && s.color) c.stroke();
      c.restore();
    }

    cross(x, y, s = {}) {
      const c = this.ctx, r = s.r || 6, px = this.X(x), py = this.Y(y);
      c.save(); this._style(Object.assign({ width: 2.5 }, s));
      c.beginPath(); c.moveTo(px - r, py - r); c.lineTo(px + r, py + r); c.moveTo(px + r, py - r); c.lineTo(px - r, py + r); c.stroke();
      c.restore();
    }

    text(str, x, y, s = {}) {
      const c = this.ctx;
      c.save();
      c.font = s.font || "12px system-ui, sans-serif";
      c.fillStyle = s.color || cssVar("--text");
      c.textAlign = s.align || "left"; c.textBaseline = s.baseline || "middle";
      const px = s.px ? x : this.X(x), py = s.px ? y : this.Y(y);
      if (s.halo) { c.strokeStyle = cssVar("--surface"); c.lineWidth = 4; c.strokeText(str, px + (s.dx || 0), py + (s.dy || 0)); }
      c.fillText(str, px + (s.dx || 0), py + (s.dy || 0));
      c.restore();
    }

    /**
     * Banded heatmap of f(x, y) over the plot area, with thin contour lines between bands.
     * `key` caches the image: pass a new key whenever f changes. transform maps raw values
     * to the banding scale (e.g. Math.log1p).
     */
    heatmap(f, { key = "", bands = 14, transform = (v) => v, cell = 2, lo, hi, strength = 1 } = {}) {
      const cols = Math.max(2, Math.floor((this.R - this.L) / cell));
      const rows = Math.max(2, Math.floor((this.B - this.T) / cell));
      const theme = cssVar("--heat-lo") + cssVar("--heat-hi");
      const fullKey = `${key}|${strength}|${cols}x${rows}|${theme}|${this.o.xmin},${this.o.xmax},${this.o.ymin},${this.o.ymax}`;
      if (!this._heat || this._heat.key !== fullKey) {
        const vals = new Float64Array(cols * rows);
        let vmin = Infinity, vmax = -Infinity;
        for (let j = 0; j < rows; j++) {
          const y = this.invY(this.T + (j + 0.5) * cell);
          for (let i = 0; i < cols; i++) {
            const v = transform(f(this.invX(this.L + (i + 0.5) * cell), y));
            vals[j * cols + i] = v;
            if (isFinite(v)) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
          }
        }
        if (lo != null) vmin = lo;
        if (hi != null) vmax = hi;
        const band = new Int16Array(cols * rows);
        for (let k = 0; k < vals.length; k++) {
          const t = (vals[k] - vmin) / (vmax - vmin || 1);
          band[k] = Math.max(0, Math.min(bands - 1, Math.floor((isFinite(t) ? t : 1) * bands)));
        }
        const a = hexToRgb(cssVar("--heat-lo")), b = hexToRgb(cssVar("--heat-hi"));
        const off = document.createElement("canvas");
        off.width = cols; off.height = rows;
        const octx = off.getContext("2d");
        const img = octx.createImageData(cols, rows);
        for (let j = 0; j < rows; j++) {
          for (let i = 0; i < cols; i++) {
            const k = j * cols + i, bnd = band[k];
            const t = strength * Math.pow(1 - bnd / (bands - 1), 1.15);
            let r = a[0] + (b[0] - a[0]) * t, g = a[1] + (b[1] - a[1]) * t, bl = a[2] + (b[2] - a[2]) * t;
            const edge = (i + 1 < cols && band[k + 1] !== bnd) || (j + 1 < rows && band[k + cols] !== bnd);
            if (edge) { r *= 0.82; g *= 0.82; bl *= 0.82; }
            const p = 4 * k;
            img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = bl; img.data[p + 3] = 255;
          }
        }
        octx.putImageData(img, 0, 0);
        this._heat = { key: fullKey, canvas: off };
      }
      const c = this.ctx;
      c.save();
      c.imageSmoothingEnabled = false;
      c.drawImage(this._heat.canvas, this.L, this.T, cols * cell, rows * cell);
      c.restore();
    }
  }

  function superscript(n) {
    const map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    return String(n).split("").map((ch) => map[ch] || ch).join("");
  }

  // ------------------------------------------------------------------ interaction

  /** Pointer position in CSS pixels relative to the canvas. */
  function pointerPos(canvas, e) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  /**
   * Generic drag handling.
   *   pick(px, py) → handle or null   (what is under the pointer)
   *   onDown(handle, px, py, e)       (may return a new handle, e.g. a freshly added point)
   *   onMove(handle, px, py, e)
   *   onUp(handle)
   */
  function draggable(canvas, { pick, onDown, onMove, onUp, cursor = "grab" }) {
    let active = null;
    canvas.addEventListener("pointerdown", (e) => {
      const [px, py] = pointerPos(canvas, e);
      let h = pick ? pick(px, py) : null;
      if (onDown) { const r = onDown(h, px, py, e); if (r !== undefined) h = r; }
      if (h != null) {
        active = h;
        canvas.setPointerCapture(e.pointerId);
        canvas.style.cursor = "grabbing";
        e.preventDefault();
      }
    });
    canvas.addEventListener("pointermove", (e) => {
      const [px, py] = pointerPos(canvas, e);
      if (active != null) { onMove && onMove(active, px, py, e); return; }
      if (pick) canvas.style.cursor = pick(px, py) != null ? cursor : "crosshair";
    });
    const end = () => { if (active != null) { onUp && onUp(active); active = null; canvas.style.cursor = ""; } };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
  }

  /** Bind a range input to an <output>; returns a getter. For log sliders pass log: true (value = 10^input). */
  function bindRange(id, { format = (v) => fmt(v), log = false, onInput } = {}) {
    const input = document.getElementById(id);
    const out = document.querySelector(`output[for="${id}"]`);
    const get = () => (log ? 10 ** parseFloat(input.value) : parseFloat(input.value));
    const show = () => { if (out) out.textContent = format(get()); };
    input.addEventListener("input", () => { show(); onInput && onInput(get()); });
    show();
    const api = {
      get,
      set(v) { input.value = log ? Math.log10(v) : v; show(); },
      el: input,
    };
    return api;
  }

  function setText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }

  /** Redraw on resize (ResizeObserver) and on light/dark changes. */
  function autoRedraw(plots, draw) {
    let queued = false;
    const schedule = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; plots.forEach((p) => p.resize()); draw(); });
    };
    const ro = new ResizeObserver(schedule);
    plots.forEach((p) => ro.observe(p.canvas));
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
    document.addEventListener("themechange", schedule);
    return schedule;
  }

  // ------------------------------------------------------------------ theme toggle (auto → light → dark)

  function initTheme() {
    const html = document.documentElement;
    const params = new URLSearchParams(location.search);
    let mode = params.get("theme");
    if (!mode) { try { mode = localStorage.getItem("mlf-theme"); } catch (e) { mode = null; } }
    const apply = (m) => {
      if (m === "light" || m === "dark") html.setAttribute("data-theme", m); else html.removeAttribute("data-theme");
      const btn = document.getElementById("theme-btn");
      if (btn) btn.textContent = m === "light" ? "☀ Light" : m === "dark" ? "☾ Dark" : "◐ Auto";
      document.dispatchEvent(new Event("themechange"));
    };
    apply(mode);
    const btn = document.getElementById("theme-btn");
    if (btn) btn.addEventListener("click", () => {
      mode = mode === "light" ? "dark" : mode === "dark" ? "auto" : "light";
      try { localStorage.setItem("mlf-theme", mode); } catch (e) { /* storage unavailable */ }
      apply(mode);
    });
  }

  document.addEventListener("DOMContentLoaded", initTheme);

  root.Plot = Plot;
  root.PlotUI = { cssVar, niceTicks, fmt, draggable, pointerPos, bindRange, setText, autoRedraw };
})(window);
