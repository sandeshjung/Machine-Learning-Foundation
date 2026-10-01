/*
 * Numerical routines shared by the explorables.
 * Pure functions (no DOM), so they can be unit-tested with Node:
 *   node docs/tests/mlmath.test.js
 * Conventions follow the notebooks: MSE uses 1/m, optimisers follow PyTorch,
 * Ridge/Lasso follow scikit-learn's objectives.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MLMath = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- random numbers

  /** Seeded uniform PRNG on [0, 1) (mulberry32), so every "new data" click is reproducible. */
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Standard normal sample (Box–Muller) from a uniform generator. */
  function randn(rand) {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // ---------------------------------------------------------------- basic statistics

  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const mean = (a) => sum(a) / a.length;

  function variance(a, ddof = 0) {
    const m = mean(a);
    return sum(a.map((v) => (v - m) ** 2)) / (a.length - ddof);
  }

  function linspace(a, b, n) {
    if (n === 1) return [a];
    return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
  }

  /** Histogram on [lo, hi) with `bins` equal bins; returns counts and densities. */
  function histogram(values, lo, hi, bins) {
    const counts = new Array(bins).fill(0);
    const width = (hi - lo) / bins;
    for (const v of values) {
      const k = Math.floor((v - lo) / width);
      if (k >= 0 && k < bins) counts[k]++;
    }
    const density = counts.map((c) => c / (values.length * width));
    const edges = linspace(lo, hi, bins + 1);
    return { counts, density, edges, width };
  }

  function normalPdf(x, mu = 0, sigma = 1) {
    const z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  }

  /** erf with |error| < 1.5e-7 (Abramowitz & Stegun 7.1.26). */
  function erf(x) {
    const s = Math.sign(x);
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  const normalCdf = (x, mu = 0, sigma = 1) => 0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));

  // ---------------------------------------------------------------- simple linear regression

  /** Sufficient statistics (all divided by m) for a 1-D least-squares problem. */
  function lineStats(xs, ys) {
    const m = xs.length;
    let Sx = 0, Sy = 0, Sxx = 0, Sxy = 0, Syy = 0;
    for (let i = 0; i < m; i++) {
      const x = xs[i], y = ys[i];
      Sx += x; Sy += y; Sxx += x * x; Sxy += x * y; Syy += y * y;
    }
    return { m, Sx: Sx / m, Sy: Sy / m, Sxx: Sxx / m, Sxy: Sxy / m, Syy: Syy / m };
  }

  /** Closed-form least-squares line y = w x + b. */
  function olsLine(xs, ys) {
    const s = lineStats(xs, ys);
    const vx = s.Sxx - s.Sx * s.Sx;
    const w = vx > 1e-12 ? (s.Sxy - s.Sx * s.Sy) / vx : 0;
    return { w, b: s.Sy - w * s.Sx };
  }

  /** MSE = (1/m) Σ (w x + b − y)², evaluated in O(1) from the statistics. */
  function lineMSE(s, w, b) {
    return s.Syy - 2 * w * s.Sxy - 2 * b * s.Sy + w * w * s.Sxx + 2 * w * b * s.Sx + b * b;
  }

  /** Gradient of the MSE with respect to (w, b). */
  function lineGrad(s, w, b) {
    return { dw: 2 * (w * s.Sxx + b * s.Sx - s.Sxy), db: 2 * (w * s.Sx + b - s.Sy) };
  }

  function r2Score(ys, preds) {
    const m = mean(ys);
    let ssr = 0, sst = 0;
    for (let i = 0; i < ys.length; i++) {
      ssr += (ys[i] - preds[i]) ** 2;
      sst += (ys[i] - m) ** 2;
    }
    return sst > 0 ? 1 - ssr / sst : 0;
  }

  // ---------------------------------------------------------------- general least squares

  /** Solve min ||A x − y||₂ with Householder QR (A: array of m rows, m ≥ n). */
  function lstsq(A, y) {
    const m = A.length, n = A[0].length;
    const R = A.map((row) => row.slice());
    const qty = y.slice();
    for (let k = 0; k < n; k++) {
      let norm = 0;
      for (let i = k; i < m; i++) norm += R[i][k] * R[i][k];
      norm = Math.sqrt(norm);
      if (norm === 0) continue;
      const alpha = R[k][k] > 0 ? -norm : norm;
      const v = new Array(m).fill(0);
      v[k] = R[k][k] - alpha;
      for (let i = k + 1; i < m; i++) v[i] = R[i][k];
      let vv = 0;
      for (let i = k; i < m; i++) vv += v[i] * v[i];
      if (vv === 0) continue;
      for (let j = k; j < n; j++) {
        let d = 0;
        for (let i = k; i < m; i++) d += v[i] * R[i][j];
        const f = (2 * d) / vv;
        for (let i = k; i < m; i++) R[i][j] -= f * v[i];
      }
      let d = 0;
      for (let i = k; i < m; i++) d += v[i] * qty[i];
      const f = (2 * d) / vv;
      for (let i = k; i < m; i++) qty[i] -= f * v[i];
    }
    const x = new Array(n).fill(0);
    for (let k = n - 1; k >= 0; k--) {
      let s = qty[k];
      for (let j = k + 1; j < n; j++) s -= R[k][j] * x[j];
      x[k] = Math.abs(R[k][k]) > 1e-300 ? s / R[k][k] : 0;
    }
    return x;
  }

  /** Solve the square system A x = b by Gaussian elimination with partial pivoting. */
  function solve(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  // ---------------------------------------------------------------- polynomial regression

  /**
   * Least-squares polynomial fit of degree d: returns coefficients [c0, c1, …, cd] for
   * c0 + c1 x + … + cd x^d. A tiny ridge term (not applied to c0) keeps the solve well
   * defined when there are fewer points than coefficients. Use x roughly within [−1, 1].
   */
  function polyFit(xs, ys, degree, ridge = 1e-10) {
    const n = degree + 1;
    const rows = xs.map((x) => {
      const r = new Array(n);
      let p = 1;
      for (let j = 0; j < n; j++) { r[j] = p; p *= x; }
      return r;
    });
    const target = ys.slice();
    const s = Math.sqrt(ridge);
    for (let j = 1; j < n; j++) {
      const r = new Array(n).fill(0);
      r[j] = s;
      rows.push(r);
      target.push(0);
    }
    return lstsq(rows, target);
  }

  function polyEval(coef, x) {
    let y = 0;
    for (let j = coef.length - 1; j >= 0; j--) y = y * x + coef[j];
    return y;
  }

  // ---------------------------------------------------------------- regularisation geometry

  /** Eigen-decomposition of a symmetric 2×2 matrix [[a, b], [b, c]]. */
  function eig2sym(H) {
    const a = H[0][0], b = H[0][1], c = H[1][1];
    const tr = a + c, det = a * c - b * b;
    const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    let v1;
    if (Math.abs(b) > 1e-12) v1 = [l1 - c, b];
    else v1 = a >= c ? [1, 0] : [0, 1];
    const nv = Math.hypot(v1[0], v1[1]);
    v1 = [v1[0] / nv, v1[1] / nv];
    return { values: [l1, l2], vectors: [v1, [-v1[1], v1[0]]] };
  }

  const softThreshold = (z, t) => Math.sign(z) * Math.max(0, Math.abs(z) - t);

  /**
   * Minimise J(w) = ½ (w − ŵ)ᵀ H (w − ŵ) + α · Ω(w), where ŵ is the least-squares solution
   * and H = XᵀX / m. Ω = ‖w‖₂² for "ridge" and ‖w‖₁ for "lasso".
   * This equals scikit-learn's Lasso objective (fit_intercept=False) up to a constant, and
   * its Ridge objective with α_sklearn = 2 m α.
   */
  function penalisedQuadratic(H, what, alpha, kind, w0) {
    const n = what.length;
    if (kind === "ridge") {
      const A = H.map((row, i) => row.map((v, j) => v + (i === j ? 2 * alpha : 0)));
      const rhs = H.map((row) => row.reduce((s, v, j) => s + v * what[j], 0));
      return solve(A, rhs);
    }
    const w = w0 ? w0.slice() : new Array(n).fill(0);
    for (let it = 0; it < 5000; it++) {
      let delta = 0;
      for (let j = 0; j < n; j++) {
        let rho = H[j][j] * what[j];
        for (let k = 0; k < n; k++) if (k !== j) rho -= H[j][k] * (w[k] - what[k]);
        const nw = softThreshold(rho, alpha) / H[j][j];
        delta = Math.max(delta, Math.abs(nw - w[j]));
        w[j] = nw;
      }
      if (delta < 1e-13) break;
    }
    return w;
  }

  /** Lasso with intercept, scikit-learn objective (1/2m)‖y − Xw − b‖² + α‖w‖₁, by coordinate descent. */
  function lasso(X, y, alpha, maxIter = 10000, tol = 1e-12) {
    const m = X.length, n = X[0].length;
    const xm = Array.from({ length: n }, (_, j) => mean(X.map((r) => r[j])));
    const ym = mean(y);
    const Xc = X.map((r) => r.map((v, j) => v - xm[j]));
    const yc = y.map((v) => v - ym);
    const colsq = Array.from({ length: n }, (_, j) => sum(Xc.map((r) => r[j] * r[j])) / m);
    const w = new Array(n).fill(0);
    const resid = yc.slice();
    for (let it = 0; it < maxIter; it++) {
      let delta = 0;
      for (let j = 0; j < n; j++) {
        if (colsq[j] === 0) continue;
        let rho = 0;
        for (let i = 0; i < m; i++) rho += Xc[i][j] * (resid[i] + Xc[i][j] * w[j]);
        rho /= m;
        const nw = softThreshold(rho, alpha) / colsq[j];
        const d = nw - w[j];
        if (d !== 0) for (let i = 0; i < m; i++) resid[i] -= Xc[i][j] * d;
        delta = Math.max(delta, Math.abs(d));
        w[j] = nw;
      }
      if (delta < tol) break;
    }
    const b = ym - w.reduce((s, v, j) => s + v * xm[j], 0);
    return { w, b };
  }

  /** Ridge with intercept, scikit-learn objective ‖y − Xw − b‖² + α‖w‖², closed form. */
  function ridge(X, y, alpha) {
    const n = X[0].length;
    const xm = Array.from({ length: n }, (_, j) => mean(X.map((r) => r[j])));
    const ym = mean(y);
    const Xc = X.map((r) => r.map((v, j) => v - xm[j]));
    const yc = y.map((v) => v - ym);
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => sum(Xc.map((r) => r[i] * r[j])) + (i === j ? alpha : 0)));
    const rhs = Array.from({ length: n }, (_, i) => sum(Xc.map((r, k) => r[i] * yc[k])));
    const w = solve(A, rhs);
    return { w, b: ym - w.reduce((s, v, j) => s + v * xm[j], 0) };
  }

  // ---------------------------------------------------------------- optimisers (PyTorch conventions)

  /**
   * Returns an optimiser with step(theta, grad) → new theta for theta = [x, y, …].
   *   gd:        θ ← θ − η g
   *   momentum:  v ← μ v + g,             θ ← θ − η v
   *   rmsprop:   s ← ρ s + (1 − ρ) g²,    θ ← θ − η g / (√s + ε)
   *   adam:      m, v moving averages with bias correction (β₁ = 0.9, β₂ = 0.999)
   */
  function makeOptimizer(kind, { lr, mu = 0.9, rho = 0.99, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 }) {
    let v = null, s = null, m = null, t = 0;
    return {
      kind,
      step(theta, g) {
        const n = theta.length;
        if (!v) { v = new Array(n).fill(0); s = new Array(n).fill(0); m = new Array(n).fill(0); }
        t++;
        const out = new Array(n);
        for (let i = 0; i < n; i++) {
          if (kind === "gd") out[i] = theta[i] - lr * g[i];
          else if (kind === "momentum") { v[i] = mu * v[i] + g[i]; out[i] = theta[i] - lr * v[i]; }
          else if (kind === "rmsprop") { s[i] = rho * s[i] + (1 - rho) * g[i] * g[i]; out[i] = theta[i] - (lr * g[i]) / (Math.sqrt(s[i]) + eps); }
          else if (kind === "adam") {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            s[i] = beta2 * s[i] + (1 - beta2) * g[i] * g[i];
            const mh = m[i] / (1 - beta1 ** t), sh = s[i] / (1 - beta2 ** t);
            out[i] = theta[i] - (lr * mh) / (Math.sqrt(sh) + eps);
          } else throw new Error("unknown optimiser " + kind);
        }
        return out;
      },
    };
  }

  return {
    rng, randn, sum, mean, variance, linspace, histogram, normalPdf, normalCdf, erf,
    lineStats, olsLine, lineMSE, lineGrad, r2Score,
    lstsq, solve, polyFit, polyEval,
    eig2sym, softThreshold, penalisedQuadratic, lasso, ridge,
    makeOptimizer,
  };
});
