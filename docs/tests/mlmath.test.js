// Checks docs/assets/mlmath.js against reference results from NumPy, scikit-learn, SciPy and PyTorch.
// Run with:  node docs/tests/mlmath.test.js   (regenerate references with make_fixtures.py)
"use strict";
const path = require("path");
const M = require(path.join(__dirname, "..", "assets", "mlmath.js"));
const fx = require(path.join(__dirname, "fixtures.json"));

let failures = 0, passed = 0;
function close(name, got, want, atol) {
  const g = [].concat(got), w = [].concat(want);
  const err = Math.max(...g.map((v, i) => Math.abs(v - w[i])));
  const ok = g.length === w.length && err <= atol;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}  (max |diff| = ${err.toExponential(2)}, atol = ${atol})`);
  ok ? passed++ : failures++;
}

// Simple linear regression
const L = fx.line, s = M.lineStats(L.x, L.y), ols = M.olsLine(L.x, L.y);
close("line: closed-form fit vs sklearn LinearRegression", [ols.w, ols.b], [L.w, L.b], 1e-10);
close("line: MSE", M.lineMSE(s, ...L.probe), L.mse, 1e-10);
const g = M.lineGrad(s, ...L.probe);
close("line: MSE gradient", [g.dw, g.db], L.grad, 1e-10);
close("line: R²", M.r2Score(L.y, L.x.map((x) => ols.w * x + ols.b)), L.r2, 1e-10);

// Polynomial least squares: the QR solve is exact, and the default tiny ridge
// (which keeps under-determined fits stable) must not visibly change predictions.
const grid = M.linspace(-1, 1, 201);
for (const c of fx.poly) {
  close(`polyfit: degree ${c.degree} vs numpy polyfit (no ridge)`, M.polyFit(c.x, c.y, c.degree, 0), c.coef, 1e-9);
  const co = M.polyFit(c.x, c.y, c.degree);
  close(`polyfit: degree ${c.degree} predictions with default ridge`, grid.map((x) => M.polyEval(co, x)), grid.map((x) => M.polyEval(c.coef, x)), 1e-5);
}

// Lasso / Ridge with intercept
for (const c of fx.lasso.cases) {
  const r = M.lasso(fx.lasso.X, fx.lasso.y, c.alpha);
  close(`lasso alpha=${c.alpha} vs sklearn Lasso`, [...r.w, r.b], [...c.w, c.b], 1e-6);
}
for (const c of fx.ridge.cases) {
  const r = M.ridge(fx.lasso.X, fx.lasso.y, c.alpha);
  close(`ridge alpha=${c.alpha} vs sklearn Ridge`, [...r.w, r.b], [...c.w, c.b], 1e-9);
}

// Penalised quadratic (regularisation geometry)
const Q = fx.quad;
for (const c of Q.lasso) close(`geometry lasso alpha=${c.alpha} vs sklearn`, M.penalisedQuadratic(Q.H, Q.what, c.alpha, "lasso"), c.w, 1e-7);
for (const c of Q.ridge) close(`geometry ridge alpha=${c.alpha} vs sklearn`, M.penalisedQuadratic(Q.H, Q.what, c.alpha, "ridge"), c.w, 1e-9);

// Optimisers vs torch.optim
const grad = ([x, y]) => [x + y, 10 * y + x];
const lrs = { gd: 0.05, momentum: 0.05, rmsprop: 0.05, adam: 0.05 };
for (const [kind, want] of Object.entries(fx.optim)) {
  const opt = M.makeOptimizer(kind, { lr: lrs[kind] });
  let th = [2.0, -1.5];
  for (let i = 0; i < 25; i++) th = opt.step(th, grad(th));
  close(`optimiser ${kind} vs torch.optim (25 steps)`, th, want, 1e-10);
}

// Normal distribution
for (const [z, p] of fx.stats.cdf) close(`normal CDF(${z}) vs scipy`, M.normalCdf(z), p, 2e-7);
for (const [x, mu, sd, p] of fx.stats.pdf) close("normal PDF vs scipy", M.normalPdf(x, mu, sd), p, 1e-12);

// Eigen-decomposition sanity check: H v = λ v
const H = [[2, 0.7], [0.7, 1]], e = M.eig2sym(H);
for (let k = 0; k < 2; k++) {
  const v = e.vectors[k], Hv = [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]];
  close(`eig2sym: H v${k} = λ v${k}`, Hv, v.map((c) => e.values[k] * c), 1e-12);
}

console.log(`\n${passed} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
