"""Regenerate fixtures.json: reference results from NumPy, scikit-learn, SciPy and PyTorch.

    python docs/tests/make_fixtures.py   # then: node docs/tests/mlmath.test.js
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge

rs = np.random.RandomState(0)
fx = {}

# Simple linear regression: closed form, MSE, gradient, R²
x = rs.uniform(0, 10, 25)
y = 1.7 + 0.8 * x + rs.normal(0, 1.2, 25)
lr = LinearRegression().fit(x[:, None], y)
w, b = 0.3, 2.0
fx["line"] = dict(
    x=x.tolist(), y=y.tolist(), w=float(lr.coef_[0]), b=float(lr.intercept_), probe=[w, b],
    mse=float(np.mean((w * x + b - y) ** 2)),
    grad=[float(np.mean(2 * (w * x + b - y) * x)), float(np.mean(2 * (w * x + b - y)))],
    r2=float(lr.score(x[:, None], y)),
)

# Polynomial least squares
xp = rs.uniform(-1, 1, 30)
yp = np.sin(np.pi * xp) + rs.normal(0, 0.2, 30)
fx["poly"] = [
    dict(degree=d, x=xp.tolist(), y=yp.tolist(), coef=np.polynomial.polynomial.polyfit(xp, yp, d).tolist())
    for d in (1, 3, 7)
]

# Lasso and Ridge with an intercept (scikit-learn objectives)
X = rs.normal(size=(40, 4))
X[:, 1] += 0.7 * X[:, 0]
yl = X @ np.array([1.5, 0.0, -2.0, 0.3]) + 0.5 + rs.normal(0, 0.5, 40)
cases = []
for a in (0.01, 0.1, 0.5):
    m = Lasso(alpha=a, tol=1e-12, max_iter=100000).fit(X, yl)
    cases.append(dict(alpha=a, w=m.coef_.tolist(), b=float(m.intercept_)))
fx["lasso"] = dict(X=X.tolist(), y=yl.tolist(), cases=cases)
cases = []
for a in (0.1, 5.0):
    m = Ridge(alpha=a).fit(X, yl)
    cases.append(dict(alpha=a, w=m.coef_.tolist(), b=float(m.intercept_)))
fx["ridge"] = dict(cases=cases)

# Penalised quadratic used by the regularisation explorable (2 features, no intercept)
X2 = rs.normal(size=(60, 2))
X2[:, 1] = 0.6 * X2[:, 0] + 0.8 * X2[:, 1]
y2 = X2 @ np.array([1.2, 0.4]) + rs.normal(0, 0.3, 60)
m_ = len(y2)
H = X2.T @ X2 / m_
what = np.linalg.lstsq(X2, y2, rcond=None)[0]
fx["quad"] = dict(
    H=H.tolist(), what=what.tolist(),
    lasso=[dict(alpha=a, w=Lasso(alpha=a, fit_intercept=False, tol=1e-14, max_iter=100000).fit(X2, y2).coef_.tolist())
           for a in (0.05, 0.3, 0.9)],
    ridge=[dict(alpha=a, w=Ridge(alpha=2 * m_ * a, fit_intercept=False).fit(X2, y2).coef_.tolist())
           for a in (0.05, 0.5)],
)


# Optimisers: 25 steps on f(x, y) = x²/2 + 5y² + xy, compared with torch.optim
def f(t):
    return 0.5 * t[0] ** 2 + 5 * t[1] ** 2 + t[0] * t[1]


optimisers = {
    "gd": lambda p: torch.optim.SGD(p, lr=0.05),
    "momentum": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "rmsprop": lambda p: torch.optim.RMSprop(p, lr=0.05, alpha=0.99, eps=1e-8),
    "adam": lambda p: torch.optim.Adam(p, lr=0.05),
}
fx["optim"] = {}
for name, make in optimisers.items():
    t = torch.tensor([2.0, -1.5], dtype=torch.float64, requires_grad=True)
    opt = make([t])
    for _ in range(25):
        opt.zero_grad()
        f(t).backward()
        opt.step()
    fx["optim"][name] = t.detach().tolist()

# Normal distribution
fx["stats"] = dict(
    cdf=[[z, float(stats.norm.cdf(z))] for z in (-3, -1.5, -0.2, 0, 0.7, 2.5)],
    pdf=[[1.3, 0.5, 2.0, float(stats.norm.pdf(1.3, 0.5, 2.0))]],
)

Path(__file__).with_name("fixtures.json").write_text(json.dumps(fx, indent=1))
print("fixtures written")
