# 00 · Cheat Sheet: Mathematical Foundation

> **Notebooks:** [linear_algebra](linear_algebra.ipynb) · [probability_statistics](probability_statistics.ipynb) · [calculus_optimization](calculus_optimization.ipynb) · [optimization_algorithms](optimization_algorithms.ipynb) · [autograd (scalar)](autograd%28scalar%29.ipynb) · [autograd (tensor)](autograd%28tensor%29.ipynb)
>
> **Full explanations:** [README](README.md)

## Linear algebra

| Concept | Formula | PyTorch |
|---|---|---|
| Dot product | $u \cdot v = \sum_i u_i v_i = \lVert u \rVert \lVert v \rVert \cos\theta$ | `u @ v` |
| L2 / L1 norm | $\lVert v \rVert_2 = \sqrt{\sum v_i^2}$, $\lVert v \rVert_1 = \sum \lvert v_i \rvert$ | `torch.linalg.norm(v, ord=...)` |
| Matrix product | $(AB)_{ij} = \sum_k A_{ik} B_{kj}$, needs $A: m \times n$, $B: n \times p$ | `A @ B` |
| Inverse | $A A^{-1} = I$ (square, full rank only) | `torch.linalg.inv`, prefer `torch.linalg.solve` |
| Eigendecomposition | $A v = \lambda v$; symmetric $A = V \Lambda V^\top$ | `torch.linalg.eigh` (symmetric), `eig` |
| SVD | $A = U \Sigma V^\top$ (any matrix) | `torch.linalg.svd` returns **$V^\top$**, not $V$ |
| Low-rank approx. | $A_k = U_k \Sigma_k V_k^\top$ (best rank-$k$ in Frobenius norm) | slice the SVD factors |

## Probability

| Distribution | PMF / PDF | Mean | Variance |
|---|---|---|---|
| Bernoulli($p$) | $p^k (1-p)^{1-k}$ | $p$ | $p(1-p)$ |
| Binomial($n, p$) | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Uniform($a, b$) | $\frac{1}{b-a}$ on $[a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Normal($\mu, \sigma^2$) | $\frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |

**Bayes' theorem:** $P(H \mid E) = \dfrac{P(E \mid H)\,P(H)}{P(E)}$, with $P(E) = P(E \mid H)P(H) + P(E \mid \neg H)P(\neg H)$.

## Calculus & optimisation

- **Gradient:** $\nabla f = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]$ points in the direction of steepest *ascent*.
- **Chain rule:** $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$. Backpropagation is the chain rule applied from the output backwards.
- **Numerical check:** central difference $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$ (error $O(h^2)$; use $h \approx 10^{-5}$).
- **Gradient descent:** $\theta \leftarrow \theta - \eta \nabla_\theta L$. **SGD** uses one sample or a mini-batch per step.
- **Convex** functions have one global minimum, so GD with a small enough $\eta$ finds it.

## Autograd essentials

| Op | Local derivative | | Op | Local derivative |
|---|---|---|---|---|
| $a + b$ | $1, 1$ | | $\tanh a$ | $1 - \tanh^2 a$ |
| $a \cdot b$ | $b, a$ | | $e^a$ | $e^a$ |
| $a^n$ | $n a^{n-1}$ | | $\ln a$ | $1/a$ |
| $AW$ (matrix) | $\bar{Z} W^\top$, $A^\top \bar{Z}$ | | broadcast | **sum** the gradient over broadcast dims |

## Pitfalls

- `.numpy()` fails on tensors that require grad, so use `.detach().numpy()`.
- Gradients **accumulate** in `.grad`, so zero them every step (`optimizer.zero_grad()`).
- In-place updates to parameters must happen inside `torch.no_grad()`.
- Learning rate too large → oscillation or divergence. For $f(x) = x^2$, anything with $\eta > 1$ diverges.
- `torch.var` is unbiased ($n-1$) by default, while NumPy's `np.var` and several scikit-learn estimators use $n$.
