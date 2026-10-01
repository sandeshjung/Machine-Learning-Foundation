# 01 · Cheat Sheet: Supervised Regression

> **Notebooks:** [linear_regression](linear_regression.ipynb) · [polynomial_overfitting](polynomial_overfitting.ipynb) · [regularization](regularization.ipynb)
>
> **Full explanations:** [README](README.md)

## Key equations

| | Formula |
|---|---|
| Model | $\hat{y} = X\theta$ (with a column of ones for the bias) |
| MSE loss | $J(\theta) = \frac{1}{m} \lVert X\theta - y \rVert^2$ |
| Gradient | $\nabla_\theta J = \frac{2}{m} X^\top (X\theta - y)$ |
| Normal equation | $\theta^* = (X^\top X)^{-1} X^\top y$ (solve, don't invert) |
| Ridge (L2) | $J + \alpha \lVert w \rVert_2^2$, closed form $\theta^* = (X^\top X + \alpha' I)^{-1} X^\top y$ |
| Lasso (L1) | $J + \alpha \lVert w \rVert_1$, no closed form (use coordinate descent) |
| Elastic Net | $J + \alpha \left[ \rho \lVert w \rVert_1 + (1-\rho) \lVert w \rVert_2^2 \right]$ |
| $R^2$ | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ (1 = perfect, 0 = predicting the mean) |

## Choosing a method

| Situation | Use |
|---|---|
| Few features (< ~10k), exact answer wanted | Normal equation / `LinearRegression` |
| Very large datasets or online learning | (Mini-batch) gradient descent |
| Non-linear relationship | Polynomial features + linear model (or a non-linear model) |
| Many correlated features | Ridge |
| Many irrelevant features, want feature selection | Lasso |
| Both | Elastic Net |

## Hyperparameters

- **Learning rate $\eta$:** too small is slow, too large diverges. Standardise features first.
- **Polynomial degree:** controls bias vs variance. Pick it with validation data, never test data.
- **$\alpha$ (regularisation):** larger means simpler models. Search on a log scale ($10^{-3} \dots 10^{2}$).

## Pitfalls

- **Scale features** before GD or regularisation. Unscaled, un-centred features create narrow loss valleys, and the penalty hits large-scale features unevenly.
- **Don't penalise the bias.**
- Library $\alpha$ conventions differ. scikit-learn's `Ridge` uses $\lVert y - Xw \rVert^2 + \alpha \lVert w \rVert^2$ (no $1/m$), and `Lasso` uses $\frac{1}{2m} \lVert \cdot \rVert^2 + \alpha \lVert w \rVert_1$. Rescale before comparing.
- Fit the scaler and polynomial transform on the **training set only**, then `transform` the test set.
- High-degree polynomials explode outside the training range, so be careful when extrapolating.
- A low training error says nothing about generalisation. Always report validation or test error.
