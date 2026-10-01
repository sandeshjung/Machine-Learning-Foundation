# 03 · Cheat Sheet: Model Evaluation & Selection

> **Notebooks:** [bias_variance](bias_variance.ipynb) · [cross_validation](cross_validation.ipynb) · [hyperparameter_tuning](hyperparameter_tuning.ipynb) · [classification_metrics](classification_metrics.ipynb) · [preprocessing_pipelines](preprocessing_pipelines.ipynb)
>
> **Full explanations:** [README](README.md)

## Bias-variance decomposition

```math
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\text{Bias}[\hat{f}(x)]^2}_{\text{too simple}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{too sensitive to the data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
```

| Symptom | Diagnosis | Remedies |
|---|---|---|
| High train error ≈ high val error | **High bias** (underfitting) | Bigger or more flexible model, more features, less regularisation, train longer |
| Low train error ≪ val error | **High variance** (overfitting) | More data, regularisation, simpler model, dropout, early stopping, augmentation |
| Both low | Good fit | Confirm once on the held-out test set |

**Learning curves** (error against training-set size): high bias means both curves plateau high and close together, and more data won't help. High variance means a large gap that shrinks with more data.

## Data splits

- **Train:** fit parameters. **Validation:** choose hyperparameters and the model. **Test:** touch **once** at the end.
- **K-fold CV:** train on $K-1$ folds and validate on the remaining one, rotating $K$ times. Report mean ± std. $K = 5$ or $10$ is typical.
- **Stratified K-fold:** keeps class proportions in every fold. Use it for classification, especially with imbalanced data.
- **Time series:** use forward-chaining splits (`TimeSeriesSplit`) and never shuffle future data into training.
- **Grouped data** (several samples per patient or user): use `GroupKFold` so a group never appears on both sides.

## Hyperparameter search

| Method | How | When |
|---|---|---|
| Grid search | Every combination on a grid | ≤ 2–3 hyperparameters, cheap models |
| Random search | Sample from distributions | Many hyperparameters. It explores each dimension better for the same budget |
| Bayesian optimisation | Model the score and pick promising points | Expensive training runs (Optuna, scikit-optimize) |

Sample scale-type parameters (`C`, `gamma`, learning rate, `alpha`) on a **log scale**, e.g. `scipy.stats.loguniform`.

## Classification metrics

| Metric | Formula | Use when |
|---|---|---|
| Precision / Recall | $\frac{TP}{TP+FP}$ / $\frac{TP}{TP+FN}$ | False alarms / misses are expensive |
| $F_\beta$ | $(1+\beta^2)\frac{PR}{\beta^2 P + R}$ | One number, $\beta > 1$ favours recall |
| Balanced accuracy, MCC | $\frac{TPR + TNR}{2}$, correlation in $[-1, 1]$ | Imbalanced classes |
| ROC-AUC | $P(s^+ > s^-)$ | Threshold-free ranking (baseline 0.5) |
| Average precision | $\sum_k (R_k - R_{k-1}) P_k$ | Ranking for a rare positive class (baseline = positive rate) |
| Brier / log-loss, reliability diagram | $\frac{1}{n}\sum (\hat{p} - y)^2$ | Are the probabilities trustworthy? |

**Threshold:** cost-optimal $t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}$ for calibrated probabilities, or maximise $F_\beta$ on validation data. **Imbalance:** threshold moving (scores unchanged), `class_weight="balanced"`, or resampling the training set only. The last two distort probabilities, so re-calibrate (`CalibratedClassifierCV`).

## Preprocessing

| Step | Default choice | Notes |
|---|---|---|
| Missing numeric | Median + missing indicator | Understand MCAR / MAR / MNAR first |
| Missing categorical | Most frequent, or a "missing" category | |
| Nominal categories | One-hot, `handle_unknown="ignore"` | |
| Ordered categories | `OrdinalEncoder(categories=[explicit order])` | Never alphabetical |
| High-cardinality categories | `TargetEncoder` (cross-fitted) | Naive target means leak |
| Skewed positive features / target | `log1p` | Invert predictions when transforming the target |
| Scale | `StandardScaler`, or `RobustScaler` with outliers | Not needed for trees |

Assemble everything with `ColumnTransformer` + `Pipeline`, then cross-validate **the whole pipeline**.

## Pitfalls

- **Data leakage:** fitting a scaler, imputer or feature selector on the full dataset before splitting. Put preprocessing inside a `Pipeline` so it's refit on every fold.
- Tuning on the test set turns it into a validation set, and your reported score becomes optimistic.
- CV scores from hyperparameter search are slightly optimistic too. Use **nested CV** for an unbiased estimate.
- A single split is noisy. Compare models on the same CV folds and look at the variance, not just the mean.
- Set `random_state` for reproducible splits and searches.
