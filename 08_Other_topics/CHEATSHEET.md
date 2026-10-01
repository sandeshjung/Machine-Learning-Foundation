# 08 · Cheat Sheet: Fairness & Interpretability

> **Notebook:** [fairness_interpretability](fairness_interpretability.ipynb)
>
> **Full explanations:** [README](README.md)

## Group fairness metrics

$\hat{Y}$ is the prediction, $Y$ the true label and $A$ the sensitive attribute.

| Criterion | Requires | Measure (report the gap between groups) |
|---|---|---|
| Demographic parity | $P(\hat{Y}=1 \mid A=a) = P(\hat{Y}=1 \mid A=b)$ | Selection rates. Disparate impact ratio < 0.8 is a red flag |
| Equal opportunity | Equal TPR: $P(\hat{Y}=1 \mid Y=1, A)$ | TPR difference |
| Equalized odds | Equal TPR **and** FPR | Max of the TPR and FPR differences |
| Calibration | $P(Y=1 \mid \hat{p}, A)$ equal across groups | Reliability curve per group |

When base rates differ between groups, these criteria **cannot all hold at once** (impossibility results), so the choice depends on the application.

## Where bias comes from and what to do

| Stage | Source | Mitigation |
|---|---|---|
| Data | Historical, representation, measurement bias | Re-weighting, re-sampling, better data collection |
| Training | Objective only optimises average accuracy | Fairness constraints, adversarial debiasing |
| Deployment | Thresholds, feedback loops | Group-aware thresholds (post-processing), monitoring |

"Fairness through unawareness" (dropping $A$) usually fails, because proxies such as zip code, occupation or marital status leak it.

## Explanation methods

| Method | Scope | Model-agnostic | Idea |
|---|---|---|---|
| Linear coefficients / tree importance | Global | No | Read the model's internals |
| Permutation importance | Global | Yes | Score drop when a feature is shuffled |
| Partial dependence / ICE | Global / local | Yes | Prediction as one feature varies |
| **LIME** | Local | Yes | Fit a weighted sparse linear surrogate around one instance |
| **SHAP** | Local, aggregates to global | Yes (Kernel), or model-specific | Shapley-value attributions |

**LIME:** $\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$. It's fast and intuitive, but unstable between runs and sensitive to the kernel width.

**SHAP:** $\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S \cup \{i\}) - f(S)]$ with $f(x) = \phi_0 + \sum_i \phi_i$. It satisfies local accuracy, consistency and missingness.

**SHAP explainers**
- `TreeExplainer`: exact and fast for trees
- `DeepExplainer` / `GradientExplainer`: neural networks
- `KernelExplainer`: any model, but slow
- `LinearExplainer`: linear models

## Pitfalls

- Explanations describe **the model, not the world**. High attribution isn't causal importance.
- Correlated features split or swap attribution, so group them or interpret with care.
- One-hot features: sum SHAP values over a category's columns to get that category's importance.
- LIME perturbations can create unrealistic samples, which makes explanations of tabular data fragile.
- Check which output you're explaining (logit vs probability, which class). SHAP base values differ.
- Always evaluate fairness **per group** on held-out data. Aggregate accuracy hides disparities.
