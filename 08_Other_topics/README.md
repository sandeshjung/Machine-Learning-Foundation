# 08 · Fairness & Interpretability

When ML models make high-stakes decisions about loans, hiring or medical care, test accuracy is not enough. Two more questions matter:

- **Fairness:** does the model treat groups differently, especially groups defined by **sensitive attributes** such as sex, race or age?
- **Interpretability:** can we understand **why** the model made a particular prediction?

> **Notebook:** [fairness_interpretability](fairness_interpretability.ipynb)
>
> **Quick revision:** [CHEATSHEET.md](CHEATSHEET.md)

## Contents

1. [The case study: UCI Adult](#1-the-case-study-uci-adult)
2. [Where bias comes from](#2-where-bias-comes-from)
3. [Group fairness metrics](#3-group-fairness-metrics)
4. [Mitigating unfairness](#4-mitigating-unfairness)
5. [Explaining predictions](#5-explaining-predictions)
6. [LIME](#6-lime)
7. [SHAP](#7-shap)
8. [LIME vs SHAP](#8-lime-vs-shap)

---

## 1. The case study: UCI Adult

**The task:** predict whether a person earns **more than \$50K a year** from census data (age, education, occupation, hours worked, …). The data includes `sex` and `race`, which makes it a standard fairness benchmark.

**The notebook pipeline:**

1. Download `adult.data` from UCI. It's cached in `./data/` after the first run.
2. Drop `fnlwgt`, rows with missing values and duplicates. Make `income` a 0/1 label.
3. Preprocess with a `ColumnTransformer`: `StandardScaler` for the numeric columns, `OneHotEncoder` for the categorical ones. That gives **107 features**.
4. Train a small MLP (`107 → 64 → 1`, ReLU) with `BCEWithLogitsLoss` and Adam for 20 epochs. It reaches about **85 % test accuracy**.

Then it:

- **measures fairness**: selection rate, TPR and FPR per group, plus the metrics below. These are computed from scratch and checked against [fairlearn](https://fairlearn.org).
- **mitigates** the gap with per-group thresholds
- **explains** the predictions with LIME and SHAP

---

## 2. Where bias comes from

| Source | Example |
|---|---|
| **Historical bias** | The data reflects a world that was already unfair (past hiring decisions) |
| **Representation bias** | Some groups are under-sampled, so the model knows them less well |
| **Measurement bias** | A proxy stands in for the real target ("arrests" for "crime") |
| **The algorithm** | Optimising *average* accuracy can trade minority-group performance for majority-group gains |
| **Deployment** | How predictions are thresholded and acted on, and the feedback loops this creates |

> [!WARNING]
> **"Fairness through unawareness" rarely works.** Dropping the sensitive column doesn't remove the information, because **proxies** such as zip code, occupation or marital status still carry it.

---

## 3. Group fairness metrics

Notation: $\hat{Y}$ is the prediction, $Y$ the true label, and $A$ the sensitive attribute, with groups $a$ and $b$.

| Criterion | Requires equal… | Formula |
|---|---|---|
| **Demographic parity** | Selection rates | $P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = b)$ |
| **Equal opportunity** | True positive rates: qualified people are found equally often | $P(\hat{Y} = 1 \mid Y = 1, A = a) = P(\hat{Y} = 1 \mid Y = 1, A = b)$ |
| **Equalised odds** | True positive rates **and** false positive rates | $P(\hat{Y} = 1 \mid Y = y, A = a) = P(\hat{Y} = 1 \mid Y = y, A = b)$ for $y \in \{0, 1\}$ |

In practice you report the **gap** (difference) or **ratio** between groups. The best-known ratio is the **disparate impact ratio**:

```math
\text{DI} = \frac{P(\hat{Y} = 1 \mid A = a)}{P(\hat{Y} = 1 \mid A = b)}
```

Values below **0.8** are a common red flag (the "80 % rule").

> [!IMPORTANT]
> When the groups have different base rates, these criteria **cannot all hold at once**. Choosing a definition of fairness is a decision about the application, not a purely technical one.

### What the notebook finds

At the default threshold of 0.5:

- **Selection rate:** men are predicted ">50K" about **3× as often** as women, a DI ratio of about **0.31**. This partly reflects different base rates in the data.
- **Equal opportunity:** among people who really earn >50K, the model finds **54 %** of women but **67 %** of men.
- **After per-group thresholds:** the TPR gap shrinks from **0.13 to about 0.00**, at a negligible cost in accuracy.

---

## 4. Mitigating unfairness

| Stage | Idea | Examples |
|---|---|---|
| **Pre-processing** | Fix the data before training | Re-weight or re-sample groups, learn fair representations |
| **In-processing** | Change the training objective | Fairness constraints or penalties, adversarial debiasing |
| **Post-processing** | Adjust the decisions after training | **Group-specific thresholds**, as used in the notebook |

The notebook includes sliders so you can move each group's threshold and watch the metrics change.

---

## 5. Explaining predictions

Explanation methods differ along two axes:

| | **Local**: explains one prediction | **Global**: explains the whole model |
|---|---|---|
| **Model-agnostic** (only needs predictions) | LIME, KernelSHAP | Permutation importance, partial dependence |
| **Model-specific** (uses the internals) | TreeSHAP, gradient methods | Tree feature importance, linear coefficients |

Averaging many local explanations, as with SHAP, also gives a global picture.

---

## 6. LIME

**LIME** stands for Local Interpretable Model-agnostic Explanations.

### 6.1 The idea

A complex model may be very non-linear overall, but **zoomed in around one point** it looks roughly linear. So LIME fits a simple model **locally** and reads off its weights.

### 6.2 How it works

To explain the prediction for an instance $x$:

1. **Perturb:** generate many samples $z$ near $x$.
2. **Query:** get the black-box predictions $f(z)$.
3. **Weight:** give samples closer to $x$ more weight, using a kernel $\pi_x(z)$.
4. **Fit** a simple model $g$, usually sparse linear, to these weighted predictions.

Formally:

```math
\xi(x) = \arg\min_{g \in G} \; \underbrace{\mathcal{L}(f, g, \pi_x)}_{\text{how badly } g \text{ mimics } f \text{ near } x} + \underbrace{\Omega(g)}_{\text{complexity of } g}
```

The coefficients of $g$ are the explanation.

### 6.3 In the notebook

`LimeTabularExplainer` explains one test instance. LIME expects probabilities for **both** classes, so the model's single sigmoid output $p$ is expanded to $[1 - p,\; p]$.

### 6.4 Caveats

- Results depend on how the samples are perturbed and on the kernel width.
- Explanations can **change between runs**.
- With one-hot features, the perturbed samples may be unrealistic, for example two categories "on" at once.

---

## 7. SHAP

**SHAP** stands for SHapley Additive exPlanations.

### 7.1 The idea

Treat the features as **players in a team game** whose payout is the prediction. Then split the payout fairly using **Shapley values** from game theory.

### 7.2 The Shapley value

Feature $i$'s Shapley value is its **average extra contribution** when it joins every possible subset $S$ of the other features:

```math
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!} \Big[ f(S \cup \{i\}) - f(S) \Big]
```

### 7.3 Additivity

The attributions add up exactly to the prediction:

```math
f(x) = \phi_0 + \sum_{i=1}^{|F|} \phi_i, \qquad \phi_0 = \mathbb{E}[f(X)] \text{ (the base value)}
```

Shapley values are the **only** attributions that satisfy all three of these properties, which is SHAP's main advantage over LIME:

- **local accuracy**: the attributions add up to the prediction
- **missingness**: an absent feature gets zero attribution
- **consistency**: if a feature matters more, its attribution doesn't go down

### 7.4 Explainers

The exact computation is exponential in the number of features, so the explainers approximate it:

| Explainer | Works with | Notes |
|---|---|---|
| `KernelExplainer` | Any model | Sampling-based, slow |
| `TreeExplainer` | Tree ensembles | Exact and fast |
| `DeepExplainer` / **`GradientExplainer`** | Neural networks | `GradientExplainer` uses *expected gradients* |
| `LinearExplainer` | Linear models | Exact |

### 7.5 In the notebook

- **Background set:** 100 random training samples.
- **Explained output:** the model's **logit** for 5 test instances.
- **Visualisation:** `shap.summary_plot`.

---

## 8. LIME vs SHAP

| | LIME | SHAP |
|---|---|---|
| Foundation | A local surrogate model | Game theory (Shapley values) |
| Guarantees | None, it's a heuristic | Local accuracy, missingness, consistency |
| Stability | Can vary between runs | More stable (deterministic for exact explainers) |
| Speed | Fast | Depends on the explainer. KernelSHAP is slow |
| Global view | Not directly | Average the local values to get global importance |

> [!WARNING]
> Explanations describe **the model, not the world**. A feature with a large attribution matters to the *model's decision*, which doesn't make it causally important.

> [!TIP]
> Interpretability tools double as **fairness audits**. A large attribution on `sex_Male` or `race_White` is a warning sign worth investigating.
