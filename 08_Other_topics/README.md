# Other Topics

> **Quick reference:** the key equations, hyperparameters and pitfalls for this module are on one page in [CHEATSHEET.md](CHEATSHEET.md).

## Responsible AI: Fairness & Interpretability

As ML models are used in high-stakes decisions (loan approvals, hiring, medical diagnosis), accuracy on a test set is no longer enough. Two further questions matter:

*   **Fairness:** Does the model disproportionately harm or benefit certain groups, especially groups defined by **sensitive attributes** such as race, sex or age?
*   **Interpretability / Explainability:** Can we understand *why* the model made a particular prediction? This builds trust, helps debugging, supports accountability, and can surface new insights about the data.

The notebook [`fairness_interpretability.ipynb`](fairness_interpretability.ipynb) trains a small MLP on the **UCI Adult** dataset and explains its predictions with **LIME** and **SHAP**. It then **measures fairness**: per-group selection rates, TPR and FPR for `sex` and `race`, and the four group-fairness metrics below, computed from scratch and checked against [`fairlearn`](https://fairlearn.org). It also mitigates the equal-opportunity gap with group-specific decision thresholds.

### Setup: The Adult (Census Income) Dataset

The task is binary classification: predict whether a person earns more than \$50K a year from census attributes (age, education, occupation, hours worked, etc.). The dataset contains sensitive attributes (`race`, `sex`), which makes it a standard benchmark for fairness research.

Pipeline used in the notebook:
1.  Download `adult.data` from the UCI repository into `./data/` (cached after the first run).
2.  Drop `fnlwgt`, missing values and duplicates, and convert `income` to a binary label.
3.  Preprocess with a `ColumnTransformer`: `StandardScaler` for numerical features and `OneHotEncoder` for categorical features (107 features after encoding).
4.  Train a 2-layer MLP (`107 → 64 → 1`, ReLU) with `BCEWithLogitsLoss` and Adam for 20 epochs, reaching about **85% test accuracy**.

## Fairness in Machine Learning

### Where Bias Comes From
*   **Data:** Historical bias (the world the data reflects was unfair), representation bias (some groups are under-sampled), measurement bias (proxies such as "arrests" standing in for "crime").
*   **Algorithm:** Objectives that optimise average accuracy can trade off minority-group performance for majority-group gains.
*   **Human interpretation:** How predictions are thresholded, used and acted on.

Simply dropping the sensitive attribute ("fairness through unawareness") rarely works, because other features (zip code, occupation, marital status) act as **proxies** for it.

### Group Fairness Metrics
Let $\large \hat{Y}$ be the prediction, $\large Y$ the true label and $\large A$ the sensitive attribute (e.g. $\large A \in \{a, b\}$).

**Demographic Parity:** Positive predictions are equally likely across groups.

$$\large
P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = b)
$$

**Equal Opportunity:** Qualified individuals are equally likely to be selected, i.e. equal **true positive rates**.

$$\large
P(\hat{Y} = 1 \mid Y = 1, A = a) = P(\hat{Y} = 1 \mid Y = 1, A = b)
$$

**Equalized Odds:** Equal true positive rates *and* equal false positive rates.

$$\large
P(\hat{Y} = 1 \mid Y = y, A = a) = P(\hat{Y} = 1 \mid Y = y, A = b), \quad y \in \{0, 1\}
$$

These criteria generally **cannot all be satisfied at once** when base rates differ between groups, so choosing a fairness definition is a decision about the application, not a purely technical one.

In practice, these are reported as a gap or ratio between groups, for example the *disparate impact ratio* $\large P(\hat{Y}=1 \mid A=a) / P(\hat{Y}=1 \mid A=b)$, where values below 0.8 are a common red flag (the "80% rule").

### What the notebook finds
On the Adult test set (threshold 0.5), the model selects men for ">50K" about 3× as often as women (disparate impact ratio ≈ 0.31, partly reflecting different base rates in the data), and it finds qualified women less often (TPR ≈ 0.54 vs 0.67). Choosing a separate threshold per group closes the TPR gap almost entirely (0.13 → 0.00) at a negligible accuracy cost. That is **post-processing** in the table below.

### Mitigation Strategies
| Stage | Idea | Examples |
|---|---|---|
| **Pre-processing** | Fix the data before training | Re-weighting or re-sampling groups, learning fair representations |
| **In-processing** | Change the learning objective | Fairness constraints or penalties, adversarial debiasing |
| **Post-processing** | Adjust predictions after training | Group-specific decision thresholds (e.g. to equalise TPRs) |

## Interpretability & Explainability (XAI)

Methods can be classified along two axes:

*   **Model-agnostic vs. model-specific:** Model-agnostic methods treat the model as a black box and only need its predictions (LIME, KernelSHAP). Model-specific methods use the model's internals (tree feature importance, gradient-based methods).
*   **Local vs. global:** Local explanations justify a *single* prediction. Global explanations describe the model's overall behaviour (e.g. averaging local attributions over many samples).

### LIME (Local Interpretable Model-agnostic Explanations)

**Idea:** A complex model may be non-linear globally, but around a single point it can be approximated well by a simple model.

For an instance $\large x$ to explain:
1.  Generate perturbed samples $\large z$ in the neighbourhood of $\large x$.
2.  Query the black-box model $\large f$ for predictions on each $\large z$.
3.  Weight each sample by its proximity to $\large x$ using a kernel $\large \pi_x(z)$.
4.  Fit an interpretable model $\large g$ (usually sparse linear) to these weighted predictions.

This amounts to solving:

$$\large
\xi(x) = \arg\min_{g \in G} \; \mathcal{L}(f, g, \pi_x) + \Omega(g)
$$

where $\large \mathcal{L}$ measures how poorly $\large g$ mimics $\large f$ near $\large x$, and $\large \Omega(g)$ penalises complexity (e.g. the number of non-zero weights). The coefficients of $\large g$ are the explanation.

In the notebook, `LimeTabularExplainer` explains a single test instance. It needs a `predict_fn` that returns probabilities for **both** classes, so the model's single sigmoid output is expanded to $\large [1 - p, \; p]$.

**Caveats:** Explanations depend on the perturbation scheme and kernel width, and can vary between runs. With one-hot features, perturbations may create unrealistic samples.

### SHAP (SHapley Additive exPlanations)

**Idea:** Treat features as players in a cooperative game whose "payout" is the prediction, and share the payout fairly using **Shapley values** from game theory.

The Shapley value of feature $\large i$ is its average marginal contribution over all subsets $\large S$ of the other features $\large F \setminus \{i\}$:

$$\large
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! \, (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]
$$

SHAP explanations are **additive**: the attributions sum to the difference between the prediction and a base value (the expected model output over a background dataset):

$$\large
f(x) = \phi_0 + \sum_{i=1}^{|F|} \phi_i, \qquad \phi_0 = \mathbb{E}[f(X)]
$$

Shapley values are the unique attributions satisfying *local accuracy*, *missingness* and *consistency*, which is SHAP's main theoretical advantage over LIME.

Exact computation is exponential in the number of features, so practical explainers approximate it:
*   **KernelExplainer:** Model-agnostic, sampling-based (slow).
*   **TreeExplainer:** Exact and fast for tree ensembles.
*   **DeepExplainer / GradientExplainer:** For neural networks. `GradientExplainer`, used in the notebook, is based on *expected gradients* (integrated gradients averaged over a background set).

In the notebook, 100 random training samples form the background set, SHAP values are computed on the model's **logit** for 5 test instances, and the results are shown with `shap.summary_plot`.

### LIME vs. SHAP
| | LIME | SHAP |
|---|---|---|
| **Foundation** | Local surrogate model | Cooperative game theory (Shapley values) |
| **Guarantees** | None; heuristic | Local accuracy, consistency, missingness |
| **Stability** | Can vary between runs | More stable (deterministic for exact explainers) |
| **Speed** | Fast | Depends on the explainer; KernelSHAP is slow |
| **Global view** | Not directly | Aggregating local values gives global importance |

Explanations describe **the model**, not the world: a feature with a large attribution is important to the model's decision, which is not the same as being causally important. Interpretability tools are also a practical way to audit fairness. For example, a large attribution on `sex_Male` or `race_White` is a warning sign worth investigating.
