# Cheat Sheet: Supervised Classification

Notebooks: [logistic_regression](logistic_regression.ipynb) · [naive_bayes](naive_bayes.ipynb) · [svm_kernels](svm_kernels.ipynb) · [knn](knn.ipynb) · [decision_trees](decision_trees.ipynb) · [ensembles](ensembles.ipynb) · Theory: [README](README.md)

## Key equations
| Model | Prediction | Loss / training |
|---|---|---|
| Logistic regression | $P(y=1 \mid x) = \sigma(w^\top x + b)$, $\sigma(z) = \frac{1}{1+e^{-z}}$ | BCE: $-\frac{1}{m}\sum [y \log \hat{p} + (1-y)\log(1-\hat{p})]$, gradient $\frac{1}{m} X^\top(\hat{p} - y)$ |
| Softmax (multi-class) | $P(y=k \mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}$ | Cross-entropy $-\log P(y = \text{true class})$ |
| Naive Bayes | $\hat{y} = \arg\max_c \log P(c) + \sum_j \log P(x_j \mid c)$ | Counting (closed form) |
| Linear SVM | $\hat{y} = \text{sign}(w^\top x + b)$ | $\frac{1}{2}\lVert w \rVert^2 + C \sum_i \max(0, 1 - y_i(w^\top x_i + b))$ |
| Kernel SVM | $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$ | Dual problem, only support vectors have $\alpha_i > 0$ |
| k-NN | Majority (or $1/d$-weighted) vote of the $k$ nearest training points | None: stores the data (lazy) |
| Decision tree | Follow $x_j \le t$ splits to a leaf, predict its majority class | Greedy splits maximising impurity decrease: Gini $1 - \sum p_k^2$, entropy $-\sum p_k \log_2 p_k$ |
| Random forest | Average the probabilities of $B$ deep trees | Each tree on a bootstrap sample, $\sqrt{d}$ random features per split |
| Gradient boosting | $F(x) = F_0 + \eta \sum_m h_m(x)$ | Each tree fits the negative gradient (residuals for MSE, $y - p$ for log-loss) |

**Naive Bayes likelihoods**
- **Gaussian:** $\mathcal{N}(x_j; \mu_{jc}, \sigma^2_{jc})$ for continuous features.
- **Multinomial:** $\frac{N_{jc} + \alpha}{N_c + \alpha d}$ for word counts.
- **Bernoulli:** $\frac{N_{jc} + \alpha}{N_c + 2\alpha}$ for binary features. It also penalises *absent* features.

**Kernels:** linear $x^\top x'$ · polynomial $(\gamma x^\top x' + r)^d$ · RBF $\exp(-\gamma \lVert x - x' \rVert^2)$

## Choosing a model
| Situation | Try |
|---|---|
| Need probabilities and interpretable weights | Logistic regression |
| Text or very high dimensions, tiny data, need speed | Naive Bayes |
| Clear margin, medium-sized data | Linear SVM |
| Non-linear boundary, < ~10k samples | RBF SVM |
| Low-dimensional data, need a quick non-parametric baseline | k-NN (scale features first) |
| Need a human-readable rule set | Shallow decision tree |
| Tabular data, strong default with little tuning | Random forest |
| Tabular data, best accuracy | Gradient boosting (`HistGradientBoosting`, XGBoost, LightGBM) with early stopping |

## Hyperparameters
- **Logistic `C` / SVM `C`:** inverse regularisation. Large $C$ gives less regularisation, so it can overfit.
- **RBF `gamma`:** the reach of each point. Large values give wiggly boundaries (overfitting), small values give near-linear ones. Tune `C` and `gamma` **together** on a log grid.
- **Naive Bayes `alpha`:** smoothing. $\alpha = 1$ is Laplace smoothing, and it prevents zero probabilities for unseen words.
- **Decision threshold:** 0.5 is arbitrary. Move it to trade precision against recall.
- **k-NN `k`:** small is jagged and overfits, large is smooth and underfits. Tune it by CV and prefer odd values.
- **Trees:** `max_depth`, `min_samples_leaf`, `ccp_alpha` (pruning). **Random forest:** `n_estimators` (more is never worse), `max_features`.
- **Boosting:** `learning_rate` × `n_estimators` (lower rate, more trees), `max_depth` 2–8, `subsample`. Use early stopping.
- **AdaBoost:** $\alpha_m = \eta\left(\log\frac{1-\varepsilon_m}{\varepsilon_m} + \log(K-1)\right)$. Misclassified sample weights grow by $e^{\alpha_m}$.

## Metrics
$\text{Precision} = \frac{TP}{TP + FP}$ · $\text{Recall} = \frac{TP}{TP + FN}$ · $F_1 = \frac{2PR}{P + R}$ · ROC-AUC is threshold-independent

## Pitfalls
- Accuracy is misleading on imbalanced classes, so check precision, recall and the confusion matrix.
- SVMs and regularised models need **scaled features**.
- SVM labels are $\pm 1$, not $0/1$, when writing the hinge loss by hand.
- Use `BCEWithLogitsLoss` on raw logits rather than `sigmoid` followed by `BCELoss`, because it's numerically stable.
- Naive Bayes probabilities are poorly calibrated (the independence assumption is usually false), even when its rankings are good.
- Compute in **log space** (sum log-probabilities) to avoid underflow.
- k-NN degrades in high dimensions (distances concentrate) and is slow at prediction time ($O(nd)$ per query).
- Single trees have high variance, and impurity-based feature importance favours high-cardinality features, so prefer permutation importance.
- Adding trees never overfits a random forest, but it **does** overfit boosting. Watch the validation curve.
