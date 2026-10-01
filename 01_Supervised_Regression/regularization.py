# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: develop_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/01_Supervised_Regression/regularization.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %% [markdown]
# # Regularization: Ridge & Lasso

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")

torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ### Why Regularization?
# - Prevents Overfitting: Especially when the number of features is large or features are highly correlated, standard linear regression models can overfit the training data, leading to poor generalization on unseen data. Overfitting often manifests as very large parameter weights.
# - Handles Multicollinearity: When features are highly correlated, the variance of the coefficient estimates can be large. Regularization helps to stabilize these estimates.
# - Feature Selection (Lasso): L1 regularization can shrink some feature weights exactly to zero, effectively performing feature selection.

# %% [markdown]
# How it works:</br></br>
# Regularization adds a penalty term to the cost function that penalizes large weights.</br>
# Cost_Regularized = Original_Cost (e.g., MSE) + Regularization_Term</br>

# %%
# Generate synthetic Data with more features

N_SAMPLES = 100
N_FEATURES = 10     # Total Features
N_INFORMATIVE_FEATURES = 5      # Number of features actually used to generate y
NOISE_LEVEL = 15.0
EFFECTIVE_RANK = N_INFORMATIVE_FEATURES     # For creating correlated features if desired

# %%
X_numpy, y_numpy, true_coefficients_numpy = make_regression(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE_FEATURES,
    noise=NOISE_LEVEL,
    coef=True, # Returns the true coefficients used to generate the data
    random_state=42,
    # effective_rank=EFFECTIVE_RANK # Can be used to introduce collinearity
)

# %%
X_numpy[:5], y_numpy[:5], y_numpy.shape

# %%
y_numpy = y_numpy.reshape(-1,1)

# %%
y_numpy.shape

# %%
print("True coefficients used by make_regression:")
print(true_coefficients_numpy) # "ideal" weights to recover for the informative features

# %%
# Standardize features (important for regularization, as it's sensitive to feature scales)
scaler = StandardScaler()
X_scaled_numpy = scaler.fit_transform(X_numpy)

# %%
X_scaled_numpy.shape

# %%
X = torch.from_numpy(X_scaled_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# %%
X[:5]

# %%
X.dtype

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.shape, y_train.shape


# %% [markdown]
# ## Standard Linear Regression (No Regularization)

# %%
def train_linear_regression(x, y, lr, epochs, reg_type=None, alpha=0.0):
    n_samples, n_features = x.shape

    weights = torch.randn(n_features, 1, requires_grad=True, dtype=torch.float32)
    bias = torch.randn(1, requires_grad=True, dtype=torch.float32)

    loss_history = []

    for epoch in range(epochs):
        # Forward pass: y_pred = X @ weights + bias
        y_predicted = x @ weights + bias 
        # Compute MSE loss
        mse_loss = torch.mean((y_predicted -y)**2)
        # Add regularization term to the loss (if any)
        total_loss = mse_loss
        if reg_type == 'l2': #Ridge
            l2_penalty = alpha * torch.sum(weights**2) #L2 norm squared of weights
            total_loss += l2_penalty
        elif reg_type == 'l1': #Lasso
            l1_penalty = alpha * torch.sum(torch.abs(weights)) #L1 norm of weights
            total_loss += l1_penalty

        loss_history.append(total_loss.item())

        # Backward pass: compute gradients
        if weights.grad is not None: weights.grad.zero_()
        if bias.grad is not None: bias.grad.zero_()
        total_loss.backward() #Gradients are computed for the total loss

        # Update parameters
        with torch.no_grad():
            weights -= lr * weights.grad
            bias -= lr * bias.grad

    return weights.detach(), bias.detach(), loss_history
            


# %%
lr = 0.01
epochs = 1000

# %%
weights_lm, bias_lm, loss_hist_lm = train_linear_regression(
    X_train, y_train, lr, epochs
    )

# %%
print(f"Standard LR - Bias: {bias_lm.item():.4f}")

# %%
print("Standard LR - Weights:")
for i, w in enumerate(weights_lm):
    print(f"  Feature {i}: {w.item():.4f}")

# %%
plt.figure(figsize=(6,4))
plt.plot(loss_hist_lm)
plt.title("Loss History - Standard Linear Regression")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); 
plt.show()

# %% [markdown]
# ## Ridge Regression (L2 Regularization)
# Cost_Ridge = MSE + alpha * sum(weights^2) </br>
# Gradient update for weights will have an additional term: - learning_rate * 2 * alpha * weights     </br>
# (This extra term comes from the derivative of alpha * sum(weights^2))

# %%
ALPHA_RIDGE = 1.0   # REgularization strength

# %%
weights_ridge, bias_ridge, loss_hist_ridge = train_linear_regression(
    X_train, y_train, lr, epochs, reg_type='l2', alpha=ALPHA_RIDGE
)

# %%
print(f"Ridge Regression (alpha={ALPHA_RIDGE}) - Bias: {bias_ridge.item():.4f}")

# %%
print(f"Ridge Regression - Weights (alpha={ALPHA_RIDGE}):")
for i, w in enumerate(weights_ridge):
    print(f"  Feature {i}: {w.item():.4f}")

# %%
plt.figure(figsize=(6,4))
plt.plot(loss_hist_ridge)
plt.title(f"Loss History - Ridge Regression (alpha={ALPHA_RIDGE})")
plt.xlabel("Epoch"); plt.ylabel("Total Loss (MSE + L2 Penalty)"); plt.show()

# %% [markdown]
# ## Lasso Regression (L1 Regularization)
# Cost_Lasso = MSE + alpha * sum(|weights|) </br>
# Gradient of sum(|weights|) w.r.t. w_j is alpha * sign(w_j) for w_j != 0.

# %%
# More advanced algorithms like Coordinate Descent or Proximal Gradient methods (e.g., ISTA)
# are typically used for Lasso to robustly handle the non-differentiability at zero.

# %%
ALPHA_LASSO = 0.1   # Regularization strength for Lasso (often smaller than Ridge alpha)

# %%
weights_lasso, bias_lasso, loss_hist_lasso = train_linear_regression(
    X_train, y_train, lr, epochs, reg_type='l1', alpha=ALPHA_LASSO
    )

# %%
weights_lasso[:5]

# %%
print(f"Lasso Regression (alpha={ALPHA_LASSO}) - Bias: {bias_lasso.item():.4f}")

# %%
print(f"Lasso Regression - Weights (alpha={ALPHA_LASSO}):")
for i, w in enumerate(weights_lasso):
    print(f"  Feature {i}: {w.item():.4f}") # Observe if some are close to zero

# %%
plt.figure(figsize=(6,4))
plt.plot(loss_hist_lasso)
plt.title(f"Loss History - Lasso Regression (alpha={ALPHA_LASSO})")
plt.xlabel("Epoch"); plt.ylabel("Total Loss (MSE + L1 Penalty)"); plt.show()

# %% [markdown]
# ## Comparing Learned Weights

# %%
# For better comparison, let's also train scikit-learn models.
# Our loss is  MSE + alpha * ||w||^2.  sklearn's Ridge minimises  ||y - Xw||^2 + alpha_sk * ||w||^2,
# which is the same objective (times n) when alpha_sk = alpha * n.
sklearn_ridge_model = SklearnRidge(alpha=ALPHA_RIDGE * len(X_train), fit_intercept=True)
sklearn_ridge_model.fit(X_train.numpy(), y_train.numpy())
weights_sklearn_ridge = sklearn_ridge_model.coef_.flatten() # sklearn weights are 1D array
bias_sklearn_ridge = sklearn_ridge_model.intercept_[0]

# %%
# Our loss is  MSE + alpha * ||w||_1.  sklearn's Lasso minimises  (1/2n)||y - Xw||^2 + alpha_sk * ||w||_1,
# which is the same objective (times 1/2) when alpha_sk = alpha / 2.
sklearn_lasso_model = SklearnLasso(alpha=ALPHA_LASSO / 2, fit_intercept=True, max_iter=2000) # Increased max_iter
sklearn_lasso_model.fit(X_train.numpy(), y_train.numpy())
weights_sklearn_lasso = sklearn_lasso_model.coef_.flatten()
bias_sklearn_lasso = sklearn_lasso_model.intercept_[0]

# %% [markdown]
# ### Verifying against scikit-learn
# With $\alpha$ rescaled as above, both sides minimise the same objective, so the coefficients should agree. Our versions use plain (sub)gradient descent for a fixed number of epochs, so they get a small tolerance. scikit-learn solves Ridge in closed form and Lasso by coordinate descent.

# %%
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from mlf_utils import check_close

sk_ols = SklearnLinearRegression().fit(X_train.numpy(), y_train.numpy())
check_close("Unregularised weights: manual GD vs sklearn", weights_lm.flatten(), sk_ols.coef_.flatten(), atol=0.01)
check_close("Ridge weights: manual GD vs sklearn", weights_ridge.flatten(), weights_sklearn_ridge, atol=0.001)
check_close("Lasso weights: manual subgradient vs sklearn", weights_lasso.flatten(), weights_sklearn_lasso, atol=0.01)

# %%
feature_indices = np.arange(N_FEATURES)
bar_width = 0.15

# %%
plt.figure(figsize=(15, 7))

plt.bar(feature_indices - 2*bar_width, weights_lm.numpy().flatten(), width=bar_width, label='Standard LR (Manual)', color='skyblue')
plt.bar(feature_indices - bar_width, weights_ridge.numpy().flatten(), width=bar_width, label=f'Ridge (Manual, alpha={ALPHA_RIDGE})', color='salmon')
plt.bar(feature_indices, weights_lasso.numpy().flatten(), width=bar_width, label=f'Lasso (Manual, alpha={ALPHA_LASSO})', color='lightgreen')
plt.bar(feature_indices + bar_width, weights_sklearn_ridge, width=bar_width, label=f'Ridge (Sklearn, alpha={ALPHA_RIDGE})', color='coral', hatch='//')
plt.bar(feature_indices + 2*bar_width, weights_sklearn_lasso, width=bar_width, label=f'Lasso (Sklearn, alpha={ALPHA_LASSO})', color='lime', hatch='xx')


# Plot true coefficients if available and meaningful (here they are for informative features)
# Remember only the first N_INFORMATIVE_FEATURES have non-zero true coefficients
true_coef_padded = np.zeros(N_FEATURES)
true_coef_padded[:N_INFORMATIVE_FEATURES] = true_coefficients_numpy[:N_INFORMATIVE_FEATURES] # Assuming make_regression puts informative first
plt.plot(feature_indices, true_coef_padded, 'ko--', label='True Coefficients (for informative features)', markersize=5, alpha=0.7)


plt.xlabel("Feature Index")
plt.ylabel("Weight Value")
plt.title("Comparison of Learned Weights by Different Models")
plt.xticks(feature_indices)
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1)) # Adjust legend position
plt.grid(True, axis='y', linestyle=':')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.show()


# %% [markdown]
# - Ridge Regression shrinks weights towards zero but rarely makes them exactly zero.
# - Lasso Regression can shrink weights exactly to zero, performing feature selection
# - (Our manual Lasso with GD might not make them perfectly zero, but sklearn's Coordinate Descent does).
# - The effect of regularization depends on the alpha value (strength).

# %% [markdown]
# ## Evaluation

# %%
def evaluate_model(weights, bias, X_test_data, y_test_data, model_name="Model"):
    with torch.no_grad():
        y_pred = X_test_data @ weights + bias
    mse = mean_squared_error(y_test_data.numpy(), y_pred.numpy())
    r2 = r2_score(y_test_data.numpy(), y_pred.numpy())
    print(f"\n{model_name} Evaluation:")
    print(f"  MSE on Test Set: {mse:.4f}")
    print(f"  RMSE on Test Set: {np.sqrt(mse):.4f}")
    print(f"  R-squared on Test Set: {r2:.4f}")
    return mse, r2


# %%
evaluate_model(weights_lm, bias_lm, X_test, y_test, "Standard LR (Manual)")
evaluate_model(weights_ridge, bias_ridge, X_test, y_test, f"Ridge (Manual, alpha={ALPHA_RIDGE})")
evaluate_model(weights_lasso, bias_lasso, X_test, y_test, f"Lasso (Manual, alpha={ALPHA_LASSO})")

# %%
# Evaluate Sklearn models
y_pred_sklearn_ridge = sklearn_ridge_model.predict(X_test.numpy())
mse_sklearn_ridge = mean_squared_error(y_test.numpy(), y_pred_sklearn_ridge)
r2_sklearn_ridge = r2_score(y_test.numpy(), y_pred_sklearn_ridge)
print(f"\nSklearn Ridge (alpha={ALPHA_RIDGE}) Evaluation:")
print(f"  MSE on Test Set: {mse_sklearn_ridge:.4f}, R2: {r2_sklearn_ridge:.4f}")

# %%
y_pred_sklearn_lasso = sklearn_lasso_model.predict(X_test.numpy())
mse_sklearn_lasso = mean_squared_error(y_test.numpy(), y_pred_sklearn_lasso)
r2_sklearn_lasso = r2_score(y_test.numpy(), y_pred_sklearn_lasso)
print(f"\nSklearn Lasso (alpha={ALPHA_LASSO}) Evaluation:")
print(f"  MSE on Test Set: {mse_sklearn_lasso:.4f}, R2: {r2_sklearn_lasso:.4f}")

# %% [markdown]
# ### Try it: regularisation strength
# Slide $\alpha$ on a log scale. Ridge shrinks every coefficient smoothly towards zero, while Lasso sets the uninformative ones to *exactly* zero (feature selection) long before the informative ones. The black markers show the true coefficients used to generate the data (the features are standardised, so they are on a different scale).
#
# *Interactive: run the notebook locally or in Colab to use the controls. GitHub only renders a static page.*

# %%
from ipywidgets import FloatLogSlider, interact

@interact(alpha=FloatLogSlider(value=0.1, base=10, min=-3, max=2, step=0.1, description="alpha", continuous_update=False))
def explore_alpha(alpha):
    n = len(X_train)
    ridge = SklearnRidge(alpha=alpha * n).fit(X_train.numpy(), y_train.numpy())
    lasso = SklearnLasso(alpha=alpha / 2, max_iter=10000).fit(X_train.numpy(), y_train.numpy())
    idx = np.arange(N_FEATURES)
    plt.figure(figsize=(10, 4))
    plt.bar(idx - 0.2, ridge.coef_.ravel(), width=0.4, label="Ridge (L2)")
    plt.bar(idx + 0.2, lasso.coef_.ravel(), width=0.4, label="Lasso (L1)")
    plt.scatter(idx, true_coefficients_numpy, color="k", marker="_", s=300, label="True (unscaled)")
    plt.axhline(0, color="gray", lw=0.8)
    plt.xticks(idx); plt.xlabel("Feature"); plt.ylabel("Coefficient"); plt.legend()
    plt.title(f"alpha = {alpha:.3g}: Lasso zeroed {int(np.sum(lasso.coef_ == 0))} of {N_FEATURES} coefficients")
    plt.show()

# %%

# %%

# %%
