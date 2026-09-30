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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/hyperparameter_tuning.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %% [markdown]
# ### Hyperparameter Tuning

# %%
import torch
import time
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from scipy.stats import expon, uniform, randint

torch.manual_seed(42)
np.random.seed(42)

# %%
device = torch.device("cpu")    # explicity CPU for this scikit-learn 

# %%
X_np, y_np = make_classification(
    n_samples=2000, 
    n_features=20, 
    n_informative=15,
    n_redundant=3, 
    n_clusters_per_class=2,
    n_classes=2, 
    class_sep=0.8,  
    random_state=42
)

# %%
X_torch = torch.from_numpy(X_np).float().to(device)
y_torch = torch.from_numpy(y_np).long().to(device)

# %%
X_torch.shape, y_torch.shape

# %%
X_train_torch, X_test_torch, y_train_torch, y_test_torch = train_test_split(
    X_torch, y_torch, test_size=0.3, random_state=42, stratify=y_torch.cpu() 
)

# %%
X_train_torch.shape, X_test_torch.shape

# %%
scaler = StandardScaler()

# %%
X_train_scaled = scaler.fit_transform(X_train_torch)
X_test_scaled = scaler.transform(X_test_torch)

# %% [markdown]
# #### Basline Model (Default Hyperparameters)

# %%
baseline_model = SVC(C=1.0, kernel='linear', random_state=42)
baseline_model.fit(X_train_scaled, y_train_torch)

# %%
y_pred_baseline_torch = torch.from_numpy(baseline_model.predict(X_test_scaled))
y_pred_baseline_torch[:5], y_test_torch[:5]

# %%
baseline_accuracy = accuracy_score(y_test_torch, y_pred_baseline_torch)
baseline_accuracy

# %%
print(classification_report(y_test_torch, y_pred_baseline_torch))

# %% [markdown]
# #### Grid Search CV

# %%
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'degree': [2, 3]
}

# %%
grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# %%
print("Starting Grid Search...")
start_time_grid = time.time()
grid_search.fit(X_train_scaled, y_train_torch)
end_time_grid = time.time()
grid_search_time = end_time_grid - start_time_grid
print(f"\nGrid Search completed in {grid_search_time:.2f} seconds.")

# %%
grid_search.best_params_, grid_search.best_score_

# %%
best_grid_model = grid_search.best_estimator_

# %%
y_pred_grid_torch = best_grid_model.predict(X_test_scaled)
grid_accuracy_test = accuracy_score(y_test_torch, y_pred_grid_torch)
grid_accuracy_test

# %%
print(classification_report(y_test_torch, y_pred_grid_torch))

# %% [markdown]
# #### Randomized Search CV

# %%
param_dist = {
    'C': expon(scale=10),  
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': expon(scale=0.1),  
    'degree': randint(2, 5), 
    'coef0': uniform(-1, 2)  
}

# %%
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=50, 
    cv=5,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# %%
print("Starting Randomized Search...")
start_time_random = time.time()
random_search.fit(X_train_scaled, y_train_torch)
end_time_random = time.time()
random_search_time = end_time_random - start_time_random

print(f"\nRandomized Search completed in {random_search_time:.2f} seconds.")

# %%
random_search.best_params_, random_search.best_score_

# %%
best_random_model = random_search.best_estimator_
y_pred_random_torch = best_random_model.predict(X_test_scaled)
random_accuracy_test = accuracy_score(y_test_torch, y_pred_random_torch)
random_accuracy_test

# %%
print(classification_report(y_test_torch, y_pred_random_torch))

# %% [markdown]
# #### Comparison and Conclusion

# %%
print("\n\n--- Hyperparameter Tuning Summary ---")
print(f"Baseline SVC Accuracy: {baseline_accuracy:.4f}")

print("\nGrid Search CV:")
print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"  Test Set Accuracy: {grid_accuracy_test:.4f}")
print(f"  Best Parameters: {grid_search.best_params_}")
print(f"  Execution Time: {grid_search_time:.2f} seconds")
num_grid_combinations = 1
for param_values_list in param_grid.values(): # Renamed to avoid conflict
    num_grid_combinations *= len(param_values_list)
print(f"  Models trained (combinations * cv_folds): {num_grid_combinations * grid_search.cv}")


print("\nRandomized Search CV:")
print(f"  Best CV Accuracy: {random_search.best_score_:.4f}")
print(f"  Test Set Accuracy: {random_accuracy_test:.4f}")
print(f"  Best Parameters: {random_search.best_params_}")
print(f"  Execution Time: {random_search_time:.2f} seconds")
print(f"  Models trained (n_iter * cv_folds): {random_search.n_iter * random_search.cv}")


# %% [markdown]
# 1.  **Performance:** Both Grid Search and Randomized Search typically find models that outperform the baseline model.
# 2.  **Execution Time:** Grid Search is exhaustive and can be slow. Randomized Search is often faster for a fixed budget of iterations.
# 3.  **Effectiveness:** Grid Search finds the best within its grid. Randomized Search explores more broadly and can find good solutions efficiently.

# %%

# %%

# %%

# %%

# %%
