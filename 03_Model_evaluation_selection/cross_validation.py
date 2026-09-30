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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/cross_validation.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %% [markdown]
# ## Cross Validation

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")

torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# **Why Cross-Validation?** 
#
# - A single train-test split can be sensitive to how the data is divided. The model's performance on one particular test set might not be representative of its true generalization ability.
# - It's especially useful when the dataset is relatively small.
# - Essential for reliable hyperparameter tuning.

# %%
# Generating synthetic classification data
# Binary classification dataset
N_SAMPLES_CV = 200
X_np_cv, y_np_cv = make_classification(
    n_samples=N_SAMPLES_CV, n_features=5, n_informative=3, n_redundant=0, 
    n_clusters_per_class=1, random_state=42, flip_y=0.05
    )

# %%
X_np_cv[:5], y_np_cv[:5]

# %%
scaler_cv = StandardScaler()
X_scaled_np_cv = scaler_cv.fit_transform(X_np_cv)

# %%
X_scaled_np_cv[:5]

# %%
X_cv = torch.from_numpy(X_scaled_np_cv).float()
y_cv = torch.from_numpy(y_np_cv).float().unsqueeze(1)

# %%
X_cv.shape, y_cv.shape


# %%
class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1) 

    def forward(self, x):
        return self.linear(x) # Return logits


# %%
def train_pytorch_model(model_instance, X_train_fold, y_train_fold,
                        criterion, optimizer, num_epochs=100, verbose=False):
    """Helper function to train a PyTorch model for one fold."""
    model_instance.train() 
    epoch_losses = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model_instance(X_train_fold) 
        
        # Compute loss
        loss = criterion(outputs, y_train_fold) # Criterion expects logits if BCEWithLogitsLoss
        epoch_losses.append(loss.item())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % (num_epochs // 2) == 0 :
             print(f"    Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return epoch_losses


# %%
def evaluate_pytorch_model(model_instance, X_val_fold, y_val_fold, threshold=0.5):
    """Helper function to evaluate a PyTorch binary classification model."""
    model_instance.eval() # Set model to evaluation mode
    with torch.no_grad():
        logits = model_instance(X_val_fold)
        probabilities = torch.sigmoid(logits) # Convert logits to probabilities
        predictions = (probabilities >= threshold).float()
    
    accuracy = accuracy_score(y_val_fold.cpu().numpy(), predictions.cpu().numpy())
    return accuracy


# %% [markdown]
# ### K-Fold Cross-Validation
#
# The dataset is divided into K equally (for neraly equally) sized "folds". The model is trained K times:
#
# - In each iteration, K-1 folds are used for training, and 1 fold is used for validation.
# - Performance metrics are collected from each validation fold.
# - The overall performance is typically the average of the metrics across all K folds.

# %%
N_SPLITS_KFold = 5 
kfold = KFold(n_splits=N_SPLITS_KFold, shuffle=True, random_state=42)

# %%
fold_accuracies_kfold = []
current_fold = 0
for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(X_cv, y_cv)):
    current_fold += 1
    print(f"\n--- Fold {current_fold}/{N_SPLITS_KFold} ---")

    # Get training and validation data for this fold
    X_train_fold, y_train_fold = X_cv[train_indices], y_cv[train_indices]
    X_val_fold, y_val_fold = X_cv[val_indices], y_cv[val_indices]
    
    print(f"  Train samples: {X_train_fold.shape[0]}, Validation samples: {X_val_fold.shape[0]}")

    # Initialize model,
    input_dim_cv = X_cv.shape[1]
    model_kfold = LogisticRegressionPyTorch(input_dim_cv)
    
    # Use BCEWithLogitsLoss for numerical stability (expects raw logits from model)
    criterion_kfold = nn.BCEWithLogitsLoss() 
    optimizer_kfold = optim.SGD(model_kfold.parameters(), lr=0.05)
    
    # Train the model
    print(f"  Training model for Fold {current_fold}...")
    train_pytorch_model(model_kfold, X_train_fold, y_train_fold,
                        criterion_kfold, optimizer_kfold, num_epochs=150, verbose=False) # Less verbose in loop
    
    # Evaluate
    val_accuracy = evaluate_pytorch_model(model_kfold, X_val_fold, y_val_fold)
    fold_accuracies_kfold.append(val_accuracy)
    print(f"  Fold {current_fold} Validation Accuracy: {val_accuracy:.4f}")


# %%
mean_accuracy_kfold = np.mean(fold_accuracies_kfold)
std_accuracy_kfold = np.std(fold_accuracies_kfold)

# %% [markdown]
# #### Sanity check against scikit-learn
# Using **the same folds**, scikit-learn's `LogisticRegression` should reach a similar mean accuracy. Our model uses 150 epochs of plain SGD and scikit-learn applies mild L2 regularisation by default, so we allow a small gap.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from mlf_utils import check_close

sk_cv_scores = cross_val_score(LogisticRegression(), X_scaled_np_cv, y_np_cv, cv=kfold)
print(f"sklearn fold accuracies: {np.round(sk_cv_scores, 4)}")
check_close("Mean K-fold accuracy vs sklearn", mean_accuracy_kfold, sk_cv_scores.mean(), atol=0.03)

# %%
print("\n--- K-Fold CV Results ---")
print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies_kfold]}")
print(f"Mean Validation Accuracy: {mean_accuracy_kfold:.4f}")
print(f"Standard Deviation of Validation Accuracy: {std_accuracy_kfold:.4f}")

# %%
plt.figure(figsize=(8, 5))
plt.bar(range(1, N_SPLITS_KFold + 1), fold_accuracies_kfold, color='skyblue', label='Fold Accuracy')
plt.axhline(mean_accuracy_kfold, color='red', linestyle='--', label=f'Mean Accuracy: {mean_accuracy_kfold:.4f}')
plt.xlabel("Fold Number")
plt.ylabel("Validation Accuracy")
plt.title(f"K-Fold Cross-Validation Accuracies (K={N_SPLITS_KFold})")
plt.xticks(range(1, N_SPLITS_KFold + 1))
plt.ylim(min(fold_accuracies_kfold) - 0.05, max(fold_accuracies_kfold) + 0.05)
plt.legend()
plt.show()

# %% [markdown]
# ### Stratified K-Fold Cross-Validation
#
# - **Problem with KFold:** If classes are imbalanced, some folds might end up with very few or even zero samples of a particular class, leading to unreliable evaluation.
# - **Stratified K-Fold:** Variation of K-Fold that returns stratified folds. Each fold is made by preserving the percentage of samples for each class as in the original dataset. Particularly, imporant for classification tasks with imbalanced class distributions.

# %%
N_SPLITS_STRATIFIED = 5
stratified_kfold = StratifiedKFold(n_splits=N_SPLITS_STRATIFIED, shuffle=True, random_state=42)

# %%
fold_accuracies_stratified = []
current_fold_strat = 0
for fold_idx, (train_indices, val_indices) in enumerate(stratified_kfold.split(X_cv, y_cv.squeeze())):
    current_fold_strat += 1
    print(f"\n--- Stratified Fold {current_fold_strat}/{N_SPLITS_STRATIFIED} ---")

    X_train_fold_s, y_train_fold_s = X_cv[train_indices], y_cv[train_indices]
    X_val_fold_s, y_val_fold_s = X_cv[val_indices], y_cv[val_indices]
    
    print(f"  Train samples: {X_train_fold_s.shape[0]}, Validation samples: {X_val_fold_s.shape[0]}")
    print(f"  Train class balance: Class 0: {(y_train_fold_s==0).sum().item()}, Class 1: {(y_train_fold_s==1).sum().item()}")
    print(f"  Val class balance:   Class 0: {(y_val_fold_s==0).sum().item()}, Class 1: {(y_val_fold_s==1).sum().item()}")


    input_dim_cv_s = X_cv.shape[1]
    model_stratified_kfold = LogisticRegressionPyTorch(input_dim_cv_s)
    criterion_stratified_kfold = nn.BCEWithLogitsLoss()
    optimizer_stratified_kfold = optim.SGD(model_stratified_kfold.parameters(), lr=0.05)
    
    print(f"  Training model for Stratified Fold {current_fold_strat}...")
    train_pytorch_model(model_stratified_kfold, X_train_fold_s, y_train_fold_s,
                        criterion_stratified_kfold, optimizer_stratified_kfold, num_epochs=150, verbose=False)
    
    val_accuracy_s = evaluate_pytorch_model(model_stratified_kfold, X_val_fold_s, y_val_fold_s)
    fold_accuracies_stratified.append(val_accuracy_s)
    print(f"  Stratified Fold {current_fold_strat} Validation Accuracy: {val_accuracy_s:.4f}")

# %%
mean_accuracy_stratified = np.mean(fold_accuracies_stratified)
std_accuracy_stratified = np.std(fold_accuracies_stratified)

# %%
print("\n--- Stratified K-Fold CV Results ---")
print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies_stratified]}")
print(f"Mean Validation Accuracy: {mean_accuracy_stratified:.4f}")
print(f"Standard Deviation of Validation Accuracy: {std_accuracy_stratified:.4f}")

# %%
plt.figure(figsize=(8, 5))
plt.bar(range(1, N_SPLITS_STRATIFIED + 1), fold_accuracies_stratified, color='lightgreen', label='Fold Accuracy')
plt.axhline(mean_accuracy_stratified, color='darkgreen', linestyle='--', label=f'Mean Accuracy: {mean_accuracy_stratified:.4f}')
plt.xlabel("Fold Number")
plt.ylabel("Validation Accuracy")
plt.title(f"Stratified K-Fold CV Accuracies (K={N_SPLITS_STRATIFIED})")
plt.xticks(range(1, N_SPLITS_STRATIFIED + 1))
plt.ylim(min(fold_accuracies_stratified) - 0.05, max(fold_accuracies_stratified) + 0.05)
plt.legend()
plt.show()

# %%
print(f"\nComparison of Mean Accuracies:")
print(f"  K-Fold (standard):        {mean_accuracy_kfold:.4f} +/- {std_accuracy_kfold:.4f}")
print(f"  Stratified K-Fold:        {mean_accuracy_stratified:.4f} +/- {std_accuracy_stratified:.4f}")
print("Stratified K-Fold often gives a more reliable estimate, especially with class imbalance.")

# %% [markdown]
# ### Using Cross-Validation for Hyperparameter Tuning (Conceptual)
# Cross-validation is essential for robust hyperparameter tuning. The general process:
# 1. Define a grid of hyperparameters to search (e.g., different learning rates,
#    regularization strengths, polynomial degrees, kernel parameters for SVM).
# 2. For each combination of hyperparameters:
#    
#    1. Perform K-Fold (or Stratified K-Fold) cross-validation.
#    2. Calculate the average validation performance (e.g., mean accuracy, mean F1-score)
#       across the K folds for that hyperparameter combination.
# 3. Select the hyperparameter combination that yields the best average validation performance.
# 4. (Optional but recommended) Retrain your model on the *entire training dataset*
#    using the best hyperparameters found.
# 5. Finally, evaluate this retrained model on a completely separate, held-out test set
#    (that was not used at all during CV or hyperparameter tuning) to get a final
#    estimate of its generalization performance.

# %%
