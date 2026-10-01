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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/04_Unsupervised_learning/pca_tsne.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE as SklearnTSNE 

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}") 
print(f"Scikit-learn Version: {sklearn.__version__}")

torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# # Dimensionality Reduction: PCA & t-SNE
#
# Dimensionality Reduction: The process of reducing the number of random variables (features)
# under consideration, by obtaining a set of principal variables.
# </br></br>
# Why Reduce Dimensions?
# - Visualization: High-dimensional data (more than 3 features) is hard to visualize.
# - Computational Efficiency: Fewer features can lead to faster model training.
# - Noise Reduction: Can remove irrelevant or noisy features.
# - Overcoming the Curse of Dimensionality: High-dimensional spaces can be sparse, making it harder for models to learn.

# %%
digits = load_digits()
X_digits_np = digits.data   # (n_samples, n_features=64)
y_digits_np = digits.target # (n_samples,)

print(f"Original Digits data shape: {X_digits_np.shape}")
print(f"Number of unique classes (digits): {len(np.unique(y_digits_np))}")

# %%
scaler_digits = StandardScaler()
X_digits_scaled_np = scaler_digits.fit_transform(X_digits_np)

# %%
X_digits_tensor = torch.from_numpy(X_digits_scaled_np).float()
y_digits_tensor = torch.from_numpy(y_digits_np).long()

# %%
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.ravel()
for i in range(10):
    axes[i].imshow(X_digits_np[i].reshape(8, 8), cmap='gray')
    axes[i].set_title(f"Label: {y_digits_np[i]}")
    axes[i].axis('off')
plt.suptitle("Sample Digits from the Dataset (Original 64D)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %% [markdown]
# ## Principal Component Analysis (PCA)

# %%
class PyTorchPCA:
    def __init__(self, n_components=None):
        self.n_components_ = n_components
        self.mean_ = None
        self.components_ = None # Principal axes in feature space (V_k.T)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X_tensor):
        """
        Assumes X_tensor is already centered if mean subtraction is desired globally.
        For strict PCA, data should be centered. Our input is StandardScaler output, so it's centered.
        """
        n_samples, n_features = X_tensor.shape
        X_centered = X_tensor # Assuming input X_tensor is already centered (e.g., by StandardScaler)

        # Perform SVD: X_centered = U @ diag(S_vec) @ V^T
        U, S_vec, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        # torch.linalg.svd returns V^T: its *rows* are the principal axes, so transpose to get one axis per column
        principal_axes = Vh.T
        explained_variance_full = (S_vec**2) / (n_samples - 1) # if n_samples > 1 else S_vec**2 / n_samples
        total_variance = torch.sum(explained_variance_full)
        explained_variance_ratio_full = explained_variance_full / total_variance

        if self.n_components_ is None:
            self.n_components_ = n_features
        
        self.components_ = principal_axes[:, :self.n_components_] 
        self.explained_variance_ = explained_variance_full[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio_full[:self.n_components_]
        
        print(f"  Selected {self.n_components_} components.")
        print(f"  Shape of components_ (V_k): {self.components_.shape}")

    def transform(self, X_tensor):
        if self.components_ is None:
            raise RuntimeError("PCA not fitted yet. Call fit() first.")
        X_centered = X_tensor
        X_pca = X_centered @ self.components_
        return X_pca

    def fit_transform(self, X_tensor):
        self.fit(X_tensor)
        return self.transform(X_tensor)


# %%
N_COMPONENTS_PCA = 2 # Reduce to 2 components for visualization
pca_torch = PyTorchPCA(n_components=N_COMPONENTS_PCA)
X_digits_pca_torch = pca_torch.fit_transform(X_digits_tensor)

# %% [markdown]
# ## Verifying against scikit-learn
# Explained variance must match exactly. Principal axes are only defined **up to sign** (both $v$ and $-v$ are valid eigenvectors), so we align each axis's sign with scikit-learn's before comparing the axes and the projected data.

# %%
from sklearn.decomposition import PCA
from mlf_utils import check_close

sk_pca = PCA(n_components=N_COMPONENTS_PCA).fit(X_digits_tensor.numpy())
check_close("Explained variance ratio vs sklearn", pca_torch.explained_variance_ratio_, sk_pca.explained_variance_ratio_, atol=1e-5)

signs = np.sign(np.sum(pca_torch.components_.numpy() * sk_pca.components_.T, axis=0))
check_close("Principal axes vs sklearn (up to sign)", pca_torch.components_.numpy() * signs, sk_pca.components_.T, atol=1e-4)
check_close("Projected data vs sklearn (up to sign)", X_digits_pca_torch.numpy() * signs, sk_pca.transform(X_digits_tensor.numpy()), atol=1e-3)

# %%
print(f"\nShape of data after PyTorch PCA: {X_digits_pca_torch.shape}") # Should be (n_samples, N_COMPONENTS_PCA)
print(f"Explained variance by component: {pca_torch.explained_variance_.numpy()}")
print(f"Explained variance ratio by component: {pca_torch.explained_variance_ratio_.numpy()}")
print(f"Total explained variance ratio: {torch.sum(pca_torch.explained_variance_ratio_).item():.4f}")

# %%
# visualize pca result
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_digits_pca_torch[:, 0].detach().numpy(),
    X_digits_pca_torch[:, 1].detach().numpy(),
    c=y_digits_np, 
    cmap='viridis',
    alpha=0.7,
    edgecolors='k',
    s=40
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"PCA of Digits Dataset ({N_COMPONENTS_PCA} Components) - PyTorch SVD")

legend_labels = [str(name) for name in digits.target_names]
handles, _ = scatter.legend_elements() 
if len(handles) > len(legend_labels): 
    handles = handles[:len(legend_labels)]
elif len(legend_labels) > len(handles): 
    legend_labels = legend_labels[:len(handles)]
plt.legend(handles=handles, labels=legend_labels, title="Digits")
plt.grid(True)
plt.show()

# %%
pca_torch_all_components = PyTorchPCA()
pca_torch_all_components.fit(X_digits_tensor)

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_torch_all_components.explained_variance_ratio_.numpy()), marker='o', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Explained Variance by Number of PCA Components")
plt.grid(True, which="both", ls="--")
plt.axhline(0.95, color='red', linestyle=':', label='95% Explained Variance') # Example threshold
plt.legend()
plt.show()

# %% [markdown]
# ## t-Distributed Stochastic Neighbor Embedding (t-SNE)
# A non-linear dimensionality reduction technique primarily used for visualization of high-dimensional datasets in low dimensions. 

# %%
# SKlearn's TSNE
N_COMPONENTS_TSNE = 2
PERPLEXITY = 30.0
N_ITER_TSNE = 1000
LEARNING_RATE_TSNE = 'auto'

# %%
X_digits_tensor.shape

# %%
tsne_sk = SklearnTSNE(
    n_components=N_COMPONENTS_TSNE,
    perplexity=PERPLEXITY,
    max_iter=N_ITER_TSNE,
    learning_rate=LEARNING_RATE_TSNE,
    init='pca',
    random_state=42,
    n_jobs=-1
    )

# %%
tsne_sk

# %%
X_digits_tsne_sk = tsne_sk.fit_transform(X_digits_tensor)

# %%

X_digits_tsne_sk[:5]

# %%
X_digits_tsne_sk.shape

# %%
plt.figure(figsize=(10, 7))
scatter_tsne = plt.scatter(
    X_digits_tsne_sk[:, 0],
    X_digits_tsne_sk[:, 1],
    c=y_digits_np, 
    cmap='viridis', 
    alpha=0.7,
    edgecolors='k',
    s=40
)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title(f"t-SNE Visualization of Digits Dataset (Perplexity={PERPLEXITY})")
legend_labels = [str(name) for name in digits.target_names]
handles, _ = scatter_tsne.legend_elements() 
if len(handles) > len(legend_labels): 
    handles = handles[:len(legend_labels)]
elif len(legend_labels) > len(handles): 
    legend_labels = legend_labels[:len(handles)]
plt.legend(handles=handles, labels=legend_labels, title="Digits")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Try it: t-SNE perplexity
# Perplexity is roughly the number of neighbours each point tries to stay close to. Low values emphasise very local structure (many small fragments), and high values preserve more global structure. This runs on a 600-point subset so each update takes a few seconds.
#
# *Interactive: run the notebook locally or in Colab to use the controls. GitHub only renders a static page.*

# %%
from ipywidgets import IntSlider, interact

subset = np.random.RandomState(0).choice(len(X_digits_scaled_np), 600, replace=False)

@interact(perplexity=IntSlider(value=30, min=2, max=100, step=2, continuous_update=False))
def explore_perplexity(perplexity):
    emb = SklearnTSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42,
                      max_iter=500).fit_transform(X_digits_scaled_np[subset])
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=y_digits_np[subset], cmap="tab10", s=15)
    plt.legend(*sc.legend_elements(), title="Digit", fontsize="small", loc="best")
    plt.title(f"t-SNE of 600 digits, perplexity = {perplexity}")
    plt.show()

# %%
