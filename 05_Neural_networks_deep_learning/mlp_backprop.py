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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/mlp_backprop.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %%
import torch
import torchvision 
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Multi-Layer Perceptrons & Backpropagation from Scratch
#
# An MLP is a feedforward neural network made of:
# - An input layer
# - One or more hidden layers
# - An output layer
#
# <p>Each layer (except the input layer) applies a linear transformation (weights and biases) followed by a non-linear activation function.</p>
#
# This notebook trains a **3-layer MLP (784 → 256 → 256 → 26, tanh)** on the EMNIST *letters* dataset **without** `loss.backward()` or `torch.optim`. The forward pass, the softmax/cross-entropy gradient, backpropagation through every layer, and the gradient-descent update are all written out manually.
#
# Theory: [Multi-Layer Perceptrons (MLPs)](README.md#1-multi-layer-perceptrons-and-backpropagation) · [Backpropagation](README.md#14-backpropagation)

# %% [markdown]
# ## Data: EMNIST Letters
# 28×28 grayscale images of handwritten letters. The dataset (about 500 MB) is downloaded into `./data` on first run.

# %%
dataset = torchvision.datasets.EMNIST(root="./data", download=True, split="letters")
dataset

# %%
letter_images = {}

for img, label in dataset:
    if label not in letter_images and 1 <= label <= 26:
        letter_images[label] = img
    if len(letter_images) == 26:
        break

sorted_labels = sorted(letter_images.keys())

fig, axes = plt.subplots(1, 26, figsize=(20, 3))
for idx, label in enumerate(sorted_labels):
    axes[idx].imshow(letter_images[label], cmap='gray')
    axes[idx].set_title(chr(label + 64))  # Convert label to letter
    axes[idx].axis('off')

plt.suptitle("EMNIST Letters", fontsize=16)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Train / test split
# Pixels are scaled to $[0, 1]$ and the first 80% of samples are used for training.

# %%
X = dataset.data
Y = dataset.targets

# %%
len(dataset) * 0.8

# %%
train_size = 99840
Xtrain = X[:train_size] /255
Ytrain = Y[:train_size]
Ytrain = Y[:train_size].unsqueeze(1)

Xtest = X[train_size:] /255
Ytest = Y[train_size:]
Ytest = Y[train_size:].unsqueeze(1)

# %%
X.shape

# %%
28*28

# %% [markdown]
# ## Network hyperparameters

# %%
ninput = 784
nhidden = 256
nhidden2 = 256
batch_size =64
nclasses = len(dataset.classes)

# %% [markdown]
# ## Parameter initialisation
# Weights use **Xavier/Glorot** scaling, $W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$, which keeps activation variance roughly constant across tanh layers (see [Weight Initialization](README.md#16-weight-initialisation)).

# %%
# Layer 1 (Input to Hidden)
W1 = torch.randn(ninput, nhidden, requires_grad=True) * torch.sqrt(torch.tensor(2.0) / (ninput + nhidden))
b1 = torch.randn(1, nhidden, requires_grad=True)
# Layer 2 (Hidden to Hidden)
W2 = torch.randn(nhidden, nhidden2, requires_grad=True)* torch.sqrt(torch.tensor(2.0) / (nhidden + nhidden2))
b2 = torch.randn(1, nhidden2, requires_grad=True)
# Layer 2 (Hidden to Output)
W3 = torch.randn(nhidden2, nclasses, requires_grad=True)* torch.sqrt(torch.tensor(2.0) / (nhidden + nhidden2))
b3 = torch.zeros(1, nclasses, requires_grad=True)

# %%
num_batches = len(Xtrain) // batch_size

# %% [markdown]
# ## Training loop: manual forward & backward pass
# For each mini-batch:
# 1. **Forward:** $Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}$, $A^{(l)} = \tanh(Z^{(l)})$, followed by a numerically stable softmax (subtract the row max) and the cross-entropy loss.
# 2. **Backward:** apply the chain rule step by step through the softmax, the log and each layer, using $\frac{d}{dz}\tanh z = 1 - \tanh^2 z$.
# 3. **Update:** $\theta \leftarrow \theta - \alpha \, \nabla_\theta L$ with learning rate $\alpha = 0.1$.

# %%
alpha = 0.1
losses = []
val_losses = []

for epoch in range(10):
    epoch_loss = 0.0
    for i in range(num_batches):
        # Create batch
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = Xtrain[start_idx:end_idx]
        Y_batch = Ytrain[start_idx:end_idx]

        # Forward pass
        Z1 = X_batch.view(-1, ninput) @ W1 + b1     # Linear Transformation
        A1 = torch.tanh(Z1)     # Activation
        Z2 = A1 @ W2 + b2       
        A2 = torch.tanh(Z2)
        Z3 = A2 @ W3 + b3
        zmax = Z3.max(dim=1, keepdim=True).values
        znorm = Z3 - zmax
        zexp = znorm.exp()
        zexp_sum = zexp.sum(dim=1, keepdim=True)
        zexp_sum_inv = zexp_sum ** (-1)
        probs = zexp * zexp_sum_inv
        log_probs = probs.log()
        L = -log_probs[torch.arange(len(Y_batch)), Y_batch.squeeze()].mean()
        epoch_loss += L.item()

        # Backward pass
        dL_dL = torch.ones_like(L)
        dL_dlogprobs = torch.zeros_like(log_probs)
        dL_dlogprobs[torch.arange(len(Y_batch)), Y_batch.squeeze()] = -dL_dL / len(Y_batch)
        dL_dprobs = dL_dlogprobs * 1 / probs
        dL_dzexp = dL_dprobs * zexp_sum_inv
        dL_dzexp_sum_inv = (dL_dprobs * zexp).sum(1, keepdim=True)
        dL_dzexp_sum = -1 * dL_dzexp_sum_inv * zexp_sum**(-2)
        dL_dzexp += dL_dzexp_sum
        dL_dznorm = dL_dzexp * zexp.clone()
        dL_dzmax = -dL_dznorm.sum(1, keepdim=True)
        dL_dZ = dL_dznorm
        dL_dZ += torch.nn.functional.one_hot(Z3.max(dim=1).indices, nclasses) * dL_dzmax
        dL_dW3 = A2.T @ dL_dZ
        dL_db3 = dL_dZ.sum(0, keepdim=True)
        dL_dA2 = dL_dZ @ W3.T
        dL_dZ2 = dL_dA2 * (1 - A2**2)
        dL_dW2 = A1.T @ dL_dZ2
        dL_db2 = dL_dZ2.sum(0, keepdim=True)
        dL_dA1 = dL_dZ2 @ W2.T
        dL_dZ1 = dL_dA1 * (1 - A1**2)
        dL_dW1 = X_batch.view(-1, ninput).T @ dL_dZ1
        dL_db1 = dL_dZ1.sum(0, keepdim=True)

        with torch.no_grad():
            W1 -= alpha * dL_dW1
            b1 -= alpha * dL_db1
            W2 -= alpha * dL_dW2
            b2 -= alpha * dL_db2
            W3 -= alpha * dL_dW3
            b3 -= alpha * dL_db3

    # Calculate validation loss
    val_loss = 0.0
    with torch.no_grad():
        for j in range(len(Xtest) // batch_size):
            start_idx = j * batch_size
            end_idx = (j + 1) * batch_size
            X_val = Xtest[start_idx:end_idx]
            Y_val = Ytest[start_idx:end_idx]

            Z1_val = X_val.view(-1, ninput) @ W1 + b1
            A1_val = torch.tanh(Z1_val)
            Z2_val = A1_val @ W2 + b2
            A2_val = torch.tanh(Z2_val)
            Z3_val = A2_val @ W3 + b3
            val_loss += -torch.nn.functional.log_softmax(Z3_val, dim=1)[torch.arange(len(Y_val)), Y_val.squeeze()].mean().item()
    val_loss /= (len(Xtest) // batch_size)

    losses.append(epoch_loss / num_batches)
    val_losses.append(val_loss)

    print(f"Epoch {epoch}, Loss: {losses[-1]}, Validation Loss: {val_losses[-1]}")

# %% [markdown]
# ## Gradient check: manual backprop vs autograd
# The loop's variables from the **last mini-batch** are still in memory. Its gradients were computed *before* the final update, so undoing that update ($W_{old} = W + \alpha \nabla_W L$) recovers the exact weights they belong to. Then we let `autograd` differentiate the same forward pass and compare all six gradients.

# %%
import torch.nn.functional as F
from mlf_utils import check_close

params_manual = {"W1": (W1, dL_dW1), "b1": (b1, dL_db1), "W2": (W2, dL_dW2),
                 "b2": (b2, dL_db2), "W3": (W3, dL_dW3), "b3": (b3, dL_db3)}
leaves = {name: (p + alpha * g).detach().requires_grad_() for name, (p, g) in params_manual.items()}

A1_ag = torch.tanh(X_batch.view(-1, ninput) @ leaves["W1"] + leaves["b1"])
A2_ag = torch.tanh(A1_ag @ leaves["W2"] + leaves["b2"])
loss_ag = F.cross_entropy(A2_ag @ leaves["W3"] + leaves["b3"], Y_batch.squeeze())
loss_ag.backward()

check_close("Loss: manual softmax + NLL vs F.cross_entropy", L, loss_ag, atol=1e-5)
for name, (_, grad_manual) in params_manual.items():
    check_close(f"dL/d{name}: manual vs autograd", grad_manual, leaves[name].grad, atol=1e-6)

# %% [markdown]
# ## Loss curves

# %%
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# %% [markdown]
# ## Test accuracy

# %%
correct = 0
total = 0

with torch.no_grad():
    for i in range(len(Xtest)):
        Z1 = Xtest[i].view(1, -1) @ W1 + b1
        A1 = torch.tanh(Z1)
        Z2 = A1 @ W2 + b2
        A2 = torch.tanh(Z2)
        Z3 = A2 @ W3 + b3
        predicted_class = torch.argmax(Z3, dim=1)

        if predicted_class == Ytest[i]:
            correct += 1
        total += 1
accuracy = correct / total
print(f"Test Accuracy: {accuracy}")

# %%
