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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/optimization_algorithms.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %% [markdown]
# # Optimization Algorithms

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from ipywidgets import FloatSlider, interact

from mlf_utils import check_close

torch.manual_seed(42)


# %% [markdown]
# ### Gradient Descent (GD)
# Algorithm:
# 1. Initialize parameters (e.g., x)
# 2. Repeat for number of iterations:
#     - Compute the objective function f(x)
#     - Compute the gradient ∇f(x)
#     - Update parameters: x = x - learning_rate * ∇f(x)

# %%
# Objective function
def f(x):
    return x ** 2

# Analytical gradient: df/dx = 2x


# %%
# Hyperparameters
lr = 0.1
n_iters = 30

# Initialize parameter
x = torch.tensor(4.0, requires_grad=True)
history = []

# %%
for i in range(n_iters):
    history.append(x.item())
    
    # Forward: compute loss
    loss = f(x)
    
    # Backward pass (compute gradient)
    # Ensure gradients from previous iterations are cleared if x is reused in a loop where backward is called multiple times on different graphs
    if x.grad is not None:
        x.grad.zero_()

    # Backward: compute df/dx
    loss.backward()
    
    # Update parameters (manual step, no_grad context to avoid tracking this update)
    with torch.no_grad():
        x -= lr * x.grad
    
    # Log progress
    if (i + 1) % 5 == 0:
        print(f"Iter {i+1:2d}: x={x.item():.4f}, loss={loss.item():.4f}, grad={x.grad.item():.4f}")

# %%
# Plot results
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history, 'o-', label='x value')
plt.axhline(0, color='r', linestyle='--', label='Optimum')
plt.title('GD: Parameter over iterations')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.legend()

plt.subplot(1,2,2)
xs = np.linspace(-4.5, 4.5, 100)
ys = xs**2
plt.plot(xs, ys, label='f(x)=x^2')
plt.scatter(history, np.array(history)**2, color='red', label='GD steps')
plt.title('GD Path on f(x)=x^2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# #### Try it: the learning rate
# For $f(x) = x^2$ the update is $x \leftarrow x - \eta \cdot 2x = (1 - 2\eta)\,x$, so the learning rate alone decides what happens:
# - $\eta < 0.5$: smooth convergence
# - $0.5 < \eta < 1$: overshoots and oscillates around the minimum, but still converges
# - $\eta = 1$: bounces between $\pm x_0$ forever
# - $\eta > 1$: diverges
#
# *Interactive: run the notebook locally or in Colab to use the controls. GitHub only renders a static page.*

# %%
def gd_trajectory(lr, x0=4.0, steps=30):
    xs = [x0]
    for _ in range(steps):
        xs.append(xs[-1] - lr * 2 * xs[-1])   # f'(x) = 2x
    return np.array(xs)

@interact(lr=FloatSlider(value=0.1, min=0.01, max=1.05, step=0.01, description="learning rate", continuous_update=False))
def explore_learning_rate(lr):
    xs = gd_trajectory(lr)
    grid = np.linspace(-5, 5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(grid, grid ** 2, color="gray", label="f(x) = x²")
    axes[0].plot(np.clip(xs, -5, 5), np.clip(xs, -5, 5) ** 2, "o-", color="tab:red", markersize=4, label="GD steps")
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-1, 26); axes[0].legend()
    axes[0].set_title(f"Gradient descent on x², lr = {lr:.2f}")
    axes[1].semilogy(np.abs(xs) + 1e-16, "o-", markersize=3)
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("|x - x*|")
    axes[1].set_title("Distance to the minimum (log scale)")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Stochastic Gradient Descent (SGD)

# %% [markdown]
# - In GD, gradient is computed using the entire dataset (for a typical ML loss). This is expensive.
# - In SGD, the gradient is computed using only a single random sample (or a small mini-batch) from the dataset at each iteration.

# %% [markdown]
# For a simple function like `f(x)=x^2`, SGD is not directly applicable in its typical ML sense as there's no "dataset" to sample from. The "stochasticity" would come from perhaps adding noise to the gradient or using noisy measurements of the function.

# %%
# Generate synthetic linear data: y = 2x - 1 + noise
N = 100
torch.manual_seed(0)
X = 2 * torch.rand(N, 1)
y = 2 * X - 1 + 0.1 * torch.randn(N, 1)

# %%
# Initialize parameters
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Hyperparameters
learning_rate = 0.1
n_iters = 30

loss_history = []

# %%
for epoch in range(n_iters):
    # Shuffle indices
    indices = torch.randperm(N)
    for i in indices:
        xi = X[i].unsqueeze(0)   
        yi = y[i].unsqueeze(0) 
        
        # Forward pass: prediction and loss
        y_pred = w * xi + b
        loss = (y_pred - yi).pow(2).mean()
        
        # Backward pass: compute gradients
        # (zero out old gradients first)
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()
        loss.backward()
        
        # Parameter update, no grad-tracking here
        with torch.no_grad():
            w[:] = w - learning_rate * w.grad
            b[:] = b - learning_rate * b.grad
        
        # Record loss for this sample
        loss_history.append(loss.item())
    
    print(f"Epoch {epoch+1}/{n_iters} — w={w.item():.3f}, b={b.item():.3f}")


# %%
plt.plot(loss_history)
plt.xlabel('Update step')
plt.ylabel('MSE loss')
plt.title('SGD: Loss per sample update')
plt.show()

plt.scatter(X.numpy(), y.numpy(), label='data')
x_line = np.linspace(0, 2, 100)
y_line = w.item() * x_line + b.item()
plt.plot(x_line, y_line, 'r-', label=f'Fit: y={w.item():.2f}x+{b.item():.2f}')
plt.legend()
plt.show()

# %% [markdown]
# ### Using PyTorch's `torch.optim`

# %%
# Parameters to optimize
x_optim = torch.tensor(4.0, requires_grad=True) 

# %%
# Optimizer needs to know which Tensors it should update.
optimizer_sgd = torch.optim.SGD([x_optim], lr=0.1)
# Separate param for Adam
optimizer_adam = torch.optim.Adam([torch.tensor(4.0, requires_grad=True)], lr=0.1) 

# %%
history_optim_sgd = []
history_optim_adam = []
# Get the tensor Adam is optimizing
x_adam_param = optimizer_adam.param_groups[0]['params'][0] 

# %%
print("\n--- Using torch.optim.SGD ---")
for i in range(n_iters):
    history_optim_sgd.append(x_optim.item())
    
    # Zero gradients from previous step
    optimizer_sgd.zero_grad()
    
    # Compute loss
    loss_optim = x_optim**2
    
    # Compute gradients w.r.t. parameters
    loss_optim.backward()
    
    # Update parameters
    optimizer_sgd.step()
    
    if (i+1)%5 == 0:
        print(f"""SGD Iter {i+1}: x = {x_optim.item():.4f}, 
              loss = {loss_optim.item():.4f}""")

# %%
print("\n--- Using torch.optim.Adam ---")
for i in range(n_iters):
    history_optim_adam.append(x_adam_param.item())
    optimizer_adam.zero_grad()
    loss_adam = x_adam_param**2
    loss_adam.backward()
    optimizer_adam.step()
    if (i+1)%5 == 0:
        print(f"""Adam Iter {i+1}: x = {x_adam_param.item():.4f}, 
              loss = {loss_adam.item():.4f}""")

# %% [markdown]
# #### Checking the manual update against `torch.optim.SGD`
# Both apply $x \leftarrow x - \eta \nabla f(x)$ from the same starting point, so the trajectories should be identical.

# %%
check_close("Manual GD vs torch.optim.SGD trajectory", history, history_optim_sgd)

# %%
plt.figure(figsize=(8,5))
plt.plot(history, 'o-', label='Manual GD')
plt.plot(history_optim_sgd, 's--', label='torch.optim.SGD') # overlaps manual gd
plt.plot(history_optim_adam, '^-', label='torch.optim.Adam')
plt.title('Comparison of Optimizers on f(x)=x^2')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.axhline(0, color='black', linestyle=':', label='Optimum x=0')
plt.legend()
plt.show()

# %% [markdown]
# ### Convex Optimization (Conceptual Introduction) ---
# Convex Function: A real-valued function f defined on an interval (or a convex set in higher dimensions) is called convex if the line segment between any two points on the graph of the function lies on or above the graph.

# %%
# Visualizing Convex vs. Non-Convex Functions ---
x_vals = np.linspace(-3, 3, 100)

# Convex function: f(x) = x^2
f_convex = x_vals**2

# Non-convex function: f(x) = x^4 - 3x^2 + x (has multiple local minima)
f_non_convex = x_vals**4 - 3*x_vals**2 + x_vals

# %%
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_vals, f_convex, label='f(x) = x^2 (Convex)')
plt.title('Convex Function')
plt.xlabel('x')
plt.ylabel('f(x)')
# Illustrate convexity property: line segment between two points
x1, x2 = -2, 1.5
y1, y2 = x1**2, x2**2
plt.plot([x1, x2], [y1, y2], 'ro-', label='Segment connecting two points')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_vals, f_non_convex, label='f(x) = x^4 - 3x^2 + x (Non-Convex)')
plt.title('Non-Convex Function (Multiple Local Minima)')
plt.xlabel('x')
plt.ylabel('f(x)')
# Highlight local minima (approximate visually)
plt.scatter([-1.3, 1.37], [(-1.3)**4 - 3*(-1.3)**2 -1.3, (1.37)**4 - 3*(1.37)**2 + 1.37], color='red', s=50, zorder=5, label='Local Minima')
plt.legend()

plt.tight_layout()
plt.show()
