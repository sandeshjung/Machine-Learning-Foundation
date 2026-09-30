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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/autograd%28tensor%29.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %% [markdown]
# ## Autograd Tensor
#
# The same engine as the scalar notebook, but each `Value` now wraps a **tensor**. The new ingredients are:
# - **Matrix multiplication**: for $Z = AW$, $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z} W^\top$ and $\frac{\partial L}{\partial W} = A^\top \frac{\partial L}{\partial Z}$
# - **Broadcasting**: when a small tensor (e.g. a bias) is broadcast in the forward pass, its gradient must be **summed** back over the broadcast dimensions
#
# The result is checked against PyTorch's built-in autograd at the end.
#
# Theory: [PyTorch Autograd: Automatic Differentiation](README.md#pytorch-autograd-automatic-differentiation)

# %% [markdown]
# #### Visualising the computation graph
# Same shared `draw_dot` helper as the scalar version ([`mlf_utils/graphs.py`](../mlf_utils/graphs.py)). For tensor nodes it shows shapes instead of values.

# %%
import torch
from mlf_utils import check_close, draw_dot  # draw_dot renders the graph with Graphviz


# %% [markdown]
# #### Undoing broadcasting in the backward pass
# `unbroadcast` sums the incoming gradient over any dimension that was added or stretched from size 1, so the gradient has the same shape as the original tensor.

# %%
def unbroadcast(target, grad):
    # If grad has extra leading dims, sum them away:
    while grad.dim() > target.dim():
        grad = grad.sum(dim=0)
    # For any dimension where target was size=1 (broadcast), sum over that axis:
    for i, size in enumerate(target.size()):
        if size == 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad


# %% [markdown]
# #### The tensor `Value` class

# %%
import torch

class Value:
    def __init__(self, value, _children=(), label="", _op=""):
        self.data = value
        self._prev = _children
        self._backward = lambda: None
        self.label = label
        self._op = _op
        self.grad = torch.zeros_like(value)

    def __add__(self, other):
        # Support both Value + Value and Value + Tensor
        if isinstance(other, Value):
            out = Value(self.data + other.data, _children=(self, other), _op="+")
            def _backward():
                # Sum out.grad back into self.grad and other.grad, but first undo any broadcasting:
                self.grad += unbroadcast(self.data, out.grad)
                other.grad += unbroadcast(other.data, out.grad)
            out._backward = _backward
            return out
        elif isinstance(other, torch.Tensor): # other is a Tensor
            out = Value(self.data + other, _children=(self,), _op="+")
            def _backward():
                self.grad += unbroadcast(self.data, out.grad)
            out._backward = _backward
            return out
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Value' and '{}'".format(type(other)))

    def __matmul__(self, other):
        if isinstance(other, Value):
            out = Value(self.data @ other.data, _children=(self, other), _op="@")
            def _backward():
                # d(a@b)/da = grad @ bᵀ
                self.grad += unbroadcast(self.data, out.grad @ other.data.T)
                # d(a@b)/db = aᵀ @ grad
                other.grad += unbroadcast(other.data, self.data.T @ out.grad)
            out._backward = _backward
            return out
        elif isinstance(other, torch.Tensor):
            out = Value(self.data @ other, _children=(self,), _op="@")
            def _backward():
                self.grad += unbroadcast(self.data, out.grad @ other.T)
            out._backward = _backward
            return out
        else:
            raise TypeError("Unsupported operand type(s) for @: 'Value' and '{}'".format(type(other)))

    def backward(self):
        # Initialize the gradient of the output value
        self.grad = torch.ones_like(self.data)
        # Propagate the gradients backward through the computation graph
        stack = [self]
        while stack:
            node = stack.pop()
            node._backward()
            stack.extend(node._prev)

    def __repr__(self):
        return str(f'Value: {self.data}')

    def __str__(self):
        return str(f'Value :{self.data}')


# %% [markdown]
# #### A single linear layer: $z = aw + b$
# We create the same random tensors twice: `w1`, `b1`, `a1` are plain PyTorch tensors used as a reference, and `a`, `b`, `w` are wrapped in our `Value` class.

# %%
# To make a tensor track operations for automatic differentiation, set its `requires_grad` attribute to `True`.
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)
a = torch.randn(2, 1, requires_grad=True)

# %%
w1 = w.clone().detach().requires_grad_(True)
b1 = b.clone().detach().requires_grad_(True)
a1 = a.clone().detach().requires_grad_(True)

# %%
a = Value(a)
a.label = 'a'
b = Value(b)
b.label = 'b'
w = Value(w)
w.label = 'w'

# %%
interm = a @ w
interm.label = 'interm'
z = interm + b
z.label = 'z'

# %%
z

# %%
draw_dot(z)

# %% [markdown]
# #### Manual backward pass
# Seed the output gradient with ones (equivalent to backpropagating `z.sum()`), then call `_backward()` from the output towards the inputs.

# %%
z.grad = torch.ones_like(z.data)

# %% [markdown]
# - To compute gradients, call .backward() on the output tensor
# - The gradients will be accumulated in the .grad attribute of the input tensors.

# %%
z._backward()

# %%
interm._backward()

# %% [markdown]
# #### Reference: PyTorch autograd
# The gradients below should match the ones from our engine.

# %%
temp = (a1 @ w1) + b1
loss = temp.sum()
loss.backward()

# %%
print("Gradient of the a tensor:", a1.grad)
print("Gradient of the w tensor:", w1.grad)
print("Gradient of the b tensor:", b1.grad)

# %%
print("Gradient of the a tensor:", a.grad)
print("Gradient of the w tensor:", w.grad)
print("Gradient of the b tensor:", b.grad)

# %% [markdown]
# #### Verifying our engine against PyTorch

# %%
for name, ours, ref in [("dz/da", a.grad, a1.grad), ("dz/dw", w.grad, w1.grad), ("dz/db", b.grad, b1.grad)]:
    check_close(name, ours, ref)

# %%
