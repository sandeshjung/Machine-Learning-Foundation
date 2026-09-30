"""Plotting helpers shared across notebooks."""

import matplotlib.pyplot as plt
import numpy as np


def show_images(images, title=None, nrow=8, num_images=None, unnormalize=False, ax=None, show=True):
    """Display a batch of image tensors (N, C, H, W) or a single image (C, H, W) as a grid.

    Args:
        unnormalize: map images from [-1, 1] back to [0, 1] (for data normalised with mean=std=0.5).
        ax: axes to draw on. Defaults to a new figure.
        show: call plt.show() at the end. Set to False when composing subplots.
    """
    import torchvision  # imported lazily to keep the package light

    images = images.detach().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if num_images is not None:
        images = images[:num_images]
    if unnormalize:
        images = images * 0.5 + 0.5

    grid = torchvision.utils.make_grid(images.clamp(0, 1), nrow=min(nrow, len(images)), padding=2)
    if ax is None:
        rows = int(np.ceil(len(images) / nrow))
        _, ax = plt.subplots(figsize=(min(nrow, len(images)) * 1.5, max(rows * 1.5, 2.0)))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_decision_regions(predict_fn, X, y, ax=None, title=None, resolution=200, padding=0.5,
                          xlabel="Feature 1", ylabel="Feature 2"):
    """Shade the regions a 2-D classifier assigns to each class and scatter the data on top.

    Args:
        predict_fn: callable mapping an (n, 2) NumPy array to n class labels
            (NumPy array or tensor).
        X, y: data to scatter (NumPy arrays or tensors).
    """
    X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
    y = y.reshape(-1)

    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = predict_fn(grid)
    Z = Z.detach().cpu().numpy() if hasattr(Z, "detach") else np.asarray(Z)
    Z = Z.reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    classes = np.unique(y)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm", levels=np.append(classes - 0.5, classes[-1] + 0.5))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(classes)))
    for c, color in zip(classes, colors):
        ax.scatter(X[y == c, 0], X[y == c, 1], color=color, edgecolors="k", alpha=0.8, label=f"Class {c:g}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize="small")
    return ax
