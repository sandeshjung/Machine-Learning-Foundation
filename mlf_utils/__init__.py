"""Shared helpers for the Machine Learning Foundation notebooks.

Only plumbing lives here (plotting, graph drawing, verification). The algorithms
being taught stay inside the notebooks so they can be read top to bottom.
"""

from .checks import check_agreement, check_close
from .graphs import draw_dot, trace
from .plotting import plot_decision_regions, show_images

__all__ = [
    "check_agreement",
    "check_close",
    "draw_dot",
    "plot_decision_regions",
    "show_images",
    "trace",
]
