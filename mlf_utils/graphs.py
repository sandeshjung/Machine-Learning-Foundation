"""Visualise the computation graphs built by the from-scratch autograd engines."""


def trace(root):
    """Walk the graph backwards from `root`, collecting all nodes and edges."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def _node_label(n):
    data = n.data
    if hasattr(data, "shape") and getattr(data, "ndim", 0) > 0:
        # Tensor-valued node: show its shape and gradient
        grad = n.grad.detach().cpu().numpy() if hasattr(n.grad, "detach") else n.grad
        return "{ %s | %s | grad %s}" % (n.label, tuple(data.shape), grad)
    return "{ %s | data %.4f | grad %.4f}" % (n.label, float(data), float(n.grad))


def draw_dot(root, rankdir="LR"):
    """Render the graph ending at `root` with Graphviz.

    Works for both the scalar and the tensor `Value` classes. Each node needs
    `data`, `grad`, `label`, `_op` and `_prev` attributes. Requires the `graphviz`
    Python package and the Graphviz `dot` binary.
    """
    from graphviz import Digraph  # imported lazily so the package works without Graphviz

    dot = Digraph(format="svg", graph_attr={"rankdir": rankdir})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=_node_label(n), shape="record")
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
