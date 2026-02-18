import networkx as nx
from llm_spice.op.operators import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def build_graph(inputs: list[Tensor], outputs: list[Tensor]) -> nx.DiGraph:
    graph = nx.DiGraph()
    frontier = []
    frontier.extend(inputs)
    while frontier:
        inp = frontier.pop(0)
        graph.add_node(inp)
        op = inp.consumer
        if op is None:
            continue
        for out in op.outputs:
            graph.add_edge(inp, out, op=op)
            frontier.append(out)

    return graph


def draw_dag(
    G: nx.DiGraph,
    figsize: tuple[int, int] = (12, 8),
    layout: str = "dot",
) -> tuple[Figure, Axes]:
    """Draw a DAG of `Tensor` nodes and `Op`-labelled edges.

    Parameters
    ----------
    G : nx.DiGraph
        Directed acyclic graph whose nodes are :class:`~llm_spice.op.operators.Tensor` objects.
    figsize : tuple[int, int], optional
        Figure size passed to :pyfunc:`matplotlib.pyplot.subplots` (default ``(12, 8)``).
    layout : str, optional
        Layout engine to use:

        * ``"dot"`` – Use Graphviz *dot* via ``pygraphviz``/``pydot`` (best looking, if available).
        * ``"spring"`` – Use NetworkX force-directed layout.
        * ``"multipartite"`` – Use ``nx.multipartite_layout`` with layers determined by topological order.

        If the requested engine is unavailable the function gracefully falls back to the next option.

    Returns
    -------
    (fig, ax)
        The Matplotlib figure and axes so that callers can further tweak or save them.
    """

    # ------------------------------------------------------------------
    # Layout computation ------------------------------------------------
    # ------------------------------------------------------------------

    # Always compute layers for fallback multipartite layout.
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer

    pos = None

    def _graphviz_layout(graph: nx.DiGraph, prog: str = "dot"):
        """Try Graphviz layout via pygraphviz or pydot, return None if not available."""
        try:
            import networkx.drawing.nx_agraph as nx_agraph  # type: ignore

            return nx_agraph.graphviz_layout(graph, prog=prog)
        except (ImportError, AttributeError):
            try:
                import networkx.drawing.nx_pydot as nx_pydot  # type: ignore

                return nx_pydot.graphviz_layout(graph, prog=prog)
            except (ImportError, AttributeError):
                return None

    if layout == "dot":
        pos = _graphviz_layout(G, "dot")
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=42)

    # Fallback to multipartite if pos still None
    if pos is None:
        pos = nx.multipartite_layout(
            G, subset_key="layer", scale=3.0
        )  # slightly spread out

    # Create node labels with tensor information
    node_labels = {}
    for node in G.nodes():
        if isinstance(node, Tensor):
            shape_str = str(node.shape)
            dtype_str = str(node.dtype)
            node_labels[node] = f"{shape_str}\n{dtype_str}"
        else:
            node_labels[node] = str(node)

    # Create edge labels with operation information
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if "op" in data and data["op"] is not None:
            edge_labels[(u, v)] = data["op"].name
        else:
            edge_labels[(u, v)] = ""

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # Drawing -----------------------------------------------------------
    # ------------------------------------------------------------------

    # Draw nodes with custom styling
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightblue",
        node_size=3000,
        linewidths=1.0,
        edgecolors="k",
        alpha=0.85,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        width=1.5,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=9,
        font_weight="bold",
        ax=ax,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="tab:red",
        ax=ax,
    )
    ax.axis("off")
    ax.margins(0.2)
    fig.tight_layout()
    return fig, ax
