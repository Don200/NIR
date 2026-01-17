from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

from synth.config import load_config
from synth.graph import load_kg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a GraphML knowledge graph.")
    parser.add_argument("--config", type=Path, help="Path to synth_config.yaml (used for kg_path).")
    parser.add_argument("--input", type=Path, help="Path to GraphML file (overrides config).")
    parser.add_argument("--output", type=Path, default=Path("artifacts/kg_visualization.png"))
    parser.add_argument("--max-nodes", type=int, default=300, help="Max nodes to render (top-degree).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", action="store_true", help="Render node labels (slow for large graphs).")
    return parser.parse_args()


def _select_nodes(g: nx.Graph, max_nodes: Optional[int]) -> Iterable[str]:
    if not max_nodes or g.number_of_nodes() <= max_nodes:
        return g.nodes()
    degree_sorted = sorted(g.degree(), key=lambda item: item[1], reverse=True)
    return [node for node, _ in degree_sorted[:max_nodes]]


def _component_colors(g: nx.Graph) -> dict[str, int]:
    components = list(nx.connected_components(g))
    color_map: dict[str, int] = {}
    for idx, component in enumerate(components):
        for node in component:
            color_map[node] = idx
    return color_map


def main() -> None:
    args = parse_args()

    if args.input:
        graph_path = args.input
    elif args.config:
        graph_path = load_config(args.config).kg_path
    else:
        raise RuntimeError("Provide --input or --config.")

    graph = load_kg(graph_path).graph
    selected = list(_select_nodes(graph, args.max_nodes))
    subgraph = graph.subgraph(selected).copy()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization.") from exc

    if subgraph.number_of_nodes() == 0:
        raise RuntimeError("Graph is empty; nothing to visualize.")

    positions = nx.spring_layout(subgraph, seed=args.seed, weight="weight")
    degrees = dict(subgraph.degree())
    node_sizes = [60 + degrees[node] * 18 for node in subgraph.nodes()]
    colors = _component_colors(subgraph)
    node_colors = [colors[node] for node in subgraph.nodes()]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(subgraph, positions, alpha=0.2, width=0.6)
    nx.draw_networkx_nodes(
        subgraph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        alpha=0.9,
        linewidths=0.3,
        edgecolors="white",
    )
    if args.labels:
        nx.draw_networkx_labels(subgraph, positions, font_size=7)

    plt.title(f"Knowledge Graph ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
    plt.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print("Saved visualization to", args.output)


if __name__ == "__main__":
    main()
