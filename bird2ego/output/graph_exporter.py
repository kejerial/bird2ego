"""GraphExporter: exports task graph to GraphML and JSON formats."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from ..graph.task_graph_builder import TaskGraphBuilder

logger = logging.getLogger(__name__)


class GraphExporter:
    """Exports task graph to GraphML and JSON formats.

    Produces:
    - task_graph.json: Nodes and edges as JSON lists
    - task_graph.graphml: NetworkX GraphML format
    """

    def __init__(
        self,
        output_dir: str,
        indent: int = 2,
    ):
        """Initialize GraphExporter.

        Args:
            output_dir: Directory to write output files.
            indent: JSON indentation level.
        """
        self.output_dir = Path(output_dir)
        self.indent = indent

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        graph_builder: TaskGraphBuilder,
    ) -> Dict[str, str]:
        """Export task graph in all formats.

        Args:
            graph_builder: TaskGraphBuilder with constructed graph.

        Returns:
            Dictionary mapping format to file path.
        """
        paths = {}

        paths["json"] = self.export_json(graph_builder)
        paths["graphml"] = self.export_graphml(graph_builder)

        logger.info(f"Exported task graph to {self.output_dir}")
        return paths

    def export_json(self, graph_builder: TaskGraphBuilder) -> str:
        """Export graph as JSON file.

        Args:
            graph_builder: TaskGraphBuilder with constructed graph.

        Returns:
            Path to output file.
        """
        graph = graph_builder.graph
        if graph is None:
            data = {"nodes": [], "edges": [], "metadata": {"error": "No graph"}}
        else:
            data = self._graph_to_json(graph)

        path = self.output_dir / "task_graph.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=self.indent, default=str)
        logger.debug(f"Wrote task graph JSON to {path}")
        return str(path)

    def export_graphml(self, graph_builder: TaskGraphBuilder) -> str:
        """Export graph as GraphML file.

        Args:
            graph_builder: TaskGraphBuilder with constructed graph.

        Returns:
            Path to output file.
        """
        path = self.output_dir / "task_graph.graphml"
        graph_builder.save_graphml(str(path))
        return str(path)

    def export_from_graph(self, graph: nx.DiGraph) -> Dict[str, str]:
        """Export a NetworkX graph directly.

        Args:
            graph: NetworkX DiGraph to export.

        Returns:
            Dictionary mapping format to file path.
        """
        paths = {}

        # JSON export
        data = self._graph_to_json(graph)
        json_path = self.output_dir / "task_graph.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=self.indent, default=str)
        paths["json"] = str(json_path)

        # GraphML export
        graphml_path = self.output_dir / "task_graph.graphml"
        self._save_graphml(graph, str(graphml_path))
        paths["graphml"] = str(graphml_path)

        return paths

    def _graph_to_json(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Convert NetworkX graph to JSON-serializable dictionary.

        Args:
            graph: NetworkX DiGraph.

        Returns:
            Dictionary with nodes, edges, and metadata.
        """
        nodes = []
        for node_id in graph.nodes():
            node_data = dict(graph.nodes[node_id])
            node_data["id"] = node_id
            nodes.append(node_data)

        edges = []
        for u, v in graph.edges():
            edge_data = dict(graph.edges[u, v])
            edge_data["source"] = u
            edge_data["target"] = v
            edges.append(edge_data)

        # Compute metadata
        metadata = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(graph),
        }

        # Causal vs temporal edge counts
        causal_count = sum(
            1 for _, _, d in graph.edges(data=True)
            if d.get("edge_type") == "causal"
        )
        temporal_count = sum(
            1 for _, _, d in graph.edges(data=True)
            if d.get("edge_type") == "temporal"
        )
        metadata["num_causal_edges"] = causal_count
        metadata["num_temporal_edges"] = temporal_count

        return {
            "schema_version": "1.0",
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }

    def _save_graphml(self, graph: nx.DiGraph, path: str) -> None:
        """Save graph as GraphML with serialized attributes.

        Args:
            graph: NetworkX DiGraph.
            path: Output file path.
        """
        # Create a copy with serialized attributes for GraphML compatibility
        export_graph = nx.DiGraph()

        for node_id in graph.nodes():
            attrs = dict(graph.nodes[node_id])
            # Serialize complex types
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value, default=str)
            export_graph.add_node(node_id, **attrs)

        for u, v in graph.edges():
            attrs = dict(graph.edges[u, v])
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value, default=str)
            export_graph.add_edge(u, v, **attrs)

        nx.write_graphml(export_graph, path)

    def load_json(self, path: str) -> nx.DiGraph:
        """Load graph from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            NetworkX DiGraph.
        """
        with open(path, "r") as f:
            data = json.load(f)

        graph = nx.DiGraph()

        for node in data.get("nodes", []):
            node_id = node.pop("id")
            graph.add_node(node_id, **node)

        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            graph.add_edge(source, target, **edge)

        return graph

    def load_graphml(self, path: str) -> nx.DiGraph:
        """Load graph from GraphML file.

        Args:
            path: Path to GraphML file.

        Returns:
            NetworkX DiGraph.
        """
        graph = nx.read_graphml(path)

        # Convert node IDs back to integers if they were
        mapping = {}
        for node in graph.nodes():
            try:
                mapping[node] = int(node)
            except (ValueError, TypeError):
                mapping[node] = node

        graph = nx.relabel_nodes(graph, mapping)

        # Deserialize JSON attributes
        for node_id in graph.nodes():
            for key, value in list(graph.nodes[node_id].items()):
                if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
                    try:
                        graph.nodes[node_id][key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass

        for u, v in graph.edges():
            for key, value in list(graph.edges[u, v].items()):
                if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
                    try:
                        graph.edges[u, v][key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass

        return graph
