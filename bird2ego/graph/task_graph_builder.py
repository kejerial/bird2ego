"""TaskGraphBuilder: builds NetworkX task graph from segments and edges."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from ..utils.timeline import ActionSegment, Timeline
from .ordering_inferencer import InferredEdge
from .postcondition_extractor import SegmentPostconditions
from .precondition_extractor import SegmentPreconditions

logger = logging.getLogger(__name__)


class TaskGraphBuilder:
    """Builds a NetworkX DiGraph representing the task structure.

    Node attributes:
    - node_id (seg_id)
    - label
    - t_start
    - t_end
    - objects

    Edge attributes:
    - edge_type ("causal" or "temporal")
    - conf
    - evidence
    """

    def __init__(self):
        """Initialize TaskGraphBuilder."""
        self._graph: Optional[nx.DiGraph] = None

    @property
    def graph(self) -> Optional[nx.DiGraph]:
        """Get the constructed graph."""
        return self._graph

    def build(
        self,
        segments: List[ActionSegment],
        edges: List[InferredEdge],
        preconditions: Optional[Dict[int, SegmentPreconditions]] = None,
        postconditions: Optional[Dict[int, SegmentPostconditions]] = None,
    ) -> nx.DiGraph:
        """Build task graph from segments and edges.

        Args:
            segments: List of action segments (nodes).
            edges: List of inferred edges.
            preconditions: Optional preconditions for each segment.
            postconditions: Optional postconditions for each segment.

        Returns:
            NetworkX DiGraph with node and edge attributes.
        """
        self._graph = nx.DiGraph()

        # Add nodes (segments)
        for segment in segments:
            node_attrs = {
                "node_id": segment.seg_id,
                "label": segment.label,
                "t_start": segment.t_start,
                "t_end": segment.t_end,
                "frame_start": segment.frame_start,
                "frame_end": segment.frame_end,
                "objects": segment.objects_involved,
                "conf": segment.conf,
            }

            # Add preconditions if available
            if preconditions and segment.seg_id in preconditions:
                node_attrs["preconditions"] = preconditions[segment.seg_id].to_dict()

            # Add postconditions if available
            if postconditions and segment.seg_id in postconditions:
                node_attrs["postconditions"] = postconditions[segment.seg_id].to_dict()

            self._graph.add_node(segment.seg_id, **node_attrs)

        # Add edges
        for edge in edges:
            edge_attrs = {
                "edge_type": edge.edge_type.value,
                "conf": edge.confidence,
                "evidence": edge.evidence,
            }
            self._graph.add_edge(edge.from_seg_id, edge.to_seg_id, **edge_attrs)

        logger.info(
            f"Built task graph: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

        return self._graph

    def build_from_timeline(
        self,
        timeline: Timeline,
        edges: List[InferredEdge],
        preconditions: Optional[Dict[int, SegmentPreconditions]] = None,
        postconditions: Optional[Dict[int, SegmentPostconditions]] = None,
    ) -> nx.DiGraph:
        """Build task graph from timeline.

        Args:
            timeline: Timeline with segments.
            edges: List of inferred edges.
            preconditions: Optional preconditions.
            postconditions: Optional postconditions.

        Returns:
            NetworkX DiGraph.
        """
        return self.build(timeline.segments, edges, preconditions, postconditions)

    def to_json(self) -> Dict[str, Any]:
        """Convert graph to JSON-serializable dictionary.

        Returns:
            Dictionary with nodes and edges lists.
        """
        if self._graph is None:
            return {"nodes": [], "edges": []}

        nodes = []
        for node_id in self._graph.nodes():
            node_data = dict(self._graph.nodes[node_id])
            node_data["id"] = node_id
            nodes.append(node_data)

        edges = []
        for u, v in self._graph.edges():
            edge_data = dict(self._graph.edges[u, v])
            edge_data["source"] = u
            edge_data["target"] = v
            edges.append(edge_data)

        return {"nodes": nodes, "edges": edges}

    def save_json(self, path: str) -> None:
        """Save graph as JSON file.

        Args:
            path: Output file path.
        """
        data = self.to_json()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved task graph JSON to {path}")

    def save_graphml(self, path: str) -> None:
        """Save graph as GraphML file.

        Args:
            path: Output file path.
        """
        if self._graph is None:
            logger.warning("No graph to save")
            return

        # NetworkX GraphML doesn't handle complex types well
        # Create a copy with serialized attributes
        export_graph = nx.DiGraph()

        for node_id in self._graph.nodes():
            attrs = dict(self._graph.nodes[node_id])
            # Serialize complex types
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value, default=str)
            export_graph.add_node(node_id, **attrs)

        for u, v in self._graph.edges():
            attrs = dict(self._graph.edges[u, v])
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value, default=str)
            export_graph.add_edge(u, v, **attrs)

        nx.write_graphml(export_graph, path)
        logger.info(f"Saved task graph GraphML to {path}")

    def get_node_count(self) -> int:
        """Get number of nodes in graph."""
        return self._graph.number_of_nodes() if self._graph else 0

    def get_edge_count(self) -> int:
        """Get number of edges in graph."""
        return self._graph.number_of_edges() if self._graph else 0

    def get_roots(self) -> List[int]:
        """Get root nodes (no incoming edges).

        Returns:
            List of root node IDs.
        """
        if self._graph is None:
            return []
        return [n for n in self._graph.nodes() if self._graph.in_degree(n) == 0]

    def get_leaves(self) -> List[int]:
        """Get leaf nodes (no outgoing edges).

        Returns:
            List of leaf node IDs.
        """
        if self._graph is None:
            return []
        return [n for n in self._graph.nodes() if self._graph.out_degree(n) == 0]

    def get_topological_order(self) -> List[int]:
        """Get nodes in topological order.

        Returns:
            List of node IDs in topological order.
        """
        if self._graph is None:
            return []
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            logger.warning("Graph has cycles, cannot compute topological order")
            return list(self._graph.nodes())

    def get_causal_chains(self) -> List[List[int]]:
        """Get all causal chains (paths through causal edges only).

        Returns:
            List of chains, each chain is a list of node IDs.
        """
        if self._graph is None:
            return []

        # Create subgraph with only causal edges
        causal_edges = [
            (u, v)
            for u, v in self._graph.edges()
            if self._graph.edges[u, v].get("edge_type") == "causal"
        ]
        causal_graph = self._graph.edge_subgraph(causal_edges)

        # Find all paths from roots to leaves
        chains = []
        roots = [n for n in causal_graph.nodes() if causal_graph.in_degree(n) == 0]
        leaves = [n for n in causal_graph.nodes() if causal_graph.out_degree(n) == 0]

        for root in roots:
            for leaf in leaves:
                try:
                    for path in nx.all_simple_paths(causal_graph, root, leaf):
                        chains.append(path)
                except nx.NetworkXNoPath:
                    continue

        return chains

    def validate(self, segments: List[ActionSegment]) -> List[str]:
        """Validate graph against segments.

        Args:
            segments: List of action segments.

        Returns:
            List of validation warnings.
        """
        warnings = []

        if self._graph is None:
            warnings.append("Graph is None")
            return warnings

        # Check node count matches segment count
        if self._graph.number_of_nodes() != len(segments):
            warnings.append(
                f"Node count ({self._graph.number_of_nodes()}) != segment count ({len(segments)})"
            )

        # Check all segment IDs are present
        seg_ids = {s.seg_id for s in segments}
        node_ids = set(self._graph.nodes())
        missing = seg_ids - node_ids
        if missing:
            warnings.append(f"Missing segment IDs in graph: {missing}")

        extra = node_ids - seg_ids
        if extra:
            warnings.append(f"Extra node IDs in graph: {extra}")

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self._graph):
            warnings.append("Graph contains cycles")

        return warnings
