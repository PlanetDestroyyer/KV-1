#!/usr/bin/env python3
"""
Small-World Knowledge Graph for KV-1

Implements brain-inspired memory architecture:
- Small-world network topology (high clustering + short paths)
- Anatomical connectivity (permanent structure)
- Functional connectivity (dynamic activation)
- Hub detection (key concepts)
- Efficient retrieval via graph traversal
- Automatic analogy discovery

Based on neuroscience research showing brain exhibits small-world properties
at multiple scales (neurons, regions, whole brain).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import time
import json
from pathlib import Path


@dataclass
class ConceptNode:
    """
    Node in knowledge graph representing a learned concept.

    Analogous to a brain region or neural population.
    """
    id: str
    name: str
    content: str  # Definition, explanation
    domain: str  # Physics, Math, Biology, etc.
    embedding: np.ndarray  # 384-D semantic vector

    # Graph connectivity
    neighbors: Set[str] = field(default_factory=set)
    degree: int = 0

    # Small-world metrics
    clustering_coef: float = 0.0
    betweenness: float = 0.0  # How often on shortest paths (hub indicator)

    # Learning metadata
    learned_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    # FEP: Prediction tracking
    prediction_error: float = 0.0  # How surprising was this concept
    complexity: float = 0.0  # How complex is this model

    def __hash__(self):
        return hash(self.id)

    def to_dict(self) -> dict:
        """Serialize (embedding saved separately)"""
        return {
            'id': self.id,
            'name': self.name,
            'content': self.content,
            'domain': self.domain,
            'neighbors': list(self.neighbors),
            'degree': self.degree,
            'clustering_coef': self.clustering_coef,
            'learned_at': self.learned_at,
            'access_count': self.access_count
        }


@dataclass
class ConceptEdge:
    """
    Edge between concepts representing relationship.

    Analogous to white matter tract (anatomical) or
    synchronized activity (functional).
    """
    source: str
    target: str
    edge_type: str  # 'semantic', 'prerequisite', 'analogy', 'example'

    # Dual connectivity (like brain!)
    anatomical_weight: float  # Permanent connection strength (0-1)
    functional_weight: float  # Current activation strength (0-1)

    # Metadata
    created_at: float = field(default_factory=time.time)
    activation_count: int = 0
    last_activated: float = 0.0

    def total_weight(self) -> float:
        """Combined connectivity"""
        return 0.7 * self.anatomical_weight + 0.3 * self.functional_weight

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'anatomical_weight': self.anatomical_weight,
            'functional_weight': self.functional_weight,
            'activation_count': self.activation_count
        }


class SmallWorldKnowledgeGraph:
    """
    Small-world knowledge graph inspired by brain connectivity.

    Key Properties:
    1. High clustering: Related concepts cluster together
    2. Short paths: Any concept reachable in few hops (like 6 degrees)
    3. Hubs: Key concepts connect many clusters
    4. Dual connectivity: Anatomical (structure) + Functional (dynamics)
    5. Efficient: O(log n) retrieval vs O(n) for flat memory

    Implements Watts-Strogatz small-world model with extensions for
    semantic similarity and domain structure.
    """

    def __init__(
        self,
        k_local: int = 4,
        rewiring_prob: float = 0.1,
        save_path: str = "data/small_world_graph.json"
    ):
        """
        Initialize small-world graph.

        Args:
            k_local: Number of local connections per node
            rewiring_prob: Probability of creating long-range shortcut
            save_path: Where to save graph state
        """
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[Tuple[str, str], ConceptEdge] = {}

        # Small-world parameters
        self.k_local = k_local
        self.rewiring_prob = rewiring_prob

        # Network statistics
        self.clustering_coefficient = 0.0
        self.average_path_length = 0.0
        self.small_world_index = 0.0

        # Domain organization (emergent clustering)
        self.domain_clusters: Dict[str, Set[str]] = defaultdict(set)

        # Hub tracking
        self.hubs: List[str] = []

        # Save path
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def add_concept(
        self,
        concept: ConceptNode,
        verbose: bool = True
    ) -> None:
        """
        Add concept to graph with small-world connectivity.

        Creates both local connections (high clustering) and
        occasional long-range shortcuts (short paths).
        """
        self.nodes[concept.id] = concept
        self.domain_clusters[concept.domain].add(concept.id)

        # Create local connections (within domain/semantic neighborhood)
        self._create_local_connections(concept)

        # Maybe create long-range shortcut (small-world property!)
        if np.random.random() < self.rewiring_prob and len(self.nodes) > 10:
            self._create_shortcut(concept, verbose=verbose)

        if verbose:
            print(f"[Added] {concept.name} | "
                  f"Domain: {concept.domain} | "
                  f"Connections: {concept.degree}")

    def _create_local_connections(self, concept: ConceptNode) -> None:
        """
        Create local connections based on semantic similarity.

        This creates HIGH CLUSTERING (like brain's local connectivity).
        """
        if len(self.nodes) <= 1:
            return

        # Find k most similar concepts
        similarities = []
        for other_id, other in self.nodes.items():
            if other_id != concept.id:
                sim = self._cosine_similarity(concept.embedding, other.embedding)
                similarities.append((sim, other_id))

        # Sort by similarity
        similarities.sort(reverse=True)

        # Connect to k_local most similar (local clustering)
        connected = 0
        for sim, other_id in similarities:
            if connected >= self.k_local:
                break

            # Prefer same domain (stronger anatomical connection)
            other = self.nodes[other_id]
            same_domain = (other.domain == concept.domain)
            anatomical = 0.8 if same_domain else 0.5

            self._add_edge(
                concept.id,
                other_id,
                edge_type='semantic',
                anatomical_weight=anatomical
            )
            connected += 1

    def _create_shortcut(self, concept: ConceptNode, verbose: bool = True) -> None:
        """
        Create long-range shortcut to reduce path length.

        This creates SHORT PATHS (small-world property!).

        Connect to moderately similar concept in DIFFERENT domain.
        This enables cross-domain transfer and analogy discovery!
        """
        candidates = []

        for other_id, other in self.nodes.items():
            # Must be different domain and not already connected
            if (other.domain != concept.domain and
                other_id not in concept.neighbors and
                other_id != concept.id):

                sim = self._cosine_similarity(concept.embedding, other.embedding)

                # Moderate similarity (0.3-0.7) = good analogy potential
                if 0.3 < sim < 0.7:
                    candidates.append((sim, other_id))

        if candidates:
            # Create shortcut to best candidate
            sim, target_id = max(candidates)
            target = self.nodes[target_id]

            self._add_edge(
                concept.id,
                target_id,
                edge_type='analogy',
                anatomical_weight=0.4  # Weaker long-range connection
            )

            if verbose:
                print(f"  [Shortcut] {concept.name} ({concept.domain}) ←→ "
                      f"{target.name} ({target.domain}) [sim: {sim:.2f}]")

    def _add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        anatomical_weight: float
    ) -> None:
        """Add bidirectional edge between nodes."""
        edge = ConceptEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            anatomical_weight=anatomical_weight,
            functional_weight=0.0  # Initially inactive
        )

        # Store both directions
        self.edges[(source, target)] = edge
        self.edges[(target, source)] = edge

        # Update adjacency
        self.nodes[source].neighbors.add(target)
        self.nodes[target].neighbors.add(source)
        self.nodes[source].degree += 1
        self.nodes[target].degree += 1

    def shortest_path(
        self,
        start_id: str,
        end_id: str,
        use_functional: bool = False
    ) -> List[str]:
        """
        Find shortest path using BFS.

        Args:
            use_functional: If True, weight by functional connectivity
                          (active paths). Otherwise use anatomical
                          (all possible paths).

        Returns:
            List of node IDs forming path from start to end
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        if start_id == end_id:
            return [start_id]

        visited = {start_id}
        queue = deque([(start_id, [start_id])])

        while queue:
            current, path = queue.popleft()

            for neighbor in self.nodes[current].neighbors:
                # Check if edge is active (if using functional)
                if use_functional:
                    edge = self.edges.get((current, neighbor))
                    if edge and edge.functional_weight < 0.1:
                        continue  # Skip inactive edges

                if neighbor == end_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # No path found

    def activate_path(self, path: List[str], strength: float = 0.1) -> None:
        """
        Activate functional connectivity along path.

        Simulates Hebbian learning: "Neurons that fire together wire together"
        When reasoning uses a path, functional connectivity strengthens.
        """
        for i in range(len(path) - 1):
            edge_key = (path[i], path[i+1])

            if edge_key in self.edges:
                edge = self.edges[edge_key]

                # Increase functional weight (saturates at 1.0)
                edge.functional_weight = min(1.0, edge.functional_weight + strength)
                edge.activation_count += 1
                edge.last_activated = time.time()

                # Update node access
                self.nodes[path[i]].access_count += 1
                self.nodes[path[i]].last_accessed = time.time()

        # Last node
        if path:
            self.nodes[path[-1]].access_count += 1
            self.nodes[path[-1]].last_accessed = time.time()

    def decay_functional_weights(self, decay_rate: float = 0.01) -> None:
        """
        Decay functional weights over time.

        Unused connections weaken (like synaptic pruning in brain).
        This maintains efficiency and prevents over-connectivity.
        """
        for edge in self.edges.values():
            edge.functional_weight = max(0.0, edge.functional_weight - decay_rate)

    def find_analogies(
        self,
        concept_id: str,
        max_distance: int = 3,
        different_domain: bool = True
    ) -> List[Tuple[str, int]]:
        """
        Find analogous concepts via graph traversal.

        Exploits small-world shortcuts to discover cross-domain analogies!

        Returns:
            List of (concept_id, distance) tuples
        """
        if concept_id not in self.nodes:
            return []

        concept = self.nodes[concept_id]
        analogies = []

        visited = {concept_id}
        queue = deque([(concept_id, 0)])

        while queue:
            current, distance = queue.popleft()

            if distance > max_distance:
                continue

            current_node = self.nodes[current]

            # Check if this is an analogy
            if current != concept_id:
                is_analogy = (
                    not different_domain or
                    current_node.domain != concept.domain
                )
                if is_analogy:
                    analogies.append((current, distance))

            # Explore neighbors
            if distance < max_distance:
                for neighbor in current_node.neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

        return analogies

    def detect_hubs(self, top_k: int = 10) -> List[str]:
        """
        Identify hub nodes (high degree centrality).

        Hubs are key concepts that connect many others.
        Learning a hub unlocks access to many connected concepts.
        """
        # Sort by degree (simple but effective)
        nodes_by_degree = sorted(
            self.nodes.items(),
            key=lambda x: x[1].degree,
            reverse=True
        )

        self.hubs = [node_id for node_id, _ in nodes_by_degree[:top_k]]
        return self.hubs

    def compute_network_statistics(self, sample_size: int = 100) -> Dict:
        """
        Compute small-world metrics.

        Returns dict with clustering coefficient, path length, small-world index.
        """
        n = len(self.nodes)
        if n < 2:
            return {}

        # 1. Clustering coefficient
        clustering_sum = 0.0
        for node in self.nodes.values():
            if node.degree < 2:
                continue

            # Count edges between neighbors
            neighbors = list(node.neighbors)
            edges_between = 0

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.nodes[neighbors[i]].neighbors:
                        edges_between += 1

            # Clustering = actual / possible
            possible = node.degree * (node.degree - 1) / 2
            node.clustering_coef = edges_between / possible if possible > 0 else 0
            clustering_sum += node.clustering_coef

        self.clustering_coefficient = clustering_sum / n

        # 2. Average path length (sample for efficiency)
        path_lengths = []
        node_ids = list(self.nodes.keys())
        actual_samples = min(sample_size, len(node_ids) * (len(node_ids) - 1) // 2)

        for _ in range(actual_samples):
            start = np.random.choice(node_ids)
            end = np.random.choice(node_ids)
            if start != end:
                path = self.shortest_path(start, end)
                if path:
                    path_lengths.append(len(path) - 1)

        self.average_path_length = np.mean(path_lengths) if path_lengths else 0.0

        # 3. Small-world index σ = (C/C_random) / (L/L_random)
        #    σ > 1 indicates small-world network
        avg_degree = np.mean([n.degree for n in self.nodes.values()])
        C_random = avg_degree / n if n > 0 else 0
        L_random = np.log(n) / np.log(avg_degree) if avg_degree > 1 else 0

        if C_random > 0 and L_random > 0 and self.average_path_length > 0:
            self.small_world_index = (
                (self.clustering_coefficient / C_random) /
                (self.average_path_length / L_random)
            )
        else:
            self.small_world_index = 0.0

        return {
            'num_nodes': n,
            'num_edges': len(self.edges) // 2,  # Undirected
            'clustering_coefficient': self.clustering_coefficient,
            'average_path_length': self.average_path_length,
            'small_world_index': self.small_world_index,
            'is_small_world': self.small_world_index > 1.0
        }

    def get_domain_summary(self) -> Dict[str, int]:
        """Get count of concepts per domain."""
        return {
            domain: len(concepts)
            for domain, concepts in self.domain_clusters.items()
        }

    def visualize_cluster(
        self,
        concept_id: str,
        depth: int = 2
    ) -> str:
        """
        Visualize local cluster around a concept.

        Returns string representation of tree structure.
        """
        if concept_id not in self.nodes:
            return "Concept not found"

        concept = self.nodes[concept_id]
        lines = [f"[{concept.domain}] {concept.name}"]

        visited = {concept_id}
        queue = deque([(concept_id, 0)])

        while queue:
            current, d = queue.popleft()

            if d >= depth:
                continue

            node = self.nodes[current]

            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_node = self.nodes[neighbor]

                    indent = "  " * (d + 1)
                    edge = self.edges.get((current, neighbor))
                    edge_info = f"[{edge.edge_type}]" if edge else ""

                    lines.append(
                        f"{indent}└─ {edge_info} [{neighbor_node.domain}] "
                        f"{neighbor_node.name}"
                    )

                    queue.append((neighbor, d + 1))

        return "\n".join(lines)

    def save(self) -> None:
        """Save graph state to disk."""
        state = {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': [edge.to_dict() for edge in set(self.edges.values())],
            'stats': {
                'clustering': self.clustering_coefficient,
                'path_length': self.average_path_length,
                'small_world_index': self.small_world_index
            }
        }

        with open(self.save_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"[Saved] Graph state to {self.save_path}")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)


def create_test_graph() -> SmallWorldKnowledgeGraph:
    """Create a test graph with sample concepts."""
    graph = SmallWorldKnowledgeGraph(k_local=3, rewiring_prob=0.15)

    # Sample concepts across domains
    concepts = [
        ("calc1", "Derivative", "Definition of derivative", "Mathematics"),
        ("calc2", "Integral", "Definition of integral", "Mathematics"),
        ("calc3", "Limit", "Definition of limit", "Mathematics"),
        ("phys1", "Velocity", "Rate of change of position", "Physics"),
        ("phys2", "Acceleration", "Rate of change of velocity", "Physics"),
        ("phys3", "Force", "Mass times acceleration", "Physics"),
        ("bio1", "Population Growth", "Rate of change of population", "Biology"),
        ("bio2", "Evolution", "Change in species over time", "Biology"),
        ("econ1", "Marginal Cost", "Rate of change of cost", "Economics"),
        ("econ2", "Optimization", "Finding maximum or minimum", "Economics"),
    ]

    for cid, name, content, domain in concepts:
        # Simple embedding (random for demo)
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)

        node = ConceptNode(
            id=cid,
            name=name,
            content=content,
            domain=domain,
            embedding=embedding
        )

        graph.add_concept(node, verbose=False)

    return graph


if __name__ == "__main__":
    print("="*70)
    print("SMALL-WORLD KNOWLEDGE GRAPH - Test")
    print("="*70)

    # Create test graph
    graph = create_test_graph()

    # Compute statistics
    stats = graph.compute_network_statistics()

    print(f"\n[Network Statistics]")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Clustering: {stats['clustering_coefficient']:.3f}")
    print(f"  Avg Path Length: {stats['average_path_length']:.2f}")
    print(f"  Small-World Index: {stats['small_world_index']:.2f}")
    print(f"  Is Small-World: {'✓ YES' if stats['is_small_world'] else '✗ NO'}")

    # Detect hubs
    print(f"\n[Hub Concepts]")
    hubs = graph.detect_hubs(top_k=5)
    for hub_id in hubs:
        hub = graph.nodes[hub_id]
        print(f"  • {hub.name} (degree: {hub.degree}, domain: {hub.domain})")

    # Find analogies
    print(f"\n[Cross-Domain Analogies]")
    concept_id = "calc1"  # Derivative
    analogies = graph.find_analogies(concept_id, max_distance=2)

    print(f"Analogies for '{graph.nodes[concept_id].name}':")
    for analogy_id, distance in analogies[:5]:
        analogy = graph.nodes[analogy_id]
        print(f"  • {analogy.name} ({analogy.domain}) - {distance} hops away")

    # Visualize cluster
    print(f"\n[Local Cluster]")
    print(graph.visualize_cluster("calc1", depth=2))

    print("\n" + "="*70)
