"""
FEP-Guided Knowledge Graph

Knowledge graph where connections minimize Free Energy!

Key Innovation:
- Connections evaluated by FE = Prediction Error + Complexity
- High FE regions = knowledge gaps = discovery opportunities
- Graph self-organizes for optimal explanatory power

This is what makes knowledge organization INTELLIGENT!
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import networkx as nx


@dataclass
class Concept:
    """A concept node in the knowledge graph."""
    id: str
    definition: str
    domain: str
    embedding: Optional[np.ndarray] = None

    # FEP metrics
    free_energy: float = 0.5  # Current FE for this concept
    prediction_error: float = 0.5  # How well neighbors predict this
    complexity: float = 0.5  # How unusual this concept is

    # Metadata
    confidence: float = 0.7
    sources: List[str] = field(default_factory=list)
    learned_at: str = ""


@dataclass
class Connection:
    """A connection between concepts."""
    source: str
    target: str
    weight: float = 1.0  # Connection strength
    connection_type: str = "semantic"  # semantic, causal, prerequisite, etc.

    # FEP metrics
    fe_reduction: float = 0.0  # How much FE does this connection reduce?
    surprisal: float = 0.5  # How unexpected is this connection?


class FEPGuidedKnowledgeGraph:
    """
    Knowledge graph that self-organizes using Free Energy Principle.

    Core Innovation:
    - Connections minimize surprise (prediction error + complexity)
    - High FE regions = gaps → discovery opportunities
    - Graph structure emerges from information theory
    """

    def __init__(self, vector_store=None):
        self.vector_store = vector_store  # For embeddings
        self.graph = nx.DiGraph()  # Directed graph

        # Concept storage
        self.concepts: Dict[str, Concept] = {}

        # Connection evaluation
        self.connection_proposals: List[Tuple[str, str, float]] = []  # (src, tgt, fe_delta)

        # FEP parameters
        self.fe_threshold = 0.6  # High FE = gap
        self.connection_threshold = 0.0  # Only connect if ΔFE < 0 (reduces FE)

        print("[FEP Graph] Initialized - Knowledge will self-organize!")

    def add_concept(
        self,
        concept_id: str,
        definition: str,
        domain: str,
        embedding: Optional[np.ndarray] = None,
        confidence: float = 0.7
    ):
        """
        Add concept to graph.

        Automatically evaluates connections using FEP!
        """
        # Create concept
        concept = Concept(
            id=concept_id,
            definition=definition,
            domain=domain,
            embedding=embedding,
            confidence=confidence
        )

        # Compute initial FE
        concept.free_energy = self._compute_initial_fe(concept)
        concept.prediction_error = concept.free_energy / 2  # Initial split
        concept.complexity = concept.free_energy / 2

        # Store
        self.concepts[concept_id] = concept
        self.graph.add_node(concept_id, concept=concept)

        # Find optimal connections using FEP
        if len(self.concepts) > 1:
            self._connect_using_fep(concept_id)

    def _compute_initial_fe(self, concept: Concept) -> float:
        """
        Compute initial free energy for a concept.

        FE = Prediction Error + Complexity

        High FE = poorly understood/connected
        """
        # Prediction error: Can we predict this from existing concepts?
        if len(self.concepts) == 0:
            prediction_error = 1.0  # First concept = max uncertainty
        else:
            # If we have embeddings, use semantic similarity
            if concept.embedding is not None and self.vector_store is not None:
                # Find similar concepts
                try:
                    similar = self.vector_store.search(concept.embedding, k=5, threshold=0.5)
                    if len(similar) > 0:
                        # Good prediction from neighbors
                        avg_sim = np.mean([sim for _, sim, _ in similar])
                        prediction_error = 1.0 - avg_sim
                    else:
                        prediction_error = 0.9  # No similar concepts
                except:
                    prediction_error = 0.7  # Default
            else:
                # No embeddings - use domain matching
                same_domain = [c for c in self.concepts.values() if c.domain == concept.domain]
                if len(same_domain) > 0:
                    prediction_error = 0.5  # Some predictability from domain
                else:
                    prediction_error = 0.8  # New domain = high uncertainty

        # Complexity: How unusual is this concept?
        # Simple heuristic: rare domains = high complexity
        domain_counts = defaultdict(int)
        for c in self.concepts.values():
            domain_counts[c.domain] += 1

        total = len(self.concepts) if len(self.concepts) > 0 else 1
        domain_freq = domain_counts.get(concept.domain, 0) / total

        if domain_freq > 0.2:  # Common domain
            complexity = 0.2
        elif domain_freq > 0.1:
            complexity = 0.4
        else:  # Rare domain
            complexity = 0.7

        # Total FE
        free_energy = prediction_error + complexity

        return free_energy

    def _connect_using_fep(self, new_concept_id: str):
        """
        Find optimal connections for new concept using FEP.

        KEY INNOVATION: Only connect if ΔFE < 0 (reduces surprise)!
        """
        new_concept = self.concepts[new_concept_id]

        # Evaluate all possible connections
        candidates = []

        for existing_id, existing_concept in self.concepts.items():
            if existing_id == new_concept_id:
                continue

            # Simulate connection
            fe_delta = self._simulate_connection(new_concept_id, existing_id)

            candidates.append((existing_id, fe_delta))

        # Sort by FE reduction (most negative = best)
        candidates.sort(key=lambda x: x[1])

        # Connect top candidates that reduce FE
        for existing_id, fe_delta in candidates[:5]:  # Top 5
            if fe_delta < self.connection_threshold:  # Reduces FE
                self._add_connection(new_concept_id, existing_id, -fe_delta)
                print(f"[FEP] Connected {new_concept_id} ↔ {existing_id} (ΔFE: {fe_delta:.3f})")

    def _simulate_connection(self, concept_a: str, concept_b: str) -> float:
        """
        Simulate adding a connection and compute FE change.

        Returns:
            ΔFE: Negative = reduces FE (good!), Positive = increases FE (bad!)
        """
        ca = self.concepts[concept_a]
        cb = self.concepts[concept_b]

        # Current FE
        current_fe_a = ca.free_energy
        current_fe_b = cb.free_energy
        current_fe_total = current_fe_a + current_fe_b

        # Predicted FE after connection
        # Connection reduces prediction error (better prediction from neighbors)
        # But may increase complexity (unexpected connections are complex)

        # Similarity between concepts (from embeddings or domain)
        if ca.embedding is not None and cb.embedding is not None:
            # Cosine similarity
            sim = np.dot(ca.embedding, cb.embedding) / (
                np.linalg.norm(ca.embedding) * np.linalg.norm(cb.embedding) + 1e-8
            )
            sim = (sim + 1) / 2  # Normalize to 0-1
        elif ca.domain == cb.domain:
            sim = 0.6  # Same domain = moderate similarity
        else:
            sim = 0.3  # Different domains = low similarity

        # Prediction error reduction
        # Similar concepts reduce prediction error more
        pe_reduction = 0.3 * sim

        # Complexity increase (unexpected connections are complex)
        # Similar concepts = expected connection = low complexity
        complexity_increase = 0.2 * (1.0 - sim)

        # Net change
        delta_fe = complexity_increase - pe_reduction

        return delta_fe

    def _add_connection(
        self,
        source: str,
        target: str,
        weight: float,
        connection_type: str = "semantic"
    ):
        """Add connection to graph."""
        conn = Connection(
            source=source,
            target=target,
            weight=weight,
            connection_type=connection_type,
            fe_reduction=weight
        )

        self.graph.add_edge(source, target, connection=conn, weight=weight)

        # Update FE for both concepts
        self._update_concept_fe(source)
        self._update_concept_fe(target)

    def _update_concept_fe(self, concept_id: str):
        """Recompute FE for a concept based on its connections."""
        if concept_id not in self.concepts:
            return

        concept = self.concepts[concept_id]
        neighbors = list(self.graph.neighbors(concept_id))

        # Prediction error: More connected = better prediction
        if len(neighbors) == 0:
            prediction_error = 1.0
        else:
            # Well-connected concepts have low prediction error
            prediction_error = max(0.1, 1.0 - (len(neighbors) / 10.0))

        # Complexity stays same for now
        complexity = concept.complexity

        # Update
        concept.prediction_error = prediction_error
        concept.free_energy = prediction_error + complexity

    def identify_high_fe_regions(
        self,
        threshold: Optional[float] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Identify knowledge gaps (high FE regions).

        These are discovery opportunities!
        """
        if threshold is None:
            threshold = self.fe_threshold

        gaps = []

        for concept_id, concept in self.concepts.items():
            if concept.free_energy >= threshold:
                gaps.append({
                    'concept': concept_id,
                    'free_energy': concept.free_energy,
                    'prediction_error': concept.prediction_error,
                    'complexity': concept.complexity,
                    'type': 'high_fe_concept'
                })

        # Also find missing connections
        missing_conns = self.find_missing_connections()
        gaps.extend(missing_conns)

        # Sort by FE (highest = biggest gaps)
        gaps.sort(key=lambda x: x['free_energy'], reverse=True)

        return gaps[:top_k]

    def find_missing_connections(self) -> List[Dict]:
        """
        Find concept pairs that should be connected but aren't.

        Uses FEP: If connecting would reduce FE, it's a missing connection!
        """
        missing = []

        # Check all pairs
        concept_ids = list(self.concepts.keys())
        for i, id_a in enumerate(concept_ids):
            for id_b in concept_ids[i+1:]:
                # Already connected?
                if self.graph.has_edge(id_a, id_b) or self.graph.has_edge(id_b, id_a):
                    continue

                # Would connection reduce FE?
                fe_delta = self._simulate_connection(id_a, id_b)

                if fe_delta < -0.1:  # Significant FE reduction
                    missing.append({
                        'concept': f"{id_a} <-> {id_b}",
                        'free_energy': -fe_delta,  # Potential reduction
                        'type': 'missing_connection',
                        'source': id_a,
                        'target': id_b,
                        'expected_fe_reduction': -fe_delta
                    })

        return missing

    def get_neighbors(self, concept_id: str) -> List[str]:
        """Get all neighbors of a concept."""
        if concept_id not in self.graph:
            return []
        return list(self.graph.neighbors(concept_id))

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get concept by ID."""
        return self.concepts.get(concept_id)

    def get_all_concepts(self) -> List[str]:
        """Get all concept IDs."""
        return list(self.concepts.keys())

    def compute_graph_free_energy(self) -> float:
        """
        Compute total free energy of the graph.

        Lower = better organized knowledge!
        """
        if len(self.concepts) == 0:
            return 0.0

        total_fe = sum(c.free_energy for c in self.concepts.values())
        avg_fe = total_fe / len(self.concepts)

        return avg_fe

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        if len(self.concepts) == 0:
            return {'status': 'empty'}

        # FE stats
        fes = [c.free_energy for c in self.concepts.values()]
        avg_fe = np.mean(fes)
        std_fe = np.std(fes)
        max_fe = max(fes)
        min_fe = min(fes)

        # Graph stats
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

        # Gap stats
        high_fe_count = sum(1 for fe in fes if fe >= self.fe_threshold)

        return {
            'status': 'active',
            'num_concepts': num_nodes,
            'num_connections': num_edges,
            'avg_connections_per_concept': avg_degree,

            # FE metrics
            'avg_free_energy': avg_fe,
            'std_free_energy': std_fe,
            'max_free_energy': max_fe,
            'min_free_energy': min_fe,

            # Gaps
            'knowledge_gaps': high_fe_count,
            'gap_percentage': 100 * high_fe_count / num_nodes if num_nodes > 0 else 0,

            # Health
            'health': 'good' if avg_fe < 0.5 else 'needs_learning'
        }

    def demonstrate_fep_organization(self):
        """Demonstrate FEP-guided organization."""
        print("\n" + "="*70)
        print("FEP-GUIDED KNOWLEDGE GRAPH")
        print("="*70)

        stats = self.get_statistics()

        if stats['status'] == 'empty':
            print("\n[!] Graph is empty - add some concepts first!")
            return

        print(f"\n📊 GRAPH STATISTICS:")
        print(f"  Concepts: {stats['num_concepts']}")
        print(f"  Connections: {stats['num_connections']}")
        print(f"  Avg connections/concept: {stats['avg_connections_per_concept']:.1f}")

        print(f"\n🔋 FREE ENERGY METRICS:")
        print(f"  Average FE: {stats['avg_free_energy']:.3f}")
        print(f"  Min FE: {stats['min_free_energy']:.3f} (well-organized)")
        print(f"  Max FE: {stats['max_free_energy']:.3f} (needs work)")
        print(f"  Std FE: {stats['std_free_energy']:.3f}")

        print(f"\n🔍 KNOWLEDGE GAPS:")
        print(f"  High-FE concepts: {stats['knowledge_gaps']}")
        print(f"  Gap percentage: {stats['gap_percentage']:.1f}%")

        print(f"\n💡 HEALTH:")
        print(f"  Status: {stats['health'].upper()}")

        # Show top gaps
        gaps = self.identify_high_fe_regions(top_k=5)
        if len(gaps) > 0:
            print(f"\n🎯 TOP DISCOVERY OPPORTUNITIES:")
            for i, gap in enumerate(gaps, 1):
                print(f"  {i}. {gap['concept']}")
                print(f"     FE: {gap['free_energy']:.3f} | Type: {gap['type']}")

        print("\n" + "="*70)


# Demo
if __name__ == "__main__":
    print("FEP-Guided Knowledge Graph")
    print("Knowledge self-organizes using Free Energy Principle!")
    print("\nUse with:")
    print("  - FAISS Vector Store (for embeddings)")
    print("  - Hypothesis Generator (to investigate gaps)")
    print("  - Discovery Orchestrator (for autonomous discovery)")
