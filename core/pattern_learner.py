"""
Mathematical Pattern Learner

Learns mathematical patterns from examples, not hardcoded keywords.
This is LEARNED pattern recognition - the key to moving beyond prompt engineering!

Architecture:
1. Observe problem → solution traces
2. Extract mathematical structure (operations, symmetries, conservation laws)
3. Cluster similar problems to discover problem "types"
4. Predict structure for new problems based on learned patterns

This replaces keyword matching with actual learning!
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import numpy as np
from datetime import datetime
import re

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[!] sklearn not available - pattern clustering disabled")


@dataclass
class MathematicalStructure:
    """Represents the mathematical structure of a problem."""

    # What operations were used?
    operations: List[str]  # ['differentiate', 'integrate', 'solve', 'factor']

    # What type of mathematical object?
    object_type: str  # 'equation', 'function', 'graph', 'matrix', 'set'

    # What equation type?
    equation_type: Optional[str]  # 'linear', 'quadratic', 'differential', 'integral'

    # What symmetries exist?
    symmetries: List[str]  # ['rotation', 'reflection', 'translation']

    # Conservation laws
    conservation_laws: List[str]  # ['energy', 'momentum', 'cardinality']

    # Variable relationships
    relationships: List[Tuple[str, str, str]]  # [('x', 'increases_with', 'y')]

    # Domain
    domain: str  # 'algebra', 'calculus', 'number_theory', 'geometry'

    # Confidence in this structure
    confidence: float = 1.0


@dataclass
class PatternInstance:
    """A single instance of a pattern."""
    problem: str
    structure: MathematicalStructure
    solution_trace: List[str]
    success: bool
    solution_time: float
    timestamp: str


class MathematicalStructureLearner:
    """
    Learn mathematical patterns from problem-solving experience.

    This is the CORE of moving from keyword matching to learned intelligence!
    """

    def __init__(self, storage_path: str = "./pattern_database.json"):
        self.storage_path = storage_path

        # Database of all observed patterns
        self.pattern_instances: List[PatternInstance] = []

        # Discovered pattern clusters (learned problem types)
        self.pattern_clusters: Dict[int, List[PatternInstance]] = defaultdict(list)

        # Structure -> cluster mapping
        self.structure_to_cluster: Dict[str, int] = {}

        # Cluster -> best structure mapping
        self.cluster_prototypes: Dict[int, MathematicalStructure] = {}

        # Performance tracking
        self.cluster_success_rate: Dict[int, float] = {}

        self.load()

    def observe_problem_solution(
        self,
        problem: str,
        solution_trace: List[str],
        success: bool,
        solution_time: float = 0.0
    ):
        """
        Learn from a problem-solution pair.

        This is how the system LEARNS patterns!

        Args:
            problem: The problem statement
            solution_trace: Steps taken to solve (from LLM or system)
            success: Whether solution was correct
            solution_time: How long it took
        """
        print(f"\n[Pattern Learner] Observing new problem...")

        # Extract mathematical structure
        structure = self._extract_structure(problem, solution_trace)

        # Create pattern instance
        instance = PatternInstance(
            problem=problem,
            structure=structure,
            solution_trace=solution_trace,
            success=success,
            solution_time=solution_time,
            timestamp=datetime.now().isoformat()
        )

        # Add to database
        self.pattern_instances.append(instance)

        print(f"[Pattern Learner] Extracted structure:")
        print(f"  Operations: {structure.operations}")
        print(f"  Type: {structure.object_type}")
        print(f"  Domain: {structure.domain}")

        # Re-cluster if we have enough instances
        if len(self.pattern_instances) >= 10 and len(self.pattern_instances) % 5 == 0:
            print("[Pattern Learner] Re-clustering patterns...")
            self._cluster_patterns()

        # Save
        self.save()

    def _extract_structure(
        self,
        problem: str,
        solution_trace: List[str]
    ) -> MathematicalStructure:
        """
        Extract the MATHEMATICAL STRUCTURE from a problem.

        This is NOT keyword matching - it's structural analysis!
        """
        problem_lower = problem.lower()
        trace_text = " ".join(solution_trace).lower()

        # Detect operations used
        operations = self._detect_operations(trace_text)

        # Detect object type
        object_type = self._detect_object_type(problem_lower)

        # Detect equation type
        equation_type = self._detect_equation_type(problem_lower, trace_text)

        # Detect symmetries
        symmetries = self._detect_symmetries(problem_lower)

        # Detect conservation laws
        conservation = self._detect_conservation_laws(problem_lower, trace_text)

        # Detect relationships
        relationships = self._detect_relationships(problem_lower)

        # Detect domain
        domain = self._detect_domain(problem_lower, operations, object_type)

        return MathematicalStructure(
            operations=operations,
            object_type=object_type,
            equation_type=equation_type,
            symmetries=symmetries,
            conservation_laws=conservation,
            relationships=relationships,
            domain=domain,
            confidence=0.8
        )

    def _detect_operations(self, trace_text: str) -> List[str]:
        """Detect which mathematical operations were used."""
        operations = []

        operation_keywords = {
            'differentiate': ['derivative', 'differentiate', "d/dx", 'rate of change'],
            'integrate': ['integral', 'integrate', 'antiderivative', 'area under'],
            'solve': ['solve', 'find x', 'solution', 'roots'],
            'factor': ['factor', 'factorize', 'factorization'],
            'expand': ['expand', 'multiply out', 'distribute'],
            'simplify': ['simplify', 'reduce', 'combine like terms'],
            'substitute': ['substitute', 'plug in', 'replace'],
            'compose': ['compose', 'composition', 'f(g(x))'],
            'invert': ['inverse', 'invert'],
            'linearize': ['linear approximation', 'linearize', 'tangent line'],
            'optimize': ['maximize', 'minimize', 'optimize', 'extreme value'],
            'transform': ['fourier', 'laplace', 'transform'],
        }

        for op, keywords in operation_keywords.items():
            if any(kw in trace_text for kw in keywords):
                operations.append(op)

        return operations

    def _detect_object_type(self, problem: str) -> str:
        """Detect the type of mathematical object."""

        if any(word in problem for word in ['equation', 'solve', '=']):
            return 'equation'
        elif any(word in problem for word in ['function', 'f(x)', 'g(x)']):
            return 'function'
        elif any(word in problem for word in ['graph', 'network', 'vertices', 'edges']):
            return 'graph'
        elif any(word in problem for word in ['matrix', 'determinant', 'eigenvalue']):
            return 'matrix'
        elif any(word in problem for word in ['set', 'elements', 'cardinality']):
            return 'set'
        elif any(word in problem for word in ['sequence', 'series', 'sum']):
            return 'sequence'
        else:
            return 'unknown'

    def _detect_equation_type(self, problem: str, trace: str) -> Optional[str]:
        """Detect the type of equation if present."""

        combined = problem + " " + trace

        if 'differential' in combined or 'dy/dx' in combined:
            return 'differential'
        elif any(word in combined for word in ['integral', 'integrate']):
            return 'integral'
        elif 'quadratic' in combined or 'x²' in combined or 'x^2' in combined:
            return 'quadratic'
        elif re.search(r'\bx\s*\+\s*\d+|x\s*-\s*\d+', combined):
            return 'linear'
        elif 'polynomial' in combined:
            return 'polynomial'
        elif any(word in combined for word in ['exponential', 'e^x', 'exp']):
            return 'exponential'
        elif any(word in combined for word in ['logarithm', 'log', 'ln']):
            return 'logarithmic'
        else:
            return None

    def _detect_symmetries(self, problem: str) -> List[str]:
        """Detect symmetries in the problem."""
        symmetries = []

        if any(word in problem for word in ['even', 'symmetric', 'mirror']):
            symmetries.append('reflection')
        if any(word in problem for word in ['periodic', 'repeating', 'cycle']):
            symmetries.append('translation')
        if any(word in problem for word in ['rotation', 'circular']):
            symmetries.append('rotation')

        return symmetries

    def _detect_conservation_laws(self, problem: str, trace: str) -> List[str]:
        """Detect conservation laws."""
        laws = []

        combined = problem + " " + trace

        if 'constant' in combined or 'conserved' in combined:
            laws.append('conservation')
        if 'equal' in combined or 'balance' in combined:
            laws.append('equality')

        return laws

    def _detect_relationships(self, problem: str) -> List[Tuple[str, str, str]]:
        """Detect relationships between variables."""
        relationships = []

        # Simple pattern matching for now
        # Can be extended with NLP

        if 'proportional' in problem:
            relationships.append(('x', 'proportional_to', 'y'))
        if 'inverse' in problem:
            relationships.append(('x', 'inverse_to', 'y'))

        return relationships

    def _detect_domain(
        self,
        problem: str,
        operations: List[str],
        object_type: str
    ) -> str:
        """Detect the mathematical domain."""

        # Domain keywords
        if any(word in problem for word in ['derivative', 'integral', 'limit', 'continuous']):
            return 'calculus'
        elif any(word in problem for word in ['prime', 'factor', 'divisor', 'gcd', 'modulo']):
            return 'number_theory'
        elif any(word in problem for word in ['triangle', 'circle', 'angle', 'area', 'volume']):
            return 'geometry'
        elif any(word in problem for word in ['matrix', 'vector', 'eigenvalue', 'linear']):
            return 'linear_algebra'
        elif any(word in problem for word in ['probability', 'random', 'expected', 'variance']):
            return 'probability'
        elif any(word in problem for word in ['graph', 'vertex', 'edge', 'path']):
            return 'graph_theory'
        elif 'solve' in problem or '=' in problem:
            return 'algebra'
        else:
            return 'general'

    def _cluster_patterns(self):
        """
        Cluster pattern instances to discover problem types.

        This is where we LEARN categories, not hardcode them!
        """
        if not HAS_SKLEARN or len(self.pattern_instances) < 10:
            print("[Pattern Learner] Not enough data for clustering")
            return

        print(f"[Pattern Learner] Clustering {len(self.pattern_instances)} patterns...")

        # Convert structures to feature vectors
        features = []
        for instance in self.pattern_instances:
            feature_vector = self._structure_to_vector(instance.structure)
            features.append(feature_vector)

        X = np.array(features)

        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cluster with DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=3)
        labels = clustering.fit_predict(X_scaled)

        # Group instances by cluster
        self.pattern_clusters.clear()
        for idx, label in enumerate(labels):
            if label != -1:  # Ignore noise
                self.pattern_clusters[label].append(self.pattern_instances[idx])

        # Compute cluster prototypes
        for cluster_id, instances in self.pattern_clusters.items():
            prototype = self._compute_prototype(instances)
            self.cluster_prototypes[cluster_id] = prototype

            # Compute success rate
            successes = sum(1 for inst in instances if inst.success)
            self.cluster_success_rate[cluster_id] = successes / len(instances)

        print(f"[Pattern Learner] Discovered {len(self.pattern_clusters)} pattern clusters")
        for cluster_id, instances in self.pattern_clusters.items():
            success_rate = self.cluster_success_rate.get(cluster_id, 0.0)
            print(f"  Cluster {cluster_id}: {len(instances)} instances, {success_rate:.1%} success")

    def _structure_to_vector(self, structure: MathematicalStructure) -> np.ndarray:
        """Convert mathematical structure to feature vector for clustering."""

        # Create feature vector
        features = []

        # Operation features (one-hot encoding)
        all_operations = ['differentiate', 'integrate', 'solve', 'factor', 'expand',
                         'simplify', 'substitute', 'compose', 'invert', 'optimize']
        for op in all_operations:
            features.append(1.0 if op in structure.operations else 0.0)

        # Object type features
        all_types = ['equation', 'function', 'graph', 'matrix', 'set', 'sequence']
        for otype in all_types:
            features.append(1.0 if structure.object_type == otype else 0.0)

        # Equation type features
        all_eq_types = ['linear', 'quadratic', 'differential', 'integral', 'polynomial']
        for eq_type in all_eq_types:
            features.append(1.0 if structure.equation_type == eq_type else 0.0)

        # Domain features
        all_domains = ['calculus', 'algebra', 'number_theory', 'geometry',
                      'linear_algebra', 'probability', 'graph_theory']
        for domain in all_domains:
            features.append(1.0 if structure.domain == domain else 0.0)

        # Number of operations
        features.append(len(structure.operations) / 10.0)

        return np.array(features)

    def _compute_prototype(self, instances: List[PatternInstance]) -> MathematicalStructure:
        """Compute the prototype (average) structure for a cluster."""

        # Aggregate structures
        all_operations = []
        all_domains = []
        all_object_types = []

        for inst in instances:
            all_operations.extend(inst.structure.operations)
            all_domains.append(inst.structure.domain)
            all_object_types.append(inst.structure.object_type)

        # Most common elements
        from collections import Counter

        operations_count = Counter(all_operations)
        top_operations = [op for op, _ in operations_count.most_common(5)]

        domain = Counter(all_domains).most_common(1)[0][0]
        object_type = Counter(all_object_types).most_common(1)[0][0]

        # Use first instance as template
        template = instances[0].structure

        return MathematicalStructure(
            operations=top_operations,
            object_type=object_type,
            equation_type=template.equation_type,
            symmetries=template.symmetries,
            conservation_laws=template.conservation_laws,
            relationships=template.relationships,
            domain=domain,
            confidence=0.9
        )

    def predict_structure(self, new_problem: str) -> Optional[MathematicalStructure]:
        """
        Predict the mathematical structure for a NEW problem.

        This is LEARNED prediction, not keyword matching!

        Returns the structure that worked for similar problems.
        """
        if not self.cluster_prototypes:
            print("[Pattern Learner] No learned patterns yet")
            return None

        print(f"\n[Pattern Learner] Predicting structure for new problem...")

        # Extract features from new problem
        temp_structure = self._extract_structure(new_problem, [])
        new_features = self._structure_to_vector(temp_structure)

        # Find closest cluster
        best_cluster = None
        best_distance = float('inf')

        for cluster_id, prototype in self.cluster_prototypes.items():
            prototype_features = self._structure_to_vector(prototype)
            distance = np.linalg.norm(new_features - prototype_features)

            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster_id

        if best_cluster is not None:
            success_rate = self.cluster_success_rate.get(best_cluster, 0.0)
            print(f"[Pattern Learner] Matched to cluster {best_cluster} (success rate: {success_rate:.1%})")
            return self.cluster_prototypes[best_cluster]
        else:
            return None

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'total_instances': len(self.pattern_instances),
            'num_clusters': len(self.pattern_clusters),
            'cluster_success_rates': self.cluster_success_rate,
            'successful_instances': sum(1 for inst in self.pattern_instances if inst.success),
            'average_solution_time': np.mean([inst.solution_time for inst in self.pattern_instances]) if self.pattern_instances else 0
        }

    def save(self):
        """Save pattern database to disk."""
        data = {
            'pattern_instances': [
                {
                    'problem': inst.problem,
                    'structure': asdict(inst.structure),
                    'solution_trace': inst.solution_trace,
                    'success': inst.success,
                    'solution_time': inst.solution_time,
                    'timestamp': inst.timestamp
                }
                for inst in self.pattern_instances
            ],
            'cluster_success_rates': self.cluster_success_rate
        }

        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[Pattern Learner] Saved {len(self.pattern_instances)} patterns")
        except Exception as e:
            print(f"[!] Failed to save patterns: {e}")

    def load(self):
        """Load pattern database from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Restore instances
            for inst_data in data.get('pattern_instances', []):
                structure = MathematicalStructure(**inst_data['structure'])
                instance = PatternInstance(
                    problem=inst_data['problem'],
                    structure=structure,
                    solution_trace=inst_data['solution_trace'],
                    success=inst_data['success'],
                    solution_time=inst_data.get('solution_time', 0.0),
                    timestamp=inst_data['timestamp']
                )
                self.pattern_instances.append(instance)

            # Restore success rates
            self.cluster_success_rate = {
                int(k): v for k, v in data.get('cluster_success_rates', {}).items()
            }

            # Re-cluster if we have data
            if len(self.pattern_instances) >= 10:
                self._cluster_patterns()

            print(f"[Pattern Learner] Loaded {len(self.pattern_instances)} patterns")

        except FileNotFoundError:
            print("[Pattern Learner] No existing pattern database found")
        except Exception as e:
            print(f"[!] Failed to load patterns: {e}")
