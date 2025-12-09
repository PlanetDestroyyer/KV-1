"""
Phase 3: Deep Mathematical Abstraction Engine

This module implements the ability to recognize when seemingly different problems
share the same underlying mathematical structure.

Key insight: "Linear algebra IS group theory" - many domains that look different
are actually the same mathematics in disguise.

Examples:
- Solving linear equations = Finding group inverses
- Polynomial roots = Eigenvalues of companion matrices
- Differential equations = Vector field flows
- Graph connectivity = Matrix rank
- Probability transitions = Stochastic linear operators

This is DEEP abstraction: seeing past surface features to mathematical essence.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class MathematicalFramework(Enum):
    """Different mathematical lenses for viewing problems"""
    LINEAR_ALGEBRA = "linear_algebra"
    GROUP_THEORY = "group_theory"
    CATEGORY_THEORY = "category_theory"
    TOPOLOGY = "topology"
    DIFFERENTIAL_GEOMETRY = "differential_geometry"
    GRAPH_THEORY = "graph_theory"
    PROBABILITY = "probability"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    ANALYSIS = "analysis"
    ALGEBRA = "algebra"
    COMBINATORICS = "combinatorics"
    NUMBER_THEORY = "number_theory"
    OPTIMIZATION = "optimization"


@dataclass
class AbstractStructure:
    """
    An abstract mathematical structure independent of domain.

    Example: The structure of "rotation" appears in:
    - Geometry (rotating shapes)
    - Complex numbers (multiplication by e^iθ)
    - Matrices (SO(n) group)
    - Quantum mechanics (spin operators)

    All are THE SAME mathematical object (SO(n) group)!
    """
    name: str
    framework: MathematicalFramework
    axioms: List[str]  # Mathematical properties that define this structure
    operations: List[str]  # Operations that must exist
    examples_from_domains: Dict[str, str]  # domain → concrete example

    # Structural fingerprint for isomorphism detection
    commutativity: bool = False
    associativity: bool = True
    has_identity: bool = True
    has_inverses: bool = True
    has_closure: bool = True
    dimension: Optional[int] = None

    def matches_structure(self, other: 'AbstractStructure', tolerance: float = 0.8) -> bool:
        """Check if two structures are isomorphic"""
        matches = 0
        total = 0

        # Check structural properties
        properties = [
            'commutativity', 'associativity', 'has_identity',
            'has_inverses', 'has_closure'
        ]

        for prop in properties:
            total += 1
            if getattr(self, prop) == getattr(other, prop):
                matches += 1

        # Check operation overlap
        op_overlap = len(set(self.operations) & set(other.operations))
        op_total = len(set(self.operations) | set(other.operations))
        if op_total > 0:
            matches += op_overlap / op_total
            total += 1

        return (matches / total) >= tolerance


@dataclass
class StructuralIsomorphism:
    """
    A mapping between two domains that preserves mathematical structure.

    Example: Linear equations ≅ Group equations
    - 'Ax = b' ↔ 'g * x = h'
    - 'Matrix A' ↔ 'Group element g'
    - 'Vector x' ↔ 'Group element x'
    - 'Solve for x' ↔ 'Find inverse'
    """
    domain_A: str
    domain_B: str
    structure_type: str  # "group", "ring", "vector_space", etc.
    mappings: Dict[str, str]  # concept_in_A → concept_in_B
    preserved_properties: List[str]
    confidence: float
    discovered_at: datetime = field(default_factory=datetime.now)

    def translate_problem(self, problem_text: str, from_domain: str, to_domain: str) -> str:
        """Translate a problem from one domain to another using isomorphism"""
        translated = problem_text

        if from_domain == self.domain_A and to_domain == self.domain_B:
            for concept_a, concept_b in self.mappings.items():
                translated = translated.replace(concept_a, concept_b)
        elif from_domain == self.domain_B and to_domain == self.domain_A:
            for concept_a, concept_b in self.mappings.items():
                translated = translated.replace(concept_b, concept_a)

        return translated


class DeepAbstractionEngine:
    """
    The core engine for deep mathematical abstraction.

    This engine:
    1. Recognizes abstract structures across different domains
    2. Detects structural isomorphisms
    3. Enables transfer learning via abstraction
    4. Selects optimal mathematical framework for each problem
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.known_structures: List[AbstractStructure] = []
        self.discovered_isomorphisms: List[StructuralIsomorphism] = []
        self.domain_to_structure: Dict[str, List[str]] = {}  # domain → structure names

        # Initialize with fundamental mathematical structures
        self._initialize_fundamental_structures()

        # Load saved abstractions if available
        if storage_path:
            self._load_from_disk()

    def _initialize_fundamental_structures(self):
        """Initialize knowledge of fundamental mathematical structures"""

        # Structure: Abelian Group (commutative group)
        self.known_structures.append(AbstractStructure(
            name="abelian_group",
            framework=MathematicalFramework.GROUP_THEORY,
            axioms=["closure", "associativity", "identity", "inverses", "commutativity"],
            operations=["addition", "inverse"],
            examples_from_domains={
                "linear_algebra": "Vector addition",
                "number_theory": "Integer addition modulo n",
                "geometry": "Translation group",
                "physics": "Symmetry transformations"
            },
            commutativity=True,
            associativity=True,
            has_identity=True,
            has_inverses=True,
            has_closure=True
        ))

        # Structure: Vector Space
        self.known_structures.append(AbstractStructure(
            name="vector_space",
            framework=MathematicalFramework.LINEAR_ALGEBRA,
            axioms=["abelian_group", "scalar_multiplication", "distributivity"],
            operations=["addition", "scalar_multiplication"],
            examples_from_domains={
                "linear_algebra": "R^n with vector addition",
                "physics": "State space in quantum mechanics",
                "differential_equations": "Solution space",
                "computer_graphics": "3D transformations",
                "machine_learning": "Feature space"
            },
            commutativity=True,
            associativity=True,
            has_identity=True,
            has_inverses=True,
            has_closure=True
        ))

        # Structure: Linear Operator
        self.known_structures.append(AbstractStructure(
            name="linear_operator",
            framework=MathematicalFramework.LINEAR_ALGEBRA,
            axioms=["linearity", "additivity", "homogeneity"],
            operations=["apply", "compose", "inverse"],
            examples_from_domains={
                "linear_algebra": "Matrix multiplication",
                "calculus": "Differential operator d/dx",
                "fourier_analysis": "Fourier transform",
                "physics": "Observable operators",
                "graph_theory": "Adjacency matrix"
            },
            commutativity=False,
            associativity=True,
            has_identity=True,
            has_inverses=False,  # Not all operators invertible
            has_closure=True
        ))

        # Structure: Graph (discrete structure)
        self.known_structures.append(AbstractStructure(
            name="graph",
            framework=MathematicalFramework.GRAPH_THEORY,
            axioms=["vertices", "edges", "connectivity"],
            operations=["traverse", "connect", "neighbor"],
            examples_from_domains={
                "graph_theory": "Undirected graph",
                "linear_algebra": "Adjacency matrix",
                "probability": "Markov chain",
                "networks": "Network topology",
                "chemistry": "Molecular structure"
            },
            commutativity=True,  # Undirected
            associativity=False,
            has_identity=False,
            has_inverses=False,
            has_closure=False
        ))

        # Update domain mappings
        for structure in self.known_structures:
            for domain in structure.examples_from_domains.keys():
                if domain not in self.domain_to_structure:
                    self.domain_to_structure[domain] = []
                self.domain_to_structure[domain].append(structure.name)

    def recognize_abstract_structure(self, problem_data: Dict[str, Any]) -> Optional[AbstractStructure]:
        """
        Recognize the abstract mathematical structure underlying a concrete problem.

        This is the KEY capability: seeing past surface details to mathematical essence.
        """
        domain = problem_data.get('domain', '')
        operations = set(problem_data.get('operations', []))
        properties = set(problem_data.get('properties', []))
        object_type = problem_data.get('object_type', '')

        best_match = None
        best_score = 0.0

        for structure in self.known_structures:
            score = 0.0

            # Check if domain is known for this structure
            if domain in structure.examples_from_domains:
                score += 0.3

            # Check operation overlap
            structure_ops = set(structure.operations)
            if operations:
                overlap = len(operations & structure_ops)
                score += 0.4 * (overlap / max(len(operations), len(structure_ops)))

            # Check property matches
            structure_props = set(structure.axioms)
            if properties:
                overlap = len(properties & structure_props)
                score += 0.3 * (overlap / max(len(properties), len(structure_props)))

            if score > best_score:
                best_score = score
                best_match = structure

        return best_match if best_score > 0.5 else None

    def detect_isomorphism(self, domain_A_data: Dict, domain_B_data: Dict) -> Optional[StructuralIsomorphism]:
        """
        Detect if two problems from different domains are structurally isomorphic.

        Example: Solving 'Ax = b' (linear algebra) is isomorphic to
        finding group inverse in 'g * x = h' (group theory).
        """
        # Recognize abstract structures
        structure_A = self.recognize_abstract_structure(domain_A_data)
        structure_B = self.recognize_abstract_structure(domain_B_data)

        if not structure_A or not structure_B:
            return None

        # Check if structures match
        if structure_A.matches_structure(structure_B):
            # Build mapping between concepts
            mappings = {}

            # Map operations
            ops_A = domain_A_data.get('operations', [])
            ops_B = domain_B_data.get('operations', [])

            # Simple heuristic: map by position if same structure
            if structure_A.name == structure_B.name:
                for i, op in enumerate(ops_A):
                    if i < len(ops_B):
                        mappings[op] = ops_B[i]

            # Create isomorphism
            isomorphism = StructuralIsomorphism(
                domain_A=domain_A_data.get('domain', 'unknown_A'),
                domain_B=domain_B_data.get('domain', 'unknown_B'),
                structure_type=structure_A.name,
                mappings=mappings,
                preserved_properties=structure_A.axioms,
                confidence=0.8  # TODO: compute from match quality
            )

            self.discovered_isomorphisms.append(isomorphism)
            return isomorphism

        return None

    def transfer_solution_via_abstraction(self, source_domain: str, target_domain: str,
                                         source_solution: str) -> Optional[str]:
        """
        Transfer a solution from one domain to another via abstract structure.

        Example: Solution to "find matrix inverse" can be transferred to
        "find group inverse" because both are the same abstract operation.
        """
        # Find isomorphism between domains
        relevant_iso = None
        for iso in self.discovered_isomorphisms:
            if (iso.domain_A == source_domain and iso.domain_B == target_domain) or \
               (iso.domain_A == target_domain and iso.domain_B == source_domain):
                relevant_iso = iso
                break

        if not relevant_iso:
            return None

        # Translate solution using isomorphism
        translated = relevant_iso.translate_problem(
            source_solution,
            from_domain=source_domain,
            to_domain=target_domain
        )

        return translated

    def select_optimal_framework(self, problem_description: str,
                                 available_frameworks: Optional[List[MathematicalFramework]] = None) -> MathematicalFramework:
        """
        Meta-reasoning: Select the best mathematical framework for solving a problem.

        Example: "Find shortest path in network" could be viewed as:
        - Graph theory (Dijkstra's algorithm)
        - Linear algebra (adjacency matrix powers)
        - Optimization (minimize distance functional)

        Which framework is best depends on problem size, structure, and what we know.
        """
        if available_frameworks is None:
            available_frameworks = list(MathematicalFramework)

        problem_lower = problem_description.lower()
        framework_scores = {fw: 0.0 for fw in available_frameworks}

        # Keyword-based heuristics (simplified)
        keywords_to_framework = {
            'matrix': MathematicalFramework.LINEAR_ALGEBRA,
            'vector': MathematicalFramework.LINEAR_ALGEBRA,
            'eigenvalue': MathematicalFramework.LINEAR_ALGEBRA,
            'group': MathematicalFramework.GROUP_THEORY,
            'symmetry': MathematicalFramework.GROUP_THEORY,
            'graph': MathematicalFramework.GRAPH_THEORY,
            'network': MathematicalFramework.GRAPH_THEORY,
            'optimize': MathematicalFramework.OPTIMIZATION,
            'minimize': MathematicalFramework.OPTIMIZATION,
            'maximize': MathematicalFramework.OPTIMIZATION,
            'probability': MathematicalFramework.PROBABILITY,
            'random': MathematicalFramework.PROBABILITY,
            'continuous': MathematicalFramework.ANALYSIS,
            'limit': MathematicalFramework.ANALYSIS,
            'derivative': MathematicalFramework.ANALYSIS,
            'topology': MathematicalFramework.TOPOLOGY,
            'prime': MathematicalFramework.NUMBER_THEORY,
            'modulo': MathematicalFramework.NUMBER_THEORY,
        }

        for keyword, framework in keywords_to_framework.items():
            if keyword in problem_lower:
                if framework in framework_scores:
                    framework_scores[framework] += 1.0

        # Return framework with highest score
        best_framework = max(framework_scores.items(), key=lambda x: x[1])
        return best_framework[0] if best_framework[1] > 0 else MathematicalFramework.ALGEBRA

    def find_unifying_abstraction(self, problems: List[Dict[str, Any]]) -> Optional[AbstractStructure]:
        """
        Find the abstract structure that unifies multiple concrete problems.

        Example: Given problems about:
        - Solving linear equations
        - Finding matrix inverses
        - Transforming coordinate systems

        Recognize they're all instances of LINEAR_OPERATOR structure.
        """
        if not problems:
            return None

        # Recognize structure in each problem
        structures = []
        for problem in problems:
            struct = self.recognize_abstract_structure(problem)
            if struct:
                structures.append(struct)

        if not structures:
            return None

        # Find most general structure that covers all
        # For now, use most common structure
        structure_counts = {}
        for struct in structures:
            name = struct.name
            structure_counts[name] = structure_counts.get(name, 0) + 1

        most_common_name = max(structure_counts.items(), key=lambda x: x[1])[0]

        for struct in self.known_structures:
            if struct.name == most_common_name:
                return struct

        return None

    def explain_abstraction(self, concrete_problem: str, abstract_structure: AbstractStructure) -> str:
        """Generate human-readable explanation of the abstraction"""
        explanation = f"DEEP ABSTRACTION INSIGHT:\n\n"
        explanation += f"This problem is fundamentally about: {abstract_structure.name}\n"
        explanation += f"Mathematical framework: {abstract_structure.framework.value}\n\n"
        explanation += f"Key properties:\n"
        for axiom in abstract_structure.axioms:
            explanation += f"  - {axiom}\n"
        explanation += f"\nThis same mathematical structure appears in:\n"
        for domain, example in abstract_structure.examples_from_domains.items():
            explanation += f"  - {domain}: {example}\n"
        explanation += f"\nTherefore, techniques from ANY of these domains can be applied!"

        return explanation

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered abstractions"""
        return {
            'known_structures': len(self.known_structures),
            'discovered_isomorphisms': len(self.discovered_isomorphisms),
            'domains_covered': len(self.domain_to_structure),
            'structures_per_domain': {
                domain: len(structs)
                for domain, structs in self.domain_to_structure.items()
            },
            'isomorphism_confidences': [
                iso.confidence for iso in self.discovered_isomorphisms
            ]
        }

    def _load_from_disk(self):
        """Load saved abstractions from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                # TODO: Deserialize discovered isomorphisms
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_to_disk(self):
        """Save discovered abstractions to disk"""
        if not self.storage_path:
            return

        data = {
            'isomorphisms': [
                {
                    'domain_A': iso.domain_A,
                    'domain_B': iso.domain_B,
                    'structure_type': iso.structure_type,
                    'mappings': iso.mappings,
                    'confidence': iso.confidence,
                    'discovered_at': iso.discovered_at.isoformat()
                }
                for iso in self.discovered_isomorphisms
            ]
        }

        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[!] Failed to save abstractions: {e}")


class FrameworkSelector:
    """
    Meta-reasoning system for selecting optimal mathematical framework.

    Given a problem, decides: "Should I think about this using linear algebra?
    Group theory? Topology? Optimization?"

    This is meta-mathematical reasoning - reasoning about which mathematics to use.
    """

    def __init__(self):
        self.framework_history: Dict[str, List[Tuple[MathematicalFramework, float]]] = {}
        self.abstraction_engine = DeepAbstractionEngine()

    def select_framework(self, problem: str, context: Optional[Dict] = None) -> Tuple[MathematicalFramework, float]:
        """
        Select best framework with confidence score.

        Returns: (framework, confidence)
        """
        # Get recommendation from abstraction engine
        framework = self.abstraction_engine.select_optimal_framework(problem)

        # Check if we have historical data for similar problems
        confidence = 0.7  # Default confidence

        # If we've solved similar problems before, boost confidence
        for past_problem, past_selections in self.framework_history.items():
            # Simple similarity check (could be improved with embeddings)
            if len(set(problem.split()) & set(past_problem.split())) > 3:
                # Similar problem found
                if past_selections and past_selections[-1][0] == framework:
                    confidence = min(0.95, confidence + 0.15)

        return framework, confidence

    def record_framework_usage(self, problem: str, framework: MathematicalFramework, success: bool):
        """Record which framework was used and whether it succeeded"""
        if problem not in self.framework_history:
            self.framework_history[problem] = []

        confidence = 1.0 if success else 0.3
        self.framework_history[problem].append((framework, confidence))
