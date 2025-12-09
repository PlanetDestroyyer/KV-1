"""
Phase 2: Compositional Reasoning Engine

This module implements the ability to:
1. Combine learned mathematical structures to solve novel problems
2. Build abstraction hierarchies (group → ring → field → vector space)
3. Transform problems between representations using structure morphisms
4. Chain multiple patterns to create complex solution strategies

This is a critical step toward AGI because it moves from pattern matching
to creative problem solving through composition.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime


class StructureType(Enum):
    """Hierarchy of mathematical structures from simple to complex"""
    SET = "set"                    # Just elements, no operations
    MAGMA = "magma"                # Set + binary operation
    SEMIGROUP = "semigroup"        # Magma + associativity
    MONOID = "monoid"              # Semigroup + identity
    GROUP = "group"                # Monoid + inverses
    ABELIAN_GROUP = "abelian_group"  # Group + commutativity
    RING = "ring"                  # Abelian group + multiplication
    FIELD = "field"                # Ring + multiplicative inverses
    VECTOR_SPACE = "vector_space"  # Field + scalar multiplication
    ALGEBRA = "algebra"            # Vector space + bilinear product

    @classmethod
    def get_hierarchy(cls) -> Dict[str, List[str]]:
        """Returns parent-child relationships in structure hierarchy"""
        return {
            "set": ["magma"],
            "magma": ["semigroup"],
            "semigroup": ["monoid"],
            "monoid": ["group"],
            "group": ["abelian_group", "ring"],
            "abelian_group": ["ring", "vector_space"],
            "ring": ["field"],
            "field": ["vector_space"],
            "vector_space": ["algebra"],
        }

    @classmethod
    def is_specialization(cls, base: str, derived: str) -> bool:
        """Check if 'derived' is a specialization of 'base'"""
        hierarchy = cls.get_hierarchy()
        visited = set()
        queue = [base]

        while queue:
            current = queue.pop(0)
            if current == derived:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(hierarchy.get(current, []))

        return False


@dataclass
class StructureMorphism:
    """
    A structure-preserving map between two mathematical structures.
    Examples: homomorphism, isomorphism, embedding
    """
    name: str
    source_structure: StructureType
    target_structure: StructureType
    transformation_type: str  # "homomorphism", "isomorphism", "embedding", "projection"
    properties_preserved: List[str]  # ["addition", "multiplication", "order", etc.]

    def can_transform(self, problem_structure: str) -> bool:
        """Check if this morphism can transform the given structure"""
        return problem_structure == self.source_structure.value


@dataclass
class CompositePattern:
    """
    A pattern created by composing multiple simpler patterns.
    Example: Solving polynomial equations = algebra + factoring + root finding
    """
    name: str
    component_patterns: List[str]  # Names of patterns being composed
    composition_order: List[int]   # Order to apply patterns
    structure_type: StructureType
    success_count: int = 0
    total_attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_attempts)

    def record_attempt(self, success: bool):
        """Record the outcome of using this composite pattern"""
        self.total_attempts += 1
        if success:
            self.success_count += 1


class CompositionEngine:
    """
    The core compositional reasoning engine.

    This engine takes learned patterns and combines them in novel ways to:
    1. Solve problems that don't match any single pattern
    2. Discover new solution strategies through composition
    3. Build abstraction hierarchies
    4. Transform problems between equivalent representations
    """

    def __init__(self):
        self.structure_hierarchy = StructureType.get_hierarchy()
        self.known_morphisms: List[StructureMorphism] = []
        self.composite_patterns: List[CompositePattern] = []
        self.abstraction_chains: Dict[str, List[StructureType]] = {}

        # Initialize common morphisms
        self._initialize_standard_morphisms()

    def _initialize_standard_morphisms(self):
        """Initialize commonly used structure morphisms"""
        # Group homomorphisms
        self.known_morphisms.append(StructureMorphism(
            name="group_to_abelian",
            source_structure=StructureType.GROUP,
            target_structure=StructureType.ABELIAN_GROUP,
            transformation_type="specialization",
            properties_preserved=["associativity", "identity", "inverses", "commutativity"]
        ))

        # Ring homomorphisms
        self.known_morphisms.append(StructureMorphism(
            name="ring_to_field",
            source_structure=StructureType.RING,
            target_structure=StructureType.FIELD,
            transformation_type="specialization",
            properties_preserved=["addition", "multiplication", "distributivity", "inverses"]
        ))

        # Field to vector space
        self.known_morphisms.append(StructureMorphism(
            name="field_to_vector_space",
            source_structure=StructureType.FIELD,
            target_structure=StructureType.VECTOR_SPACE,
            transformation_type="embedding",
            properties_preserved=["addition", "scalar_multiplication", "field_axioms"]
        ))

        # Polynomial ring morphisms
        self.known_morphisms.append(StructureMorphism(
            name="evaluate_polynomial",
            source_structure=StructureType.RING,
            target_structure=StructureType.FIELD,
            transformation_type="homomorphism",
            properties_preserved=["addition", "multiplication"]
        ))

    def identify_structure_type(self, pattern_data: Dict[str, Any]) -> StructureType:
        """
        Identify the mathematical structure type from pattern data.
        This is where we recognize what kind of math object we're dealing with.
        """
        operations = pattern_data.get('operations', set())
        object_type = pattern_data.get('object_type', '')
        properties = pattern_data.get('properties', set())

        # Vector space indicators
        if 'vector' in object_type or 'span' in operations or 'basis' in operations:
            return StructureType.VECTOR_SPACE

        # Field indicators
        if 'field' in object_type or ('division' in operations and 'multiplication' in operations):
            return StructureType.FIELD

        # Ring indicators
        if 'ring' in object_type or ('multiplication' in operations and 'addition' in operations):
            return StructureType.RING

        # Group indicators
        if 'group' in object_type or 'inverse' in operations:
            if 'commutative' in properties or 'abelian' in properties:
                return StructureType.ABELIAN_GROUP
            return StructureType.GROUP

        # Monoid indicators
        if 'identity' in properties and 'associative' in properties:
            return StructureType.MONOID

        # Semigroup indicators
        if 'associative' in properties:
            return StructureType.SEMIGROUP

        # Default to set
        return StructureType.SET

    def build_abstraction_chain(self, start_structure: StructureType,
                                target_structure: StructureType) -> Optional[List[StructureType]]:
        """
        Build a chain of abstractions from start to target structure.
        Example: GROUP → ABELIAN_GROUP → RING → FIELD
        """
        if start_structure == target_structure:
            return [start_structure]

        # BFS to find path in hierarchy
        queue = [(start_structure, [start_structure])]
        visited = set()

        while queue:
            current, path = queue.pop(0)

            if current.value in visited:
                continue
            visited.add(current.value)

            # Check if we can reach target from current
            children = self.structure_hierarchy.get(current.value, [])
            for child_name in children:
                child = StructureType(child_name)
                new_path = path + [child]

                if child == target_structure:
                    return new_path

                queue.append((child, new_path))

        return None  # No path found

    def find_applicable_morphisms(self, source_structure: StructureType) -> List[StructureMorphism]:
        """Find all morphisms that can be applied to the given structure"""
        return [m for m in self.known_morphisms
                if m.source_structure == source_structure]

    def compose_patterns(self, pattern1_data: Dict, pattern2_data: Dict,
                        problem_context: str) -> Optional[CompositePattern]:
        """
        Attempt to compose two patterns into a more complex pattern.

        This is where creativity happens: combining patterns in novel ways
        to solve problems that neither pattern alone could solve.
        """
        struct1 = self.identify_structure_type(pattern1_data)
        struct2 = self.identify_structure_type(pattern2_data)

        # Check if structures are compatible for composition
        chain = self.build_abstraction_chain(struct1, struct2)
        if chain is None:
            # Try reverse direction
            chain = self.build_abstraction_chain(struct2, struct1)
            if chain is None:
                return None  # Structures incompatible

        # Find morphisms that could connect them
        connecting_morphisms = []
        for morphism in self.known_morphisms:
            if (morphism.source_structure == struct1 and morphism.target_structure == struct2) or \
               (morphism.source_structure == struct2 and morphism.target_structure == struct1):
                connecting_morphisms.append(morphism)

        if not connecting_morphisms:
            return None  # No way to connect

        # Create composite pattern
        pattern_name1 = pattern1_data.get('name', 'pattern1')
        pattern_name2 = pattern2_data.get('name', 'pattern2')

        composite = CompositePattern(
            name=f"{pattern_name1}_composed_with_{pattern_name2}",
            component_patterns=[pattern_name1, pattern_name2],
            composition_order=[0, 1],  # Apply pattern1 first, then pattern2
            structure_type=chain[-1] if chain else struct1
        )

        self.composite_patterns.append(composite)
        return composite

    def decompose_problem(self, problem_description: str,
                         available_patterns: List[Dict]) -> List[Tuple[str, float]]:
        """
        Decompose a complex problem into simpler subproblems that match known patterns.

        Returns list of (pattern_name, confidence) tuples.
        """
        problem_lower = problem_description.lower()
        matching_patterns = []

        # Identify all potentially relevant patterns
        for pattern in available_patterns:
            pattern_name = pattern.get('name', '')
            structure_type = self.identify_structure_type(pattern)

            # Calculate relevance score
            score = 0.0

            # Check for operation overlap
            problem_ops = self._extract_operations(problem_lower)
            pattern_ops = pattern.get('operations', set())
            if pattern_ops:
                overlap = len(problem_ops & pattern_ops)
                score += 0.4 * (overlap / len(pattern_ops))

            # Check for object type match
            if pattern.get('object_type', '') in problem_lower:
                score += 0.3

            # Check for domain match
            if pattern.get('domain', '') in problem_lower:
                score += 0.3

            if score > 0.2:  # Threshold for relevance
                matching_patterns.append((pattern_name, score))

        # Sort by relevance
        matching_patterns.sort(key=lambda x: x[1], reverse=True)
        return matching_patterns

    def _extract_operations(self, text: str) -> Set[str]:
        """Extract mathematical operations mentioned in text"""
        operations = set()
        op_keywords = {
            'add', 'addition', 'sum', 'plus',
            'subtract', 'subtraction', 'minus', 'difference',
            'multiply', 'multiplication', 'product', 'times',
            'divide', 'division', 'quotient',
            'differentiate', 'derivative', 'diff',
            'integrate', 'integration', 'integral',
            'solve', 'equation', 'root', 'factor',
            'expand', 'simplify', 'substitute',
            'inverse', 'transpose', 'determinant', 'eigenvalue'
        }

        for op in op_keywords:
            if op in text:
                operations.add(op)

        return operations

    def create_solution_strategy(self, problem: str,
                                 available_patterns: List[Dict]) -> Dict[str, Any]:
        """
        Create a multi-step solution strategy by composing patterns.

        This is the core of compositional reasoning: given a problem and
        a library of learned patterns, construct a novel solution path.
        """
        # Step 1: Decompose problem
        relevant_patterns = self.decompose_problem(problem, available_patterns)

        if not relevant_patterns:
            return {
                'success': False,
                'reason': 'No relevant patterns found',
                'strategy': []
            }

        # Step 2: Build composition chain
        strategy_steps = []
        used_patterns = set()

        for pattern_name, confidence in relevant_patterns[:5]:  # Top 5 patterns
            if pattern_name in used_patterns:
                continue

            # Find pattern data
            pattern_data = next((p for p in available_patterns
                               if p.get('name') == pattern_name), None)
            if not pattern_data:
                continue

            structure_type = self.identify_structure_type(pattern_data)

            strategy_steps.append({
                'pattern': pattern_name,
                'confidence': confidence,
                'structure': structure_type.value,
                'operations': list(pattern_data.get('operations', [])),
                'estimated_success': pattern_data.get('success_rate', 0.5)
            })

            used_patterns.add(pattern_name)

        # Step 3: Find morphisms to connect patterns
        morphism_chain = []
        for i in range(len(strategy_steps) - 1):
            struct1 = StructureType(strategy_steps[i]['structure'])
            struct2 = StructureType(strategy_steps[i + 1]['structure'])

            # Find connecting morphism
            for morphism in self.known_morphisms:
                if morphism.source_structure == struct1 and \
                   morphism.target_structure == struct2:
                    morphism_chain.append({
                        'from': struct1.value,
                        'to': struct2.value,
                        'via': morphism.name
                    })
                    break

        return {
            'success': True,
            'strategy': strategy_steps,
            'morphisms': morphism_chain,
            'total_confidence': sum(s['confidence'] for s in strategy_steps) / len(strategy_steps),
            'requires_composition': len(strategy_steps) > 1
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about compositional reasoning"""
        return {
            'total_morphisms': len(self.known_morphisms),
            'composite_patterns': len(self.composite_patterns),
            'successful_compositions': sum(1 for cp in self.composite_patterns
                                          if cp.success_rate > 0.5),
            'abstraction_chains': len(self.abstraction_chains),
            'average_composition_success': (
                sum(cp.success_rate for cp in self.composite_patterns) /
                max(1, len(self.composite_patterns))
            )
        }


class AbstractionBuilder:
    """
    Builds abstraction hierarchies by recognizing when one structure
    is a specialization of another.

    Example: Recognize that "solving linear equations" is a special case
    of "solving polynomial equations" which is a special case of "solving
    algebraic equations".
    """

    def __init__(self):
        self.abstractions: Dict[str, List[str]] = {}  # general -> [specific]
        self.composition_engine = CompositionEngine()

    def learn_abstraction(self, specific_pattern: Dict, general_pattern: Dict) -> bool:
        """
        Learn that specific_pattern is a specialization of general_pattern.

        Returns True if the abstraction relationship is valid.
        """
        spec_type = self.composition_engine.identify_structure_type(specific_pattern)
        gen_type = self.composition_engine.identify_structure_type(general_pattern)

        # Check if specific is actually more specialized than general
        if StructureType.is_specialization(gen_type.value, spec_type.value):
            gen_name = general_pattern.get('name', 'general')
            spec_name = specific_pattern.get('name', 'specific')

            if gen_name not in self.abstractions:
                self.abstractions[gen_name] = []

            if spec_name not in self.abstractions[gen_name]:
                self.abstractions[gen_name].append(spec_name)

            return True

        return False

    def generalize_pattern(self, pattern: Dict) -> Optional[StructureType]:
        """
        Find the most general structure that this pattern belongs to.
        This enables transfer learning across problem domains.
        """
        current_type = self.composition_engine.identify_structure_type(pattern)

        # Walk up the hierarchy
        for general, specifics in StructureType.get_hierarchy().items():
            if current_type.value in specifics:
                return StructureType(general)

        return current_type

    def find_similar_patterns(self, pattern: Dict,
                             all_patterns: List[Dict]) -> List[Tuple[str, float]]:
        """
        Find patterns that are at the same abstraction level.
        This enables analogical reasoning.
        """
        pattern_type = self.composition_engine.identify_structure_type(pattern)
        similar = []

        for other in all_patterns:
            if other.get('name') == pattern.get('name'):
                continue  # Skip self

            other_type = self.composition_engine.identify_structure_type(other)

            if other_type == pattern_type:
                # Same abstraction level
                similarity = self._compute_similarity(pattern, other)
                similar.append((other.get('name', 'unknown'), similarity))

        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    def _compute_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Compute similarity between two patterns"""
        score = 0.0

        # Operation overlap
        ops1 = pattern1.get('operations', set())
        ops2 = pattern2.get('operations', set())
        if ops1 and ops2:
            overlap = len(ops1 & ops2)
            union = len(ops1 | ops2)
            score += 0.5 * (overlap / union)

        # Domain similarity
        if pattern1.get('domain') == pattern2.get('domain'):
            score += 0.3

        # Object type similarity
        if pattern1.get('object_type') == pattern2.get('object_type'):
            score += 0.2

        return score
