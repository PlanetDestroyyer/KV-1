"""
Neurosymbolic Learning System for KV-1

Implements AI-native learning that operates on:
1. Vectors (semantic embeddings) - neural component
2. Formulas (symbolic rules) - symbolic component
3. Compositional reasoning (combine formulas algebraically)

This allows the AI to:
- Learn concepts as mathematical objects (vectors + rules)
- Discover new formulas by composing existing ones
- Understand relationships without language constraints
- Transfer knowledge algebraically

Example:
    Learn: is_prime(n) = ∀d∈(2,√n): n%d≠0
    Learn: is_even(n) = n%2=0
    Discover: is_prime(n) ∧ ¬is_even(n) → n>2
    Apply: Generate new understanding of odd primes
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class Concept:
    """A concept represented in AI-native format."""
    name: str
    vector: np.ndarray  # Semantic embedding
    formulas: List[str]  # Symbolic rules
    examples: List[Any]  # Concrete instances
    learned_at: str
    confidence: float = 1.0

    def __hash__(self):
        return hash(self.name)


@dataclass
class Formula:
    """A symbolic rule that can be composed with others."""
    rule: str  # Symbolic representation
    vector: np.ndarray  # Embedding of the rule
    inputs: List[str]  # Input variable names
    output: str  # Output variable name
    learned_from: List[str]  # Which concepts taught this

    def apply(self, concept_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply formula as vector transformation."""
        # Simple composition: average of inputs
        if not self.inputs:
            return self.vector

        input_vecs = [concept_vectors.get(inp, np.zeros_like(self.vector))
                      for inp in self.inputs]
        return np.mean(input_vecs, axis=0) if input_vecs else self.vector


class NeurosymbolicMemory:
    """
    AI-native memory that stores concepts as vectors + formulas.

    Key innovations:
    1. Semantic search - find related concepts by vector similarity
    2. Formula learning - extract patterns from examples
    3. Compositional reasoning - combine formulas algebraically
    4. Transfer learning - apply learned patterns to new domains
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        # Vector component (neural)
        self.embedder = None
        if SentenceTransformer:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"[!] Could not load embedder: {e}")
                self.embedding_dim = 384
        else:
            self.embedding_dim = 384

        # Memory stores
        self.concepts: Dict[str, Concept] = {}
        self.formulas: Dict[str, Formula] = {}
        self.relationships: Dict[Tuple[str, str], np.ndarray] = {}

        print(f"[+] Neurosymbolic Memory initialized (dim={self.embedding_dim})")

    def embed(self, text: str) -> np.ndarray:
        """Convert text to vector embedding."""
        if self.embedder:
            return self.embedder.encode(text, convert_to_numpy=True)
        else:
            # Fallback: random hash-based embedding
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def learn_concept(
        self,
        name: str,
        definition: str,
        examples: List[Any],
        confidence: float = 1.0
    ) -> Concept:
        """
        Learn a concept in AI-native format.

        Args:
            name: Concept name
            definition: Human-readable definition (for embedding)
            examples: Concrete examples to extract patterns from
            confidence: How confident we are (from validation)

        Returns:
            Concept object with vector + extracted formulas
        """
        # 1. Create semantic embedding
        text_repr = f"{name}: {definition}"
        vector = self.embed(text_repr)

        # 2. Extract formulas from definition and examples
        formulas = self._extract_formulas(name, definition, examples)

        # 3. Create concept
        concept = Concept(
            name=name,
            vector=vector,
            formulas=formulas,
            examples=examples,
            learned_at=datetime.now().isoformat(),
            confidence=confidence
        )

        # 4. Store in memory
        self.concepts[name] = concept

        # 5. Discover relationships with existing concepts
        self._discover_relationships(concept)

        # 6. Try to compose new formulas
        self._compose_formulas(concept)

        print(f"[Neurosymbolic] Learned '{name}' (vector + {len(formulas)} formulas)")

        return concept

    def find_similar(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find semantically similar concepts (vector search).

        Returns:
            List of (concept_name, similarity_score) tuples
        """
        if not self.concepts:
            return []

        query_vec = self.embed(query)

        similarities = []
        for name, concept in self.concepts.items():
            sim = self._cosine_similarity(query_vec, concept.vector)
            if sim >= threshold:
                similarities.append((name, float(sim)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def has_concept(self, name: str, similarity_threshold: float = 0.85) -> bool:
        """
        Check if we already know this concept (semantically).

        Returns True if we have a very similar concept already.
        """
        similar = self.find_similar(name, top_k=1, threshold=similarity_threshold)
        return len(similar) > 0

    def _extract_formulas(self, name: str, definition: str, examples: List[Any]) -> List[str]:
        """
        Extract symbolic formulas from definition and examples.

        Uses pattern matching to find mathematical rules.
        """
        formulas = []

        # Pattern 1: Divisibility rules
        if "divisib" in definition.lower():
            # Extract: "divisible by X" → formula: n % X = 0
            matches = re.findall(r'divisible by (\d+)', definition.lower())
            for match in matches:
                formulas.append(f"n % {match} = 0")

        # Pattern 2: Comparison rules
        if re.search(r'greater than (\d+)', definition.lower()):
            matches = re.findall(r'greater than (\d+)', definition.lower())
            for match in matches:
                formulas.append(f"n > {match}")

        if re.search(r'less than (\d+)', definition.lower()):
            matches = re.findall(r'less than (\d+)', definition.lower())
            for match in matches:
                formulas.append(f"n < {match}")

        # Pattern 3: Range rules
        if "between" in definition.lower():
            # Extract: "between X and Y"
            matches = re.findall(r'between (\d+) and (\d+)', definition.lower())
            for low, high in matches:
                formulas.append(f"{low} <= n <= {high}")

        # Pattern 4: Learn from examples (inductive)
        if examples and all(isinstance(x, (int, float)) for x in examples[:3]):
            # Try to find patterns in numeric examples
            example_formulas = self._induce_formula_from_examples(examples)
            formulas.extend(example_formulas)

        return formulas

    def _induce_formula_from_examples(self, examples: List[Any]) -> List[str]:
        """
        Induce formulas from examples (inductive learning).

        Example: [2, 4, 6, 8] → "n % 2 = 0"
        """
        formulas = []

        if not examples or len(examples) < 2:
            return formulas

        nums = [x for x in examples if isinstance(x, (int, float))]
        if len(nums) < 2:
            return formulas

        # Check if all even
        if all(n % 2 == 0 for n in nums):
            formulas.append("n % 2 = 0")

        # Check if all odd
        if all(n % 2 == 1 for n in nums):
            formulas.append("n % 2 = 1")

        # Check arithmetic sequence
        if len(nums) >= 3:
            diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            if len(set(diffs)) == 1:
                # Constant difference - arithmetic sequence
                d = diffs[0]
                formulas.append(f"n[i+1] = n[i] + {d}")

        return formulas

    def _discover_relationships(self, new_concept: Concept):
        """
        Discover relationships between new concept and existing ones.

        Uses vector arithmetic: king - man + woman ≈ queen
        Applied to concepts: prime - odd + even ≈ composite
        """
        for name, existing in self.concepts.items():
            if name == new_concept.name:
                continue

            # Compute relationship vector
            relation = new_concept.vector - existing.vector

            # Store bidirectional relationships
            self.relationships[(new_concept.name, name)] = relation
            self.relationships[(name, new_concept.name)] = -relation

    def _compose_formulas(self, concept: Concept):
        """
        Try to compose new formulas from existing ones.

        Example:
            Formula A: is_prime(n) = n > 1 ∧ ∀d: n%d≠0
            Formula B: is_even(n) = n%2=0
            Compose: is_prime(n) ∧ is_even(n) → n=2 (only even prime)
        """
        # For now, simple composition: combine formulas from same concept
        if len(concept.formulas) >= 2:
            # AND composition
            combined = " ∧ ".join(concept.formulas)

            # Store as a new derived formula
            if combined not in self.formulas:
                formula_vec = self.embed(combined)
                self.formulas[combined] = Formula(
                    rule=combined,
                    vector=formula_vec,
                    inputs=["n"],
                    output="bool",
                    learned_from=[concept.name]
                )
                print(f"[Composed] New formula: {combined[:80]}...")

    def apply_formula_to_concept(self, formula_name: str, concept_name: str) -> Optional[np.ndarray]:
        """
        Apply a learned formula to a concept (vector transformation).

        This allows the AI to "think" mathematically in vector space!
        """
        if formula_name not in self.formulas or concept_name not in self.concepts:
            return None

        formula = self.formulas[formula_name]
        concept = self.concepts[concept_name]

        # Apply transformation
        result = formula.vector + concept.vector  # Simple addition

        # Find what concept this result is closest to
        similarities = []
        for name, other_concept in self.concepts.items():
            sim = self._cosine_similarity(result, other_concept.vector)
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        if similarities:
            best_match, score = similarities[0]
            print(f"[Formula Application] {formula_name} + {concept_name} ≈ {best_match} (sim={score:.2f})")

        return result

    def transfer_knowledge(self, source_concept: str, target_domain: str) -> List[str]:
        """
        Transfer formulas from source concept to target domain.

        Example: Transfer "prime number" formulas to "prime polynomial"
        """
        if source_concept not in self.concepts:
            return []

        source = self.concepts[source_concept]

        # Find concepts in target domain (simple keyword match for now)
        target_concepts = [
            name for name in self.concepts.keys()
            if target_domain.lower() in name.lower()
        ]

        transferred = []
        for formula in source.formulas:
            # Adapt formula to target domain (simple substitution)
            adapted = formula.replace("n", "p")  # n → p for polynomials
            transferred.append(adapted)

        if transferred:
            print(f"[Transfer] {len(transferred)} formulas: {source_concept} → {target_domain}")

        return transferred

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_concept_graph(self) -> Dict[str, List[str]]:
        """
        Build a graph of concept relationships.

        Returns dict: concept_name → [related_concepts]
        """
        graph = {name: [] for name in self.concepts.keys()}

        for (concept_a, concept_b), relation in self.relationships.items():
            # Strong relationship if vector magnitude is small
            strength = np.linalg.norm(relation)
            if strength < 1.0:  # Arbitrary threshold
                graph[concept_a].append(concept_b)

        return graph

    def summarize(self) -> str:
        """Summarize what the AI has learned."""
        summary = []
        summary.append(f"Neurosymbolic Memory Status:")
        summary.append(f"  Concepts learned: {len(self.concepts)}")
        summary.append(f"  Formulas discovered: {len(self.formulas)}")
        summary.append(f"  Relationships mapped: {len(self.relationships)}")

        if self.concepts:
            summary.append(f"\nTop concepts:")
            for i, (name, concept) in enumerate(list(self.concepts.items())[:5], 1):
                summary.append(f"  {i}. {name} ({len(concept.formulas)} formulas, conf={concept.confidence:.2f})")

        return "\n".join(summary)
