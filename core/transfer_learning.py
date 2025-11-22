"""
Transfer Learning Module

Enables the AGI to transfer knowledge across domains:
- Recognize similar patterns in different contexts
- Apply solutions from one domain to another
- Build cross-domain conceptual bridges

Example: Use "exponential growth" from biology in finance (compound interest)
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TransferMapping:
    """Maps a concept from one domain to another"""
    source_concept: str
    source_domain: str
    target_concept: str
    target_domain: str
    similarity: float
    mapping_type: str  # "structural", "functional", "analogical"


class TransferLearner:
    """
    Enables cross-domain knowledge transfer.

    Core insight: Intelligence is about recognizing patterns,
    not memorizing domain-specific facts!
    """

    def __init__(self, memory, embedder=None):
        """
        Args:
            memory: HybridMemory instance with learned concepts
            embedder: SentenceTransformer for computing embeddings
        """
        self.memory = memory
        self.embedder = embedder
        self.transfer_history: List[TransferMapping] = []

    def find_transferable_knowledge(
        self,
        target_concept: str,
        target_domain: str,
        similarity_threshold: float = 0.75,
        max_results: int = 5
    ) -> List[TransferMapping]:
        """
        Find concepts from OTHER domains that could help with target concept.

        Example:
            target: "compound interest" (finance)
            finds: "exponential growth" (biology) - same pattern!
        """
        if not self.embedder:
            print("[Transfer] No embedder available, skipping")
            return []

        # Encode target concept
        target_embedding = self.embedder.encode(target_concept)

        transfers = []

        # Search ALL concepts in memory
        for concept_name, concept_data in self.memory.concepts.items():
            # Skip if same domain (not transfer learning)
            if hasattr(concept_data, 'domain'):
                if concept_data.domain == target_domain:
                    continue

            # Compute structural similarity
            if hasattr(concept_data, 'tensor') and concept_data.tensor is not None:
                # Use vector similarity
                try:
                    similarity = self._cosine_similarity(
                        target_embedding,
                        concept_data.tensor
                    )

                    if similarity >= similarity_threshold:
                        # Found a transferable concept!
                        mapping = TransferMapping(
                            source_concept=concept_name,
                            source_domain=getattr(concept_data, 'domain', 'unknown'),
                            target_concept=target_concept,
                            target_domain=target_domain,
                            similarity=similarity,
                            mapping_type="structural"
                        )
                        transfers.append(mapping)

                except Exception as e:
                    # Skip if tensor dimensions don't match
                    continue

        # Sort by similarity
        transfers.sort(key=lambda x: x.similarity, reverse=True)

        return transfers[:max_results]

    def apply_transfer(
        self,
        source_concept: str,
        target_problem: str,
        llm
    ) -> Optional[str]:
        """
        Apply knowledge from source concept to solve target problem.

        Args:
            source_concept: Concept to transfer FROM
            target_problem: Problem to solve
            llm: LLM bridge for reasoning

        Returns:
            Solution using transferred knowledge
        """
        # Get source concept details
        source_data = self.memory.concepts.get(source_concept)
        if not source_data:
            return None

        # Build transfer prompt
        prompt = f"""You are applying knowledge from one domain to another (TRANSFER LEARNING).

SOURCE CONCEPT: {source_concept}
Definition: {getattr(source_data, 'definition', 'No definition')}
Examples: {getattr(source_data, 'examples', [])}

TARGET PROBLEM: {target_problem}

TASK: Apply the PATTERN/STRUCTURE from the source concept to solve the target problem.

Think step-by-step:
1. What is the core PATTERN in the source concept?
2. How does this pattern map to the target problem?
3. What's the solution using this transferred knowledge?

Respond with:
PATTERN: [core pattern identified]
MAPPING: [how source maps to target]
SOLUTION: [answer to target problem]
"""

        response = llm.generate(
            system_prompt="You are an AGI performing transfer learning across domains. Find the common pattern and apply it.",
            user_input=prompt
        )

        text = response.get("text", "")

        # Record successful transfer
        print(f"[Transfer] Applied '{source_concept}' pattern to new problem")
        print(f"[Transfer] Cross-domain learning activated! 🧠")

        return text

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors"""
        # Handle different tensor types
        if hasattr(vec1, 'cpu'):  # PyTorch tensor
            vec1 = vec1.cpu().numpy()
        if hasattr(vec2, 'cpu'):
            vec2 = vec2.cpu().numpy()

        # Flatten if needed
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()

        # Ensure same dimensions
        if len(vec1) != len(vec2):
            # Pad shorter vector
            max_len = max(len(vec1), len(vec2))
            if len(vec1) < max_len:
                vec1 = np.pad(vec1, (0, max_len - len(vec1)))
            if len(vec2) < max_len:
                vec2 = np.pad(vec2, (0, max_len - len(vec2)))

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def suggest_transfer_opportunities(self, current_problem: str) -> List[str]:
        """
        Suggest concepts from other domains that might help.

        Returns list of suggestions like:
        "Try applying 'exponential growth' (biology) to this problem"
        """
        # Extract keywords from problem
        keywords = current_problem.lower().split()

        suggestions = []
        for concept_name in self.memory.concepts.keys():
            # Simple keyword matching for now
            if any(keyword in concept_name.lower() for keyword in keywords):
                suggestions.append(
                    f"Consider '{concept_name}' (from {getattr(self.memory.concepts[concept_name], 'domain', 'unknown')})"
                )

        return suggestions[:5]
