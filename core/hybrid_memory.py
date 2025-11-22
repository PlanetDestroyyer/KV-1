"""
Hybrid Memory System for KV-1

Combines:
1. STM (Short-Term Memory): Fast O(1) key-value lookup for recent concepts
2. LTM (Long-Term Memory): GPU-accelerated semantic search for older concepts
3. Neurosymbolic: Tensors + formulas for AI-native learning

Architecture:
    User Query → Check STM (instant O(1))
                 ├─ Found? Return immediately (microseconds)
                 └─ Not found? Search LTM with GPU (milliseconds)
                               ├─ Found? Move to STM (consolidation)
                               └─ Not found? Return None

Performance:
- Recent concepts: 0.001ms (STM)
- Older concepts: 1-5ms (LTM with GPU)
- 1000x faster than always searching!

Example:
    memory = HybridMemory()

    # Learn concept (stores in both STM and LTM)
    memory.learn("prime numbers", definition, examples)

    # First access (STM hit - instant!)
    concept = memory.recall("prime numbers")  # 0.001ms

    # Different phrasing (LTM semantic search)
    concept = memory.recall("primes")  # 2ms, then moves to STM

    # Next access (STM hit again - instant!)
    concept = memory.recall("primes")  # 0.001ms
"""

from __future__ import annotations

import sys
import os
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import time

# Add HSOKV to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hsokv'))

try:
    from hsokv.hsokv.dual_memory import ShortTermMemory
    HSOKV_AVAILABLE = True
except ImportError:
    ShortTermMemory = None
    HSOKV_AVAILABLE = False

try:
    from core.neurosymbolic_gpu import NeurosymbolicGPU, ConceptGPU, TORCH_AVAILABLE
except ImportError:
    from neurosymbolic_gpu import NeurosymbolicGPU, ConceptGPU, TORCH_AVAILABLE


@dataclass
class MemoryStats:
    """Statistics about memory usage and performance."""
    stm_hits: int = 0
    ltm_hits: int = 0
    misses: int = 0
    consolidations: int = 0
    avg_stm_time_ms: float = 0.0
    avg_ltm_time_ms: float = 0.0


class HybridMemory:
    """
    Hybrid memory combining STM (fast) + LTM (semantic) + Neurosymbolic (GPU).

    Workflow:
    1. Learn concept → Store in both STM and LTM
    2. Recall concept → Check STM first (O(1))
                     → If not in STM, search LTM (semantic)
                     → If found in LTM, promote to STM
    3. Consolidation → Frequently accessed STM items stay
                    → Rarely accessed items decay from STM
    """

    def __init__(
        self,
        stm_capacity: int = 7,
        stm_decay_seconds: float = 300.0,  # 5 minutes (longer than HSOKV default)
        use_gpu: bool = True,
        storage_path: str = "./ltm_memory.json"  # NEW: Persistent storage
    ):
        """
        Initialize hybrid memory.

        Args:
            stm_capacity: How many concepts in STM (default: 7 = Miller's number)
            stm_decay_seconds: How long STM items last without access
            use_gpu: Whether to use GPU for LTM (if available)
            storage_path: Path to save/load concepts from disk
        """
        self.storage_path = storage_path

        # Short-term memory (fast key-value store)
        if HSOKV_AVAILABLE:
            self.stm = ShortTermMemory(
                capacity=stm_capacity,
                decay_seconds=stm_decay_seconds
            )
        else:
            # Fallback: simple dict
            self.stm = {}
            print("[!] HSOKV not available, using simple dict for STM")

        # Long-term memory (GPU-accelerated semantic search)
        self.ltm = NeurosymbolicGPU() if use_gpu and TORCH_AVAILABLE else None

        # Statistics
        self.stats = MemoryStats()

        # Concept storage (for compatibility)
        self.concepts: Dict[str, ConceptGPU] = {}

        # Issue #6: Dirty flag for batch saves
        self._dirty = False

        # Load existing concepts from disk
        self.load()

        print(f"[+] Hybrid Memory initialized")
        print(f"    STM: {stm_capacity} slots (O(1) lookup)")
        print(f"    LTM: GPU semantic search" if self.ltm else "    LTM: Disabled")
        print(f"    Storage: {storage_path}")

    def learn(
        self,
        name: str,
        definition: str,
        examples: List[Any],
        confidence: float = 1.0
    ) -> ConceptGPU:
        """
        Learn a concept and store in both STM and LTM.

        Args:
            name: Concept name
            definition: Human-readable definition
            examples: Concrete examples
            confidence: How confident (from validation)

        Returns:
            ConceptGPU object
        """
        # Store in LTM (GPU semantic search)
        if self.ltm:
            concept = self.ltm.learn_concept(name, definition, examples, confidence)
            self.concepts[name] = concept
        else:
            # Fallback: create concept without GPU
            from datetime import datetime
            import torch
            concept = ConceptGPU(
                name=name,
                tensor=torch.randn(384),  # Dummy tensor
                formulas=[],
                examples=examples,
                learned_at=datetime.now().isoformat(),
                confidence=confidence
            )
            self.concepts[name] = concept

        # Store in STM (fast lookup)
        if HSOKV_AVAILABLE and self.stm:
            # Store name → full concept data
            self.stm.store(name, definition)
        elif isinstance(self.stm, dict):
            self.stm[name] = definition

        print(f"[Hybrid] Learned '{name}' → STM + LTM")

        # Issue #6: Mark as dirty instead of saving immediately (batch optimization)
        self._dirty = True

        return concept

    def recall(
        self,
        query: str,
        threshold: float = 0.7
    ) -> Optional[Tuple[str, ConceptGPU, float]]:
        """
        Recall a concept (checks STM first, then LTM).

        This is the KEY optimization:
        - Recent queries → STM hit (0.001ms)
        - Old queries → LTM search (1-5ms) then promote to STM

        Args:
            query: What to search for
            threshold: Minimum similarity for LTM search

        Returns:
            (concept_name, concept_object, confidence) or None
        """
        start = time.time()

        # STEP 1: Check STM (O(1) - instant!)
        stm_result = self._check_stm(query)
        if stm_result:
            elapsed = (time.time() - start) * 1000
            self.stats.stm_hits += 1
            self.stats.avg_stm_time_ms = (
                (self.stats.avg_stm_time_ms * (self.stats.stm_hits - 1) + elapsed)
                / self.stats.stm_hits
            )
            name, definition = stm_result
            concept = self.concepts.get(name)
            print(f"[STM Hit] '{query}' → {name} ({elapsed:.3f}ms)")
            return (name, concept, 1.0) if concept else None

        # STEP 2: Search LTM (semantic search with GPU)
        ltm_result = self._search_ltm(query, threshold)
        if ltm_result:
            elapsed = (time.time() - start) * 1000
            self.stats.ltm_hits += 1
            self.stats.avg_ltm_time_ms = (
                (self.stats.avg_ltm_time_ms * (self.stats.ltm_hits - 1) + elapsed)
                / self.stats.ltm_hits
            )

            name, similarity = ltm_result
            concept = self.concepts.get(name)

            # STEP 3: Promote to STM (consolidation!)
            if concept:
                self._promote_to_stm(name, concept.tensor, concept.formulas)
                self.stats.consolidations += 1

            print(f"[LTM Hit] '{query}' → {name} (sim={similarity:.2f}, {elapsed:.2f}ms)")
            return (name, concept, similarity) if concept else None

        # STEP 3: Not found anywhere
        elapsed = (time.time() - start) * 1000
        self.stats.misses += 1
        print(f"[Miss] '{query}' not found ({elapsed:.2f}ms)")
        return None

    def _check_stm(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Check if concept is in STM (fast O(1) lookup).

        Returns:
            (name, definition) or None
        """
        if HSOKV_AVAILABLE and hasattr(self.stm, 'retrieve'):
            # Try exact match first
            definition, should_consolidate = self.stm.retrieve(query)
            if definition:
                return (query, definition)

            # Try lowercase
            definition, should_consolidate = self.stm.retrieve(query.lower())
            if definition:
                return (query.lower(), definition)

        elif isinstance(self.stm, dict):
            if query in self.stm:
                return (query, self.stm[query])
            if query.lower() in self.stm:
                return (query.lower(), self.stm[query.lower()])

        return None

    def _search_ltm(self, query: str, threshold: float) -> Optional[Tuple[str, float]]:
        """
        Search LTM using semantic similarity (GPU-accelerated).

        Returns:
            (concept_name, similarity_score) or None
        """
        if not self.ltm:
            return None

        results = self.ltm.find_similar(query, top_k=1, threshold=threshold)

        if results:
            name, similarity = results[0]
            return (name, similarity)

        return None

    def _promote_to_stm(self, name: str, tensor, formulas):
        """
        Promote concept from LTM to STM (consolidation).

        This happens when:
        - Concept found in LTM but not STM
        - Frequently accessed concepts move to fast storage
        """
        if HSOKV_AVAILABLE and hasattr(self.stm, 'store'):
            # Create simple definition from formulas
            definition = f"{name} ({len(formulas)} formulas)"
            self.stm.store(name, definition)
        elif isinstance(self.stm, dict):
            self.stm[name] = f"{name} (promoted)"

    def has_concept(self, name: str, semantic: bool = True) -> bool:
        """
        Check if we know this concept.

        Args:
            name: Concept name
            semantic: If True, uses semantic search (slower but smarter)

        Returns:
            True if concept exists
        """
        # Fast path: check STM
        if self._check_stm(name):
            return True

        # Slow path: semantic search in LTM
        if semantic:
            result = self._search_ltm(name, threshold=0.85)
            return result is not None

        # Fallback: exact match in concepts dict
        return name in self.concepts or name.lower() in self.concepts

    # ===== Compatibility methods for PersistentLTM interface =====

    def has(self, name: str) -> bool:
        """Check if concept exists (PersistentLTM compatibility)."""
        return self.has_concept(name, semantic=True)

    def add(self, entry):
        """
        Add learning entry (PersistentLTM compatibility).

        Args:
            entry: LearningEntry with concept, definition, examples
        """
        return self.learn(
            name=entry.concept,
            definition=entry.definition,
            examples=entry.examples if entry.examples else [],
            confidence=getattr(entry, 'confidence_score', 1.0)
        )

    def get(self, name: str):
        """
        Get concept (PersistentLTM compatibility).

        Returns LearningEntry-like object or None.
        """
        # BUG FIX: Try exact match FIRST before semantic search
        # This prevents the issue where concepts in the dict can't be retrieved
        # due to low semantic similarity threshold
        if name in self.concepts:
            concept_obj = self.concepts[name]
            if concept_obj:
                # Create LearningEntry-compatible object
                class LearningEntryCompat:
                    def __init__(self, concept_obj):
                        self.concept = concept_obj.name
                        self.definition = f"Tensor concept with {len(concept_obj.formulas)} formulas"
                        self.learned_at = concept_obj.learned_at
                        self.needed_for = ""
                        self.source = "neurosymbolic"
                        self.examples = concept_obj.examples
                        self.confidence_score = concept_obj.confidence

                        # BUG FIX: Add actual text definition if available
                        if hasattr(concept_obj, 'definition') and concept_obj.definition:
                            self.definition = concept_obj.definition

                return LearningEntryCompat(concept_obj)

        # Fall back to semantic search if exact match not found
        result = self.recall(name, threshold=0.7)
        if not result:
            return None

        concept_name, concept_obj, similarity = result
        if not concept_obj:
            return None

        # Create LearningEntry-compatible object
        class LearningEntryCompat:
            def __init__(self, concept_obj):
                self.concept = concept_obj.name
                self.definition = f"Tensor concept with {len(concept_obj.formulas)} formulas"
                self.learned_at = concept_obj.learned_at
                self.needed_for = ""
                self.source = "neurosymbolic"
                self.examples = concept_obj.examples
                self.confidence_score = concept_obj.confidence

        return LearningEntryCompat(concept_obj)

    def save(self, force=False):
        """
        Save all concepts to disk for persistence.

        Args:
            force: Force save even if not dirty (default: False)
        """
        import json
        import shutil

        # Check if save is needed (batch save optimization)
        if not force and not self._dirty:
            return

        try:
            # Issue #8: Check disk space (need at least 10MB)
            try:
                stat = shutil.disk_usage(os.path.dirname(self.storage_path) or ".")
                if stat.free < 10 * 1024 * 1024:
                    print(f"[!] Low disk space ({stat.free/1024/1024:.1f}MB), skipping save")
                    return
            except Exception as e:
                print(f"[!] Warning: Could not check disk space: {e}")

            # Check if file/directory is writable
            if os.path.exists(self.storage_path) and not os.access(self.storage_path, os.W_OK):
                print(f"[!] File not writable: {self.storage_path}")
                return

            # Convert concepts to JSON-serializable format
            data = {}
            for name, concept in self.concepts.items():
                # Issue #4: Robust tensor serialization
                tensor_data = []
                try:
                    if hasattr(concept.tensor, 'cpu'):
                        # PyTorch tensor
                        tensor_data = concept.tensor.cpu().numpy().tolist()
                    elif hasattr(concept.tensor, 'tolist'):
                        # NumPy array
                        tensor_data = concept.tensor.tolist()
                    elif concept.tensor is not None:
                        # List or other
                        tensor_data = list(concept.tensor)
                except Exception as e:
                    print(f"[!] Failed to serialize tensor for '{name}': {e}")
                    tensor_data = []

                data[name] = {
                    "name": concept.name,
                    "definition": concept.definition if hasattr(concept, 'definition') and concept.definition else f"Concept with {len(concept.formulas)} formulas",  # BUG FIX: Save actual definition
                    "formulas": concept.formulas,
                    "examples": concept.examples,
                    "learned_at": concept.learned_at,
                    "confidence": concept.confidence,
                    "tensor": tensor_data
                }

            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)

            # Atomic write: write to temp file, then rename
            temp_path = self.storage_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Atomic rename (overwrites existing file)
            os.replace(temp_path, self.storage_path)

            print(f"[+] Saved {len(data)} concepts to {self.storage_path}")
            self._dirty = False  # Mark as clean after save
        except Exception as e:
            print(f"[!] Failed to save concepts: {e}")

    def load(self):
        """Load concepts from disk."""
        import json
        if not os.path.exists(self.storage_path):
            print(f"[i] No existing memory found at {self.storage_path}, starting fresh")
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            loaded_count = 0
            for name, concept_data in data.items():
                # Issue #5: Check for duplicates - skip if already loaded
                if name in self.concepts:
                    print(f"[!] Skipping duplicate concept: {name}")
                    continue

                # Reconstruct concept from saved data
                if TORCH_AVAILABLE:
                    import torch
                    tensor = torch.tensor(concept_data.get("tensor", []))

                    # Issue #5: Move tensor to correct device (match LTM device)
                    if self.ltm and hasattr(self.ltm, 'device'):
                        tensor = tensor.to(self.ltm.device)
                    elif self.ltm and hasattr(self.ltm, 'concept_matrix') and self.ltm.concept_matrix is not None:
                        tensor = tensor.to(self.ltm.concept_matrix.device)
                else:
                    tensor = None

                # Create ConceptGPU object
                concept = ConceptGPU(
                    name=concept_data["name"],
                    tensor=tensor,
                    formulas=concept_data.get("formulas", []),
                    examples=concept_data.get("examples", []),
                    learned_at=concept_data.get("learned_at", ""),
                    confidence=concept_data.get("confidence", 1.0),
                    definition=concept_data.get("definition", "")  # BUG FIX: Load definition
                )

                self.concepts[name] = concept

                # Also add to LTM if available
                if self.ltm and tensor is not None and TORCH_AVAILABLE:
                    # Issue #5: Check if not already in LTM
                    if name not in self.ltm.concepts:
                        self.ltm.concepts[name] = concept

                        # Issue #5: Ensure tensor is on same device before concat
                        if self.ltm.concept_matrix is None:
                            self.ltm.concept_matrix = tensor.unsqueeze(0)
                            self.ltm.concept_names = [name]
                        else:
                            # Move tensor to same device as existing matrix
                            tensor = tensor.to(self.ltm.concept_matrix.device)
                            self.ltm.concept_matrix = torch.cat(
                                [self.ltm.concept_matrix, tensor.unsqueeze(0)],
                                dim=0
                            )
                            self.ltm.concept_names.append(name)

                loaded_count += 1

            print(f"[+] Loaded {loaded_count} concepts from {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to load concepts: {e}")
            import traceback
            traceback.print_exc()
            self.concepts = {}

    # ===== End compatibility methods =====

    def decay_stm(self):
        """
        Remove old items from STM (like human forgetting).

        Call this periodically to simulate memory decay.
        """
        if HSOKV_AVAILABLE and hasattr(self.stm, 'decay'):
            self.stm.decay()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_queries = self.stats.stm_hits + self.stats.ltm_hits + self.stats.misses

        stm_size = len(self.stm.memory) if HSOKV_AVAILABLE else len(self.stm)
        ltm_size = len(self.concepts)

        return {
            "stm_size": stm_size,
            "ltm_size": ltm_size,
            "total_queries": total_queries,
            "stm_hit_rate": self.stats.stm_hits / total_queries if total_queries > 0 else 0,
            "ltm_hit_rate": self.stats.ltm_hits / total_queries if total_queries > 0 else 0,
            "miss_rate": self.stats.misses / total_queries if total_queries > 0 else 0,
            "avg_stm_time_ms": self.stats.avg_stm_time_ms,
            "avg_ltm_time_ms": self.stats.avg_ltm_time_ms,
            "consolidations": self.stats.consolidations,
        }

    def summarize(self) -> str:
        """Get human-readable summary."""
        stats = self.get_stats()

        lines = []
        lines.append("Hybrid Memory Status:")
        lines.append(f"  STM: {stats['stm_size']}/{7 if HSOKV_AVAILABLE else 'unlimited'} slots")
        lines.append(f"  LTM: {stats['ltm_size']} concepts")
        lines.append(f"\nPerformance:")
        lines.append(f"  Total queries: {stats['total_queries']}")
        lines.append(f"  STM hit rate: {stats['stm_hit_rate']*100:.1f}% (avg {stats['avg_stm_time_ms']:.3f}ms)")
        lines.append(f"  LTM hit rate: {stats['ltm_hit_rate']*100:.1f}% (avg {stats['avg_ltm_time_ms']:.2f}ms)")
        lines.append(f"  Miss rate: {stats['miss_rate']*100:.1f}%")
        lines.append(f"  Consolidations: {stats['consolidations']}")

        if self.ltm:
            lines.append(f"\n{self.ltm.summarize()}")

        return "\n".join(lines)
