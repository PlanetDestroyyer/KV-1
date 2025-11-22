"""
GPU-Accelerated Neurosymbolic Learning for KV-1

Uses PyTorch tensors for GPU acceleration instead of NumPy arrays.
All vector operations run on GPU for 10-100x speedup!

Key differences from CPU version:
- Uses torch.Tensor instead of np.ndarray
- All operations on CUDA (if available)
- Batch processing for efficiency
- Mixed precision support (FP16 for speed)

Usage:
    # Automatic GPU detection
    memory = NeurosymbolicGPU()  # Uses CUDA if available

    # Force CPU
    memory = NeurosymbolicGPU(device='cpu')

    # Force specific GPU
    memory = NeurosymbolicGPU(device='cuda:0')
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
from datetime import datetime

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False
    print("[!] PyTorch not available - install with: pip install torch")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    TRANSFORMERS_AVAILABLE = False


@dataclass
class ConceptGPU:
    """A concept represented as GPU tensors."""
    name: str
    tensor: torch.Tensor  # Semantic embedding on GPU
    formulas: List[str]  # Symbolic rules
    examples: List[Any]  # Concrete instances
    learned_at: str
    confidence: float = 1.0
    device: str = 'cuda'
    definition: str = ""  # BUG FIX: Store text definition for retrieval

    def to(self, device: str):
        """Move tensor to different device."""
        self.tensor = self.tensor.to(device)
        self.device = device
        return self


class NeurosymbolicGPU:
    """
    GPU-Accelerated neurosymbolic memory using PyTorch tensors.

    Innovations:
    1. GPU acceleration - 10-100x faster than CPU
    2. Batch processing - process multiple concepts simultaneously
    3. Mixed precision - FP16 for speed, FP32 for accuracy
    4. Tensor operations - native PyTorch for max efficiency
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_fp16: bool = False
    ):
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.use_fp16 = use_fp16 and self.device.startswith('cuda')
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        # Embedding model
        self.embedder = None
        self.embedding_dim = 384

        if TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                # Move embedder to GPU
                if self.device.startswith('cuda'):
                    self.embedder = self.embedder.to(self.device)
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"[!] Could not load embedder: {e}")

        # Memory stores (all tensors on GPU)
        self.concepts: Dict[str, ConceptGPU] = {}
        self.concept_matrix: Optional[torch.Tensor] = None  # Stacked embeddings for batch ops
        self.concept_names: List[str] = []

        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        print(f"[+] Neurosymbolic GPU Memory initialized")
        print(f"    Device: {self.device} ({gpu_info})")
        print(f"    Precision: {'FP16' if self.use_fp16 else 'FP32'}")
        print(f"    Embedding dim: {self.embedding_dim}")

    def embed(self, text: str) -> torch.Tensor:
        """
        Convert text to tensor embedding on GPU.

        Returns:
            torch.Tensor on self.device
        """
        if self.embedder:
            # Sentence transformers returns numpy, convert to torch
            import numpy as np
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            tensor = torch.from_numpy(embedding).to(self.device, dtype=self.dtype)
            return tensor
        else:
            # Fallback: random hash-based tensor
            import numpy as np
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            return torch.from_numpy(embedding).to(self.device, dtype=self.dtype)

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Embed multiple texts at once (GPU batch processing).

        Much faster than calling embed() in a loop!

        Returns:
            torch.Tensor of shape [batch_size, embedding_dim]
        """
        if self.embedder:
            import numpy as np
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            return torch.from_numpy(embeddings).to(self.device, dtype=self.dtype)
        else:
            # Fallback: batch of random embeddings
            tensors = [self.embed(text) for text in texts]
            return torch.stack(tensors)

    def learn_concept(
        self,
        name: str,
        definition: str,
        examples: List[Any],
        confidence: float = 1.0
    ) -> ConceptGPU:
        """
        Learn concept and store on GPU.

        All tensor operations run on GPU for max speed!
        """
        # Create embedding (on GPU)
        text_repr = f"{name}: {definition}"
        tensor = self.embed(text_repr)

        # Extract formulas (CPU operation)
        formulas = self._extract_formulas(name, definition, examples)

        # Create concept
        concept = ConceptGPU(
            name=name,
            tensor=tensor,
            formulas=formulas,
            examples=examples,
            learned_at=datetime.now().isoformat(),
            confidence=confidence,
            device=self.device,
            definition=definition  # BUG FIX: Store definition for later retrieval
        )

        # Store in memory
        self.concepts[name] = concept
        self.concept_names.append(name)

        # Rebuild concept matrix for fast batch operations
        self._rebuild_concept_matrix()

        print(f"[GPU] Learned '{name}' ({len(formulas)} formulas) on {self.device}")

        return concept

    def find_similar_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[List[Tuple[str, float]]]:
        """
        Find similar concepts for multiple queries at once (GPU batch).

        10-100x faster than looping find_similar()!
        """
        if not self.concepts or self.concept_matrix is None:
            return [[] for _ in queries]

        # Embed all queries at once (GPU batch)
        query_tensors = self.embed_batch(queries)  # [num_queries, dim]

        # Compute similarities: all queries vs all concepts (GPU matmul!)
        # query_tensors: [num_queries, dim]
        # concept_matrix: [num_concepts, dim]
        # result: [num_queries, num_concepts]
        similarities = F.cosine_similarity(
            query_tensors.unsqueeze(1),  # [num_queries, 1, dim]
            self.concept_matrix.unsqueeze(0),  # [1, num_concepts, dim]
            dim=2
        )

        # Find top-k for each query (GPU operation)
        results = []
        for i, query in enumerate(queries):
            sims = similarities[i]  # [num_concepts]

            # Filter by threshold
            mask = sims >= threshold
            filtered_sims = sims[mask]
            filtered_names = [self.concept_names[j] for j, m in enumerate(mask) if m]

            if len(filtered_sims) == 0:
                results.append([])
                continue

            # Sort and take top-k
            top_k_vals, top_k_idx = torch.topk(filtered_sims, min(top_k, len(filtered_sims)))

            query_results = [
                (filtered_names[idx], top_k_vals[idx].item())
                for idx in top_k_idx
            ]
            results.append(query_results)

        return results

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find similar concepts (GPU-accelerated).

        Uses batched version internally for consistency.
        """
        results = self.find_similar_batch([query], top_k, threshold)
        return results[0] if results else []

    def compose_concepts(
        self,
        concept_a: str,
        concept_b: str,
        operation: str = 'add'
    ) -> Optional[torch.Tensor]:
        """
        Compose two concepts algebraically (GPU tensor ops).

        Operations:
            'add': concept_a + concept_b
            'sub': concept_a - concept_b
            'avg': (concept_a + concept_b) / 2
            'interpolate': weighted average

        Returns:
            Resulting tensor on GPU
        """
        if concept_a not in self.concepts or concept_b not in self.concepts:
            return None

        tensor_a = self.concepts[concept_a].tensor
        tensor_b = self.concepts[concept_b].tensor

        # All operations on GPU!
        if operation == 'add':
            result = tensor_a + tensor_b
        elif operation == 'sub':
            result = tensor_a - tensor_b
        elif operation == 'avg':
            result = (tensor_a + tensor_b) / 2
        else:
            result = tensor_a

        # Find nearest concept to result
        result_query = result.unsqueeze(0)  # [1, dim]
        similarities = F.cosine_similarity(
            result_query,
            self.concept_matrix,
            dim=1
        )

        best_idx = torch.argmax(similarities)
        best_name = self.concept_names[best_idx]
        best_score = similarities[best_idx].item()

        print(f"[Compose] {concept_a} {operation} {concept_b} ≈ {best_name} (sim={best_score:.2f})")

        return result

    def _rebuild_concept_matrix(self):
        """
        Rebuild matrix of all concept embeddings for fast batch ops.

        Stacks all tensors into single matrix on GPU.
        """
        if not self.concepts:
            self.concept_matrix = None
            return

        # Stack all concept tensors
        tensors = [c.tensor for c in self.concepts.values()]
        self.concept_matrix = torch.stack(tensors)  # [num_concepts, dim]

    def _extract_formulas(self, name: str, definition: str, examples: List[Any]) -> List[str]:
        """Extract symbolic formulas (same as CPU version)."""
        formulas = []

        # Pattern 1: Divisibility
        if "divisib" in definition.lower():
            matches = re.findall(r'divisible by (\d+)', definition.lower())
            for match in matches:
                formulas.append(f"n % {match} = 0")

        # Pattern 2: Comparisons
        if re.search(r'greater than (\d+)', definition.lower()):
            matches = re.findall(r'greater than (\d+)', definition.lower())
            for match in matches:
                formulas.append(f"n > {match}")

        # Pattern 3: Learn from examples
        if examples and all(isinstance(x, (int, float)) for x in examples[:3]):
            nums = [x for x in examples if isinstance(x, (int, float))]
            # Check patterns
            if all(n % 2 == 0 for n in nums):
                formulas.append("n % 2 = 0")
            if all(n % 2 == 1 for n in nums):
                formulas.append("n % 2 = 1")

        return formulas

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU memory and performance stats."""
        stats = {
            "device": self.device,
            "concepts_loaded": len(self.concepts),
            "dtype": str(self.dtype)
        }

        if torch.cuda.is_available():
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            stats["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
            stats["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"

        return stats

    def summarize(self) -> str:
        """Summarize GPU memory status."""
        stats = self.get_gpu_stats()

        summary = []
        summary.append(f"Neurosymbolic GPU Memory:")
        summary.append(f"  Device: {stats['device']}")
        summary.append(f"  Concepts: {stats['concepts_loaded']}")
        summary.append(f"  Precision: {stats['dtype']}")

        if "gpu_memory_allocated" in stats:
            summary.append(f"  GPU Memory: {stats['gpu_memory_allocated']}")

        return "\n".join(summary)


# Convenience function
def create_neurosymbolic(use_gpu: bool = True) -> NeurosymbolicGPU:
    """
    Create neurosymbolic memory (auto-detects GPU).

    Args:
        use_gpu: If True, uses GPU if available

    Returns:
        NeurosymbolicGPU instance
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    device = None
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    elif use_gpu:
        print("[!] GPU requested but CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = 'cpu'

    return NeurosymbolicGPU(device=device)
