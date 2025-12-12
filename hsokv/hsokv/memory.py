"""
Key-Value memory with 3-stage lifecycle.

This is the core data structure that stores and retrieves memories
using the revolutionary 3-stage lifecycle.

Now with Faiss GPU-accelerated similarity search!
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    LOGGER = logging.getLogger(__name__)
    LOGGER.warning("Faiss not available, falling back to PyTorch similarity search")

from .config import MemoryConfig
from .lifecycle import MemoryLifecycle, MemoryStage

LOGGER = logging.getLogger(__name__)


class KeyValueMemory:
    """
    Stores and retrieves memories with human-like 3-stage lifecycle.

    Key features:
    - L2-normalized embeddings (cosine similarity)
    - 3-stage lifecycle (LEARNING → REINFORCEMENT → MATURE)
    - Pure recall for new memories
    - Confidence-based weighted retrieval
    - Stage-aware pruning protection
    """

    def __init__(self, embedding_dim: int, config: MemoryConfig):
        """
        Initialize memory system with Faiss GPU index.

        Args:
            embedding_dim: Dimension of embedding vectors
            config: Memory configuration
        """
        self.embedding_dim = embedding_dim
        self.config = config
        self.lifecycle = MemoryLifecycle(config)

        # Storage
        device = torch.device(config.device)
        self.keys = torch.empty(0, embedding_dim, device=device)
        self.values: List[torch.Tensor] = []
        self.metadata: List[Dict] = []
        self.labels: List[str] = []  # Human-readable labels

        # Faiss index for GPU-accelerated similarity search
        self.use_faiss = FAISS_AVAILABLE
        self.faiss_index = None
        if self.use_faiss:
            self._init_faiss_index()

    def _init_faiss_index(self):
        """Initialize Faiss GPU index for fast similarity search."""
        # IndexFlatIP = Inner Product (for normalized vectors, same as cosine similarity)
        cpu_index = faiss.IndexFlatIP(self.embedding_dim)

        # Move to GPU if CUDA available
        if self.config.device == "cuda" and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                LOGGER.info(f"Faiss GPU index initialized (dim={self.embedding_dim})")
            except Exception as e:
                LOGGER.warning(f"Failed to initialize Faiss GPU index: {e}, using CPU")
                self.faiss_index = cpu_index
        else:
            self.faiss_index = cpu_index
            LOGGER.info(f"Faiss CPU index initialized (dim={self.embedding_dim})")

    def __len__(self) -> int:
        return len(self.metadata)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """L2 normalize for cosine similarity."""
        norm = torch.sqrt(torch.sum(tensor ** 2, dim=-1, keepdim=True) + 1e-12)
        return tensor / norm

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        label: str,
        confidence: Optional[float] = None,
        is_first_exposure: bool = True,
    ) -> int:
        """
        Store a new memory.

        Args:
            key: Query embedding (what we search with)
            value: Value embedding (what we retrieve)
            label: Human-readable label
            confidence: Initial confidence (uses config default if None)
            is_first_exposure: True for new concepts (enables lifecycle)

        Returns:
            Index of stored memory
        """
        # Normalize and move to device
        device = self.keys.device if len(self.keys) > 0 else torch.device(self.config.device)
        key = self._normalize(key.detach().to(device))
        value = self._normalize(value.detach().to(device))

        # Store key
        if len(self.keys) == 0:
            self.keys = key.unsqueeze(0)
        else:
            self.keys = torch.cat([self.keys, key.unsqueeze(0)], dim=0)

        # Add to Faiss index
        if self.use_faiss and self.faiss_index is not None:
            key_numpy = key.cpu().numpy().reshape(1, -1).astype('float32')
            self.faiss_index.add(key_numpy)

        # Store value
        self.values.append(value)

        # Store label
        self.labels.append(label)

        # Store metadata
        meta = {
            "confidence": confidence if confidence is not None else self.config.initial_confidence,
            "retrieval_count": 0,
            "success_rate": 0.0,
            "is_first_exposure": is_first_exposure,
            "created_at": len(self.metadata),
        }
        self.metadata.append(meta)

        # Enforce capacity
        if len(self.metadata) > self.config.max_entries:
            self._enforce_capacity()

        return len(self.metadata) - 1

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Retrieve memories using 3-stage lifecycle.

        Args:
            query: Query embedding (single vector or batch)
            top_k: Number of memories to retrieve (uses config default if None)

        Returns:
            (retrieved_value, details) where:
            - retrieved_value: Weighted combination of memory values
            - details: Dict with retrieval_indices, similarities, stages
        """
        if len(self.metadata) == 0:
            # No memories - return zero vector
            device = query.device
            if query.dim() == 1:
                return torch.zeros_like(query), {"retrieval_indices": [], "avg_similarity": 0.0}
            else:
                return torch.zeros_like(query), {"retrieval_indices": [[] for _ in range(len(query))]}

        k = top_k if top_k is not None else self.config.top_k
        single = query.dim() == 1

        if single:
            query = query.unsqueeze(0)

        # Normalize query
        query = self._normalize(query)

        # Compute similarities using Faiss or PyTorch
        if self.use_faiss and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            # Faiss GPU-accelerated search
            k = min(k, self.faiss_index.ntotal)
            query_numpy = query.cpu().numpy().astype('float32')

            # Search returns (distances, indices)
            similarities_np, topk_idx_np = self.faiss_index.search(query_numpy, k)

            # Convert back to torch tensors
            topk_sim = torch.from_numpy(similarities_np).to(query.device)
            topk_idx = torch.from_numpy(topk_idx_np).to(query.device).long()
        else:
            # Fallback to PyTorch similarity search
            keys = self.keys.to(query.device)
            similarities = torch.clamp(F.linear(query, keys), min=0.0)
            k = min(k, keys.size(0))
            topk_sim, topk_idx = similarities.topk(k, dim=-1)

        # Process each query
        outputs = []
        all_details = []

        for i in range(query.size(0)):
            indices = topk_idx[i]
            sims = topk_sim[i]

            # Get best match stage
            best_idx = indices[0].item()
            best_stage = self.lifecycle.get_stage(self.metadata[best_idx])

            # LEARNING STAGE: Pure recall (return only best match)
            if best_stage == MemoryStage.LEARNING and self.config.use_pure_recall:
                best_value = self.values[best_idx].to(query.device)

                # Update retrieval count
                self.metadata[best_idx]["retrieval_count"] += 1

                outputs.append(best_value)
                all_details.append({
                    "retrieval_indices": [best_idx],
                    "stages": [best_stage.value],
                    "avg_similarity": float(sims[0].item()),
                })
                continue

            # REINFORCEMENT/MATURE: Weighted aggregation
            weights = []
            values = []
            stages_used = []

            for j, idx in enumerate(indices):
                entry_id = idx.item()
                meta = self.metadata[entry_id]
                similarity = float(sims[j].item())

                # Skip low similarity
                if similarity < self.config.similarity_threshold:
                    continue

                # Get stage and confidence boost
                stage = self.lifecycle.get_stage(meta)
                confidence = meta["confidence"]
                boost = self.lifecycle.get_confidence_boost(stage, meta["retrieval_count"])

                effective_confidence = min(confidence * boost, 1.0)
                weight = effective_confidence * similarity

                weights.append(weight)
                values.append(self.values[entry_id].to(query.device))
                stages_used.append(stage.value)

                # Update retrieval count
                self.metadata[entry_id]["retrieval_count"] += 1

                # Graduate if needed
                if self.lifecycle.should_graduate(meta):
                    self.metadata[entry_id]["is_first_exposure"] = False

            if not weights:
                # No matches - return query itself
                outputs.append(query[i])
                all_details.append({
                    "retrieval_indices": [],
                    "stages": [],
                    "avg_similarity": 0.0,
                })
                continue

            # Weighted average
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=query.device)
            stacked_values = torch.stack(values)
            aggregated = (weight_tensor.unsqueeze(-1) * stacked_values).sum(dim=0) / weight_tensor.sum()

            outputs.append(aggregated)
            all_details.append({
                "retrieval_indices": indices[:len(weights)].tolist(),
                "stages": stages_used,
                "avg_similarity": float(torch.tensor(weights).mean().item()),
            })

        result = torch.stack(outputs)
        if single:
            result = result.squeeze(0)
            all_details = all_details[0]

        return result, all_details

    def update_confidence(self, entry_id: int, success: bool):
        """
        Update memory confidence based on retrieval success.

        Args:
            entry_id: Memory index
            success: Whether the retrieval was successful
        """
        if entry_id < 0 or entry_id >= len(self.metadata):
            return

        meta = self.metadata[entry_id]
        count = meta["retrieval_count"]

        # Update success rate
        old_count = max(1, count - 1)
        signal = 1.0 if success else 0.0
        meta["success_rate"] = (meta["success_rate"] * old_count + signal) / count

        # Update confidence
        adjustment = 0.1 * (signal - 0.5)
        meta["confidence"] = float(np.clip(meta["confidence"] + adjustment, 0.05, 1.0))

    def prune(self):
        """
        Remove low-confidence MATURE memories.

        Protected memories (LEARNING and REINFORCEMENT stages) are kept.
        """
        if not self.metadata:
            return

        keep_mask = []
        for idx, meta in enumerate(self.metadata):
            stage = self.lifecycle.get_stage(meta)

            # Check protection
            if self.lifecycle.should_protect_from_pruning(stage):
                keep_mask.append(True)
            else:
                # MATURE: prune based on confidence
                keep_mask.append(meta["confidence"] >= self.config.confidence_threshold)

        # Apply mask
        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]

        if len(keep_indices) == 0:
            # Clear all
            device = self.keys.device
            self.keys = torch.empty(0, self.embedding_dim, device=device)
            self.values = []
            self.metadata = []
            self.labels = []
        else:
            # Keep selected
            keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=self.keys.device)
            self.keys = self.keys[keep_tensor]
            self.values = [self.values[i] for i in keep_indices]
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.labels = [self.labels[i] for i in keep_indices]

        # Rebuild Faiss index after pruning
        if self.use_faiss and self.faiss_index is not None:
            self._rebuild_faiss_index()

    def _enforce_capacity(self):
        """Enforce maximum capacity by pruning."""
        self.prune()

        # If still over capacity, remove lowest confidence MATURE memories
        if len(self.metadata) > self.config.max_entries:
            confidences = [m["confidence"] for m in self.metadata]
            keep_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
            keep_indices = keep_indices[:self.config.max_entries]

            keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=self.keys.device)
            self.keys = self.keys[keep_tensor]
            self.values = [self.values[i] for i in keep_indices]
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.labels = [self.labels[i] for i in keep_indices]

            # Rebuild Faiss index
            if self.use_faiss and self.faiss_index is not None:
                self._rebuild_faiss_index()

    def _rebuild_faiss_index(self):
        """Rebuild Faiss index from scratch (after pruning)."""
        if not self.use_faiss or self.faiss_index is None:
            return

        # Reset index
        self.faiss_index.reset()

        # Re-add all keys
        if len(self.keys) > 0:
            keys_numpy = self.keys.cpu().numpy().astype('float32')
            self.faiss_index.add(keys_numpy)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if not self.metadata:
            return {
                "total": 0,
                "learning": 0,
                "reinforcement": 0,
                "mature": 0,
            }

        stages = {stage: 0 for stage in MemoryStage}
        for meta in self.metadata:
            stage = self.lifecycle.get_stage(meta)
            stages[stage] += 1

        return {
            "total": len(self.metadata),
            "learning": stages[MemoryStage.LEARNING],
            "reinforcement": stages[MemoryStage.REINFORCEMENT],
            "mature": stages[MemoryStage.MATURE],
        }
