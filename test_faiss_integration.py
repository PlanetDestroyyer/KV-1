#!/usr/bin/env python3
"""
Quick test to verify Faiss integration works correctly.
"""

import sys
import torch

# Add hsokv to path
sys.path.insert(0, './hsokv')

from hsokv.hsokv.memory import KeyValueMemory
from hsokv.hsokv.config import MemoryConfig

def test_faiss_integration():
    """Test basic Faiss integration."""
    print("="*60)
    print("Testing Faiss GPU Integration")
    print("="*60)

    # Create config
    config = MemoryConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_entries=1000,
        top_k=5
    )

    print(f"\nDevice: {config.device}")

    # Create memory system
    memory = KeyValueMemory(embedding_dim=768, config=config)

    print(f"Faiss available: {memory.use_faiss}")
    if memory.use_faiss:
        print(f"Faiss index type: {type(memory.faiss_index).__name__}")
        print(f"Faiss on GPU: {'Gpu' in type(memory.faiss_index).__name__}")

    # Store some test memories
    print("\n" + "-"*60)
    print("Storing test memories...")
    print("-"*60)

    test_concepts = [
        ("prime numbers", "Numbers divisible only by 1 and themselves"),
        ("quadratic formula", "Formula to solve ax² + bx + c = 0"),
        ("derivative", "Rate of change of a function"),
        ("integral", "Area under a curve"),
        ("factorial", "Product of all positive integers up to n"),
    ]

    for concept, definition in test_concepts:
        # Create random embeddings (in real use, these would come from sentence-transformers)
        key_embedding = torch.randn(768)
        value_embedding = torch.randn(768)

        memory.store(
            key=key_embedding,
            value=value_embedding,
            label=concept,
            confidence=0.8
        )
        print(f"  ✓ Stored: {concept}")

    print(f"\nTotal memories: {len(memory)}")
    if memory.use_faiss:
        print(f"Faiss index size: {memory.faiss_index.ntotal}")

    # Test retrieval
    print("\n" + "-"*60)
    print("Testing retrieval...")
    print("-"*60)

    query = torch.randn(768)
    retrieved, details = memory.retrieve(query, top_k=3)

    print(f"Query shape: {query.shape}")
    print(f"Retrieved shape: {retrieved.shape}")
    print(f"Top-3 indices: {details['retrieval_indices']}")
    print(f"Top-3 concepts: {[test_concepts[i][0] for i in details['retrieval_indices'][:3]]}")
    print(f"Avg similarity: {details['avg_similarity']:.4f}")

    # Test batch retrieval
    print("\n" + "-"*60)
    print("Testing batch retrieval...")
    print("-"*60)

    batch_query = torch.randn(2, 768)
    batch_retrieved, batch_details = memory.retrieve(batch_query, top_k=2)

    print(f"Batch query shape: {batch_query.shape}")
    print(f"Batch retrieved shape: {batch_retrieved.shape}")

    # Test pruning and Faiss rebuild
    print("\n" + "-"*60)
    print("Testing pruning (Faiss rebuild)...")
    print("-"*60)

    before_count = len(memory)
    memory.prune()
    after_count = len(memory)

    print(f"Before pruning: {before_count} memories")
    print(f"After pruning: {after_count} memories")
    if memory.use_faiss:
        print(f"Faiss index size after prune: {memory.faiss_index.ntotal}")

    # Verify Faiss index matches memory count
    if memory.use_faiss:
        assert memory.faiss_index.ntotal == len(memory), "Faiss index out of sync!"
        print("✓ Faiss index synced with memory")

    # Performance comparison
    print("\n" + "-"*60)
    print("Performance comparison (100 queries)...")
    print("-"*60)

    import time

    # Faiss timing
    if memory.use_faiss:
        start = time.time()
        for _ in range(100):
            q = torch.randn(768)
            memory.retrieve(q, top_k=5)
        faiss_time = time.time() - start
        print(f"Faiss search (100 queries): {faiss_time:.4f}s ({faiss_time/100*1000:.2f}ms per query)")

    # Fallback timing (disable Faiss temporarily)
    original_use_faiss = memory.use_faiss
    memory.use_faiss = False

    start = time.time()
    for _ in range(100):
        q = torch.randn(768)
        memory.retrieve(q, top_k=5)
    pytorch_time = time.time() - start
    print(f"PyTorch search (100 queries): {pytorch_time:.4f}s ({pytorch_time/100*1000:.2f}ms per query)")

    memory.use_faiss = original_use_faiss

    if memory.use_faiss:
        speedup = pytorch_time / faiss_time
        print(f"\n⚡ Faiss speedup: {speedup:.2f}x faster")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    try:
        test_faiss_integration()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
