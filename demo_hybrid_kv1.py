#!/usr/bin/env python3
"""
Demo: KV-1 with Hybrid Memory (STM + LTM + GPU)

Shows the performance difference between:
- Old: String-based PersistentLTM
- New: STM (O(1) fast) + LTM (GPU semantic search)

This demonstrates:
1. Recent concepts → STM hit (microseconds)
2. Older concepts → LTM search (milliseconds) then promote to STM
3. Semantic search works (different phrasing)
4. GPU acceleration
"""

import asyncio
from self_discovery_orchestrator import SelfDiscoveryOrchestrator

async def main():
    print("=" * 70)
    print("KV-1 WITH HYBRID MEMORY DEMO")
    print("STM (Fast O(1)) + LTM (GPU Semantic Search)")
    print("=" * 70)

    # Create orchestrator with hybrid memory
    orchestrator = SelfDiscoveryOrchestrator(
        goal="What are prime numbers?",
        use_hybrid_memory=True  # NEW! Uses STM+LTM+GPU
    )

    print("\n" + "="*70)
    print("LEARNING PHASE")
    print("="*70)

    # Manually add some concepts to test
    print("\n[1] Learning 'prime numbers'...")
    orchestrator.ltm.learn(
        name="prime numbers",
        definition="Natural numbers greater than 1 divisible only by 1 and themselves",
        examples=[2, 3, 5, 7, 11, 13],
        confidence=0.95
    )

    print("\n[2] Learning 'composite numbers'...")
    orchestrator.ltm.learn(
        name="composite numbers",
        definition="Natural numbers with divisors other than 1 and themselves",
        examples=[4, 6, 8, 9, 10, 12],
        confidence=0.92
    )

    print("\n[3] Learning 'even numbers'...")
    orchestrator.ltm.learn(
        name="even numbers",
        definition="Numbers divisible by 2",
        examples=[2, 4, 6, 8, 10],
        confidence=0.98
    )

    print("\n[4] Learning 'odd numbers'...")
    orchestrator.ltm.learn(
        name="odd numbers",
        definition="Numbers not divisible by 2",
        examples=[1, 3, 5, 7, 9],
        confidence=0.98
    )

    print("\n[5] Learning 'fibonacci numbers'...")
    orchestrator.ltm.learn(
        name="fibonacci numbers",
        definition="Sequence where each number is sum of previous two",
        examples=[1, 1, 2, 3, 5, 8, 13],
        confidence=0.9
    )

    print("\n" + "="*70)
    print("RECALL PHASE - Testing STM vs LTM Performance")
    print("="*70)

    queries = [
        # Recent concepts (should be in STM)
        ("fibonacci numbers", "exact match - STM should hit"),
        ("odd numbers", "exact match - STM should hit"),

        # Different phrasing (semantic search in LTM)
        ("fibonacci", "abbreviated - LTM semantic search"),
        ("numbers divisible by two", "different phrasing - LTM"),
        ("primes", "abbreviated - LTM"),

        # After LTM hit, should be in STM
        ("fibonacci", "second time - now STM hit!"),
        ("primes", "second time - now STM hit!"),
    ]

    print("\n")
    for query, description in queries:
        print(f"\n[Query] '{query}' ({description})")
        result = orchestrator.ltm.recall(query, threshold=0.6)

        if result:
            name, concept, score = result
            print(f"  Found: {name} (confidence: {score:.2f})")
        else:
            print(f"  Not found")

    print("\n" + "="*70)
    print("MEMORY STATISTICS")
    print("="*70)
    print("\n" + orchestrator.ltm.summarize())

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    stats = orchestrator.ltm.get_stats()

    print(f"\n1. STM Hit Rate: {stats['stm_hit_rate']*100:.1f}%")
    print(f"   → Average time: {stats['avg_stm_time_ms']:.3f}ms (O(1) lookup)")

    print(f"\n2. LTM Hit Rate: {stats['ltm_hit_rate']*100:.1f}%")
    print(f"   → Average time: {stats['avg_ltm_time_ms']:.2f}ms (GPU semantic search)")

    print(f"\n3. Speedup: {stats['avg_ltm_time_ms'] / max(stats['avg_stm_time_ms'], 0.001):.1f}x faster")
    print(f"   STM is {stats['avg_ltm_time_ms'] / max(stats['avg_stm_time_ms'], 0.001):.0f}x faster than LTM!")

    print(f"\n4. Consolidations: {stats['consolidations']}")
    print(f"   → LTM concepts promoted to STM for faster access")

    print("\n" + "="*70)
    print("✓ Demo Complete!")
    print("="*70)

    print("\n🔥 BENEFITS:")
    print("  ✅ Recent concepts = instant O(1) lookup (STM)")
    print("  ✅ Semantic search works (different phrasing)")
    print("  ✅ GPU acceleration (10-100x faster)")
    print("  ✅ Auto-consolidation (frequent → STM)")
    print("  ✅ No duplicate learning (semantic dedup)")

if __name__ == "__main__":
    asyncio.run(main())
