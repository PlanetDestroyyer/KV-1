#!/usr/bin/env python3
"""
Demo: AI-Native Learning with Neurosymbolic Memory

Shows how KV-1 can learn using vectors + formulas instead of just text.

This demonstrates:
1. Learning concepts as vectors (semantic embeddings)
2. Extracting formulas from definitions
3. Finding similar concepts (no string matching!)
4. Composing formulas to discover new rules
5. Transferring knowledge algebraically
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.neurosymbolic_memory import NeurosymbolicMemory


def demo_basic_learning():
    """Demo 1: Learn concepts in AI-native format."""
    print("=" * 70)
    print("DEMO 1: Learning Concepts as Vectors + Formulas")
    print("=" * 70)

    memory = NeurosymbolicMemory()

    # Learn "prime numbers"
    print("\n[Learning] Prime Numbers")
    memory.learn_concept(
        name="prime numbers",
        definition="Natural numbers greater than 1 that have no divisors other than 1 and themselves",
        examples=[2, 3, 5, 7, 11, 13, 17, 19],
        confidence=0.95
    )

    # Learn "even numbers"
    print("\n[Learning] Even Numbers")
    memory.learn_concept(
        name="even numbers",
        definition="Numbers divisible by 2",
        examples=[2, 4, 6, 8, 10, 12],
        confidence=0.98
    )

    # Learn "odd numbers"
    print("\n[Learning] Odd Numbers")
    memory.learn_concept(
        name="odd numbers",
        definition="Numbers not divisible by 2",
        examples=[1, 3, 5, 7, 9, 11],
        confidence=0.98
    )

    print("\n" + memory.summarize())


def demo_semantic_search():
    """Demo 2: Find concepts using semantic similarity (not string matching!)."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Semantic Search (Vector-based, not string matching)")
    print("=" * 70)

    memory = NeurosymbolicMemory()

    # Learn various concepts
    memory.learn_concept("prime numbers", "Numbers with no divisors except 1 and itself", [2, 3, 5, 7])
    memory.learn_concept("composite numbers", "Numbers with divisors other than 1 and itself", [4, 6, 8, 9])
    memory.learn_concept("factorization", "Breaking a number into prime factors", ["12 = 2×2×3"])
    memory.learn_concept("divisibility", "When one number divides another evenly", [])

    # Search with different phrasing
    queries = [
        "prime number",  # Singular vs plural
        "numbers that are prime",  # Different phrasing
        "factoring",  # Related concept
        "division",  # Related concept
    ]

    for query in queries:
        print(f"\n[Search] '{query}'")
        results = memory.find_similar(query, top_k=3, threshold=0.5)
        for name, score in results:
            print(f"  → {name} (similarity: {score:.2f})")


def demo_formula_composition():
    """Demo 3: Compose formulas to discover new knowledge."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Formula Composition (Discovering New Rules)")
    print("=" * 70)

    memory = NeurosymbolicMemory()

    # Learn concepts that will compose
    print("\n[Learning] Concepts with formulas...")

    c1 = memory.learn_concept(
        "prime numbers",
        "Numbers greater than 1 divisible only by 1 and themselves",
        [2, 3, 5, 7, 11],
        confidence=0.95
    )
    print(f"  Formulas extracted: {c1.formulas}")

    c2 = memory.learn_concept(
        "even numbers",
        "Numbers divisible by 2",
        [2, 4, 6, 8, 10],
        confidence=0.98
    )
    print(f"  Formulas extracted: {c2.formulas}")

    print("\n[Composed Formulas]")
    for formula_name, formula in memory.formulas.items():
        print(f"  • {formula_name}")
        print(f"    Learned from: {formula.learned_from}")


def demo_concept_relationships():
    """Demo 4: Discover relationships between concepts (vector arithmetic)."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Concept Relationships (Vector Arithmetic)")
    print("=" * 70)

    memory = NeurosymbolicMemory()

    # Learn related concepts
    memory.learn_concept("prime", "Indivisible numbers", [2, 3, 5, 7])
    memory.learn_concept("composite", "Divisible numbers", [4, 6, 8, 9])
    memory.learn_concept("even", "Divisible by 2", [2, 4, 6, 8])
    memory.learn_concept("odd", "Not divisible by 2", [1, 3, 5, 7])

    # Show concept graph
    print("\n[Concept Relationships Graph]")
    graph = memory.get_concept_graph()
    for concept, related in graph.items():
        if related:
            print(f"  {concept} ↔ {', '.join(related)}")


def demo_knowledge_transfer():
    """Demo 5: Transfer formulas to new domains."""
    print("\n\n" + "=" * 70)
    print("DEMO 5: Knowledge Transfer (Apply learned formulas to new domains)")
    print("=" * 70)

    memory = NeurosymbolicMemory()

    # Learn in number domain
    print("\n[Learning] Number domain...")
    memory.learn_concept(
        "prime numbers",
        "Numbers greater than 1 divisible only by 1 and themselves",
        [2, 3, 5, 7, 11]
    )

    # Learn polynomial concept
    print("\n[Learning] Polynomial domain...")
    memory.learn_concept(
        "prime polynomial",
        "Polynomials that cannot be factored",
        ["x^2 + 1"]
    )

    # Transfer knowledge
    print("\n[Transfer] Applying 'prime numbers' knowledge to polynomials...")
    transferred = memory.transfer_knowledge("prime numbers", "polynomial")
    for formula in transferred:
        print(f"  Adapted formula: {formula}")


def demo_comparison():
    """Compare old (string) vs new (vector) approach."""
    print("\n\n" + "=" * 70)
    print("COMPARISON: String Matching vs Semantic Search")
    print("=" * 70)

    memory = NeurosymbolicMemory()

    # Learn "prime numbers"
    memory.learn_concept("prime numbers", "Indivisible numbers > 1", [2, 3, 5, 7])

    test_queries = [
        ("prime numbers", "Exact match"),
        ("prime number", "Singular vs plural"),
        ("primes", "Abbreviation"),
        ("indivisible numbers", "Different phrasing"),
        ("numbers with no factors", "Semantic equivalent"),
    ]

    print("\n[String Matching Results]")
    string_store = {"prime numbers": "concept data"}
    for query, description in test_queries:
        found = query in string_store
        print(f"  '{query}' ({description}): {'✓ FOUND' if found else '✗ NOT FOUND'}")

    print("\n[Vector/Semantic Search Results]")
    for query, description in test_queries:
        results = memory.find_similar(query, top_k=1, threshold=0.5)
        found = len(results) > 0
        score = results[0][1] if found else 0.0
        print(f"  '{query}' ({description}): {'✓ FOUND' if found else '✗ NOT FOUND'} (sim={score:.2f})")


def main():
    """Run all demos."""
    print("\n" + "🧠" * 35)
    print("KV-1 NEUROSYMBOLIC LEARNING DEMO")
    print("Learning with Vectors + Formulas (AI-Native Format)")
    print("🧠" * 35)

    try:
        demo_basic_learning()
        demo_semantic_search()
        demo_formula_composition()
        demo_concept_relationships()
        demo_knowledge_transfer()
        demo_comparison()

        print("\n\n" + "=" * 70)
        print("✓ All demos completed!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("1. ✅ Concepts stored as vectors (no string matching!)")
        print("2. ✅ Formulas extracted automatically from definitions")
        print("3. ✅ New formulas composed from existing ones")
        print("4. ✅ Knowledge transfers across domains algebraically")
        print("5. ✅ AI learns in its native format (math, not text)")
        print("\nThis is how KV-1 becomes a truly learning system!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
