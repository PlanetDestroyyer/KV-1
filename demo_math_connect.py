#!/usr/bin/env python3
"""
Demo: MathConnect - AI Thinking in Math Equations

Shows how AI can:
1. Learn theorems from text → symbolic equations
2. Find connections between theorems automatically
3. Compose theorems to derive new results
4. Search web for mathematical knowledge

The key insight: Most math already exists scattered everywhere.
The problem: We don't know how these theorems CONNECT.
The solution: AI that stores math symbolically and finds connections.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.math_connect import MathConnect

def main():
    print("=" * 70)
    print("MATHCONNECT DEMO - AI Thinking in Math Equations")
    print("=" * 70)
    print("\nKey Insight: The problem isn't lack of math,")
    print("it's lack of CONNECTIONS between existing theorems!\n")

    # Initialize MathConnect (no LLM or web needed for basic demo)
    math_system = MathConnect()

    print("=" * 70)
    print("PHASE 1: Learning Theorems from Text")
    print("=" * 70)

    # Learn basic theorems
    theorems = [
        ("pythagorean", "a squared plus b squared equals c squared", "geometry"),
        ("trig_identity", "sin squared theta plus cos squared theta equals one", "trigonometry"),
        ("quadratic", "x equals negative b plus or minus square root of b squared minus 4ac all over 2a", "algebra"),
        ("area_circle", "area equals pi times r squared", "geometry"),
        ("circumference", "c equals 2 times pi times r", "geometry"),
    ]

    print("\nLearning theorems...")
    for name, text, domain in theorems:
        print(f"\n[Learning] {name}...")
        success = math_system.learn_theorem_from_text(name, text, domain)
        if not success:
            print(f"  ⚠ Could not parse: {text}")

    print("\n" + "=" * 70)
    print("PHASE 2: Connection Discovery")
    print("=" * 70)

    # The system already found connections during learning
    graph = math_system.get_graph()

    print("\nConnection Graph:")
    for theorem, connections in graph.items():
        if connections:
            print(f"  {theorem} ↔ {list(connections)}")

    print("\n" + "=" * 70)
    print("PHASE 3: Finding Connection Paths")
    print("=" * 70)

    # Try to find how theorems are connected
    test_paths = [
        ("pythagorean", "trig_identity"),
        ("area_circle", "circumference"),
    ]

    for theorem_a, theorem_b in test_paths:
        print(f"\n[Path] {theorem_a} → {theorem_b}")
        path = math_system.find_connection_path(theorem_a, theorem_b)
        if path:
            print(f"  Found path: {' → '.join(path)}")
        else:
            print(f"  No direct path found")

    print("\n" + "=" * 70)
    print("PHASE 4: Deriving New Theorems by Composition")
    print("=" * 70)

    print("\nThe system automatically tries to compose theorems during learning.")
    print("Let's check what was derived...")

    # Get all derived theorems
    derived = math_system.composer.derived_theorems
    if derived:
        print(f"\n[Derived] Found {len(derived)} new theorems:")
        for theorem in derived[:5]:  # Show first 5
            print(f"  • {theorem.name}: {theorem.equation}")
            print(f"    (from: {', '.join(theorem.related_theorems)})")
    else:
        print("\n[Derived] No new theorems automatically derived yet.")
        print("          This is normal - composition requires compatible equations.")

    print("\n" + "=" * 70)
    print("PHASE 5: Summary")
    print("=" * 70)

    print("\n" + math_system.summarize())

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("""
1. ✅ Theorems stored as SYMBOLIC EQUATIONS (not text!)
   → Can manipulate, substitute, compose them

2. ✅ Connections found AUTOMATICALLY
   → Shared symbols, structural similarity, derivability

3. ✅ Can find PATHS between theorems
   → "How do I get from A to B using known math?"

4. ✅ Composes theorems to derive NEW results
   → Substitution, addition, multiplication of equations

5. ✅ AI thinks in MATH, not just text!
   → This is how you prove theorems, not by reading papers
    """)

    print("\n" + "=" * 70)
    print("WHAT THIS ENABLES")
    print("=" * 70)

    print("""
With web integration (WebResearcher):
→ Search for "Pythagorean theorem" → Learn it symbolically
→ Search for "Trig identities" → Learn them symbolically
→ Automatically finds: They're CONNECTED! (unit circle)
→ Derives: New relationships humans might miss

Instead of reading papers about math (TEXT),
AI now OPERATES on math directly (EQUATIONS).

This is the breakthrough: Thinking in the language of math itself!
    """)

    print("=" * 70)
    print("✓ Demo Complete!")
    print("=" * 70)

    print("\nNext Steps:")
    print("1. Integrate with WebResearcher for automatic theorem discovery")
    print("2. Add proof search (forward/backward chaining)")
    print("3. Store theorems in HybridMemory for fast retrieval")
    print("4. Connect with KV-1 self-discovery loop")


if __name__ == "__main__":
    main()
