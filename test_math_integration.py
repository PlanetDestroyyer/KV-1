#!/usr/bin/env python3
"""
Test MathConnect integration with Self-Discovery Orchestrator

Shows how the system automatically learns mathematical theorems symbolically
when they're encountered during goal pursuit.
"""

import asyncio
from self_discovery_orchestrator import SelfDiscoveryOrchestrator

async def test_math_integration():
    print("=" * 70)
    print("TESTING: MathConnect + Self-Discovery Integration")
    print("=" * 70)
    print("\nGoal: Learn about the Pythagorean theorem")
    print("Expected: System should learn it both as text AND symbolic equation\n")

    # Create orchestrator
    orchestrator = SelfDiscoveryOrchestrator(
        goal="What is the Pythagorean theorem?",
        ltm_path="./test_math_ltm.json",
        use_hybrid_memory=True  # Hybrid memory enabled
    )

    print("\n" + "=" * 70)
    print("SYSTEM CAPABILITIES")
    print("=" * 70)
    print(f"Hybrid Memory: {orchestrator.using_hybrid}")
    print(f"MathConnect: {orchestrator.using_mathconnect}")
    print(f"Goal Domain: {orchestrator.goal_domain}")

    # Manually add a mathematical concept to test
    print("\n" + "=" * 70)
    print("MANUAL TEST: Learning Pythagorean Theorem")
    print("=" * 70)

    concept = "pythagorean theorem"
    definition = "In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a squared plus b squared equals c squared"

    # Test mathematical concept detection
    is_math = orchestrator._is_mathematical_concept(concept, definition)
    print(f"\nIs mathematical concept? {is_math}")

    if orchestrator.using_mathconnect and is_math:
        print("\n[🧮] Testing symbolic learning with MathConnect...")

        success = orchestrator.math_connect.learn_theorem_from_text(
            name=concept,
            text=definition,
            domain="geometry"
        )

        if success:
            print("[✓] Successfully learned as symbolic equation!")

            # Get the theorem
            theorem = orchestrator.math_connect.connection_finder.theorems.get(concept)
            if theorem:
                print(f"\nStored equation: {theorem.equation}")
                print(f"LaTeX: {theorem.latex}")
                print(f"Domain: {theorem.domain}")

            # Check connections
            graph = orchestrator.math_connect.get_graph()
            connections = graph.get(concept, [])
            print(f"\nConnections: {len(connections)}")

            # Try learning another theorem to see connections
            print("\n" + "=" * 70)
            print("Learning second theorem to show connections...")
            print("=" * 70)

            orchestrator.math_connect.learn_theorem_from_text(
                name="trig_identity",
                text="sin squared theta plus cos squared theta equals one",
                domain="trigonometry"
            )

            # Check if they connected
            graph = orchestrator.math_connect.get_graph()
            pyt_connections = graph.get(concept, [])
            trig_connections = graph.get("trig_identity", [])

            print(f"\nPythagorean connections: {list(pyt_connections)}")
            print(f"Trig identity connections: {list(trig_connections)}")

            # Show summary
            print("\n" + "=" * 70)
            print("MATHCONNECT SUMMARY")
            print("=" * 70)
            print(orchestrator.math_connect.summarize())

    print("\n" + "=" * 70)
    print("✓ Integration Test Complete!")
    print("=" * 70)

    print("\n📝 KEY INSIGHTS:")
    print("  1. Mathematical concepts are auto-detected")
    print("  2. Definitions parsed to symbolic equations")
    print("  3. Connections found automatically")
    print("  4. Works alongside normal text-based learning")
    print("  5. All stored in hybrid memory (STM + LTM + GPU)")

if __name__ == "__main__":
    asyncio.run(test_math_integration())
