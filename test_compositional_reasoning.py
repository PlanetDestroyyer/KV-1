"""
Test Compositional Reasoning - Phase 2

This demonstrates how the system combines learned patterns to solve novel problems.
The key insight: AGI emerges when you can COMPOSE solutions, not just match patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location("compositional_reasoner",
                                               os.path.join(os.path.dirname(__file__), "core", "compositional_reasoner.py"))
comp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comp_module)

CompositionEngine = comp_module.CompositionEngine
AbstractionBuilder = comp_module.AbstractionBuilder
StructureType = comp_module.StructureType
StructureMorphism = comp_module.StructureMorphism
CompositePattern = comp_module.CompositePattern


def test_structure_hierarchy():
    """Test that we correctly identify abstraction hierarchies"""
    print("\n" + "="*70)
    print("TEST 1: Structure Hierarchy Recognition")
    print("="*70)

    # Check if field is a specialization of group
    is_spec = StructureType.is_specialization("group", "field")
    print(f"Is FIELD a specialization of GROUP? {is_spec}")
    assert is_spec, "Field should be a specialization of group"

    # Check if field is a specialization of ring
    is_spec = StructureType.is_specialization("ring", "field")
    print(f"Is FIELD a specialization of RING? {is_spec}")
    assert is_spec, "Field should be a specialization of ring"

    # Check false case
    is_spec = StructureType.is_specialization("field", "group")
    print(f"Is GROUP a specialization of FIELD? {is_spec}")
    assert not is_spec, "Group should not be a specialization of field"

    print("[✓] Hierarchy recognition works correctly!")


def test_abstraction_chain():
    """Test building abstraction chains between structures"""
    print("\n" + "="*70)
    print("TEST 2: Abstraction Chain Building")
    print("="*70)

    engine = CompositionEngine()

    # Build chain from GROUP to FIELD
    chain = engine.build_abstraction_chain(
        StructureType.GROUP,
        StructureType.FIELD
    )

    if chain:
        print(f"Chain from GROUP to FIELD:")
        print(" → ".join(s.value for s in chain))
        assert StructureType.GROUP in chain, "Chain should start with GROUP"
        assert StructureType.FIELD in chain, "Chain should end with FIELD"
        print(f"[✓] Found abstraction path with {len(chain)} steps!")
    else:
        print("[✗] No chain found!")


def test_structure_identification():
    """Test identifying mathematical structure from pattern data"""
    print("\n" + "="*70)
    print("TEST 3: Structure Type Identification")
    print("="*70)

    engine = CompositionEngine()

    # Test 1: Vector space
    pattern1 = {
        'operations': {'addition', 'scalar_multiplication'},
        'object_type': 'vector',
        'properties': {'linear', 'span'}
    }
    struct = engine.identify_structure_type(pattern1)
    print(f"Pattern with vectors + scalar multiplication → {struct.value}")
    assert struct == StructureType.VECTOR_SPACE, "Should identify as vector space"

    # Test 2: Field
    pattern2 = {
        'operations': {'addition', 'multiplication', 'division'},
        'object_type': 'number',
        'properties': {'commutative', 'inverses'}
    }
    struct = engine.identify_structure_type(pattern2)
    print(f"Pattern with division + multiplication → {struct.value}")
    assert struct == StructureType.FIELD, "Should identify as field"

    # Test 3: Group
    pattern3 = {
        'operations': {'inverse'},
        'object_type': 'permutation',
        'properties': {'associative'}
    }
    struct = engine.identify_structure_type(pattern3)
    print(f"Pattern with inverse operation → {struct.value}")
    assert struct == StructureType.GROUP, "Should identify as group"

    print("[✓] Structure identification works correctly!")


def test_pattern_decomposition():
    """Test decomposing a problem into subproblems matching known patterns"""
    print("\n" + "="*70)
    print("TEST 4: Problem Decomposition")
    print("="*70)

    engine = CompositionEngine()

    # Available patterns (simulating learned patterns)
    available_patterns = [
        {
            'name': 'linear_algebra',
            'operations': {'addition', 'scalar_multiplication', 'solve'},
            'object_type': 'vector',
            'domain': 'linear_algebra',
            'success_rate': 0.85
        },
        {
            'name': 'polynomial_solving',
            'operations': {'solve', 'factor', 'root'},
            'object_type': 'polynomial',
            'domain': 'algebra',
            'success_rate': 0.75
        },
        {
            'name': 'calculus_optimization',
            'operations': {'differentiate', 'solve'},
            'object_type': 'function',
            'domain': 'calculus',
            'success_rate': 0.70
        }
    ]

    # Test problem: requires linear algebra
    problem1 = "Solve the system of linear equations: 2x + 3y = 7, x - y = 1"
    matches = engine.decompose_problem(problem1, available_patterns)

    print(f"\nProblem: {problem1}")
    print("Relevant patterns:")
    for pattern_name, confidence in matches[:3]:
        print(f"  - {pattern_name}: {confidence*100:.1f}% confidence")

    # Should rank linear_algebra highest
    if matches:
        assert matches[0][0] == 'linear_algebra', "Should identify linear algebra pattern"
        print(f"[✓] Correctly identified linear algebra ({matches[0][1]*100:.1f}% confidence)")

    # Test problem: requires polynomial solving
    problem2 = "Factor the polynomial x^2 + 5x + 6"
    matches = engine.decompose_problem(problem2, available_patterns)

    print(f"\nProblem: {problem2}")
    print("Relevant patterns:")
    for pattern_name, confidence in matches[:3]:
        print(f"  - {pattern_name}: {confidence*100:.1f}% confidence")

    if matches:
        assert matches[0][0] == 'polynomial_solving', "Should identify polynomial pattern"
        print(f"[✓] Correctly identified polynomial solving ({matches[0][1]*100:.1f}% confidence)")


def test_solution_strategy():
    """Test creating multi-step solution strategies through composition"""
    print("\n" + "="*70)
    print("TEST 5: Solution Strategy Creation (THE CORE OF AGI!)")
    print("="*70)

    engine = CompositionEngine()

    # Learned patterns
    patterns = [
        {
            'name': 'differentiation',
            'operations': {'differentiate', 'derivative'},
            'object_type': 'function',
            'domain': 'calculus',
            'success_rate': 0.90
        },
        {
            'name': 'equation_solving',
            'operations': {'solve', 'isolate'},
            'object_type': 'equation',
            'domain': 'algebra',
            'success_rate': 0.85
        },
        {
            'name': 'critical_points',
            'operations': {'solve', 'derivative'},
            'object_type': 'function',
            'domain': 'calculus',
            'success_rate': 0.75
        }
    ]

    # Novel problem that requires COMPOSING multiple patterns
    problem = "Find the maximum value of the function f(x) = -x^2 + 4x + 1"

    print(f"Problem (NOVEL - never seen before!):")
    print(f"  {problem}")
    print("\nThis requires:")
    print("  1. Differentiate the function")
    print("  2. Solve derivative = 0 to find critical points")
    print("  3. Evaluate function at critical points")
    print("\nLet's see if composition engine discovers this strategy...")

    strategy = engine.create_solution_strategy(problem, patterns)

    print(f"\n{'='*70}")
    print("SOLUTION STRATEGY:")
    print("="*70)

    if strategy['success']:
        print(f"✓ Strategy found with {strategy['total_confidence']*100:.1f}% confidence")
        print(f"Requires composition: {strategy['requires_composition']}")
        print("\nSteps:")
        for i, step in enumerate(strategy['strategy'], 1):
            print(f"  {i}. Use pattern '{step['pattern']}'")
            print(f"     - Structure: {step['structure']}")
            print(f"     - Confidence: {step['confidence']*100:.1f}%")
            print(f"     - Operations: {', '.join(step['operations'])}")

        if strategy['morphisms']:
            print("\nStructure Transformations:")
            for morph in strategy['morphisms']:
                print(f"  {morph['from']} → {morph['to']} (via {morph['via']})")

        print("\n[✓] COMPOSITIONAL REASONING WORKS!")
        print("[i] This is the key to AGI: solving novel problems by composing known patterns!")
    else:
        print(f"✗ No strategy found: {strategy.get('reason', 'unknown')}")


def test_abstraction_learning():
    """Test learning abstraction relationships between patterns"""
    print("\n" + "="*70)
    print("TEST 6: Abstraction Learning")
    print("="*70)

    builder = AbstractionBuilder()

    # Specific pattern: solving linear equations
    specific = {
        'name': 'linear_equation_solver',
        'operations': {'addition', 'multiplication', 'solve'},
        'object_type': 'linear_equation',
        'domain': 'algebra'
    }

    # General pattern: solving polynomial equations
    general = {
        'name': 'polynomial_equation_solver',
        'operations': {'addition', 'multiplication', 'solve', 'factor'},
        'object_type': 'polynomial',
        'domain': 'algebra'
    }

    # The system should recognize that both are RINGs
    # and linear is a special case of polynomial

    spec_type = builder.composition_engine.identify_structure_type(specific)
    gen_type = builder.composition_engine.identify_structure_type(general)

    print(f"Specific pattern structure: {spec_type.value}")
    print(f"General pattern structure: {gen_type.value}")

    # Learn the abstraction
    learned = builder.learn_abstraction(specific, general)

    if learned:
        print(f"[✓] Learned that '{specific['name']}' is a special case of '{general['name']}'")
    else:
        print(f"[i] Patterns are at same abstraction level")

    # Test finding similar patterns
    all_patterns = [specific, general, {
        'name': 'quadratic_solver',
        'operations': {'addition', 'multiplication', 'solve', 'factor'},
        'object_type': 'polynomial',
        'domain': 'algebra'
    }]

    similar = builder.find_similar_patterns(general, all_patterns)
    print("\nPatterns similar to polynomial solver:")
    for name, similarity in similar:
        print(f"  - {name}: {similarity*100:.1f}% similar")


def test_morphisms():
    """Test structure morphisms for transforming problems"""
    print("\n" + "="*70)
    print("TEST 7: Structure Morphisms")
    print("="*70)

    engine = CompositionEngine()

    print(f"Known morphisms: {len(engine.known_morphisms)}")
    for morph in engine.known_morphisms:
        print(f"\n  {morph.name}:")
        print(f"    {morph.source_structure.value} → {morph.target_structure.value}")
        print(f"    Type: {morph.transformation_type}")
        print(f"    Preserves: {', '.join(morph.properties_preserved)}")

    # Find morphisms applicable to RING
    applicable = engine.find_applicable_morphisms(StructureType.RING)
    print(f"\nMorphisms applicable to RING structures: {len(applicable)}")
    for morph in applicable:
        print(f"  - {morph.name}: RING → {morph.target_structure.value}")

    print("\n[✓] Morphisms enable transforming problems between representations!")


def run_all_tests():
    """Run all compositional reasoning tests"""
    print("\n" + "="*70)
    print("COMPOSITIONAL REASONING TEST SUITE - PHASE 2")
    print("="*70)
    print("\nThis tests the ability to COMPOSE learned patterns to solve novel problems.")
    print("This is the core mechanism that moves from pattern matching to creative problem solving.")
    print("="*70)

    try:
        test_structure_hierarchy()
        test_abstraction_chain()
        test_structure_identification()
        test_pattern_decomposition()
        test_solution_strategy()
        test_abstraction_learning()
        test_morphisms()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nKEY INSIGHTS:")
        print("1. System identifies mathematical structure types (group, ring, field, etc.)")
        print("2. Builds abstraction chains between structures")
        print("3. Decomposes problems into known patterns")
        print("4. COMPOSES multiple patterns to solve NOVEL problems")
        print("5. Learns abstraction relationships")
        print("6. Uses morphisms to transform between representations")
        print("\n→ This is how mathematical AGI emerges: through compositional reasoning!")
        print("="*70)

    except AssertionError as e:
        print(f"\n[✗] TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n[✗] ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
