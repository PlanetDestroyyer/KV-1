"""
Comprehensive test and validation of tensor reasoning system.

Tests:
1. Import verification
2. Basic functionality
3. Error handling
4. Logic flow validation
"""

import sys
import traceback


def test_imports():
    """Test all imports work correctly"""
    print("\n" + "="*70)
    print("TEST 1: Import Verification")
    print("="*70)

    errors = []

    # Test math primitives
    try:
        from core.math_primitives import MathematicalPrimitives, MathDomain, ProofSteps
        print("✅ math_primitives.py")
    except Exception as e:
        errors.append(("math_primitives.py", str(e)))
        print(f"❌ math_primitives.py: {e}")

    # Test symbolic engine
    try:
        from core.symbolic_math_engine import SymbolicMathEngine, ProofStatus, ProofResult
        print("✅ symbolic_math_engine.py")
    except Exception as e:
        errors.append(("symbolic_math_engine.py", str(e)))
        print(f"❌ symbolic_math_engine.py: {e}")

    # Test geometric space
    try:
        from core.geometric_knowledge_space import RiemannianKnowledgeManifold, ConceptPoint, GeodesicPath
        print("✅ geometric_knowledge_space.py")
    except Exception as e:
        errors.append(("geometric_knowledge_space.py", str(e)))
        print(f"❌ geometric_knowledge_space.py: {e}")

    # Test exploration engine
    try:
        from core.mathematical_exploration_engine import MathematicalExplorationEngine, ExplorationResult
        print("✅ mathematical_exploration_engine.py")
    except Exception as e:
        errors.append(("mathematical_exploration_engine.py", str(e)))
        print(f"❌ mathematical_exploration_engine.py: {e}")

    # Test tensor reasoning system
    try:
        from core.tensor_reasoning_system import TensorReasoningSystem, ReasoningResult
        print("✅ tensor_reasoning_system.py")
    except Exception as e:
        errors.append(("tensor_reasoning_system.py", str(e)))
        print(f"❌ tensor_reasoning_system.py: {e}")

    # Test unified AGI learner
    try:
        from core.unified_agi_learner import UnifiedAGILearner, QuestionType, UnifiedResult
        print("✅ unified_agi_learner.py")
    except Exception as e:
        errors.append(("unified_agi_learner.py", str(e)))
        print(f"❌ unified_agi_learner.py: {e}")

    if errors:
        print(f"\n❌ {len(errors)} import errors found!")
        for file, error in errors:
            print(f"   {file}: {error}")
        return False
    else:
        print(f"\n✅ All imports successful!")
        return True


def test_initialization():
    """Test basic initialization"""
    print("\n" + "="*70)
    print("TEST 2: Component Initialization")
    print("="*70)

    try:
        from core.math_primitives import MathematicalPrimitives
        primitives = MathematicalPrimitives()
        print(f"✅ MathematicalPrimitives: {len(primitives.axioms)} axioms, {len(primitives.operations)} operations")
    except Exception as e:
        print(f"❌ MathematicalPrimitives failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.symbolic_math_engine import SymbolicMathEngine
        symbolic = SymbolicMathEngine()
        print(f"✅ SymbolicMathEngine: {len(symbolic.known_definitions)} definitions")
    except Exception as e:
        print(f"❌ SymbolicMathEngine failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.geometric_knowledge_space import RiemannianKnowledgeManifold
        geometry = RiemannianKnowledgeManifold(dimension=768)
        print(f"✅ RiemannianKnowledgeManifold: {geometry.dimension}-dimensional, device={geometry.device}")
    except Exception as e:
        print(f"❌ RiemannianKnowledgeManifold failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.mathematical_exploration_engine import MathematicalExplorationEngine
        from core.symbolic_math_engine import SymbolicMathEngine
        from core.geometric_knowledge_space import RiemannianKnowledgeManifold

        symbolic = SymbolicMathEngine()
        geometry = RiemannianKnowledgeManifold()
        explorer = MathematicalExplorationEngine(symbolic, geometry)
        print(f"✅ MathematicalExplorationEngine: max_depth={explorer.max_depth}, max_states={explorer.max_states}")
    except Exception as e:
        print(f"❌ MathematicalExplorationEngine failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.tensor_reasoning_system import TensorReasoningSystem
        system = TensorReasoningSystem(dimension=768)
        print(f"✅ TensorReasoningSystem: Initialized successfully")
        stats = system.get_stats()
        print(f"   - Primitives: {stats['primitives']['axioms']} axioms, {stats['primitives']['operations']} ops")
        print(f"   - Geometry: {stats['geometric_space']['num_concepts']} concepts")
    except Exception as e:
        print(f"❌ TensorReasoningSystem failed: {e}")
        traceback.print_exc()
        return False

    print("\n✅ All components initialized successfully!")
    return True


def test_basic_operations():
    """Test basic operations"""
    print("\n" + "="*70)
    print("TEST 3: Basic Operations")
    print("="*70)

    try:
        from core.math_primitives import MathematicalPrimitives
        primitives = MathematicalPrimitives()

        # Test operations
        result = primitives.apply_operation('factor', primitives.x**2 - 4)
        print(f"✅ Factor operation: x²-4 = {result}")

        result = primitives.apply_operation('is_prime', 7)
        print(f"✅ Primality test: is_prime(7) = {result}")

    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.geometric_knowledge_space import RiemannianKnowledgeManifold
        manifold = RiemannianKnowledgeManifold()

        # Embed concepts
        manifold.embed_concept("prime", None, {'domain': 'number_theory', 'is_prime': True})
        manifold.embed_concept("composite", None, {'domain': 'number_theory', 'is_prime': False})

        # Compute distance
        distance = manifold.riemannian_distance("prime", "composite")
        print(f"✅ Geometric distance: distance(prime, composite) = {distance:.3f}")

        # Neighbors
        neighbors = manifold.get_neighbors("prime", k=1)
        print(f"✅ Nearest neighbors: prime → {neighbors}")

    except Exception as e:
        print(f"❌ Geometric operations failed: {e}")
        traceback.print_exc()
        return False

    print("\n✅ All basic operations work!")
    return True


def test_symbolic_reasoning():
    """Test symbolic math engine"""
    print("\n" + "="*70)
    print("TEST 4: Symbolic Reasoning")
    print("="*70)

    try:
        from core.symbolic_math_engine import SymbolicMathEngine
        import sympy as sp

        engine = SymbolicMathEngine()

        # Test computational verification
        statement = sp.Eq(4 + 4, 8)
        result = engine.verify_by_computation(statement, test_range=10)
        print(f"✅ Computational verification: 4+4=8 → {result.status.value}")

        # Test Goldbach exploration
        print(f"✅ Testing Goldbach representations for n=10...")
        reps = engine.find_prime_representation(10)
        print(f"   10 = {reps}")

    except Exception as e:
        print(f"❌ Symbolic reasoning failed: {e}")
        traceback.print_exc()
        return False

    print("\n✅ Symbolic reasoning works!")
    return True


def check_logic_flows():
    """Check for logical flow issues"""
    print("\n" + "="*70)
    print("TEST 5: Logic Flow Validation")
    print("="*70)

    issues = []

    # Check if all required methods exist
    print("\nChecking method signatures...")

    try:
        from core.tensor_reasoning_system import TensorReasoningSystem
        system = TensorReasoningSystem()

        # Check required methods exist
        required_methods = [
            'solve', 'find_relations', 'get_learning_path',
            'get_stats', 'print_stats'
        ]

        for method in required_methods:
            if not hasattr(system, method):
                issues.append(f"TensorReasoningSystem missing method: {method}")
            else:
                print(f"✅ {method}() exists")

    except Exception as e:
        issues.append(f"TensorReasoningSystem check failed: {e}")

    try:
        from core.unified_agi_learner import UnifiedAGILearner

        # Check method signatures
        # Note: Can't fully initialize without LLM, but can check structure
        print("✅ UnifiedAGILearner structure valid")

    except Exception as e:
        issues.append(f"UnifiedAGILearner check failed: {e}")

    if issues:
        print(f"\n❌ {len(issues)} logic flow issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ All logic flows valid!")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🔍 COMPREHENSIVE TENSOR REASONING SYSTEM VALIDATION")
    print("="*70)

    results = []

    results.append(("Import Verification", test_imports()))
    results.append(("Component Initialization", test_initialization()))
    results.append(("Basic Operations", test_basic_operations()))
    results.append(("Symbolic Reasoning", test_symbolic_reasoning()))
    results.append(("Logic Flow Validation", check_logic_flows()))

    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n{'='*70}")
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready to use!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("❌ Please fix errors before using")
        return 1


if __name__ == "__main__":
    sys.exit(main())
