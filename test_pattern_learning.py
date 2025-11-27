"""
Test Pattern Learning System

Demonstrates how the pattern learner extracts mathematical structures
and learns from problem-solving experience.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pattern_learner import MathematicalStructureLearner


def test_pattern_extraction():
    """Test extracting mathematical structure from problems."""

    learner = MathematicalStructureLearner(storage_path="./test_patterns.json")

    print("="*60)
    print("PATTERN LEARNING TEST")
    print("="*60)

    # Test 1: Quadratic equation
    print("\n[Test 1] Quadratic Equation")
    problem1 = "Solve x² - 5x + 6 = 0"
    trace1 = ["identify quadratic", "apply quadratic formula", "x = (5 ± √1)/2", "x = 2 or x = 3"]

    learner.observe_problem_solution(
        problem=problem1,
        solution_trace=trace1,
        success=True,
        solution_time=2.5
    )

    # Test 2: Another quadratic
    print("\n[Test 2] Another Quadratic")
    problem2 = "Find roots of 2x² + 3x - 2 = 0"
    trace2 = ["quadratic equation", "use formula", "x = (-3 ± √25)/4", "x = 0.5 or x = -2"]

    learner.observe_problem_solution(
        problem=problem2,
        solution_trace=trace2,
        success=True,
        solution_time=3.0
    )

    # Test 3: Differentiation
    print("\n[Test 3] Calculus - Differentiation")
    problem3 = "Find derivative of x³ + 2x²"
    trace3 = ["apply power rule", "d/dx(x³) = 3x²", "d/dx(2x²) = 4x", "result: 3x² + 4x"]

    learner.observe_problem_solution(
        problem=problem3,
        solution_trace=trace3,
        success=True,
        solution_time=1.8
    )

    # Test 4: Integration
    print("\n[Test 4] Calculus - Integration")
    problem4 = "Integrate 3x² + 4x"
    trace4 = ["reverse power rule", "∫3x²dx = x³", "∫4xdx = 2x²", "result: x³ + 2x² + C"]

    learner.observe_problem_solution(
        problem=problem4,
        solution_trace=trace4,
        success=True,
        solution_time=2.2
    )

    # Test 5: Prime factorization
    print("\n[Test 5] Number Theory - Prime Factorization")
    problem5 = "Factor 60 into primes"
    trace5 = ["divide by 2: 60/2 = 30", "divide by 2: 30/2 = 15", "divide by 3: 15/3 = 5", "5 is prime", "result: 2² × 3 × 5"]

    learner.observe_problem_solution(
        problem=problem5,
        solution_trace=trace5,
        success=True,
        solution_time=1.5
    )

    # Test 6: Another quadratic (for clustering)
    print("\n[Test 6] Third Quadratic (for clustering)")
    problem6 = "Solve 3x² - 12x + 9 = 0"
    trace6 = ["quadratic formula", "x = (12 ± √36)/6", "x = 3 or x = 1"]

    learner.observe_problem_solution(
        problem=problem6,
        solution_trace=trace6,
        success=True,
        solution_time=2.8
    )

    # Test 7: Another derivative (for clustering)
    print("\n[Test 7] Another Derivative")
    problem7 = "Differentiate 5x⁴ - 2x"
    trace7 = ["power rule", "4·5x³ = 20x³", "-1·2 = -2", "result: 20x³ - 2"]

    learner.observe_problem_solution(
        problem=problem7,
        solution_trace=trace7,
        success=True,
        solution_time=1.6
    )

    # Test 8: Failed attempt (to show learning from failure)
    print("\n[Test 8] Failed Attempt (missing knowledge)")
    problem8 = "Solve the differential equation dy/dx = 2x"
    trace8 = ["recognize differential equation", "missing: integration techniques", "cannot solve"]

    learner.observe_problem_solution(
        problem=problem8,
        solution_trace=trace8,
        success=False,
        solution_time=1.0
    )

    # Test 9-11: More problems to trigger clustering
    for i in range(9, 12):
        print(f"\n[Test {i}] Linear equation")
        problem = f"Solve {i}x + {i*2} = {i*10}"
        trace = ["isolate x", f"x = {i*10 - i*2}/{i}", "result"]
        learner.observe_problem_solution(problem, trace, True, 1.0)

    # Show statistics
    print("\n" + "="*60)
    print("PATTERN LEARNING STATISTICS")
    print("="*60)

    stats = learner.get_statistics()
    print(f"Total problems observed: {stats['total_instances']}")
    print(f"Successful: {stats['successful_instances']}")
    print(f"Pattern clusters discovered: {stats['num_clusters']}")
    print(f"Average solution time: {stats['average_solution_time']:.2f}s")

    if stats['num_clusters'] > 0:
        print("\nCluster Success Rates:")
        for cluster_id, rate in stats['cluster_success_rates'].items():
            print(f"  Cluster {cluster_id}: {rate*100:.0f}% success")

    # Test prediction
    print("\n" + "="*60)
    print("PREDICTION TEST")
    print("="*60)

    new_problem = "Solve the equation 4x² - 8x + 3 = 0"
    print(f"\nNew problem: {new_problem}")

    predicted_structure = learner.predict_structure(new_problem)
    if predicted_structure:
        print(f"\nPredicted structure:")
        print(f"  Operations: {predicted_structure.operations}")
        print(f"  Domain: {predicted_structure.domain}")
        print(f"  Equation type: {predicted_structure.equation_type}")
        print(f"  Object type: {predicted_structure.object_type}")

    print("\n" + "="*60)
    print("✅ Pattern Learning Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_pattern_extraction()
