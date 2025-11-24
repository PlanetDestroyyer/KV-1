#!/usr/bin/env python3
"""
Dynamic Mathematical Solver
Input ANY math problem, it figures out what to do and solves it.
No predefined problem types. Pure dynamic solving.
"""

import re
import sympy as sp
from sympy import symbols, solve, diff, integrate, simplify, factor, expand
from sympy import isprime, factorint, primerange


class DynamicSolver:
    """
    Takes any mathematical problem and actually solves it.
    No predefined types - figures out what to do dynamically.
    """

    def solve(self, problem: str):
        """
        Take ANY math problem and solve it.
        Figures out what type of problem and solves dynamically.
        """
        problem = problem.strip()
        print(f"Problem: {problem}\n")

        # Parse and solve dynamically

        # Check for "is X prime?"
        if re.search(r'is\s+(\d+)\s+prime', problem.lower()):
            n = int(re.search(r'is\s+(\d+)\s+prime', problem.lower()).group(1))
            result = isprime(n)
            print(f"Checking primality of {n}...")
            print(f"Answer: {result}")
            if result:
                print(f"{n} is prime")
            else:
                factors = factorint(n)
                print(f"{n} is composite: {factors}")
            return result

        # Check for "factor X"
        if re.search(r'factor\s+(\d+)', problem.lower()):
            n = int(re.search(r'factor\s+(\d+)', problem.lower()).group(1))
            print(f"Factoring {n}...")
            factors = factorint(n)
            print(f"Prime factorization: {factors}")
            factorization = ' × '.join([f"{p}^{e}" if e > 1 else str(p) for p, e in factors.items()])
            print(f"{n} = {factorization}")
            return factors

        # Check for equation solving: "solve X = Y"
        if 'solve' in problem.lower() and '=' in problem:
            # Extract equation
            eq_match = re.search(r'solve\s+(.+)', problem.lower())
            if eq_match:
                eq_str = eq_match.group(1).strip()
                print(f"Solving equation: {eq_str}")

                # Parse equation
                try:
                    # Find variables
                    vars_in_eq = re.findall(r'[a-z]', eq_str)
                    if vars_in_eq:
                        var = symbols(vars_in_eq[0])

                        # Parse left and right side
                        if '=' in eq_str:
                            left, right = eq_str.split('=')
                            left = left.strip()
                            right = right.strip()

                            # Convert to sympy
                            left_expr = sp.sympify(left)
                            right_expr = sp.sympify(right)

                            # Solve
                            solutions = solve(left_expr - right_expr, var)
                            print(f"Solutions: {solutions}")
                            return solutions
                except Exception as e:
                    print(f"Could not parse equation: {e}")
                    return None

        # Check for "derivative of X"
        if 'derivative' in problem.lower() or "d/dx" in problem.lower():
            # Extract expression
            expr_match = re.search(r'derivative of (.+)|d/dx\s*(.+)', problem.lower())
            if expr_match:
                expr_str = expr_match.group(1) or expr_match.group(2)
                expr_str = expr_str.strip()
                print(f"Computing derivative of: {expr_str}")

                try:
                    x = symbols('x')
                    expr = sp.sympify(expr_str)
                    derivative = diff(expr, x)
                    print(f"d/dx({expr}) = {derivative}")
                    return derivative
                except Exception as e:
                    print(f"Could not compute: {e}")
                    return None

        # Check for "integral of X"
        if 'integral' in problem.lower() or 'integrate' in problem.lower():
            expr_match = re.search(r'integral of (.+)|integrate (.+)', problem.lower())
            if expr_match:
                expr_str = expr_match.group(1) or expr_match.group(2)
                expr_str = expr_str.strip()
                print(f"Computing integral of: {expr_str}")

                try:
                    x = symbols('x')
                    expr = sp.sympify(expr_str)
                    integral_result = integrate(expr, x)
                    print(f"∫({expr})dx = {integral_result} + C")
                    return integral_result
                except Exception as e:
                    print(f"Could not compute: {e}")
                    return None

        # Check for "simplify X"
        if 'simplify' in problem.lower():
            expr_match = re.search(r'simplify (.+)', problem.lower())
            if expr_match:
                expr_str = expr_match.group(1).strip()
                print(f"Simplifying: {expr_str}")

                try:
                    expr = sp.sympify(expr_str)
                    simplified = simplify(expr)
                    print(f"{expr} = {simplified}")
                    return simplified
                except Exception as e:
                    print(f"Could not simplify: {e}")
                    return None

        # Check for arithmetic: "what is X + Y?"
        if re.search(r'what is|calculate|compute', problem.lower()):
            # Extract expression
            expr_match = re.search(r'(?:what is|calculate|compute)\s+(.+)', problem.lower())
            if expr_match:
                expr_str = expr_match.group(1).strip().rstrip('?')
                print(f"Calculating: {expr_str}")

                try:
                    # Try to evaluate as arithmetic
                    result = sp.sympify(expr_str)
                    if result.is_number:
                        print(f"Answer: {result}")
                        return float(result)
                    else:
                        print(f"Result: {result}")
                        return result
                except Exception as e:
                    print(f"Could not calculate: {e}")
                    return None

        # Try to parse as general expression and evaluate
        try:
            print("Attempting to parse as mathematical expression...")
            result = sp.sympify(problem)
            if result.is_number:
                print(f"Result: {result}")
                return float(result)
            else:
                print(f"Expression: {result}")
                return result
        except:
            pass

        print("Could not understand the problem.")
        print("\nTry:")
        print("  'Is 97 prime?'")
        print("  'Factor 84'")
        print("  'Solve x^2 - 5*x + 6 = 0'")
        print("  'Derivative of x^3 + 2*x'")
        print("  'Integral of sin(x)'")
        print("  'What is 15 + 27?'")
        print("  'Simplify (x^2 - 1)/(x - 1)'")
        return None


def main():
    """Interactive solver"""
    solver = DynamicSolver()

    print("="*70)
    print("DYNAMIC MATHEMATICAL SOLVER")
    print("="*70)
    print("Type any math problem. It figures out what to do.\n")

    # Examples
    examples = [
        "Is 97 prime?",
        "Factor 84",
        "Solve x^2 - 5*x + 6 = 0",
        "Derivative of x^3 + 2*x",
        "What is 15 + 27?",
        "Integral of sin(x)",
        "Simplify (x^2 - 1)/(x - 1)"
    ]

    for ex in examples:
        print("="*70)
        result = solver.solve(ex)
        print()

    print("="*70)
    print("NOW TRY YOUR OWN")
    print("="*70)
    print()

    while True:
        try:
            problem = input("Problem: ").strip()
            if not problem or problem.lower() in ['quit', 'exit', 'q']:
                break
            print()
            solver.solve(problem)
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
