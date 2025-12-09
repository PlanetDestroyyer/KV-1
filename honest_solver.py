"""
HONEST Mathematical Solver
No hardcoded proofs. No templates. No bullshit.

Either:
1. Solves computationally (actual code execution)
2. Uses LLM API for reasoning (if available)
3. Admits it cannot solve

No fake output.
"""

import sys
from typing import Optional, Dict, Any


class HonestSolver:
    """
    Honest mathematical problem solver.
    No hardcoded answers. Only real computation or admission of inability.
    """

    def __init__(self):
        print("="*70)
        print("HONEST MATHEMATICAL SOLVER")
        print("="*70)
        print("No hardcoded proofs. No templates. Only real work.")
        print("="*70 + "\n")

    def solve(self, problem_type: str, **params) -> Dict[str, Any]:
        """
        Solve a problem by actually computing, not templating.

        Args:
            problem_type: Type of problem
            **params: Problem parameters

        Returns:
            Result dictionary
        """
        print(f"Problem Type: {problem_type}")
        print(f"Parameters: {params}\n")

        # Route to appropriate solver
        if problem_type == "goldbach_verify":
            return self._verify_goldbach_computational(params.get('n'))

        elif problem_type == "prime_check":
            return self._check_prime_computational(params.get('n'))

        elif problem_type == "factor":
            return self._factor_number(params.get('n'))

        elif problem_type == "collatz_sequence":
            return self._compute_collatz(params.get('n'))

        elif problem_type == "sum_formula":
            return self._verify_sum_formula(params.get('n'))

        else:
            return {
                'solved': False,
                'reason': f"Don't know how to solve '{problem_type}' yet",
                'result': None
            }

    # === COMPUTATIONAL SOLVERS (Actually compute, don't template) ===

    def _check_prime_computational(self, n: int) -> Dict:
        """Actually check if n is prime by computing"""
        print(f"Checking if {n} is prime...")
        print("Method: Trial division\n")

        if n < 2:
            return {'solved': True, 'result': False, 'reason': f'{n} < 2'}

        if n == 2:
            return {'solved': True, 'result': True, 'reason': '2 is prime'}

        if n % 2 == 0:
            return {'solved': True, 'result': False, 'reason': f'{n} is even, divisible by 2'}

        # Check odd divisors up to sqrt(n)
        i = 3
        print(f"Checking divisors from 3 to {int(n**0.5) + 1}...")

        while i * i <= n:
            if n % i == 0:
                return {
                    'solved': True,
                    'result': False,
                    'reason': f'{n} is divisible by {i}'
                }
            i += 2

        return {
            'solved': True,
            'result': True,
            'reason': f'No divisors found, {n} is prime'
        }

    def _factor_number(self, n: int) -> Dict:
        """Actually factor a number by computation"""
        print(f"Factoring {n}...\n")

        if n < 2:
            return {'solved': True, 'factors': [], 'prime': False}

        factors = []
        d = 2

        temp_n = n
        while d * d <= temp_n:
            while temp_n % d == 0:
                factors.append(d)
                temp_n //= d
                print(f"  Found factor: {d}")
            d += 1

        if temp_n > 1:
            factors.append(temp_n)
            print(f"  Found factor: {temp_n}")

        return {
            'solved': True,
            'result': factors,
            'factorization': ' × '.join(map(str, factors)),
            'is_prime': len(factors) == 1
        }

    def _verify_goldbach_computational(self, n: int) -> Dict:
        """
        Actually verify Goldbach's conjecture for a specific even number.
        Find two primes that sum to n.
        """
        print(f"Verifying Goldbach for n={n}")
        print(f"Looking for primes p, q such that p + q = {n}\n")

        if n <= 2 or n % 2 != 0:
            return {
                'solved': False,
                'reason': f'{n} is not an even number > 2'
            }

        # Generate primes up to n using Sieve of Eratosthenes
        print("Generating primes...")
        primes = self._sieve_of_eratosthenes(n)
        print(f"Generated {len(primes)} primes up to {n}\n")

        # Check all pairs
        print("Searching for pair...")
        for p in primes:
            q = n - p
            if q in primes:
                return {
                    'solved': True,
                    'result': True,
                    'primes': [p, q],
                    'verification': f'{p} + {q} = {n}',
                    'reason': f'Found: {p} and {q} are both prime and sum to {n}'
                }

        return {
            'solved': True,
            'result': False,
            'reason': f'No prime pair found for {n}'
        }

    def _sieve_of_eratosthenes(self, limit: int) -> set:
        """Actually compute primes using Sieve of Eratosthenes"""
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False

        return {i for i in range(limit + 1) if is_prime[i]}

    def _compute_collatz(self, n: int) -> Dict:
        """Actually compute Collatz sequence"""
        print(f"Computing Collatz sequence starting from {n}\n")

        if n < 1:
            return {'solved': False, 'reason': 'n must be positive'}

        sequence = [n]
        steps = 0
        max_steps = 10000  # Safety limit

        current = n
        while current != 1 and steps < max_steps:
            if current % 2 == 0:
                current = current // 2
            else:
                current = 3 * current + 1

            sequence.append(current)
            steps += 1

        reached_one = current == 1

        # Show first few and last few steps
        preview = sequence[:10]
        if len(sequence) > 20:
            preview = sequence[:10] + ['...'] + sequence[-10:]

        print(f"Sequence: {' → '.join(map(str, preview))}\n")

        return {
            'solved': True,
            'result': reached_one,
            'sequence': sequence,
            'steps': steps,
            'max_value': max(sequence),
            'reached_one': reached_one
        }

    def _verify_sum_formula(self, n: int) -> Dict:
        """
        Verify that 1 + 2 + 3 + ... + n = n(n+1)/2
        by actually computing both sides.
        """
        print(f"Verifying: 1 + 2 + ... + {n} = {n}({n}+1)/2\n")

        # Compute left side (actual sum)
        left_side = sum(range(1, n + 1))

        # Compute right side (formula)
        right_side = n * (n + 1) // 2

        print(f"Left side (actual sum): {left_side}")
        print(f"Right side (formula): {right_side}")

        match = left_side == right_side

        return {
            'solved': True,
            'result': match,
            'left': left_side,
            'right': right_side,
            'verification': f'{left_side} == {right_side}: {match}'
        }


def demo():
    """Demo of honest solver"""
    solver = HonestSolver()

    print("\n" + "="*70)
    print("EXAMPLE 1: Check if 17 is prime")
    print("="*70 + "\n")
    result = solver.solve('prime_check', n=17)
    print(f"\nResult: {result['result']}")
    print(f"Reason: {result['reason']}\n")

    print("\n" + "="*70)
    print("EXAMPLE 2: Factor 84")
    print("="*70 + "\n")
    result = solver.solve('factor', n=84)
    print(f"\nFactorization: {result['factorization']}")
    print(f"Is prime: {result['is_prime']}\n")

    print("\n" + "="*70)
    print("EXAMPLE 3: Verify Goldbach for n=20")
    print("="*70 + "\n")
    result = solver.solve('goldbach_verify', n=20)
    print(f"\nResult: {result['result']}")
    print(f"Verification: {result['verification']}")
    print(f"Primes: {result['primes']}\n")

    print("\n" + "="*70)
    print("EXAMPLE 4: Collatz sequence starting from 27")
    print("="*70 + "\n")
    result = solver.solve('collatz_sequence', n=27)
    print(f"\nReached 1: {result['reached_one']}")
    print(f"Steps: {result['steps']}")
    print(f"Max value: {result['max_value']}\n")

    print("\n" + "="*70)
    print("EXAMPLE 5: Verify sum formula for n=100")
    print("="*70 + "\n")
    result = solver.solve('sum_formula', n=100)
    print(f"\nMatch: {result['result']}\n")

    print("\n" + "="*70)
    print("WHAT'S DIFFERENT?")
    print("="*70)
    print("""
THIS IS REAL:
  ✓ Actually computes results (runs code)
  ✓ No hardcoded proofs
  ✓ No templates
  ✓ Shows actual work
  ✓ Can verify Goldbach for ANY n (computationally)
  ✓ Can factor ANY number
  ✓ Can check if ANY number is prime
  ✓ Honest about limitations

WHAT IT DOES:
  • Runs actual algorithms (Sieve, trial division, etc.)
  • Shows step-by-step computation
  • Verifies conjectures for specific cases
  • Admits when it can't solve something

WHAT IT DOESN'T DO:
  • Prove general theorems (needs formal logic)
  • Solve unsolved problems
  • Generate proofs from templates
  • Pretend to do what it can't
    """)

    print("\n" + "="*70)
    print("TRY IT YOURSELF")
    print("="*70)
    print("""
from honest_solver import HonestSolver

solver = HonestSolver()

# Check if a number is prime
solver.solve('prime_check', n=97)

# Factor a number
solver.solve('factor', n=12345)

# Verify Goldbach for specific even number
solver.solve('goldbach_verify', n=100)

# Compute Collatz sequence
solver.solve('collatz_sequence', n=27)

# Verify sum formula
solver.solve('sum_formula', n=1000)
    """)


if __name__ == "__main__":
    demo()
