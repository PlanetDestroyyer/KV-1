"""
Mathematical Primitives

The fundamental building blocks for mathematical reasoning.
Like letters and words for a child learning to write.

Contains:
- Axioms (foundational truths)
- Operations (what we can do)
- Theorems (known results)
- Definitions (what things mean)
"""

from typing import List, Dict, Set, Callable, Any
from dataclasses import dataclass
from enum import Enum
import sympy as sp
from sympy import symbols, Symbol, Integer, Rational, Prime
from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent
from sympy.logic.boolalg import ForAll, Exists


class MathDomain(Enum):
    """Mathematical domains"""
    NUMBER_THEORY = "number_theory"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    TOPOLOGY = "topology"


@dataclass
class MathAxiom:
    """A mathematical axiom - fundamental truth"""
    name: str
    domain: MathDomain
    symbolic_form: Any  # SymPy expression
    natural_language: str

    def __repr__(self):
        return f"Axiom({self.name}: {self.symbolic_form})"


@dataclass
class MathOperation:
    """A mathematical operation we can perform"""
    name: str
    operation: Callable
    domain: MathDomain
    description: str

    def apply(self, *args):
        """Apply the operation"""
        return self.operation(*args)

    def __repr__(self):
        return f"Operation({self.name})"


@dataclass
class MathTheorem:
    """A known mathematical theorem"""
    name: str
    domain: MathDomain
    statement: Any  # SymPy expression
    proof_sketch: str
    related_theorems: List[str]

    def __repr__(self):
        return f"Theorem({self.name})"


class MathematicalPrimitives:
    """
    Repository of mathematical primitives.

    These are the "letters and words" the system uses to explore.
    Like a child learning to write, we combine these primitives
    to construct proofs and discover new mathematics.
    """

    def __init__(self):
        # Symbolic variables
        self.n, self.m, self.k = symbols('n m k', integer=True)
        self.p, self.q = symbols('p q', integer=True, positive=True)
        self.x, self.y, self.z = symbols('x y z', real=True)
        self.a, self.b, self.c = symbols('a b c')

        # Initialize primitives
        self.axioms = self._init_axioms()
        self.operations = self._init_operations()
        self.theorems = self._init_theorems()
        self.definitions = self._init_definitions()

    def _init_axioms(self) -> Dict[str, MathAxiom]:
        """Initialize fundamental axioms"""
        axioms = {}

        # Number theory axioms
        axioms["peano_1"] = MathAxiom(
            name="Peano Axiom 1",
            domain=MathDomain.NUMBER_THEORY,
            symbolic_form=sp.sympify("1 in N"),  # 1 is a natural number
            natural_language="1 is a natural number"
        )

        axioms["peano_successor"] = MathAxiom(
            name="Peano Successor",
            domain=MathDomain.NUMBER_THEORY,
            symbolic_form="n in N implies n+1 in N",
            natural_language="Every natural number has a successor"
        )

        # Well-ordering principle
        axioms["well_ordering"] = MathAxiom(
            name="Well-Ordering Principle",
            domain=MathDomain.NUMBER_THEORY,
            symbolic_form="Every non-empty set of natural numbers has a least element",
            natural_language="Every non-empty subset of natural numbers has a minimum"
        )

        # Division algorithm
        axioms["division_algorithm"] = MathAxiom(
            name="Division Algorithm",
            domain=MathDomain.NUMBER_THEORY,
            symbolic_form="ForAll(a,b: exists q,r: a = b*q + r and 0 <= r < b)",
            natural_language="Every integer can be divided with quotient and remainder"
        )

        # Fundamental theorem of arithmetic (as axiom)
        axioms["fundamental_theorem_arithmetic"] = MathAxiom(
            name="Fundamental Theorem of Arithmetic",
            domain=MathDomain.NUMBER_THEORY,
            symbolic_form="Every integer > 1 has unique prime factorization",
            natural_language="Every integer has a unique decomposition into primes"
        )

        return axioms

    def _init_operations(self) -> Dict[str, MathOperation]:
        """Initialize mathematical operations"""
        ops = {}

        # Arithmetic operations
        ops["add"] = MathOperation(
            name="Addition",
            operation=lambda a, b: a + b,
            domain=MathDomain.ALGEBRA,
            description="Add two numbers or expressions"
        )

        ops["multiply"] = MathOperation(
            name="Multiplication",
            operation=lambda a, b: a * b,
            domain=MathDomain.ALGEBRA,
            description="Multiply two numbers or expressions"
        )

        ops["divide"] = MathOperation(
            name="Division",
            operation=lambda a, b: a / b if b != 0 else None,
            domain=MathDomain.ALGEBRA,
            description="Divide two numbers or expressions"
        )

        # Symbolic operations
        ops["factor"] = MathOperation(
            name="Factor",
            operation=sp.factor,
            domain=MathDomain.ALGEBRA,
            description="Factor an expression"
        )

        ops["expand"] = MathOperation(
            name="Expand",
            operation=sp.expand,
            domain=MathDomain.ALGEBRA,
            description="Expand an expression"
        )

        ops["simplify"] = MathOperation(
            name="Simplify",
            operation=sp.simplify,
            domain=MathDomain.ALGEBRA,
            description="Simplify an expression"
        )

        # Number theory operations
        ops["is_prime"] = MathOperation(
            name="Primality Test",
            operation=sp.isprime,
            domain=MathDomain.NUMBER_THEORY,
            description="Check if a number is prime"
        )

        ops["prime_factors"] = MathOperation(
            name="Prime Factorization",
            operation=sp.factorint,
            domain=MathDomain.NUMBER_THEORY,
            description="Find prime factorization"
        )

        ops["gcd"] = MathOperation(
            name="Greatest Common Divisor",
            operation=sp.gcd,
            domain=MathDomain.NUMBER_THEORY,
            description="Find GCD of two numbers"
        )

        ops["lcm"] = MathOperation(
            name="Least Common Multiple",
            operation=sp.lcm,
            domain=MathDomain.NUMBER_THEORY,
            description="Find LCM of two numbers"
        )

        # Logical operations
        ops["logical_and"] = MathOperation(
            name="Logical AND",
            operation=And,
            domain=MathDomain.LOGIC,
            description="Logical conjunction"
        )

        ops["logical_or"] = MathOperation(
            name="Logical OR",
            operation=Or,
            domain=MathDomain.LOGIC,
            description="Logical disjunction"
        )

        ops["implies"] = MathOperation(
            name="Implication",
            operation=Implies,
            domain=MathDomain.LOGIC,
            description="Logical implication"
        )

        ops["negation"] = MathOperation(
            name="Negation",
            operation=Not,
            domain=MathDomain.LOGIC,
            description="Logical negation"
        )

        # Calculus operations
        ops["derivative"] = MathOperation(
            name="Derivative",
            operation=lambda expr, var: sp.diff(expr, var),
            domain=MathDomain.CALCULUS,
            description="Take derivative"
        )

        ops["integral"] = MathOperation(
            name="Integral",
            operation=lambda expr, var: sp.integrate(expr, var),
            domain=MathDomain.CALCULUS,
            description="Integrate expression"
        )

        ops["limit"] = MathOperation(
            name="Limit",
            operation=lambda expr, var, point: sp.limit(expr, var, point),
            domain=MathDomain.CALCULUS,
            description="Compute limit"
        )

        return ops

    def _init_theorems(self) -> Dict[str, MathTheorem]:
        """Initialize known theorems"""
        theorems = {}

        # Euclid's theorem on infinitude of primes
        theorems["euclid_primes"] = MathTheorem(
            name="Euclid's Theorem",
            domain=MathDomain.NUMBER_THEORY,
            statement="There are infinitely many prime numbers",
            proof_sketch="Assume finitely many primes p1,...,pn. Consider N=p1*p2*...*pn+1. N is not divisible by any pi, so either N is prime or has a prime factor not in list. Contradiction.",
            related_theorems=["fundamental_theorem_arithmetic"]
        )

        # Pythagorean theorem
        p, q, r = symbols('p q r', positive=True)
        theorems["pythagorean"] = MathTheorem(
            name="Pythagorean Theorem",
            domain=MathDomain.GEOMETRY,
            statement=sp.Eq(p**2 + q**2, r**2),
            proof_sketch="For right triangle with legs a,b and hypotenuse c: a²+b²=c²",
            related_theorems=["euclidean_geometry"]
        )

        # Quadratic formula
        a, b, c, x = symbols('a b c x')
        theorems["quadratic_formula"] = MathTheorem(
            name="Quadratic Formula",
            domain=MathDomain.ALGEBRA,
            statement=sp.Eq(x, (-b + sp.sqrt(b**2 - 4*a*c))/(2*a)),
            proof_sketch="Solutions to ax²+bx+c=0 are x=(-b±√(b²-4ac))/(2a)",
            related_theorems=["completing_square"]
        )

        return theorems

    def _init_definitions(self) -> Dict[str, str]:
        """Initialize mathematical definitions"""
        return {
            "prime": "p is prime ⟺ p > 1 ∧ ∀d: (d|p → d∈{1,p})",
            "composite": "n is composite ⟺ n > 1 ∧ ∃d: (1 < d < n ∧ d|n)",
            "even": "n is even ⟺ ∃k: n = 2k",
            "odd": "n is odd ⟺ ∃k: n = 2k+1",
            "divides": "a|b ⟺ ∃k: b = ak",
            "gcd": "gcd(a,b) = max{d: d|a ∧ d|b}",
            "coprime": "gcd(a,b) = 1",
            "perfect_square": "n is perfect square ⟺ ∃k: n = k²",
        }

    def get_axioms_for_domain(self, domain: MathDomain) -> List[MathAxiom]:
        """Get all axioms for a specific domain"""
        return [ax for ax in self.axioms.values() if ax.domain == domain]

    def get_operations_for_domain(self, domain: MathDomain) -> List[MathOperation]:
        """Get all operations for a specific domain"""
        return [op for op in self.operations.values() if op.domain == domain]

    def get_theorems_for_domain(self, domain: MathDomain) -> List[MathTheorem]:
        """Get all theorems for a specific domain"""
        return [th for th in self.theorems.values() if th.domain == domain]

    def all_operations(self) -> List[MathOperation]:
        """Get all available operations"""
        return list(self.operations.values())

    def apply_operation(self, op_name: str, *args):
        """Apply a named operation"""
        if op_name not in self.operations:
            raise ValueError(f"Unknown operation: {op_name}")
        return self.operations[op_name].apply(*args)


class ProofSteps:
    """
    Common proof techniques and strategies.
    Like sentence structures for writing.
    """

    @staticmethod
    def direct_proof(hypothesis, conclusion):
        """Direct proof: P → Q"""
        return {
            'type': 'direct',
            'assume': hypothesis,
            'prove': conclusion,
            'strategy': 'Use definitions and axioms to derive conclusion from hypothesis'
        }

    @staticmethod
    def proof_by_contradiction(statement):
        """Proof by contradiction: ¬P leads to contradiction"""
        return {
            'type': 'contradiction',
            'assume': Not(statement),
            'goal': 'Derive contradiction',
            'strategy': 'Assume negation, derive impossibility, conclude original statement true'
        }

    @staticmethod
    def proof_by_induction(base_case, inductive_step):
        """Proof by induction on natural numbers"""
        return {
            'type': 'induction',
            'base': base_case,
            'step': inductive_step,
            'strategy': 'Prove P(1), then prove P(n) → P(n+1)'
        }

    @staticmethod
    def proof_by_contrapositive(hypothesis, conclusion):
        """Contrapositive: P → Q equivalent to ¬Q → ¬P"""
        return {
            'type': 'contrapositive',
            'original': Implies(hypothesis, conclusion),
            'contrapositive': Implies(Not(conclusion), Not(hypothesis)),
            'strategy': 'Prove ¬Q → ¬P instead of P → Q'
        }

    @staticmethod
    def case_analysis(cases):
        """Proof by cases: Prove for all possible cases"""
        return {
            'type': 'cases',
            'cases': cases,
            'strategy': 'Divide into exhaustive cases, prove each separately'
        }


# Global instance for easy access
MATH_PRIMITIVES = MathematicalPrimitives()
