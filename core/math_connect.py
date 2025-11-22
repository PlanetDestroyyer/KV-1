"""
MathConnect - Mathematical Connection Engine

The insight: Most math already exists scattered across papers, books, and web.
The problem: We don't know how these theorems CONNECT.
The solution: AI that learns theorems, stores as equations, finds connections.

Architecture:
1. Web Search → Find theorems and formulas
2. Math Parser → Text to symbolic equations
3. Equation DB → Store all known math
4. Connection Finder → Which theorems relate?
5. Composer → Combine theorems to derive new results

Example:
  Learn: Pythagorean theorem (a² + b² = c²)
  Learn: Trigonometric identity (sin²θ + cos²θ = 1)
  Connect: They're the SAME THING! (unit circle)
  Derive: New relationships automatically

This is how AI discovers connections humans miss!
"""

from __future__ import annotations

import re
import sympy
from sympy import symbols, sympify, simplify, Eq, solve
from sympy.parsing.latex import parse_latex
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

try:
    from core.web_researcher import WebResearcher
    from core.llm import LLMBridge
except ImportError:
    WebResearcher = None
    LLMBridge = None


@dataclass
class MathTheorem:
    """A mathematical theorem stored symbolically."""
    name: str
    equation: sympy.Expr  # Symbolic expression
    latex: str  # Original LaTeX representation
    source: str  # Where it came from
    domain: str  # algebra, calculus, geometry, etc.
    related_theorems: List[str]  # Connected theorems
    learned_at: str

    def __str__(self):
        return f"{self.name}: {self.equation}"


class MathParser:
    """Parse mathematical text to symbolic equations."""

    def __init__(self, llm: Optional[LLMBridge] = None):
        self.llm = llm

    def parse_text_to_equation(self, text: str) -> Optional[sympy.Expr]:
        """
        Parse natural language math to SymPy equation.

        Examples:
          "a squared plus b squared equals c squared"
          → a**2 + b**2 = c**2

          "the sum from n equals 1 to infinity of 1 over n squared equals pi squared over 6"
          → Sum(1/n**2, (n, 1, oo)) = pi**2/6
        """
        # Try direct parsing common patterns
        equation = self._parse_common_patterns(text)
        if equation is not None:
            return equation

        # Try LaTeX parsing if looks like LaTeX
        if '\\' in text or '{' in text:
            try:
                return parse_latex(text)
            except:
                pass

        # Fall back to LLM to help parse
        if self.llm:
            return self._parse_with_llm(text)

        return None

    def _parse_common_patterns(self, text: str) -> Optional[sympy.Expr]:
        """Parse common mathematical phrasings."""
        text = text.lower().strip()

        # Pattern: "sin squared theta plus cos squared theta equals one"
        if 'sin' in text and 'cos' in text and 'squared' in text:
            try:
                theta = symbols('theta', real=True)
                return Eq(sympy.sin(theta)**2 + sympy.cos(theta)**2, 1)
            except:
                pass

        # Pattern: "area equals pi times r squared"
        if 'area' in text and 'pi' in text and 'r' in text and 'squared' in text:
            try:
                A, r = symbols('A r', real=True, positive=True)
                return Eq(A, sympy.pi * r**2)
            except:
                pass

        # Issue #7: Pattern: "c equals 2 times pi times r" or "circumference" (more flexible)
        if re.search(r'circumference|C\s*[=:≈]', text, re.I) and re.search(r'pi|π', text, re.I):
            try:
                C, r = symbols('C r', real=True, positive=True)
                return Eq(C, 2 * sympy.pi * r)
            except:
                pass

        # Pattern: quadratic formula
        if 'quadratic' in text or ('negative b' in text and 'square root' in text):
            try:
                x, a, b, c = symbols('x a b c', real=True)
                # x = (-b ± √(b²-4ac)) / 2a
                discriminant = sympy.sqrt(b**2 - 4*a*c)
                return Eq(x, (-b + discriminant) / (2*a))
            except:
                pass

        # Pattern: "a squared plus b squared equals c squared" (Pythagorean)
        if 'pythagorean' in text or (re.search(r'\ba\b.*squared.*\bb\b.*squared.*\bc\b.*squared', text)):
            try:
                a, b, c = symbols('a b c', real=True, positive=True)
                return Eq(a**2 + b**2, c**2)
            except:
                pass

        # Try direct sympify with word replacements
        try:
            # Replace common words with symbols
            equation_text = text.replace('equals', '==')
            equation_text = equation_text.replace('squared', '**2')
            equation_text = equation_text.replace('cubed', '**3')
            equation_text = equation_text.replace(' times ', '*')
            equation_text = equation_text.replace(' plus ', '+')
            equation_text = equation_text.replace(' minus ', '-')
            equation_text = equation_text.replace(' over ', '/')
            return sympify(equation_text)
        except:
            pass

        return None

    def _parse_with_llm(self, text: str) -> Optional[sympy.Expr]:
        """Use LLM to help parse complex mathematical expressions."""
        if not self.llm:
            return None

        system_prompt = "You are a math parser. Convert natural language to SymPy Python code."
        user_input = f"""Convert this to SymPy code:
"{text}"

Output ONLY the SymPy code, nothing else. Use symbols() for variables.
Example: "a squared plus b squared" → Eq(a**2 + b**2, c**2)"""

        try:
            result = self.llm.generate(system_prompt, user_input, execute=True)
            code = result.get("text", "")

            # Try to execute the SymPy code
            # (In production, use safer evaluation!)
            local_vars = {}
            exec(f"from sympy import *\n{code}", {}, local_vars)

            # Find the equation in local_vars
            for val in local_vars.values():
                if isinstance(val, (sympy.Expr, sympy.Eq)):
                    return val
        except:
            pass

        return None


class ConnectionFinder:
    """Find connections between mathematical theorems."""

    def __init__(self):
        self.theorems: Dict[str, MathTheorem] = {}
        self.connection_graph: Dict[str, Set[str]] = {}  # theorem → related theorems

    def add_theorem(self, theorem: MathTheorem):
        """Add theorem and find connections to existing ones."""
        self.theorems[theorem.name] = theorem
        self.connection_graph[theorem.name] = set()

        # Find connections to all existing theorems
        for existing_name, existing_theorem in self.theorems.items():
            if existing_name == theorem.name:
                continue

            # Check if they're related
            if self._are_related(theorem, existing_theorem):
                self.connection_graph[theorem.name].add(existing_name)
                self.connection_graph[existing_name].add(theorem.name)

                print(f"[Connection] {theorem.name} ↔ {existing_name}")

    def _are_related(self, t1: MathTheorem, t2: MathTheorem) -> bool:
        """Check if two theorems are mathematically related."""

        # Same domain?
        if t1.domain == t2.domain:
            # Check if they share symbols
            symbols1 = t1.equation.free_symbols
            symbols2 = t2.equation.free_symbols

            if symbols1 & symbols2:  # Intersection
                return True

        # Structural similarity (both have same form?)
        # E.g., both are Eq(something**2, something_else)
        if self._structurally_similar(t1.equation, t2.equation):
            return True

        # Can one be derived from the other?
        if self._can_derive(t1.equation, t2.equation):
            return True

        return False

    def _structurally_similar(self, eq1: sympy.Expr, eq2: sympy.Expr) -> bool:
        """Check if equations have similar structure."""
        # Simple heuristic: same operators at top level
        try:
            if type(eq1) == type(eq2):
                # Both are same type (Add, Mul, Pow, etc.)
                return True
        except:
            pass

        return False

    def _can_derive(self, eq1: sympy.Expr, eq2: sympy.Expr) -> bool:
        """Check if eq2 can be derived from eq1 (or vice versa)."""
        try:
            # Try simplifying one to get the other
            if simplify(eq1 - eq2) == 0:
                return True

            # Try substitution
            # (This is expensive, so we're conservative)
        except:
            pass

        return False

    def find_path(self, theorem_a: str, theorem_b: str) -> Optional[List[str]]:
        """
        Find chain of theorems connecting A to B.

        Returns:
          [theorem_a, intermediate1, intermediate2, ..., theorem_b]
        """
        # BFS to find shortest path
        from collections import deque

        if theorem_a not in self.theorems or theorem_b not in self.theorems:
            return None

        queue = deque([(theorem_a, [theorem_a])])
        visited = {theorem_a}

        while queue:
            current, path = queue.popleft()

            if current == theorem_b:
                return path

            for neighbor in self.connection_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


class TheoremComposer:
    """Compose theorems to derive new results."""

    def __init__(self):
        self.derived_theorems: List[MathTheorem] = []

    def compose(self, theorem1: MathTheorem, theorem2: MathTheorem) -> Optional[MathTheorem]:
        """
        Combine two theorems to derive something new.

        Examples:
          Pythagorean: a² + b² = c²
          Trig identity: sin²θ + cos²θ = 1
          → Derive: On unit circle (c=1), a=cosθ, b=sinθ
        """
        eq1 = theorem1.equation
        eq2 = theorem2.equation

        # Try various composition strategies
        derived = None

        # Strategy 1: Substitution
        derived = self._try_substitution(eq1, eq2, theorem1.name, theorem2.name)
        if derived:
            return derived

        # Strategy 2: Addition/Subtraction
        derived = self._try_addition(eq1, eq2, theorem1.name, theorem2.name)
        if derived:
            return derived

        # Strategy 3: Multiplication/Division
        derived = self._try_multiplication(eq1, eq2, theorem1.name, theorem2.name)
        if derived:
            return derived

        return None

    def _try_substitution(self, eq1, eq2, name1, name2) -> Optional[MathTheorem]:
        """Try substituting one equation into another."""
        try:
            # If eq1 is Eq(a, b), substitute a with b in eq2
            if isinstance(eq1, Eq):
                lhs, rhs = eq1.args
                # Substitute lhs with rhs in eq2
                substituted = eq2.subs(lhs, rhs)
                simplified = simplify(substituted)

                if simplified != eq2:  # Something changed
                    return MathTheorem(
                        name=f"{name1}_into_{name2}",
                        equation=simplified,
                        latex=sympy.latex(simplified),
                        source=f"Derived from {name1} and {name2}",
                        domain="derived",
                        related_theorems=[name1, name2],
                        learned_at=datetime.now().isoformat()
                    )
        except:
            pass

        return None

    def _try_addition(self, eq1, eq2, name1, name2) -> Optional[MathTheorem]:
        """Try adding equations."""
        try:
            if isinstance(eq1, Eq) and isinstance(eq2, Eq):
                # Add both sides
                new_lhs = eq1.lhs + eq2.lhs
                new_rhs = eq1.rhs + eq2.rhs
                combined = Eq(simplify(new_lhs), simplify(new_rhs))

                return MathTheorem(
                    name=f"{name1}_plus_{name2}",
                    equation=combined,
                    latex=sympy.latex(combined),
                    source=f"Sum of {name1} and {name2}",
                    domain="derived",
                    related_theorems=[name1, name2],
                    learned_at=datetime.now().isoformat()
                )
        except:
            pass

        return None

    def _try_multiplication(self, eq1, eq2, name1, name2) -> Optional[MathTheorem]:
        """Try multiplying equations."""
        try:
            if isinstance(eq1, Eq) and isinstance(eq2, Eq):
                new_lhs = eq1.lhs * eq2.lhs
                new_rhs = eq1.rhs * eq2.rhs
                combined = Eq(simplify(new_lhs), simplify(new_rhs))

                return MathTheorem(
                    name=f"{name1}_times_{name2}",
                    equation=combined,
                    latex=sympy.latex(combined),
                    source=f"Product of {name1} and {name2}",
                    domain="derived",
                    related_theorems=[name1, name2],
                    learned_at=datetime.now().isoformat()
                )
        except:
            pass

        return None


class MathConnect:
    """
    Main system: Learn theorems from web, find connections, derive new results.

    This is the "thinking in math" system that connects existing knowledge.
    """

    def __init__(self, llm: Optional[LLMBridge] = None, web: Optional[WebResearcher] = None):
        self.parser = MathParser(llm)
        self.connection_finder = ConnectionFinder()
        self.composer = TheoremComposer()
        self.llm = llm
        self.web = web

        print("[+] MathConnect initialized")
        print("    - Learns theorems from web")
        print("    - Finds connections between math")
        print("    - Derives new results by composition")

    def learn_theorem_from_text(self, name: str, text: str, domain: str = "unknown") -> bool:
        """
        Learn a theorem from natural language.

        Example:
          learn_theorem_from_text(
              "pythagorean",
              "a squared plus b squared equals c squared",
              "geometry"
          )
        """
        equation = self.parser.parse_text_to_equation(text)

        # Issue #11: Validate equation type (not just None check)
        if equation is None or not isinstance(equation, (sympy.Expr, sympy.Eq, sympy.Basic)):
            print(f"[!] Could not parse to valid equation: {text}")
            print(f"[!] Got type: {type(equation)}")
            return False

        theorem = MathTheorem(
            name=name,
            equation=equation,
            latex=sympy.latex(equation),
            source="manual",
            domain=domain,
            related_theorems=[],
            learned_at=datetime.now().isoformat()
        )

        self.connection_finder.add_theorem(theorem)
        print(f"[Learned] {name}: {equation}")

        # Try composing with existing theorems
        self._try_compose_with_all(theorem)

        return True

    def search_and_learn(self, topic: str, max_theorems: int = 5) -> int:
        """
        Search web for theorems on topic and learn them.

        Example:
          search_and_learn("pythagorean theorem")
          → Finds theorem, learns it, finds connections
        """
        if not self.web:
            print("[!] No web researcher available")
            return 0

        # Search for topic
        result = self.web.fetch(f"{topic} theorem formula", mode="scrape")

        if not result or not result.text:
            print(f"[!] No results for: {topic}")
            return 0

        # Extract equations from text
        equations = self._extract_equations_from_text(result.text)

        learned = 0
        for i, eq_text in enumerate(equations[:max_theorems]):
            success = self.learn_theorem_from_text(
                name=f"{topic}_{i+1}",
                text=eq_text,
                domain=topic.split()[0]  # First word as domain
            )
            if success:
                learned += 1

        print(f"[Search] Learned {learned} theorems about '{topic}'")
        return learned

    def _extract_equations_from_text(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        equations = []

        # Look for LaTeX-style equations
        latex_pattern = r'\$\$?(.*?)\$\$?'
        matches = re.findall(latex_pattern, text)
        equations.extend(matches)

        # Look for common equation patterns
        # "a + b = c" style
        eq_pattern = r'([a-z](\s*[\+\-\*/]\s*[a-z])*\s*=\s*[a-z0-9]+)'
        matches = re.findall(eq_pattern, text.lower())
        equations.extend([m[0] for m in matches])

        return equations[:10]  # Limit to avoid spam

    def _try_compose_with_all(self, new_theorem: MathTheorem):
        """Try composing new theorem with all existing ones."""
        # Create a list copy to avoid modifying dict during iteration
        existing_theorems = list(self.connection_finder.theorems.items())

        for existing_name, existing_theorem in existing_theorems:
            if existing_name == new_theorem.name:
                continue

            derived = self.composer.compose(new_theorem, existing_theorem)
            if derived:
                print(f"[Derived] {derived.name}: {derived.equation}")
                self.connection_finder.add_theorem(derived)

    def find_connection_path(self, theorem_a: str, theorem_b: str) -> Optional[List[str]]:
        """
        Find how two theorems are connected.

        Returns chain of theorems linking them.
        """
        return self.connection_finder.find_path(theorem_a, theorem_b)

    def get_graph(self) -> Dict[str, List[str]]:
        """Get the full connection graph."""
        return {
            name: list(connections)
            for name, connections in self.connection_finder.connection_graph.items()
        }

    def summarize(self) -> str:
        """Summary of learned mathematics."""
        num_theorems = len(self.connection_finder.theorems)
        num_connections = sum(len(c) for c in self.connection_finder.connection_graph.values()) // 2
        num_derived = len(self.composer.derived_theorems)

        lines = []
        lines.append("MathConnect Status:")
        lines.append(f"  Theorems learned: {num_theorems}")
        lines.append(f"  Connections found: {num_connections}")
        lines.append(f"  New theorems derived: {num_derived}")

        if self.connection_finder.theorems:
            lines.append("\nTop theorems:")
            for i, (name, theorem) in enumerate(list(self.connection_finder.theorems.items())[:5], 1):
                lines.append(f"  {i}. {name}: {theorem.equation}")

        return "\n".join(lines)
