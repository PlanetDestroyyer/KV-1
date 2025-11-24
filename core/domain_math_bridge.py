#!/usr/bin/env python3
"""
Domain-to-Math Bridge: Universal Problem Solver

Maps problems from ANY domain to mathematical structures, solves them using
true mathematical reasoning, then interprets results back to the original domain.

Key Insight: Mathematics is the language the universe is written in.
Every domain (physics, economics, politics, social science) can be mapped to math.

Architecture:
    Problem in Domain X
           ↓
    [Domain Recognizer] → Identify domain
           ↓
    [Structure Mapper] → Map to mathematical structure
           ↓
    [Problem Translator] → Formulate as math problem
           ↓
    [True Math Reasoner] → Solve using mathematical reasoning
           ↓
    [Result Interpreter] → Translate back to domain language
           ↓
    Solution in Domain X

Cross-Domain Transfer:
    If Problem A (physics) and Problem B (economics) map to the same
    mathematical structure, then solutions transfer between them!
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import re


class Domain(Enum):
    """Known domains that can be mapped to mathematics."""
    PHYSICS = "physics"
    ECONOMICS = "economics"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    SOCIAL_SCIENCE = "social_science"
    POLITICS = "politics"
    ENGINEERING = "engineering"
    COMPUTER_SCIENCE = "computer_science"
    MEDICINE = "medicine"
    ENVIRONMENTAL_SCIENCE = "environmental_science"
    PSYCHOLOGY = "psychology"
    LINGUISTICS = "linguistics"
    PURE_MATH = "pure_math"
    UNKNOWN = "unknown"


class MathStructure(Enum):
    """Mathematical structures that domains map to."""
    # Continuous Mathematics
    DIFFERENTIAL_EQUATIONS = "differential_equations"
    PARTIAL_DIFFERENTIAL_EQUATIONS = "partial_differential_equations"
    CALCULUS = "calculus"
    REAL_ANALYSIS = "real_analysis"

    # Algebraic Structures
    LINEAR_ALGEBRA = "linear_algebra"
    ABSTRACT_ALGEBRA = "abstract_algebra"
    GROUP_THEORY = "group_theory"
    VECTOR_SPACES = "vector_spaces"

    # Discrete Mathematics
    GRAPH_THEORY = "graph_theory"
    COMBINATORICS = "combinatorics"
    DISCRETE_OPTIMIZATION = "discrete_optimization"
    LOGIC = "logic"

    # Probabilistic/Statistical
    PROBABILITY_THEORY = "probability_theory"
    STATISTICS = "statistics"
    STOCHASTIC_PROCESSES = "stochastic_processes"
    MARKOV_CHAINS = "markov_chains"

    # Optimization
    OPTIMIZATION = "optimization"
    GAME_THEORY = "game_theory"
    DECISION_THEORY = "decision_theory"

    # Geometric
    TOPOLOGY = "topology"
    DIFFERENTIAL_GEOMETRY = "differential_geometry"
    EUCLIDEAN_GEOMETRY = "euclidean_geometry"

    # Dynamical Systems
    DYNAMICAL_SYSTEMS = "dynamical_systems"
    CHAOS_THEORY = "chaos_theory"
    CONTROL_THEORY = "control_theory"

    # Information/Computation
    INFORMATION_THEORY = "information_theory"
    COMPUTATIONAL_COMPLEXITY = "computational_complexity"
    AUTOMATA_THEORY = "automata_theory"


@dataclass
class DomainMapping:
    """Mapping from a domain to mathematical structures."""
    domain: Domain
    primary_structures: List[MathStructure]  # Main mathematical tools
    secondary_structures: List[MathStructure]  # Supporting tools
    key_concepts: Dict[str, str]  # Domain concept → Math concept
    example_problems: List[str]
    confidence: float = 1.0


@dataclass
class MathFormulation:
    """A problem translated to mathematical form."""
    original_problem: str
    domain: Domain
    math_structure: MathStructure

    # Mathematical representation
    variables: Dict[str, str]  # name → description
    constraints: List[str]
    objective: Optional[str]  # For optimization problems
    equations: List[str]
    initial_conditions: List[str]

    # Metadata
    assumptions: List[str]
    simplifications: List[str]

    def __str__(self):
        parts = [f"Problem: {self.original_problem}"]
        parts.append(f"Domain: {self.domain.value}")
        parts.append(f"Mathematical Structure: {self.math_structure.value}")

        if self.variables:
            parts.append("\nVariables:")
            for var, desc in self.variables.items():
                parts.append(f"  {var}: {desc}")

        if self.equations:
            parts.append("\nEquations:")
            for eq in self.equations:
                parts.append(f"  {eq}")

        if self.constraints:
            parts.append("\nConstraints:")
            for c in self.constraints:
                parts.append(f"  {c}")

        if self.objective:
            parts.append(f"\nObjective: {self.objective}")

        return "\n".join(parts)


@dataclass
class DomainSolution:
    """Solution translated back to domain language."""
    original_problem: str
    domain: Domain
    math_solution: str
    domain_interpretation: str
    confidence: float
    assumptions_used: List[str]
    limitations: List[str]
    related_concepts: List[str]


class DomainRecognizer:
    """Identifies which domain a problem belongs to."""

    def __init__(self):
        # Keywords that indicate specific domains
        self.domain_keywords = {
            Domain.PHYSICS: [
                "force", "energy", "momentum", "velocity", "acceleration",
                "mass", "gravity", "friction", "wave", "particle",
                "electric", "magnetic", "quantum", "relativity", "thermodynamics",
                "motion", "kinetic", "potential", "field", "charge"
            ],
            Domain.ECONOMICS: [
                "price", "demand", "supply", "market", "cost", "profit",
                "utility", "equilibrium", "elasticity", "inflation", "gdp",
                "investment", "consumption", "production", "revenue", "interest",
                "stock", "bond", "trade", "tax", "subsidy"
            ],
            Domain.BIOLOGY: [
                "population", "species", "growth", "evolution", "gene",
                "protein", "cell", "organism", "reproduction", "mutation",
                "selection", "ecosystem", "predator", "prey", "bacteria",
                "virus", "immunity", "metabolism", "enzyme", "dna"
            ],
            Domain.POLITICS: [
                "vote", "election", "candidate", "policy", "coalition",
                "party", "government", "power", "majority", "minority",
                "parliament", "congress", "senate", "representative", "democracy",
                "legislation", "ballot", "campaign", "constituent", "referendum"
            ],
            Domain.SOCIAL_SCIENCE: [
                "society", "group", "network", "influence", "behavior",
                "community", "relationship", "interaction", "culture", "norm",
                "institution", "organization", "hierarchy", "status", "role",
                "trust", "cooperation", "conflict", "inequality", "mobility"
            ],
            Domain.CHEMISTRY: [
                "molecule", "atom", "reaction", "bond", "element",
                "compound", "ion", "electron", "proton", "neutron",
                "oxidation", "reduction", "catalyst", "equilibrium", "acid",
                "base", "ph", "solvent", "solution", "concentration"
            ],
            Domain.ENGINEERING: [
                "design", "system", "component", "efficiency", "optimization",
                "constraint", "load", "stress", "strain", "material",
                "circuit", "signal", "control", "feedback", "stability"
            ],
        }

    def recognize(self, problem: str) -> Tuple[Domain, float]:
        """
        Identify domain from problem text.

        Returns:
            (domain, confidence) where confidence is 0.0-1.0
        """
        problem_lower = problem.lower()

        # Count keyword matches for each domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in problem_lower)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return Domain.UNKNOWN, 0.0

        # Get domain with highest score
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        domain, score = best_domain

        # Calculate confidence (normalize by max possible score)
        max_keywords = len(self.domain_keywords[domain])
        confidence = min(score / 5.0, 1.0)  # 5+ keywords = full confidence

        return domain, confidence


class StructureMapper:
    """Maps domain problems to mathematical structures."""

    def __init__(self):
        # Define mappings from domains to mathematical structures
        self.domain_mappings: Dict[Domain, DomainMapping] = {
            Domain.PHYSICS: DomainMapping(
                domain=Domain.PHYSICS,
                primary_structures=[
                    MathStructure.DIFFERENTIAL_EQUATIONS,
                    MathStructure.VECTOR_SPACES,
                    MathStructure.CALCULUS,
                ],
                secondary_structures=[
                    MathStructure.LINEAR_ALGEBRA,
                    MathStructure.DIFFERENTIAL_GEOMETRY,
                ],
                key_concepts={
                    "motion": "differential equations (Newton's laws)",
                    "energy": "conservation laws (scalar fields)",
                    "field": "vector fields (differential geometry)",
                    "wave": "partial differential equations",
                    "quantum": "linear operators on Hilbert spaces",
                },
                example_problems=[
                    "Projectile motion",
                    "Simple harmonic oscillator",
                    "Electromagnetic wave propagation",
                ]
            ),

            Domain.ECONOMICS: DomainMapping(
                domain=Domain.ECONOMICS,
                primary_structures=[
                    MathStructure.OPTIMIZATION,
                    MathStructure.GAME_THEORY,
                    MathStructure.CALCULUS,
                ],
                secondary_structures=[
                    MathStructure.STATISTICS,
                    MathStructure.DYNAMICAL_SYSTEMS,
                ],
                key_concepts={
                    "market equilibrium": "fixed point of dynamical system",
                    "utility maximization": "constrained optimization",
                    "strategic interaction": "game theory (Nash equilibrium)",
                    "price dynamics": "differential equations",
                    "risk": "probability theory",
                },
                example_problems=[
                    "Supply and demand equilibrium",
                    "Portfolio optimization",
                    "Auction design",
                ]
            ),

            Domain.BIOLOGY: DomainMapping(
                domain=Domain.BIOLOGY,
                primary_structures=[
                    MathStructure.DYNAMICAL_SYSTEMS,
                    MathStructure.DIFFERENTIAL_EQUATIONS,
                    MathStructure.STOCHASTIC_PROCESSES,
                ],
                secondary_structures=[
                    MathStructure.GRAPH_THEORY,
                    MathStructure.OPTIMIZATION,
                ],
                key_concepts={
                    "population dynamics": "differential equations (Lotka-Volterra)",
                    "evolution": "optimization on fitness landscape",
                    "gene networks": "graph theory + dynamical systems",
                    "random mutations": "stochastic processes",
                    "epidemic spread": "SIR model (differential equations)",
                },
                example_problems=[
                    "Predator-prey dynamics",
                    "Epidemic modeling",
                    "Genetic drift",
                ]
            ),

            Domain.POLITICS: DomainMapping(
                domain=Domain.POLITICS,
                primary_structures=[
                    MathStructure.GAME_THEORY,
                    MathStructure.GRAPH_THEORY,
                    MathStructure.COMBINATORICS,
                ],
                secondary_structures=[
                    MathStructure.OPTIMIZATION,
                    MathStructure.DECISION_THEORY,
                ],
                key_concepts={
                    "voting": "social choice theory (Arrow's theorem)",
                    "coalition formation": "cooperative game theory",
                    "power distribution": "weighted voting games",
                    "influence networks": "graph theory",
                    "strategic voting": "non-cooperative game theory",
                },
                example_problems=[
                    "Fair voting systems",
                    "Coalition stability",
                    "Gerrymandering detection",
                ]
            ),

            Domain.SOCIAL_SCIENCE: DomainMapping(
                domain=Domain.SOCIAL_SCIENCE,
                primary_structures=[
                    MathStructure.GRAPH_THEORY,
                    MathStructure.STATISTICS,
                    MathStructure.GAME_THEORY,
                ],
                secondary_structures=[
                    MathStructure.STOCHASTIC_PROCESSES,
                    MathStructure.DYNAMICAL_SYSTEMS,
                ],
                key_concepts={
                    "social networks": "graph theory",
                    "influence spread": "epidemic models on networks",
                    "cooperation": "evolutionary game theory",
                    "inequality": "statistical distributions (Gini coefficient)",
                    "segregation": "Schelling model (cellular automata)",
                },
                example_problems=[
                    "Information diffusion in networks",
                    "Trust formation",
                    "Social norm emergence",
                ]
            ),
        }

    def map_to_structure(self, problem: str, domain: Domain) -> MathStructure:
        """Determine which mathematical structure best fits the problem."""

        if domain not in self.domain_mappings:
            return MathStructure.OPTIMIZATION  # Default fallback

        mapping = self.domain_mappings[domain]
        problem_lower = problem.lower()

        # Check for keywords that suggest specific structures
        structure_keywords = {
            MathStructure.DIFFERENTIAL_EQUATIONS: [
                "change", "rate", "derivative", "differential", "flow",
                "dynamics", "evolve", "motion", "velocity"
            ],
            MathStructure.OPTIMIZATION: [
                "maximize", "minimize", "optimal", "best", "efficient",
                "cost", "benefit", "utility", "profit", "loss"
            ],
            MathStructure.GAME_THEORY: [
                "strategy", "player", "payoff", "equilibrium", "competition",
                "cooperation", "nash", "dominance", "coalition"
            ],
            MathStructure.GRAPH_THEORY: [
                "network", "connection", "link", "node", "path",
                "graph", "connected", "neighbor", "degree", "flow"
            ],
            MathStructure.PROBABILITY_THEORY: [
                "random", "probability", "chance", "risk", "uncertain",
                "stochastic", "expected", "variance", "distribution"
            ],
        }

        # Score each structure
        scores = {}
        for structure in mapping.primary_structures:
            if structure in structure_keywords:
                score = sum(1 for kw in structure_keywords[structure]
                           if kw in problem_lower)
                if score > 0:
                    scores[structure] = score

        # Return structure with highest score, or first primary structure
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return mapping.primary_structures[0]


class ProblemTranslator:
    """Translates domain problems to mathematical formulations."""

    def __init__(self):
        self.structure_mapper = StructureMapper()

    def translate(self, problem: str, domain: Domain) -> MathFormulation:
        """
        Translate a domain problem to mathematical formulation.

        This is the core of the bridge: converting natural language
        domain problems into precise mathematical statements.
        """

        # Determine appropriate mathematical structure
        math_structure = self.structure_mapper.map_to_structure(problem, domain)

        # Dispatch to structure-specific translator
        if math_structure == MathStructure.DIFFERENTIAL_EQUATIONS:
            return self._translate_to_differential_equations(problem, domain)
        elif math_structure == MathStructure.OPTIMIZATION:
            return self._translate_to_optimization(problem, domain)
        elif math_structure == MathStructure.GAME_THEORY:
            return self._translate_to_game_theory(problem, domain)
        elif math_structure == MathStructure.GRAPH_THEORY:
            return self._translate_to_graph_theory(problem, domain)
        else:
            return self._translate_generic(problem, domain, math_structure)

    def _translate_to_differential_equations(
        self, problem: str, domain: Domain
    ) -> MathFormulation:
        """Translate to differential equation formulation."""

        # Extract key quantities that change over time
        variables = {}
        equations = []
        initial_conditions = []

        if "population" in problem.lower():
            variables["P(t)"] = "Population at time t"
            variables["t"] = "Time"
            equations.append("dP/dt = rP(1 - P/K)")  # Logistic growth
            initial_conditions.append("P(0) = P₀")

        elif "motion" in problem.lower() or "velocity" in problem.lower():
            variables["x(t)"] = "Position at time t"
            variables["v(t)"] = "Velocity at time t"
            variables["a(t)"] = "Acceleration at time t"
            equations.append("v(t) = dx/dt")
            equations.append("a(t) = dv/dt = d²x/dt²")

        return MathFormulation(
            original_problem=problem,
            domain=domain,
            math_structure=MathStructure.DIFFERENTIAL_EQUATIONS,
            variables=variables,
            constraints=[],
            objective=None,
            equations=equations,
            initial_conditions=initial_conditions,
            assumptions=["Continuous time", "Deterministic dynamics"],
            simplifications=["Ignore random fluctuations"]
        )

    def _translate_to_optimization(
        self, problem: str, domain: Domain
    ) -> MathFormulation:
        """Translate to optimization problem."""

        variables = {}
        constraints = []
        objective = ""

        if "maximize" in problem.lower():
            objective = "maximize f(x)"
            variables["x"] = "Decision variable(s)"
        elif "minimize" in problem.lower():
            objective = "minimize f(x)"
            variables["x"] = "Decision variable(s)"

        # Extract constraints
        if "budget" in problem.lower() or "cost" in problem.lower():
            constraints.append("Σ cᵢxᵢ ≤ B (budget constraint)")

        return MathFormulation(
            original_problem=problem,
            domain=domain,
            math_structure=MathStructure.OPTIMIZATION,
            variables=variables,
            constraints=constraints,
            objective=objective,
            equations=[],
            initial_conditions=[],
            assumptions=["Objective is well-defined", "Constraints are feasible"],
            simplifications=[]
        )

    def _translate_to_game_theory(
        self, problem: str, domain: Domain
    ) -> MathFormulation:
        """Translate to game-theoretic formulation."""

        variables = {
            "N": "Set of players",
            "S": "Strategy spaces for each player",
            "u": "Utility functions"
        }

        equations = [
            "Nash equilibrium: s* such that ∀i: uᵢ(sᵢ*, s₋ᵢ*) ≥ uᵢ(sᵢ, s₋ᵢ*)"
        ]

        return MathFormulation(
            original_problem=problem,
            domain=domain,
            math_structure=MathStructure.GAME_THEORY,
            variables=variables,
            constraints=[],
            objective="Find Nash equilibrium",
            equations=equations,
            initial_conditions=[],
            assumptions=["Rational players", "Common knowledge of game"],
            simplifications=[]
        )

    def _translate_to_graph_theory(
        self, problem: str, domain: Domain
    ) -> MathFormulation:
        """Translate to graph-theoretic formulation."""

        variables = {
            "G": "Graph G = (V, E)",
            "V": "Set of vertices (nodes)",
            "E": "Set of edges (connections)"
        }

        return MathFormulation(
            original_problem=problem,
            domain=domain,
            math_structure=MathStructure.GRAPH_THEORY,
            variables=variables,
            constraints=[],
            objective=None,
            equations=[],
            initial_conditions=[],
            assumptions=["Graph structure captures relevant relationships"],
            simplifications=[]
        )

    def _translate_generic(
        self, problem: str, domain: Domain, structure: MathStructure
    ) -> MathFormulation:
        """Generic translation when specific method not available."""

        return MathFormulation(
            original_problem=problem,
            domain=domain,
            math_structure=structure,
            variables={"x": "Primary variable(s)"},
            constraints=[],
            objective=None,
            equations=[],
            initial_conditions=[],
            assumptions=[],
            simplifications=[]
        )


class ResultInterpreter:
    """Translates mathematical solutions back to domain language."""

    def interpret(
        self,
        math_solution: str,
        formulation: MathFormulation
    ) -> DomainSolution:
        """
        Interpret mathematical solution in domain-specific terms.

        This reverses the translation: mathematical answer → domain meaning.
        """

        domain = formulation.domain

        # Dispatch to domain-specific interpreter
        if domain == Domain.PHYSICS:
            interpretation = self._interpret_physics(math_solution, formulation)
        elif domain == Domain.ECONOMICS:
            interpretation = self._interpret_economics(math_solution, formulation)
        elif domain == Domain.BIOLOGY:
            interpretation = self._interpret_biology(math_solution, formulation)
        elif domain == Domain.POLITICS:
            interpretation = self._interpret_politics(math_solution, formulation)
        elif domain == Domain.SOCIAL_SCIENCE:
            interpretation = self._interpret_social_science(math_solution, formulation)
        else:
            interpretation = self._interpret_generic(math_solution, formulation)

        return DomainSolution(
            original_problem=formulation.original_problem,
            domain=domain,
            math_solution=math_solution,
            domain_interpretation=interpretation,
            confidence=0.8,  # TODO: Calculate actual confidence
            assumptions_used=formulation.assumptions,
            limitations=[],
            related_concepts=[]
        )

    def _interpret_physics(self, solution: str, formulation: MathFormulation) -> str:
        """Interpret solution in physics terms."""

        interpretation = "Physical Interpretation:\n"

        if "differential equation" in solution.lower():
            interpretation += "The solution describes how the physical system evolves over time. "

        if "equilibrium" in solution.lower():
            interpretation += "The system reaches a stable state where forces balance. "

        return interpretation

    def _interpret_economics(self, solution: str, formulation: MathFormulation) -> str:
        """Interpret solution in economic terms."""

        interpretation = "Economic Interpretation:\n"

        if "equilibrium" in solution.lower():
            interpretation += "The market reaches equilibrium where supply equals demand. "

        if "optimal" in solution.lower() or "maximize" in solution.lower():
            interpretation += "This represents the best allocation of resources given constraints. "

        return interpretation

    def _interpret_biology(self, solution: str, formulation: MathFormulation) -> str:
        """Interpret solution in biological terms."""

        interpretation = "Biological Interpretation:\n"

        if "population" in formulation.original_problem.lower():
            interpretation += "The population dynamics show how species abundance changes over time. "

        if "equilibrium" in solution.lower():
            interpretation += "The ecosystem reaches a balanced state. "

        return interpretation

    def _interpret_politics(self, solution: str, formulation: MathFormulation) -> str:
        """Interpret solution in political terms."""

        interpretation = "Political Interpretation:\n"

        if "coalition" in formulation.original_problem.lower():
            interpretation += "The solution identifies stable political alliances. "

        if "voting" in formulation.original_problem.lower():
            interpretation += "This shows the outcome of the voting process. "

        return interpretation

    def _interpret_social_science(self, solution: str, formulation: MathFormulation) -> str:
        """Interpret solution in social science terms."""

        interpretation = "Social Science Interpretation:\n"

        if "network" in formulation.original_problem.lower():
            interpretation += "The network structure reveals patterns of social connections. "

        return interpretation

    def _interpret_generic(self, solution: str, formulation: MathFormulation) -> str:
        """Generic interpretation."""
        return f"The mathematical solution provides insights into {formulation.domain.value}."


class DomainMathBridge:
    """
    Main interface: Universal problem solver via mathematical reasoning.

    Takes ANY domain problem → Maps to math → Solves → Interprets back.
    """

    def __init__(self, true_math_reasoner=None):
        self.domain_recognizer = DomainRecognizer()
        self.structure_mapper = StructureMapper()
        self.problem_translator = ProblemTranslator()
        self.result_interpreter = ResultInterpreter()
        self.true_math_reasoner = true_math_reasoner

        # Cross-domain knowledge: problems that share mathematical structure
        self.cross_domain_mappings: Dict[MathStructure, List[Domain]] = {}
        self._build_cross_domain_mappings()

    def _build_cross_domain_mappings(self):
        """Build mappings showing which domains use which math structures."""

        for mapping in self.structure_mapper.domain_mappings.values():
            for structure in mapping.primary_structures + mapping.secondary_structures:
                if structure not in self.cross_domain_mappings:
                    self.cross_domain_mappings[structure] = []
                if mapping.domain not in self.cross_domain_mappings[structure]:
                    self.cross_domain_mappings[structure].append(mapping.domain)

    def solve(self, problem: str) -> DomainSolution:
        """
        Solve a problem from ANY domain using mathematical reasoning.

        Steps:
        1. Recognize domain
        2. Map to mathematical structure
        3. Translate to math formulation
        4. Solve using math reasoning
        5. Interpret back to domain language

        Returns:
            DomainSolution with interpretation
        """

        # Step 1: Recognize domain
        domain, confidence = self.domain_recognizer.recognize(problem)

        print(f"[Domain Recognition] {domain.value} (confidence: {confidence:.2f})")

        # Step 2: Translate to mathematical formulation
        formulation = self.problem_translator.translate(problem, domain)

        print(f"[Mathematical Structure] {formulation.math_structure.value}")
        print(f"\n{formulation}")

        # Step 3: Solve using mathematical reasoning
        # (This would use TrueMathReasoner if integrated)
        math_solution = self._solve_math_problem(formulation)

        print(f"\n[Mathematical Solution] {math_solution}")

        # Step 4: Interpret back to domain language
        solution = self.result_interpreter.interpret(math_solution, formulation)

        return solution

    def _solve_math_problem(self, formulation: MathFormulation) -> str:
        """
        Solve the mathematical formulation.

        In full integration, this would use TrueMathReasoner.
        For now, returns template solutions.
        """

        if formulation.math_structure == MathStructure.DIFFERENTIAL_EQUATIONS:
            return "Solution: x(t) = x₀ * exp(λt) (exponential growth/decay)"

        elif formulation.math_structure == MathStructure.OPTIMIZATION:
            return "Solution: x* = argmax f(x) subject to constraints"

        elif formulation.math_structure == MathStructure.GAME_THEORY:
            return "Solution: Nash equilibrium at s* where no player can improve"

        elif formulation.math_structure == MathStructure.GRAPH_THEORY:
            return "Solution: Connected components, shortest paths, centrality measures"

        else:
            return "Solution: [Mathematical solution would go here]"

    def find_analogies(self, problem: str) -> List[Tuple[Domain, str]]:
        """
        Find analogous problems in OTHER domains via shared math structure.

        This enables cross-domain transfer learning!

        Example:
            - "Epidemic spread" (biology) uses same math as
            - "Information diffusion" (social science)
            → Solution methods transfer!
        """

        # Recognize domain and map to structure
        domain, _ = self.domain_recognizer.recognize(problem)
        formulation = self.problem_translator.translate(problem, domain)
        structure = formulation.math_structure

        # Find other domains that use this structure
        related_domains = self.cross_domain_mappings.get(structure, [])

        analogies = []
        for other_domain in related_domains:
            if other_domain != domain:
                # Get example problems from that domain
                mapping = self.structure_mapper.domain_mappings.get(other_domain)
                if mapping and mapping.example_problems:
                    analogies.append((other_domain, mapping.example_problems[0]))

        return analogies

    def explain_mathematical_connection(
        self, problem1: str, problem2: str
    ) -> str:
        """
        Explain why two problems from different domains are mathematically related.

        Example:
            problem1: "Population growth" (biology)
            problem2: "Compound interest" (economics)
            → Both use exponential growth: P(t) = P₀ * exp(rt)
        """

        # Translate both problems
        domain1, _ = self.domain_recognizer.recognize(problem1)
        domain2, _ = self.domain_recognizer.recognize(problem2)

        form1 = self.problem_translator.translate(problem1, domain1)
        form2 = self.problem_translator.translate(problem2, domain2)

        if form1.math_structure == form2.math_structure:
            explanation = f"""
Both problems use the same mathematical structure: {form1.math_structure.value}

Problem 1 ({domain1.value}):
{form1.original_problem}
→ Maps to: {form1.equations if form1.equations else 'same structure'}

Problem 2 ({domain2.value}):
{form2.original_problem}
→ Maps to: {form2.equations if form2.equations else 'same structure'}

Because they share the same mathematical structure, solutions and insights
transfer between domains! This is the power of mathematical abstraction.
"""
            return explanation
        else:
            return f"These problems use different mathematical structures: {form1.math_structure.value} vs {form2.math_structure.value}"

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the bridge system."""

        return {
            "domains_supported": len(Domain) - 1,  # Exclude UNKNOWN
            "math_structures": len(MathStructure),
            "domain_mappings": len(self.structure_mapper.domain_mappings),
            "cross_domain_connections": sum(
                len(domains) for domains in self.cross_domain_mappings.values()
            )
        }


def demo_quick():
    """Quick demonstration of domain-to-math bridge."""

    print("="*70)
    print("DOMAIN-TO-MATH BRIDGE - Quick Demo")
    print("="*70)

    bridge = DomainMathBridge()

    # Test problems from different domains
    problems = [
        "How does population growth change over time in a limited environment?",
        "What is the optimal investment strategy to maximize returns?",
        "How do political coalitions form in multi-party systems?",
        "How does information spread through social networks?"
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n[Problem {i}] {problem}")
        print("-" * 70)

        solution = bridge.solve(problem)
        print(f"\n{solution.domain_interpretation}")

    # Show cross-domain analogies
    print("\n" + "="*70)
    print("CROSS-DOMAIN ANALOGIES")
    print("="*70)

    problem = "How does an epidemic spread through a population?"
    print(f"\nProblem: {problem}")
    analogies = bridge.find_analogies(problem)

    print("\nAnalogous problems in other domains:")
    for domain, example in analogies:
        print(f"  • {domain.value}: {example}")

    print("\n" + "="*70)


if __name__ == "__main__":
    demo_quick()
