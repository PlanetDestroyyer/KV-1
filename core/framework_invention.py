"""
Phase 4: Framework Invention Engine

Creates new mathematical frameworks when existing ones are insufficient.
This is TRUE mathematical creativity - inventing new ways to think about problems!

Architecture:
1. Recognize when existing frameworks are inadequate
2. Analyze the gap between problem requirements and available tools
3. Synthesize new frameworks by combining/extending existing ones
4. Validate the new framework through problem-solving
5. Store invented frameworks for future reuse

Example:
    Problem: "Describe continuous change in curved space"
    Existing: Linear algebra (flat space only)
    Invention: Differential geometry (generalizes to curved spaces)
    Result: New framework created!
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re


@dataclass
class MathematicalFramework:
    """Represents a mathematical framework (algebra, topology, etc.)."""

    name: str  # "differential geometry", "category theory"
    description: str  # What it does
    axioms: List[str]  # Foundational rules
    operations: List[str]  # Available operations
    objects: List[str]  # What it operates on
    applications: List[str]  # Where it's useful
    invented_at: str  # When created
    confidence: float  # How well-validated (0-1)
    parent_frameworks: List[str]  # What it extends/combines


@dataclass
class FrameworkGap:
    """Represents a gap in available frameworks."""

    problem_domain: str  # "curved space navigation"
    required_capabilities: List[str]  # "measure angles on spheres"
    missing_tools: List[str]  # "non-Euclidean distance"
    existing_frameworks: List[str]  # "linear algebra", "calculus"
    gap_severity: float  # How critical (0-1)


class FrameworkInventor:
    """
    Invents new mathematical frameworks when needed.

    This is the KEY to open-ended intelligence:
    - Don't just learn existing math
    - CREATE new math when you hit limitations!
    """

    def __init__(self, storage_path: str = "./invented_frameworks.json"):
        self.storage_path = storage_path
        self.frameworks: Dict[str, MathematicalFramework] = {}

        # Seed with known frameworks
        self._seed_known_frameworks()

        # Load previously invented frameworks
        self.load()

        print("[+] 🔬 Framework Inventor: Creates NEW math when needed!")

    def _seed_known_frameworks(self):
        """Initialize with fundamental mathematical frameworks."""
        known = [
            MathematicalFramework(
                name="arithmetic",
                description="Basic operations on numbers",
                axioms=["commutativity", "associativity", "distributivity"],
                operations=["add", "subtract", "multiply", "divide"],
                objects=["numbers"],
                applications=["counting", "measurement"],
                invented_at="ancient",
                confidence=1.0,
                parent_frameworks=[]
            ),
            MathematicalFramework(
                name="algebra",
                description="Symbolic manipulation and equation solving",
                axioms=["substitution", "equivalence", "inverse operations"],
                operations=["solve", "factor", "expand", "simplify"],
                objects=["variables", "equations", "expressions"],
                applications=["problem solving", "generalization"],
                invented_at="ancient",
                confidence=1.0,
                parent_frameworks=["arithmetic"]
            ),
            MathematicalFramework(
                name="calculus",
                description="Analysis of continuous change",
                axioms=["limits", "continuity", "differentiability"],
                operations=["differentiate", "integrate", "optimize"],
                objects=["functions", "derivatives", "integrals"],
                applications=["physics", "optimization", "rates of change"],
                invented_at="17th century",
                confidence=1.0,
                parent_frameworks=["algebra"]
            ),
            MathematicalFramework(
                name="linear_algebra",
                description="Study of vector spaces and linear transformations",
                axioms=["vector addition", "scalar multiplication", "linearity"],
                operations=["matrix multiplication", "eigenvalues", "projection"],
                objects=["vectors", "matrices", "linear maps"],
                applications=["computer graphics", "quantum mechanics", "data science"],
                invented_at="19th century",
                confidence=1.0,
                parent_frameworks=["algebra"]
            ),
            MathematicalFramework(
                name="group_theory",
                description="Study of algebraic structures with single operation",
                axioms=["closure", "associativity", "identity", "inverse"],
                operations=["group action", "conjugation", "quotient"],
                objects=["groups", "subgroups", "homomorphisms"],
                applications=["symmetry", "cryptography", "physics"],
                invented_at="19th century",
                confidence=1.0,
                parent_frameworks=["algebra"]
            ),
        ]

        for framework in known:
            self.frameworks[framework.name] = framework

    def detect_framework_gap(
        self,
        problem: str,
        attempted_frameworks: List[str],
        failure_reason: str
    ) -> Optional[FrameworkGap]:
        """
        Detect when existing frameworks are insufficient.

        Args:
            problem: The problem we're trying to solve
            attempted_frameworks: Frameworks already tried
            failure_reason: Why they failed

        Returns:
            FrameworkGap if detected, None if existing frameworks sufficient
        """
        # Extract what the problem needs
        required_capabilities = self._extract_requirements(problem)

        # Find what's missing from attempted frameworks
        available_tools = []
        for fw_name in attempted_frameworks:
            if fw_name in self.frameworks:
                available_tools.extend(self.frameworks[fw_name].operations)

        missing_tools = [req for req in required_capabilities if req not in available_tools]

        # If significant tools are missing, there's a gap
        if len(missing_tools) > len(required_capabilities) / 2:
            gap_severity = len(missing_tools) / len(required_capabilities) if required_capabilities else 1.0

            return FrameworkGap(
                problem_domain=self._extract_domain(problem),
                required_capabilities=required_capabilities,
                missing_tools=missing_tools,
                existing_frameworks=attempted_frameworks,
                gap_severity=gap_severity
            )

        return None

    def invent_framework(
        self,
        gap: FrameworkGap,
        llm_bridge
    ) -> Optional[MathematicalFramework]:
        """
        Invent a new mathematical framework to fill the gap.

        This is the CREATIVE step - synthesizing something NEW!

        Args:
            gap: The identified gap in frameworks
            llm_bridge: LLM for creative synthesis

        Returns:
            New framework if successful, None otherwise
        """
        print(f"[🔬] Inventing new framework for: {gap.problem_domain}")
        print(f"    Missing tools: {gap.missing_tools[:3]}...")

        # Prompt LLM to invent new framework
        prompt = f"""You are a mathematical framework inventor.

Problem domain: {gap.problem_domain}
Required capabilities: {', '.join(gap.required_capabilities)}
Missing tools: {', '.join(gap.missing_tools)}
Existing frameworks tried: {', '.join(gap.existing_frameworks)}

Invent a NEW mathematical framework that provides the missing capabilities.

Your framework should:
1. Extend/combine existing frameworks creatively
2. Introduce new objects or operations
3. Have clear axioms
4. Be applicable to the problem domain

Respond in JSON format:
{{
    "name": "framework name (lowercase, underscores)",
    "description": "what it does",
    "axioms": ["axiom1", "axiom2", ...],
    "operations": ["operation1", "operation2", ...],
    "objects": ["object1", "object2", ...],
    "applications": ["application1", ...],
    "parent_frameworks": ["framework1", "framework2"]
}}
"""

        response = llm_bridge.generate(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                framework_data = json.loads(json_match.group())

                # Create framework
                framework = MathematicalFramework(
                    name=framework_data.get("name", "unnamed_framework"),
                    description=framework_data.get("description", ""),
                    axioms=framework_data.get("axioms", []),
                    operations=framework_data.get("operations", []),
                    objects=framework_data.get("objects", []),
                    applications=framework_data.get("applications", []),
                    invented_at=datetime.now().isoformat(),
                    confidence=0.5,  # Start with low confidence, validate later
                    parent_frameworks=framework_data.get("parent_frameworks", [])
                )

                print(f"[✓] Invented: {framework.name}")
                print(f"    Operations: {framework.operations[:3]}...")

                # Store for future use
                self.frameworks[framework.name] = framework
                self.save()

                return framework
            else:
                print("[!] Failed to parse framework from LLM response")
                return None

        except Exception as e:
            print(f"[!] Framework invention failed: {e}")
            return None

    def validate_framework(
        self,
        framework: MathematicalFramework,
        test_problems: List[str],
        llm_bridge
    ) -> float:
        """
        Validate invented framework by testing on problems.

        Args:
            framework: Framework to validate
            test_problems: Problems to test it on
            llm_bridge: LLM for problem solving

        Returns:
            Validation confidence (0-1)
        """
        if not test_problems:
            return 0.5  # Neutral if no tests

        successes = 0
        for problem in test_problems[:3]:  # Test on first 3 problems
            prompt = f"""Using the {framework.name} framework:
Operations: {', '.join(framework.operations)}
Objects: {', '.join(framework.objects)}
Axioms: {', '.join(framework.axioms)}

Solve: {problem}

Can this framework solve it? Respond YES or NO."""

            response = llm_bridge.generate(prompt)
            if "YES" in response.upper():
                successes += 1

        confidence = successes / min(len(test_problems), 3)

        # Update framework confidence
        framework.confidence = max(framework.confidence, confidence)
        self.save()

        return confidence

    def _extract_requirements(self, problem: str) -> List[str]:
        """Extract what operations/capabilities the problem needs."""
        requirements = []

        # Keywords that suggest operations
        keywords = {
            "find": "search",
            "solve": "solve",
            "prove": "prove",
            "optimize": "optimize",
            "calculate": "calculate",
            "measure": "measure",
            "transform": "transform",
            "map": "map",
            "classify": "classify",
            "compare": "compare"
        }

        problem_lower = problem.lower()
        for keyword, operation in keywords.items():
            if keyword in problem_lower:
                requirements.append(operation)

        return requirements if requirements else ["general_problem_solving"]

    def _extract_domain(self, problem: str) -> str:
        """Extract the mathematical domain from problem text."""
        domains = {
            "number": "number_theory",
            "prime": "number_theory",
            "geometry": "geometry",
            "shape": "geometry",
            "angle": "geometry",
            "calculus": "analysis",
            "derivative": "analysis",
            "integral": "analysis",
            "limit": "analysis",
            "matrix": "linear_algebra",
            "vector": "linear_algebra",
            "probability": "probability_theory",
            "random": "probability_theory",
            "graph": "graph_theory",
            "network": "graph_theory"
        }

        problem_lower = problem.lower()
        for keyword, domain in domains.items():
            if keyword in problem_lower:
                return domain

        return "general_mathematics"

    def get_framework(self, name: str) -> Optional[MathematicalFramework]:
        """Get framework by name."""
        return self.frameworks.get(name)

    def list_frameworks(self) -> List[str]:
        """List all available frameworks."""
        return list(self.frameworks.keys())

    def save(self):
        """Save invented frameworks to disk."""
        try:
            data = {name: asdict(fw) for name, fw in self.frameworks.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[+] Saved {len(data)} frameworks to {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to save frameworks: {e}")

    def load(self):
        """Load invented frameworks from disk."""
        import os
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for name, fw_data in data.items():
                if name not in self.frameworks:  # Don't overwrite seeds
                    self.frameworks[name] = MathematicalFramework(**fw_data)

            print(f"[+] Loaded {len(data)} frameworks from {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to load frameworks: {e}")

    def summarize(self) -> str:
        """Get human-readable summary."""
        lines = []
        lines.append("Framework Inventor Status:")
        lines.append(f"  Total frameworks: {len(self.frameworks)}")

        invented = [fw for fw in self.frameworks.values() if fw.invented_at != "ancient" and "century" not in fw.invented_at]
        lines.append(f"  Invented by system: {len(invented)}")

        if invented:
            lines.append("\nRecently invented:")
            for fw in invented[:3]:
                lines.append(f"  - {fw.name}: {fw.description}")
                lines.append(f"    Confidence: {fw.confidence:.2f}")

        return "\n".join(lines)
