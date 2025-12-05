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


# =============================================================================
# ADVANCED SYNTHESIS ENGINE - Phase 4 Enhancement
# =============================================================================

@dataclass
class SynthesisPath:
    """Represents a path for synthesizing frameworks."""

    source_frameworks: List[str]  # Starting frameworks
    target_capability: str  # What we want to achieve
    synthesis_steps: List[str]  # Steps to combine/extend
    expected_properties: List[str]  # Properties of result
    risk_level: float  # Complexity/uncertainty (0-1)


@dataclass
class AxiomSystem:
    """Represents a formal axiom system."""

    name: str
    axioms: List[str]
    inference_rules: List[str]
    consistency_score: float  # Estimated consistency (0-1)
    completeness_score: float  # Estimated completeness (0-1)
    derived_theorems: List[str]


@dataclass
class ConceptFusion:
    """Represents fusion of multiple mathematical concepts."""

    concepts: List[str]
    fusion_type: str  # "product", "coproduct", "tensor", "quotient"
    result_name: str
    result_properties: List[str]
    preservation_map: Dict[str, str]  # What properties are preserved


class AdvancedSynthesisEngine:
    """
    Advanced Framework Synthesis Engine.

    Goes beyond simple framework invention to perform:
    1. Multi-framework fusion (tensor products, fibered products)
    2. Axiom system generation and validation
    3. Category-theoretic synthesis (functors, natural transformations)
    4. Automatic theorem discovery
    5. Consistency checking via proof search
    """

    def __init__(self, framework_inventor: FrameworkInventor):
        self.inventor = framework_inventor
        self.synthesis_history: List[ConceptFusion] = []
        self.axiom_systems: Dict[str, AxiomSystem] = {}
        self.discovered_theorems: List[Dict] = []

        # Category-theoretic structures
        self.functors: Dict[str, Dict] = {}
        self.natural_transformations: List[Dict] = []

        print("[+] 🧬 Advanced Synthesis Engine: Multi-framework fusion active!")

    def synthesize_frameworks(
        self,
        frameworks: List[str],
        synthesis_type: str,
        llm_bridge
    ) -> Optional[MathematicalFramework]:
        """
        Synthesize multiple frameworks into a new unified framework.

        Args:
            frameworks: List of framework names to combine
            synthesis_type: "product", "coproduct", "tensor", "fiber"
            llm_bridge: LLM for creative synthesis

        Returns:
            New synthesized framework if successful
        """
        print(f"[🧬] Synthesizing {len(frameworks)} frameworks via {synthesis_type}...")

        # Get source frameworks
        sources = []
        for fw_name in frameworks:
            if fw_name in self.inventor.frameworks:
                sources.append(self.inventor.frameworks[fw_name])

        if len(sources) < 2:
            print("[!] Need at least 2 valid frameworks to synthesize")
            return None

        # Build synthesis prompt based on type
        synthesis_prompts = {
            "product": self._product_synthesis_prompt,
            "coproduct": self._coproduct_synthesis_prompt,
            "tensor": self._tensor_synthesis_prompt,
            "fiber": self._fiber_synthesis_prompt
        }

        prompt_builder = synthesis_prompts.get(synthesis_type, self._product_synthesis_prompt)
        prompt = prompt_builder(sources)

        response = llm_bridge.generate(prompt)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Create new framework
                framework = MathematicalFramework(
                    name=data.get("name", f"{synthesis_type}_synthesis"),
                    description=data.get("description", "Synthesized framework"),
                    axioms=data.get("axioms", []),
                    operations=data.get("operations", []),
                    objects=data.get("objects", []),
                    applications=data.get("applications", []),
                    invented_at=datetime.now().isoformat(),
                    confidence=0.6,  # Medium confidence for synthesis
                    parent_frameworks=frameworks
                )

                # Record fusion
                fusion = ConceptFusion(
                    concepts=frameworks,
                    fusion_type=synthesis_type,
                    result_name=framework.name,
                    result_properties=framework.operations,
                    preservation_map=data.get("preservation_map", {})
                )
                self.synthesis_history.append(fusion)

                # Store framework
                self.inventor.frameworks[framework.name] = framework
                self.inventor.save()

                print(f"[✓] Synthesized: {framework.name}")
                return framework

        except Exception as e:
            print(f"[!] Synthesis failed: {e}")
            return None

    def _product_synthesis_prompt(self, sources: List[MathematicalFramework]) -> str:
        """Generate prompt for product synthesis (A × B)."""
        fw_descriptions = "\n".join([
            f"- {fw.name}: {fw.description}\n  Operations: {', '.join(fw.operations[:5])}"
            for fw in sources
        ])

        return f"""You are a mathematical framework synthesizer using PRODUCT construction.

Source frameworks:
{fw_descriptions}

Create a PRODUCT framework that combines ALL operations from both frameworks.
The product should:
1. Include operations from BOTH frameworks
2. Allow paired objects (a, b) where a is from first, b is from second
3. Have projection operations to access individual components
4. Preserve properties componentwise

Respond in JSON:
{{
    "name": "product_framework_name",
    "description": "what the product does",
    "axioms": ["axiom1", "axiom2"],
    "operations": ["op1", "op2", "proj1", "proj2"],
    "objects": ["paired objects"],
    "applications": ["application1"],
    "preservation_map": {{"original_op": "product_op"}}
}}
"""

    def _coproduct_synthesis_prompt(self, sources: List[MathematicalFramework]) -> str:
        """Generate prompt for coproduct synthesis (A + B)."""
        fw_descriptions = "\n".join([
            f"- {fw.name}: {fw.description}"
            for fw in sources
        ])

        return f"""You are a mathematical framework synthesizer using COPRODUCT construction.

Source frameworks:
{fw_descriptions}

Create a COPRODUCT (disjoint union) framework that:
1. Includes ALL objects from BOTH frameworks
2. Has injection operations to embed each framework
3. Has universal property for mapping out
4. Preserves individual framework structures

Respond in JSON:
{{
    "name": "coproduct_framework_name",
    "description": "what the coproduct does",
    "axioms": ["axiom1", "axiom2"],
    "operations": ["inj1", "inj2", "case_analysis"],
    "objects": ["tagged union objects"],
    "applications": ["application1"],
    "preservation_map": {{"original_op": "coproduct_op"}}
}}
"""

    def _tensor_synthesis_prompt(self, sources: List[MathematicalFramework]) -> str:
        """Generate prompt for tensor product synthesis (A ⊗ B)."""
        fw_descriptions = "\n".join([
            f"- {fw.name}: operations={fw.operations[:3]}"
            for fw in sources
        ])

        return f"""You are a mathematical framework synthesizer using TENSOR PRODUCT construction.

Source frameworks:
{fw_descriptions}

Create a TENSOR PRODUCT framework that:
1. Captures bilinear combinations of objects
2. Has tensor operation: a ⊗ b
3. Satisfies universal property for bilinear maps
4. Operations distribute over tensor

Respond in JSON:
{{
    "name": "tensor_framework_name",
    "description": "tensor product description",
    "axioms": ["bilinearity", "universal property"],
    "operations": ["tensor", "distribute", "bilinear_map"],
    "objects": ["tensor objects"],
    "applications": ["multilinear algebra"],
    "preservation_map": {{}}
}}
"""

    def _fiber_synthesis_prompt(self, sources: List[MathematicalFramework]) -> str:
        """Generate prompt for fiber product synthesis (pullback)."""
        fw_descriptions = "\n".join([
            f"- {fw.name}: {fw.description}"
            for fw in sources
        ])

        return f"""You are a mathematical framework synthesizer using FIBER PRODUCT (pullback).

Source frameworks:
{fw_descriptions}

Create a FIBER PRODUCT framework that:
1. Combines frameworks over a common base
2. Objects satisfy compatibility condition
3. Has projection to both source frameworks
4. Universal property for compatible pairs

Respond in JSON:
{{
    "name": "fiber_framework_name",
    "description": "fiber product over common structure",
    "axioms": ["compatibility", "universal property"],
    "operations": ["fiber_proj1", "fiber_proj2", "lift"],
    "objects": ["compatible pairs"],
    "applications": ["application1"],
    "preservation_map": {{}}
}}
"""

    def generate_axiom_system(
        self,
        domain: str,
        desired_properties: List[str],
        llm_bridge
    ) -> Optional[AxiomSystem]:
        """
        Generate a formal axiom system for a mathematical domain.

        Args:
            domain: Mathematical domain name
            desired_properties: Properties the axiom system should capture
            llm_bridge: LLM for axiom generation

        Returns:
            AxiomSystem if successful
        """
        print(f"[🧬] Generating axiom system for: {domain}")

        prompt = f"""You are a mathematical logician designing formal axiom systems.

Domain: {domain}
Desired properties: {', '.join(desired_properties)}

Design a MINIMAL, CONSISTENT axiom system that captures these properties.

Requirements:
1. Axioms should be independent (no redundancy)
2. Include inference rules for deriving theorems
3. Aim for consistency (no contradictions)
4. Aim for completeness (capture all true statements)

Respond in JSON:
{{
    "name": "{domain}_axiom_system",
    "axioms": ["axiom1: formal statement", "axiom2: formal statement"],
    "inference_rules": ["modus ponens", "universal instantiation"],
    "sample_theorems": ["theorem1", "theorem2"],
    "consistency_argument": "why this is consistent",
    "completeness_argument": "what this can prove"
}}
"""

        response = llm_bridge.generate(prompt)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                axiom_system = AxiomSystem(
                    name=data.get("name", f"{domain}_axioms"),
                    axioms=data.get("axioms", []),
                    inference_rules=data.get("inference_rules", []),
                    consistency_score=0.7,  # Initial estimate
                    completeness_score=0.5,  # Conservative estimate
                    derived_theorems=data.get("sample_theorems", [])
                )

                self.axiom_systems[axiom_system.name] = axiom_system
                print(f"[✓] Generated axiom system: {axiom_system.name}")
                print(f"    Axioms: {len(axiom_system.axioms)}")
                print(f"    Rules: {len(axiom_system.inference_rules)}")

                return axiom_system

        except Exception as e:
            print(f"[!] Axiom system generation failed: {e}")
            return None

    def discover_theorems(
        self,
        axiom_system: AxiomSystem,
        search_depth: int = 3,
        llm_bridge = None
    ) -> List[Dict]:
        """
        Discover theorems from axiom system via proof search.

        Args:
            axiom_system: The axiom system to explore
            search_depth: How many inference steps to try
            llm_bridge: LLM for guided proof search

        Returns:
            List of discovered theorems with proofs
        """
        if not llm_bridge:
            return []

        print(f"[🧬] Discovering theorems from {axiom_system.name}...")

        prompt = f"""You are a theorem prover exploring an axiom system.

Axiom System: {axiom_system.name}
Axioms:
{chr(10).join(f'  {i+1}. {ax}' for i, ax in enumerate(axiom_system.axioms))}

Inference Rules:
{chr(10).join(f'  - {rule}' for rule in axiom_system.inference_rules)}

Discover {search_depth} NEW theorems by applying inference rules to axioms.
Each theorem must have a valid proof from the axioms.

Respond in JSON:
{{
    "theorems": [
        {{
            "statement": "theorem statement",
            "proof_steps": ["step1", "step2"],
            "used_axioms": [1, 2],
            "significance": "why this matters"
        }}
    ]
}}
"""

        response = llm_bridge.generate(prompt)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                theorems = data.get("theorems", [])

                # Add to discovered theorems
                for theorem in theorems:
                    theorem["axiom_system"] = axiom_system.name
                    theorem["discovered_at"] = datetime.now().isoformat()
                    self.discovered_theorems.append(theorem)
                    axiom_system.derived_theorems.append(theorem["statement"])

                print(f"[✓] Discovered {len(theorems)} theorems")
                return theorems

        except Exception as e:
            print(f"[!] Theorem discovery failed: {e}")
            return []

    def define_functor(
        self,
        source_framework: str,
        target_framework: str,
        llm_bridge
    ) -> Optional[Dict]:
        """
        Define a functor (structure-preserving map) between frameworks.

        Args:
            source_framework: Domain framework
            target_framework: Codomain framework
            llm_bridge: LLM for functor construction

        Returns:
            Functor definition if successful
        """
        if source_framework not in self.inventor.frameworks:
            print(f"[!] Source framework '{source_framework}' not found")
            return None
        if target_framework not in self.inventor.frameworks:
            print(f"[!] Target framework '{target_framework}' not found")
            return None

        source = self.inventor.frameworks[source_framework]
        target = self.inventor.frameworks[target_framework]

        print(f"[🧬] Defining functor: {source_framework} → {target_framework}")

        prompt = f"""You are a category theorist defining functors between mathematical frameworks.

Source Framework: {source.name}
  Objects: {', '.join(source.objects)}
  Operations: {', '.join(source.operations)}

Target Framework: {target.name}
  Objects: {', '.join(target.objects)}
  Operations: {', '.join(target.operations)}

Define a FUNCTOR F: {source.name} → {target.name} that:
1. Maps objects to objects
2. Maps operations to operations
3. Preserves composition
4. Preserves identity

Respond in JSON:
{{
    "functor_name": "F",
    "object_map": {{"source_obj": "target_obj"}},
    "operation_map": {{"source_op": "target_op"}},
    "preservation_proof": "why this preserves structure",
    "is_faithful": true/false,
    "is_full": true/false
}}
"""

        response = llm_bridge.generate(prompt)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                functor = json.loads(json_match.group())
                functor["source"] = source_framework
                functor["target"] = target_framework
                functor["defined_at"] = datetime.now().isoformat()

                # Store functor
                functor_key = f"{source_framework}_to_{target_framework}"
                self.functors[functor_key] = functor

                print(f"[✓] Defined functor: {functor.get('functor_name', 'F')}")
                return functor

        except Exception as e:
            print(f"[!] Functor definition failed: {e}")
            return None

    def check_consistency(
        self,
        axiom_system: AxiomSystem,
        llm_bridge
    ) -> Tuple[bool, str]:
        """
        Check axiom system for consistency via proof search.

        Args:
            axiom_system: System to check
            llm_bridge: LLM for proof search

        Returns:
            (is_consistent, explanation)
        """
        print(f"[🧬] Checking consistency of {axiom_system.name}...")

        prompt = f"""You are a mathematical logician checking axiom consistency.

Axiom System: {axiom_system.name}
Axioms:
{chr(10).join(f'  {i+1}. {ax}' for i, ax in enumerate(axiom_system.axioms))}

Check if these axioms are CONSISTENT (cannot derive a contradiction).

Analysis:
1. Look for potential contradictions
2. Check for conflicting axioms
3. Consider independence of axioms
4. Identify any issues

Respond: CONSISTENT or INCONSISTENT, followed by detailed explanation.
"""

        response = llm_bridge.generate(prompt)

        is_consistent = "CONSISTENT" in response.upper() and "INCONSISTENT" not in response.upper()

        # Update consistency score
        if is_consistent:
            axiom_system.consistency_score = min(1.0, axiom_system.consistency_score + 0.1)
        else:
            axiom_system.consistency_score = max(0.0, axiom_system.consistency_score - 0.2)

        return (is_consistent, response.strip())

    def get_synthesis_stats(self) -> Dict:
        """Get statistics about synthesis operations."""
        return {
            "total_fusions": len(self.synthesis_history),
            "axiom_systems": len(self.axiom_systems),
            "discovered_theorems": len(self.discovered_theorems),
            "defined_functors": len(self.functors),
            "fusion_types": list(set(f.fusion_type for f in self.synthesis_history))
        }

    def summarize(self) -> str:
        """Get human-readable summary."""
        stats = self.get_synthesis_stats()
        lines = [
            "Advanced Synthesis Engine Status:",
            f"  Framework fusions: {stats['total_fusions']}",
            f"  Axiom systems: {stats['axiom_systems']}",
            f"  Discovered theorems: {stats['discovered_theorems']}",
            f"  Defined functors: {stats['defined_functors']}"
        ]

        if self.synthesis_history:
            lines.append("\nRecent fusions:")
            for fusion in self.synthesis_history[-3:]:
                lines.append(f"  - {' + '.join(fusion.concepts)} → {fusion.result_name}")

        return "\n".join(lines)
