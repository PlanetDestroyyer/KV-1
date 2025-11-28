"""
Phase 5: Physical Grounding Engine

Connects abstract mathematics to physical reality.
This bridges the gap between pure math and the real world!

Architecture:
1. Map mathematical concepts to physical phenomena
2. Use physical intuition to guide mathematical reasoning
3. Generate physical examples for abstract concepts
4. Validate mathematical results against physical reality
5. Discover new math by observing physical patterns

Example:
    Math: "Group theory (abstract algebra)"
    Physical: "Crystal symmetries, molecule rotations"
    Grounding: Abstract symmetry operations ↔ Real rotations/reflections
    Result: Better understanding through concrete examples!
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re


@dataclass
class PhysicalMapping:
    """Maps a mathematical concept to physical phenomenon."""

    math_concept: str  # "derivative", "eigenvector"
    physical_phenomenon: str  # "velocity", "vibrational mode"
    description: str  # How they relate
    examples: List[str]  # Concrete examples
    units: Optional[str]  # Physical units if applicable
    measurable: bool  # Can be physically measured?
    confidence: float  # How well-grounded (0-1)
    learned_at: str  # When created


@dataclass
class PhysicalDomain:
    """Represents a domain of physical reality."""

    name: str  # "mechanics", "thermodynamics", "electromagnetism"
    description: str  # What it covers
    fundamental_laws: List[str]  # "Newton's laws", "Maxwell equations"
    observable_quantities: List[str]  # "position", "momentum", "energy"
    typical_math: List[str]  # "calculus", "linear_algebra"
    examples: List[str]  # Real-world examples


class PhysicalGroundingEngine:
    """
    Grounds abstract mathematics in physical reality.

    This is KEY to:
    - Making math intuitive through physical examples
    - Validating mathematical predictions
    - Discovering new mathematics from physics
    """

    def __init__(self, storage_path: str = "./physical_groundings.json"):
        self.storage_path = storage_path
        self.mappings: Dict[str, PhysicalMapping] = {}
        self.domains: Dict[str, PhysicalDomain] = {}

        # Initialize physical domains
        self._seed_physical_domains()

        # Load previous groundings
        self.load()

        print("[+] 🌍 Physical Grounding: Connects math to reality!")

    def _seed_physical_domains(self):
        """Initialize fundamental physical domains."""
        domains = [
            PhysicalDomain(
                name="classical_mechanics",
                description="Motion of macroscopic objects",
                fundamental_laws=["Newton's laws", "conservation of energy", "conservation of momentum"],
                observable_quantities=["position", "velocity", "acceleration", "force", "mass"],
                typical_math=["calculus", "differential equations", "vector calculus"],
                examples=["falling apple", "swinging pendulum", "orbiting planet"]
            ),
            PhysicalDomain(
                name="thermodynamics",
                description="Heat, energy, and entropy",
                fundamental_laws=["conservation of energy", "entropy increases", "absolute zero"],
                observable_quantities=["temperature", "pressure", "volume", "entropy", "heat"],
                typical_math=["partial derivatives", "statistical mechanics", "probability"],
                examples=["steam engine", "refrigerator", "melting ice"]
            ),
            PhysicalDomain(
                name="electromagnetism",
                description="Electric and magnetic phenomena",
                fundamental_laws=["Maxwell's equations", "Lorentz force", "charge conservation"],
                observable_quantities=["electric field", "magnetic field", "current", "voltage"],
                typical_math=["vector calculus", "differential equations", "complex analysis"],
                examples=["lightning", "electromagnet", "radio waves"]
            ),
            PhysicalDomain(
                name="quantum_mechanics",
                description="Behavior at atomic scales",
                fundamental_laws=["Schrödinger equation", "uncertainty principle", "wave-particle duality"],
                observable_quantities=["wavefunction", "energy levels", "spin", "probability amplitude"],
                typical_math=["linear algebra", "complex analysis", "functional analysis"],
                examples=["atomic spectra", "electron tunneling", "quantum entanglement"]
            ),
            PhysicalDomain(
                name="relativity",
                description="Space, time, and gravitation",
                fundamental_laws=["equivalence principle", "spacetime curvature", "speed of light constant"],
                observable_quantities=["spacetime interval", "proper time", "curvature tensor"],
                typical_math=["differential geometry", "tensor calculus", "non-Euclidean geometry"],
                examples=["GPS satellites", "gravitational lensing", "black holes"]
            )
        ]

        for domain in domains:
            self.domains[domain.name] = domain

    def ground_concept(
        self,
        math_concept: str,
        math_description: str,
        llm_bridge
    ) -> Optional[PhysicalMapping]:
        """
        Find physical grounding for a mathematical concept.

        Args:
            math_concept: Name of mathematical concept
            math_description: What it means mathematically
            llm_bridge: LLM for generating mappings

        Returns:
            PhysicalMapping if successful, None otherwise
        """
        print(f"[🌍] Grounding '{math_concept}' in physical reality...")

        # Prompt LLM to find physical interpretation
        prompt = f"""You are a physicist connecting mathematics to physical reality.

Mathematical concept: {math_concept}
Mathematical description: {math_description}

Find a PHYSICAL interpretation or application of this concept.

Consider:
- What does it represent in the real world?
- What physical quantity or phenomenon does it describe?
- Can it be measured or observed?
- What are concrete examples from physics?

Respond in JSON format:
{{
    "physical_phenomenon": "name of physical thing",
    "description": "how math concept relates to physical phenomenon",
    "examples": ["example1", "example2", "example3"],
    "units": "physical units (or null if dimensionless)",
    "measurable": true/false
}}

If no physical grounding exists, respond: {{"no_grounding": true}}
"""

        response = llm_bridge.generate(prompt)

        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                if data.get("no_grounding"):
                    print(f"    [i] No physical grounding found (pure abstraction)")
                    return None

                mapping = PhysicalMapping(
                    math_concept=math_concept,
                    physical_phenomenon=data.get("physical_phenomenon", "unknown"),
                    description=data.get("description", ""),
                    examples=data.get("examples", []),
                    units=data.get("units"),
                    measurable=data.get("measurable", False),
                    confidence=0.7,  # Medium confidence initially
                    learned_at=datetime.now().isoformat()
                )

                print(f"    [✓] Grounded in: {mapping.physical_phenomenon}")
                if mapping.units:
                    print(f"    Units: {mapping.units}")
                print(f"    Examples: {mapping.examples[:2]}")

                # Store mapping
                self.mappings[math_concept] = mapping
                self.save()

                return mapping
            else:
                print("[!] Failed to parse grounding response")
                return None

        except Exception as e:
            print(f"[!] Physical grounding failed: {e}")
            return None

    def find_physical_domain(
        self,
        problem: str,
        llm_bridge
    ) -> Optional[str]:
        """
        Identify which physical domain a problem belongs to.

        Args:
            problem: Problem description
            llm_bridge: LLM for domain detection

        Returns:
            Domain name if identified, None otherwise
        """
        # Simple keyword matching first
        problem_lower = problem.lower()
        keywords_to_domain = {
            "mechanics": ["motion", "force", "velocity", "acceleration", "momentum"],
            "thermodynamics": ["heat", "temperature", "entropy", "energy transfer"],
            "electromagnetism": ["electric", "magnetic", "current", "voltage", "field"],
            "quantum": ["quantum", "atomic", "electron", "photon", "wavefunction"],
            "relativity": ["spacetime", "gravity", "speed of light", "black hole"]
        }

        for domain, keywords in keywords_to_domain.items():
            if any(kw in problem_lower for kw in keywords):
                full_domain = f"{domain}_mechanics" if domain == "classical" else domain
                if full_domain in self.domains:
                    return full_domain

        return None

    def generate_physical_examples(
        self,
        math_concept: str,
        count: int = 3,
        llm_bridge = None
    ) -> List[str]:
        """
        Generate concrete physical examples for abstract math concept.

        Args:
            math_concept: Mathematical concept name
            count: Number of examples to generate
            llm_bridge: LLM for example generation

        Returns:
            List of physical examples
        """
        # Check if we already have grounding
        if math_concept in self.mappings:
            return self.mappings[math_concept].examples[:count]

        if not llm_bridge:
            return []

        print(f"[🌍] Generating physical examples for '{math_concept}'...")

        prompt = f"""Generate {count} concrete PHYSICAL examples that illustrate the mathematical concept '{math_concept}'.

Each example should:
- Be from real-world physics
- Be observable or measurable
- Clearly show the math concept in action

Respond as a numbered list:
1. Example 1
2. Example 2
3. Example 3
"""

        response = llm_bridge.generate(prompt)

        # Parse numbered list
        examples = []
        for line in response.split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                example = re.sub(r'^\d+\.\s*', '', line.strip())
                if example:
                    examples.append(example)

        print(f"    [✓] Generated {len(examples)} examples")
        return examples[:count]

    def validate_mathematical_result(
        self,
        math_result: str,
        physical_context: Optional[str],
        llm_bridge
    ) -> Tuple[bool, str]:
        """
        Validate a mathematical result against physical reality.

        Args:
            math_result: Mathematical conclusion or formula
            physical_context: Physical interpretation if known
            llm_bridge: LLM for validation

        Returns:
            (is_physically_reasonable, explanation)
        """
        if not physical_context:
            # Can't validate without physical context
            return (True, "No physical context for validation")

        print(f"[🌍] Validating mathematical result against physics...")

        prompt = f"""You are a physicist checking if mathematical results make physical sense.

Mathematical result: {math_result}
Physical context: {physical_context}

Questions:
1. Is this result physically reasonable?
2. Does it violate any physical laws?
3. Are the units correct?
4. Is the magnitude sensible?

Respond: VALID or INVALID, followed by explanation.
"""

        response = llm_bridge.generate(prompt)

        is_valid = "VALID" in response.upper() and "INVALID" not in response.upper()
        explanation = response.strip()

        if is_valid:
            print(f"    [✓] Physically reasonable")
        else:
            print(f"    [!] Physical violation detected")

        return (is_valid, explanation)

    def discover_math_from_physics(
        self,
        physical_observation: str,
        llm_bridge
    ) -> Optional[str]:
        """
        Discover mathematical patterns from physical observations.

        This is reverse grounding: Physics → Math

        Args:
            physical_observation: Observed physical phenomenon
            llm_bridge: LLM for pattern extraction

        Returns:
            Mathematical description if pattern found
        """
        print(f"[🌍] Extracting math from physical observation...")

        prompt = f"""You are a mathematical physicist discovering mathematical patterns.

Physical observation: {physical_observation}

What mathematical structure or pattern does this reveal?

Consider:
- Symmetries
- Conservation laws
- Functional relationships
- Geometric structures

Describe the mathematical pattern in precise terms.
"""

        response = llm_bridge.generate(prompt)
        return response.strip() if response else None

    def get_grounding(self, math_concept: str) -> Optional[PhysicalMapping]:
        """Get physical grounding for concept if it exists."""
        return self.mappings.get(math_concept)

    def save(self):
        """Save groundings to disk."""
        try:
            data = {
                "mappings": {name: asdict(mapping) for name, mapping in self.mappings.items()},
                "domains": {name: asdict(domain) for name, domain in self.domains.items()}
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[+] Saved {len(self.mappings)} groundings to {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to save groundings: {e}")

    def load(self):
        """Load groundings from disk."""
        import os
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load mappings
            for name, mapping_data in data.get("mappings", {}).items():
                self.mappings[name] = PhysicalMapping(**mapping_data)

            # Load domains (don't overwrite seeds)
            for name, domain_data in data.get("domains", {}).items():
                if name not in self.domains:
                    self.domains[name] = PhysicalDomain(**domain_data)

            print(f"[+] Loaded {len(self.mappings)} groundings from {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to load groundings: {e}")

    def summarize(self) -> str:
        """Get human-readable summary."""
        lines = []
        lines.append("Physical Grounding Status:")
        lines.append(f"  Known domains: {len(self.domains)}")
        lines.append(f"  Math→Physics mappings: {len(self.mappings)}")

        measurable = [m for m in self.mappings.values() if m.measurable]
        lines.append(f"  Measurable quantities: {len(measurable)}")

        if self.mappings:
            lines.append("\nRecent groundings:")
            for name, mapping in list(self.mappings.items())[:3]:
                lines.append(f"  - {name} → {mapping.physical_phenomenon}")
                if mapping.units:
                    lines.append(f"    Units: {mapping.units}")

        return "\n".join(lines)
