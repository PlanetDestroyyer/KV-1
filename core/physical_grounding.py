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


# =============================================================================
# PHYSICAL SIMULATION ENGINE - Phase 5 Enhancement
# =============================================================================

@dataclass
class SimulationState:
    """State of a physical simulation at a point in time."""

    time: float  # Simulation time
    variables: Dict[str, float]  # Variable name -> value
    derivatives: Dict[str, float]  # Rate of change
    energy: float  # Total system energy
    metadata: Dict  # Additional info


@dataclass
class SimulationResult:
    """Result of running a physical simulation."""

    domain: str  # Physics domain
    initial_state: SimulationState
    final_state: SimulationState
    trajectory: List[SimulationState]  # States over time
    conservation_check: Dict[str, bool]  # Which quantities conserved
    mathematical_verification: str  # How math predicts this
    runtime_seconds: float


@dataclass
class PhysicalExperiment:
    """Represents a virtual physical experiment."""

    name: str
    hypothesis: str  # What we're testing
    setup: Dict  # Initial conditions
    procedure: List[str]  # Steps to run
    expected_outcome: str  # Predicted result
    actual_outcome: Optional[str]  # After running
    validates_math: Optional[str]  # Mathematical concept validated


class PhysicalSimulationEngine:
    """
    Physical Simulation Engine for grounding mathematics in reality.

    Provides:
    1. Numerical simulation of physical systems
    2. Virtual experiments to validate mathematical predictions
    3. Conservation law verification
    4. Parameter space exploration
    5. Physical intuition building through interactive simulation
    """

    def __init__(self, grounding_engine: PhysicalGroundingEngine):
        self.grounding = grounding_engine
        self.simulations: List[SimulationResult] = []
        self.experiments: List[PhysicalExperiment] = []
        self.conservation_laws: Dict[str, str] = {
            "energy": "Total energy remains constant in isolated systems",
            "momentum": "Total momentum remains constant without external forces",
            "angular_momentum": "Total angular momentum is conserved",
            "charge": "Total electric charge is conserved",
            "mass": "Total mass is conserved (non-relativistic)"
        }

        print("[+] 🔬 Physical Simulation Engine: Virtual experiments active!")

    def simulate_system(
        self,
        domain: str,
        equations: List[str],
        initial_conditions: Dict[str, float],
        time_span: Tuple[float, float],
        dt: float = 0.01,
        llm_bridge = None
    ) -> Optional[SimulationResult]:
        """
        Simulate a physical system numerically.

        Args:
            domain: Physics domain (mechanics, electromagnetism, etc.)
            equations: Differential equations governing the system
            initial_conditions: Starting values for variables
            time_span: (start_time, end_time)
            dt: Time step
            llm_bridge: LLM for interpretation

        Returns:
            SimulationResult with trajectory
        """
        import time
        start_time = time.time()

        print(f"[🔬] Simulating {domain} system...")
        print(f"    Equations: {equations[:2]}...")
        print(f"    Time span: {time_span[0]} to {time_span[1]}")

        # Initialize state
        t = time_span[0]
        state = SimulationState(
            time=t,
            variables=initial_conditions.copy(),
            derivatives={},
            energy=self._compute_energy(domain, initial_conditions),
            metadata={"domain": domain}
        )

        trajectory = [state]
        initial_state = state

        # Simple Euler integration (for demonstration)
        # In production, use scipy.integrate.solve_ivp
        try:
            while t < time_span[1]:
                # Compute derivatives based on domain
                derivatives = self._compute_derivatives(domain, state.variables, llm_bridge)
                state.derivatives = derivatives

                # Update variables
                new_vars = {}
                for var, val in state.variables.items():
                    if var in derivatives:
                        new_vars[var] = val + derivatives[var] * dt
                    else:
                        new_vars[var] = val

                t += dt
                new_state = SimulationState(
                    time=t,
                    variables=new_vars,
                    derivatives=derivatives,
                    energy=self._compute_energy(domain, new_vars),
                    metadata={"domain": domain}
                )
                trajectory.append(new_state)
                state = new_state

        except Exception as e:
            print(f"[!] Simulation error: {e}")
            return None

        # Check conservation laws
        conservation = self._check_conservation(initial_state, state, domain)

        # Create result
        result = SimulationResult(
            domain=domain,
            initial_state=initial_state,
            final_state=state,
            trajectory=trajectory,
            conservation_check=conservation,
            mathematical_verification=self._verify_math(equations, trajectory, llm_bridge),
            runtime_seconds=time.time() - start_time
        )

        self.simulations.append(result)

        print(f"[✓] Simulation complete: {len(trajectory)} steps")
        print(f"    Energy conserved: {conservation.get('energy', 'N/A')}")

        return result

    def _compute_derivatives(
        self,
        domain: str,
        variables: Dict[str, float],
        llm_bridge = None
    ) -> Dict[str, float]:
        """Compute derivatives based on physics domain."""
        derivatives = {}

        if domain == "classical_mechanics":
            # F = ma, so a = F/m
            # For simple harmonic oscillator: a = -ω²x
            if "x" in variables and "v" in variables:
                omega_sq = variables.get("omega_sq", 1.0)
                derivatives["x"] = variables["v"]
                derivatives["v"] = -omega_sq * variables["x"]

        elif domain == "projectile_motion":
            # Simple projectile under gravity
            g = variables.get("g", 9.81)
            if "vy" in variables:
                derivatives["vy"] = -g
            if "vx" in variables:
                derivatives["vx"] = 0  # No air resistance
            if "x" in variables and "vx" in variables:
                derivatives["x"] = variables["vx"]
            if "y" in variables and "vy" in variables:
                derivatives["y"] = variables["vy"]

        elif domain == "thermodynamics":
            # Heat conduction: dT/dt = k * (T_env - T)
            k = variables.get("k", 0.1)
            T_env = variables.get("T_env", 300)
            if "T" in variables:
                derivatives["T"] = k * (T_env - variables["T"])

        elif domain == "electromagnetism":
            # RC circuit: dQ/dt = (V - Q/C) / R
            R = variables.get("R", 1.0)
            C = variables.get("C", 1.0)
            V = variables.get("V", 1.0)
            if "Q" in variables:
                derivatives["Q"] = (V - variables["Q"] / C) / R

        elif domain == "population_dynamics":
            # Logistic growth: dP/dt = r*P*(1 - P/K)
            r = variables.get("r", 0.1)
            K = variables.get("K", 1000)
            if "P" in variables:
                P = variables["P"]
                derivatives["P"] = r * P * (1 - P / K)

        return derivatives

    def _compute_energy(self, domain: str, variables: Dict[str, float]) -> float:
        """Compute total energy of system."""
        energy = 0.0

        if domain == "classical_mechanics":
            # E = KE + PE = 0.5*m*v² + 0.5*k*x²
            m = variables.get("m", 1.0)
            k = variables.get("k", 1.0)
            v = variables.get("v", 0.0)
            x = variables.get("x", 0.0)
            energy = 0.5 * m * v**2 + 0.5 * k * x**2

        elif domain == "projectile_motion":
            m = variables.get("m", 1.0)
            g = variables.get("g", 9.81)
            vx = variables.get("vx", 0.0)
            vy = variables.get("vy", 0.0)
            y = variables.get("y", 0.0)
            v_sq = vx**2 + vy**2
            energy = 0.5 * m * v_sq + m * g * y

        elif domain == "thermodynamics":
            # Internal energy ∝ temperature
            T = variables.get("T", 300)
            n = variables.get("n", 1.0)  # moles
            energy = 1.5 * n * 8.314 * T  # Ideal gas

        return energy

    def _check_conservation(
        self,
        initial: SimulationState,
        final: SimulationState,
        domain: str
    ) -> Dict[str, bool]:
        """Check which quantities are conserved."""
        conservation = {}

        # Energy conservation (within tolerance)
        energy_diff = abs(final.energy - initial.energy)
        tolerance = max(abs(initial.energy) * 0.01, 0.001)  # 1% tolerance
        conservation["energy"] = energy_diff < tolerance

        # Momentum conservation
        if "vx" in initial.variables and "vy" in initial.variables:
            m = initial.variables.get("m", 1.0)
            p_init = m * (initial.variables.get("vx", 0)**2 + initial.variables.get("vy", 0)**2)**0.5
            p_final = m * (final.variables.get("vx", 0)**2 + final.variables.get("vy", 0)**2)**0.5
            # Momentum only conserved if no external force
            if domain != "projectile_motion":  # Gravity is external
                conservation["momentum"] = abs(p_final - p_init) < p_init * 0.01

        return conservation

    def _verify_math(
        self,
        equations: List[str],
        trajectory: List[SimulationState],
        llm_bridge
    ) -> str:
        """Verify that simulation matches mathematical predictions."""
        if not llm_bridge or not equations:
            return "Mathematical verification requires LLM and equations"

        # Sample trajectory points
        n = len(trajectory)
        samples = [trajectory[0], trajectory[n//2], trajectory[-1]]

        prompt = f"""Verify this physical simulation against mathematical predictions.

Governing equations: {', '.join(equations)}

Trajectory samples:
- t=0: {samples[0].variables}
- t=mid: {samples[1].variables}
- t=end: {samples[2].variables}

Does this trajectory match what the equations predict?
Brief analysis (2-3 sentences):"""

        return llm_bridge.generate(prompt).strip()

    def run_virtual_experiment(
        self,
        name: str,
        hypothesis: str,
        setup: Dict,
        domain: str,
        llm_bridge
    ) -> PhysicalExperiment:
        """
        Run a virtual physical experiment.

        Args:
            name: Experiment name
            hypothesis: What we expect to happen
            setup: Initial conditions and parameters
            domain: Physics domain
            llm_bridge: LLM for interpretation

        Returns:
            PhysicalExperiment with results
        """
        print(f"[🔬] Running experiment: {name}")
        print(f"    Hypothesis: {hypothesis}")

        experiment = PhysicalExperiment(
            name=name,
            hypothesis=hypothesis,
            setup=setup,
            procedure=[
                f"Initialize {domain} system",
                "Set initial conditions",
                "Run simulation",
                "Analyze results"
            ],
            expected_outcome=hypothesis,
            actual_outcome=None,
            validates_math=None
        )

        # Run simulation
        result = self.simulate_system(
            domain=domain,
            equations=setup.get("equations", []),
            initial_conditions=setup.get("initial_conditions", {}),
            time_span=setup.get("time_span", (0, 10)),
            dt=setup.get("dt", 0.01),
            llm_bridge=llm_bridge
        )

        if result:
            # Analyze outcome
            experiment.actual_outcome = self._analyze_experiment(result, hypothesis, llm_bridge)
            experiment.validates_math = self._check_math_validation(result, llm_bridge)

        self.experiments.append(experiment)

        print(f"[✓] Experiment complete")
        if experiment.actual_outcome:
            print(f"    Outcome: {experiment.actual_outcome[:100]}...")

        return experiment

    def _analyze_experiment(
        self,
        result: SimulationResult,
        hypothesis: str,
        llm_bridge
    ) -> str:
        """Analyze experiment results."""
        if not llm_bridge:
            return f"Final state: {result.final_state.variables}"

        prompt = f"""Analyze this physics experiment result.

Hypothesis: {hypothesis}

Results:
- Initial: {result.initial_state.variables}
- Final: {result.final_state.variables}
- Conservation: {result.conservation_check}
- Duration: {result.runtime_seconds:.3f}s

Did the experiment confirm or refute the hypothesis?
Brief analysis:"""

        return llm_bridge.generate(prompt).strip()

    def _check_math_validation(
        self,
        result: SimulationResult,
        llm_bridge
    ) -> str:
        """Check what mathematical concepts are validated."""
        if not llm_bridge:
            return "Differential equations, numerical integration"

        prompt = f"""What mathematical concepts does this simulation validate?

Domain: {result.domain}
Conservation laws verified: {result.conservation_check}
Trajectory length: {len(result.trajectory)} points

List 2-3 mathematical concepts this validates:"""

        return llm_bridge.generate(prompt).strip()

    def explore_parameter_space(
        self,
        domain: str,
        base_conditions: Dict[str, float],
        parameter_to_vary: str,
        values: List[float],
        llm_bridge = None
    ) -> List[SimulationResult]:
        """
        Explore parameter space by running multiple simulations.

        Args:
            domain: Physics domain
            base_conditions: Base initial conditions
            parameter_to_vary: Which parameter to sweep
            values: Values to try
            llm_bridge: LLM for interpretation

        Returns:
            List of simulation results
        """
        print(f"[🔬] Parameter sweep: {parameter_to_vary} = {values}")

        results = []
        for val in values:
            conditions = base_conditions.copy()
            conditions[parameter_to_vary] = val

            result = self.simulate_system(
                domain=domain,
                equations=[],
                initial_conditions=conditions,
                time_span=(0, 5),
                llm_bridge=llm_bridge
            )

            if result:
                results.append(result)

        print(f"[✓] Completed {len(results)} parameter variations")
        return results

    def build_physical_intuition(
        self,
        math_concept: str,
        llm_bridge
    ) -> Dict:
        """
        Build physical intuition for a mathematical concept.

        Args:
            math_concept: Mathematical concept to understand
            llm_bridge: LLM for generating intuition

        Returns:
            Dict with physical intuition
        """
        print(f"[🔬] Building intuition for: {math_concept}")

        prompt = f"""Help build physical intuition for the mathematical concept: {math_concept}

Provide:
1. A simple physical analogy
2. A real-world example you can visualize
3. What changes when this concept is varied
4. Common misconceptions to avoid

Format as structured explanation:"""

        intuition_text = llm_bridge.generate(prompt)

        # Design a simple experiment
        experiment_prompt = f"""Design a simple thought experiment to demonstrate: {math_concept}

Describe:
1. Setup (what physical system?)
2. What to observe
3. How it demonstrates the concept

Keep it simple and visual:"""

        experiment_text = llm_bridge.generate(experiment_prompt)

        intuition = {
            "concept": math_concept,
            "physical_intuition": intuition_text.strip(),
            "thought_experiment": experiment_text.strip(),
            "related_physics": self._find_related_physics(math_concept)
        }

        print(f"[✓] Intuition built for {math_concept}")
        return intuition

    def _find_related_physics(self, math_concept: str) -> List[str]:
        """Find physics domains related to a math concept."""
        related = []

        # Keyword mapping
        concept_to_physics = {
            "derivative": ["classical_mechanics", "thermodynamics"],
            "integral": ["classical_mechanics", "electromagnetism"],
            "differential equation": ["classical_mechanics", "quantum_mechanics"],
            "linear algebra": ["quantum_mechanics", "electromagnetism"],
            "vector": ["classical_mechanics", "electromagnetism"],
            "tensor": ["relativity", "electromagnetism"],
            "probability": ["quantum_mechanics", "thermodynamics"],
            "wave": ["quantum_mechanics", "electromagnetism"],
            "field": ["electromagnetism", "quantum_mechanics"],
            "symmetry": ["quantum_mechanics", "relativity"]
        }

        concept_lower = math_concept.lower()
        for keyword, domains in concept_to_physics.items():
            if keyword in concept_lower:
                related.extend(domains)

        return list(set(related)) if related else ["general_physics"]

    def get_simulation_stats(self) -> Dict:
        """Get statistics about simulations run."""
        if not self.simulations:
            return {"total_simulations": 0}

        total_steps = sum(len(s.trajectory) for s in self.simulations)
        domains = list(set(s.domain for s in self.simulations))

        return {
            "total_simulations": len(self.simulations),
            "total_experiments": len(self.experiments),
            "total_trajectory_steps": total_steps,
            "domains_explored": domains,
            "avg_runtime": sum(s.runtime_seconds for s in self.simulations) / len(self.simulations)
        }

    def summarize(self) -> str:
        """Get human-readable summary."""
        stats = self.get_simulation_stats()
        lines = [
            "Physical Simulation Engine Status:",
            f"  Simulations run: {stats.get('total_simulations', 0)}",
            f"  Experiments: {stats.get('total_experiments', 0)}",
            f"  Domains: {', '.join(stats.get('domains_explored', []))}",
        ]

        if self.experiments:
            lines.append("\nRecent experiments:")
            for exp in self.experiments[-3:]:
                lines.append(f"  - {exp.name}: {exp.hypothesis[:50]}...")

        return "\n".join(lines)
