"""
Unified AGI Controller

The central orchestrator that integrates all AGI subsystems:
- Phase 1: Pattern Learning
- Phase 2: Compositional Reasoning
- Phase 3: Deep Abstraction
- Phase 4: Framework Invention (with Advanced Synthesis)
- Phase 5: Physical Grounding (with Simulation)
- Phase 6: Multimodal Reasoning
- Phase 7: Autonomous Agent System

Architecture:
1. Unified interface for all cognitive capabilities
2. Dynamic capability routing based on task type
3. Cross-system information flow
4. Meta-learning and self-improvement
5. Coherent world model maintenance
"""

from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
import time


class CognitiveCapability(Enum):
    """Available cognitive capabilities."""
    PATTERN_LEARNING = "pattern_learning"
    COMPOSITIONAL_REASONING = "compositional_reasoning"
    DEEP_ABSTRACTION = "deep_abstraction"
    FRAMEWORK_INVENTION = "framework_invention"
    ADVANCED_SYNTHESIS = "advanced_synthesis"
    PHYSICAL_GROUNDING = "physical_grounding"
    PHYSICAL_SIMULATION = "physical_simulation"
    MULTIMODAL_REASONING = "multimodal_reasoning"
    AUTONOMOUS_AGENT = "autonomous_agent"
    META_LEARNING = "meta_learning"


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = 1  # Single capability
    SIMPLE = 2   # 2-3 capabilities
    MODERATE = 3 # Multiple capabilities, some integration
    COMPLEX = 4  # All capabilities, deep integration
    NOVEL = 5    # Requires new framework invention


@dataclass
class CognitiveTask:
    """A task for the AGI system."""

    id: str
    description: str
    task_type: str  # "solve", "learn", "create", "reason", "explore"
    complexity: TaskComplexity
    required_capabilities: List[CognitiveCapability]
    input_data: Dict = field(default_factory=dict)
    context: Dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CognitiveResult:
    """Result of cognitive processing."""

    task_id: str
    success: bool
    output: Any
    capabilities_used: List[str]
    reasoning_trace: List[Dict]  # Step-by-step reasoning
    confidence: float
    processing_time: float
    insights: List[str] = field(default_factory=list)


@dataclass
class WorldModel:
    """The AGI's internal model of the world."""

    concepts: Dict[str, Dict]  # Known concepts
    relationships: List[Dict]  # Concept relationships
    physical_laws: Dict[str, str]  # Understood physics
    mathematical_structures: Dict[str, Dict]  # Known math structures
    active_hypotheses: List[Dict]  # Current hypotheses
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class UnifiedAGIController:
    """
    The Unified AGI Controller - Central Orchestrator.

    Integrates all 7 phases of the AGI system into a coherent whole.

    Responsibilities:
    1. Task analysis and capability routing
    2. Cross-system coordination
    3. World model maintenance
    4. Meta-cognitive oversight
    5. Continuous improvement
    """

    def __init__(
        self,
        llm_bridge = None,
        storage_path: str = "./agi_controller_state.json"
    ):
        self.llm = llm_bridge
        self.storage_path = storage_path

        # World model
        self.world_model = WorldModel(
            concepts={},
            relationships=[],
            physical_laws={},
            mathematical_structures={},
            active_hypotheses=[]
        )

        # Cognitive systems (lazy loaded)
        self._systems = {}
        self._system_status = {}

        # Processing history
        self.task_history: List[CognitiveTask] = []
        self.result_history: List[CognitiveResult] = []

        # Meta-learning statistics
        self.capability_performance: Dict[str, Dict] = {
            cap.value: {"successes": 0, "failures": 0, "avg_time": 0}
            for cap in CognitiveCapability
        }

        # Discovery system components (NEW: FEP + Compound + CoT)
        self._discovery_systems = {}
        self._init_discovery_systems()

        # Load state
        self.load()

        print("[+] Unified AGI Controller initialized")
        print("    Integrating 7 cognitive phases + autonomous discovery")
        print("    Discovery systems: FEP, Bayesian, Contradictions, Compound Growth")

    def _get_system(self, capability: CognitiveCapability) -> Optional[Any]:
        """Lazy load and return cognitive system for capability."""
        if capability.value in self._systems:
            return self._systems[capability.value]

        try:
            if capability == CognitiveCapability.PATTERN_LEARNING:
                from .pattern_learner import MathematicalStructureLearner
                self._systems[capability.value] = MathematicalStructureLearner()

            elif capability == CognitiveCapability.COMPOSITIONAL_REASONING:
                from .compositional_reasoner import CompositionEngine
                self._systems[capability.value] = CompositionEngine()

            elif capability == CognitiveCapability.DEEP_ABSTRACTION:
                from .deep_abstraction import DeepAbstractionEngine
                self._systems[capability.value] = DeepAbstractionEngine()

            elif capability in [CognitiveCapability.FRAMEWORK_INVENTION, CognitiveCapability.ADVANCED_SYNTHESIS]:
                from .framework_invention import FrameworkInventor, AdvancedSynthesisEngine
                inventor = FrameworkInventor()
                self._systems[CognitiveCapability.FRAMEWORK_INVENTION.value] = inventor
                self._systems[CognitiveCapability.ADVANCED_SYNTHESIS.value] = AdvancedSynthesisEngine(inventor)

            elif capability in [CognitiveCapability.PHYSICAL_GROUNDING, CognitiveCapability.PHYSICAL_SIMULATION]:
                from .physical_grounding import PhysicalGroundingEngine, PhysicalSimulationEngine
                grounding = PhysicalGroundingEngine()
                self._systems[CognitiveCapability.PHYSICAL_GROUNDING.value] = grounding
                self._systems[CognitiveCapability.PHYSICAL_SIMULATION.value] = PhysicalSimulationEngine(grounding)

            elif capability == CognitiveCapability.MULTIMODAL_REASONING:
                from .multimodal_reasoning import MultimodalReasoningEngine
                self._systems[capability.value] = MultimodalReasoningEngine()

            elif capability == CognitiveCapability.AUTONOMOUS_AGENT:
                from .autonomous_agent import AutonomousAgent
                self._systems[capability.value] = AutonomousAgent(
                    name="AGI_Core",
                    llm_bridge=self.llm
                )

            elif capability == CognitiveCapability.META_LEARNING:
                from .meta_learner import MetaLearner
                self._systems[capability.value] = MetaLearner()

            self._system_status[capability.value] = "active"
            return self._systems.get(capability.value)

        except ImportError as e:
            print(f"[!] Could not load system for {capability.value}: {e}")
            self._system_status[capability.value] = "unavailable"
            return None

    def _init_discovery_systems(self):
        """
        Initialize autonomous discovery system components.

        NEW: FEP + Compound Knowledge Growth + CoT Pattern Mining
        """
        try:
            # 1. FAISS Vector Store (for RAG and similarity search)
            from .faiss_vector_store import FAISSVectorStore
            self._discovery_systems['vector_store'] = FAISSVectorStore(dimension=384)

            # 2. FEP-Guided Knowledge Graph
            from .fep_knowledge_graph import FEPGuidedKnowledgeGraph
            self._discovery_systems['knowledge_graph'] = FEPGuidedKnowledgeGraph(
                vector_store=self._discovery_systems['vector_store']
            )

            # 3. Bayesian Evidence Evaluator
            from .bayesian_evidence_evaluator import BayesianEvidenceEvaluator
            self._discovery_systems['bayesian'] = BayesianEvidenceEvaluator()

            # 4. Contradiction Detector
            from .contradiction_detector import ContradictionDetector
            self._discovery_systems['contradictions'] = ContradictionDetector(
                vector_store=self._discovery_systems['vector_store'],
                bayesian_evaluator=self._discovery_systems['bayesian'],
                knowledge_graph=self._discovery_systems['knowledge_graph']
            )

            # 5. Compound Growth Tracker
            from .compound_growth_tracker import CompoundGrowthTracker
            self._discovery_systems['compound'] = CompoundGrowthTracker()

            # 6. Hypothesis Generator
            from .hypothesis_generator import HypothesisGenerator
            self._discovery_systems['hypothesis_gen'] = HypothesisGenerator(
                llm_bridge=self.llm,
                knowledge_graph=self._discovery_systems['knowledge_graph'],
                fep_learner=None  # Optional FEP learner
            )

            # 7. CoT Pattern Miner
            from .cot_pattern_miner import CoTPatternMiner
            self._discovery_systems['cot_miner'] = CoTPatternMiner()

            # 8. Experiment Designer
            from .experiment_designer import ExperimentDesigner
            self._discovery_systems['experiment_designer'] = ExperimentDesigner(
                llm_bridge=self.llm
            )

            # 9. Theory Synthesizer
            from .theory_synthesizer import TheorySynthesizer
            self._discovery_systems['theory_synth'] = TheorySynthesizer(
                llm_bridge=self.llm,
                bayesian_evaluator=self._discovery_systems['bayesian'],
                knowledge_graph=self._discovery_systems['knowledge_graph']
            )

            # 10. Discovery Orchestrator (THE HEART!)
            from .discovery_orchestrator import DiscoveryOrchestrator
            self._discovery_systems['orchestrator'] = DiscoveryOrchestrator(
                knowledge_graph=self._discovery_systems['knowledge_graph'],
                hypothesis_generator=self._discovery_systems['hypothesis_gen'],
                bayesian_evaluator=self._discovery_systems['bayesian'],
                contradiction_detector=self._discovery_systems['contradictions'],
                compound_tracker=self._discovery_systems['compound'],
                experiment_designer=self._discovery_systems['experiment_designer'],
                theory_synthesizer=self._discovery_systems['theory_synth'],
                cot_miner=self._discovery_systems['cot_miner'],
                llm_bridge=self.llm
            )

            print("[✓] Discovery systems initialized successfully")
            print(f"    Components: {len(self._discovery_systems)}")

        except ImportError as e:
            print(f"[!] Could not initialize discovery systems: {e}")
            print("    Discovery capabilities will be unavailable")

    def discover(self, domain: str, initial_observations: Optional[List[str]] = None,
                max_iterations: int = 5, verbose: bool = True):
        """
        Run autonomous discovery loop!

        NEW: Full autonomous discovery using FEP + Compound Growth + CoT

        Args:
            domain: Domain to explore
            initial_observations: Starting observations
            max_iterations: Max discovery iterations
            verbose: Print progress

        Returns:
            DiscoverySession with all discoveries
        """
        if 'orchestrator' not in self._discovery_systems:
            print("[!] Discovery orchestrator not available")
            return None

        orchestrator = self._discovery_systems['orchestrator']
        return orchestrator.discover(
            domain=domain,
            initial_observations=initial_observations,
            max_iterations=max_iterations,
            verbose=verbose
        )

    def analyze_task(self, description: str, context: Dict = None) -> CognitiveTask:
        """
        Analyze a task and determine required capabilities.

        Args:
            description: Task description
            context: Additional context

        Returns:
            CognitiveTask with analysis
        """
        context = context or {}

        print(f"[AGI] Analyzing task: {description[:50]}...")

        # Determine task type
        task_type = self._classify_task_type(description)

        # Determine required capabilities
        capabilities = self._determine_capabilities(description, task_type, context)

        # Estimate complexity
        complexity = self._estimate_complexity(description, capabilities)

        task = CognitiveTask(
            id=f"task_{len(self.task_history)}_{int(time.time())}",
            description=description,
            task_type=task_type,
            complexity=complexity,
            required_capabilities=capabilities,
            input_data=context.get("input_data", {}),
            context=context
        )

        self.task_history.append(task)

        print(f"    Type: {task_type}")
        print(f"    Complexity: {complexity.name}")
        print(f"    Capabilities: {[c.value for c in capabilities]}")

        return task

    def _classify_task_type(self, description: str) -> str:
        """Classify the type of task."""
        desc_lower = description.lower()

        type_keywords = {
            "solve": ["solve", "calculate", "compute", "find", "determine"],
            "learn": ["learn", "understand", "study", "memorize", "comprehend"],
            "create": ["create", "invent", "design", "generate", "synthesize"],
            "reason": ["reason", "prove", "explain", "derive", "deduce"],
            "explore": ["explore", "investigate", "analyze", "examine", "discover"]
        }

        for task_type, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return task_type

        return "reason"  # Default

    def _determine_capabilities(
        self,
        description: str,
        task_type: str,
        context: Dict
    ) -> List[CognitiveCapability]:
        """Determine which capabilities are needed for a task."""
        capabilities = []
        desc_lower = description.lower()

        # Always include pattern learning for any task
        capabilities.append(CognitiveCapability.PATTERN_LEARNING)

        # Task type based capabilities
        if task_type == "solve":
            capabilities.append(CognitiveCapability.COMPOSITIONAL_REASONING)

        if task_type == "create":
            capabilities.append(CognitiveCapability.FRAMEWORK_INVENTION)
            capabilities.append(CognitiveCapability.ADVANCED_SYNTHESIS)

        if task_type == "learn":
            capabilities.append(CognitiveCapability.META_LEARNING)

        # Content based capabilities
        content_capability_map = {
            "abstract": CognitiveCapability.DEEP_ABSTRACTION,
            "physical": CognitiveCapability.PHYSICAL_GROUNDING,
            "simulation": CognitiveCapability.PHYSICAL_SIMULATION,
            "image": CognitiveCapability.MULTIMODAL_REASONING,
            "visual": CognitiveCapability.MULTIMODAL_REASONING,
            "audio": CognitiveCapability.MULTIMODAL_REASONING,
            "diagram": CognitiveCapability.MULTIMODAL_REASONING,
            "autonomous": CognitiveCapability.AUTONOMOUS_AGENT,
            "agent": CognitiveCapability.AUTONOMOUS_AGENT,
            "framework": CognitiveCapability.FRAMEWORK_INVENTION,
            "synthesize": CognitiveCapability.ADVANCED_SYNTHESIS,
            "category": CognitiveCapability.ADVANCED_SYNTHESIS,
            "functor": CognitiveCapability.ADVANCED_SYNTHESIS
        }

        for keyword, capability in content_capability_map.items():
            if keyword in desc_lower and capability not in capabilities:
                capabilities.append(capability)

        # Context based capabilities
        if context.get("has_images"):
            if CognitiveCapability.MULTIMODAL_REASONING not in capabilities:
                capabilities.append(CognitiveCapability.MULTIMODAL_REASONING)

        if context.get("needs_simulation"):
            if CognitiveCapability.PHYSICAL_SIMULATION not in capabilities:
                capabilities.append(CognitiveCapability.PHYSICAL_SIMULATION)

        return capabilities

    def _estimate_complexity(
        self,
        description: str,
        capabilities: List[CognitiveCapability]
    ) -> TaskComplexity:
        """Estimate task complexity."""
        num_capabilities = len(capabilities)

        # Base complexity on number of capabilities
        if num_capabilities == 1:
            complexity = TaskComplexity.TRIVIAL
        elif num_capabilities <= 3:
            complexity = TaskComplexity.SIMPLE
        elif num_capabilities <= 5:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.COMPLEX

        # Check for novel/framework invention needs
        novel_keywords = ["new framework", "invent", "novel approach", "unprecedented"]
        if any(kw in description.lower() for kw in novel_keywords):
            complexity = TaskComplexity.NOVEL

        return complexity

    def process(
        self,
        task: Union[str, CognitiveTask],
        context: Dict = None
    ) -> CognitiveResult:
        """
        Process a cognitive task using the appropriate systems.

        Args:
            task: Task description or CognitiveTask object
            context: Additional context

        Returns:
            CognitiveResult with output
        """
        start_time = time.time()

        # Convert string to task if needed
        if isinstance(task, str):
            task = self.analyze_task(task, context)

        print(f"\n[AGI] Processing task: {task.description[:50]}...")
        print(f"    Using {len(task.required_capabilities)} cognitive systems")

        reasoning_trace = []
        output = None
        success = True
        capabilities_used = []

        try:
            # Phase 1: Information gathering
            reasoning_trace.append({
                "phase": "information_gathering",
                "action": "Collecting relevant information from context and world model"
            })
            relevant_info = self._gather_information(task)

            # Phase 2: Capability orchestration
            for capability in task.required_capabilities:
                system = self._get_system(capability)
                if not system:
                    reasoning_trace.append({
                        "phase": "capability_error",
                        "capability": capability.value,
                        "error": "System not available"
                    })
                    continue

                reasoning_trace.append({
                    "phase": "capability_execution",
                    "capability": capability.value,
                    "action": f"Engaging {capability.value} system"
                })

                # Execute capability
                cap_output = self._execute_capability(
                    capability, system, task, relevant_info
                )
                capabilities_used.append(capability.value)

                # Integrate output
                if cap_output:
                    reasoning_trace.append({
                        "phase": "capability_result",
                        "capability": capability.value,
                        "result_summary": str(cap_output)[:100]
                    })
                    output = self._integrate_output(output, cap_output, capability)

            # Phase 3: Synthesis
            if len(capabilities_used) > 1:
                reasoning_trace.append({
                    "phase": "synthesis",
                    "action": "Synthesizing results from multiple capabilities"
                })
                output = self._synthesize_results(output, task, reasoning_trace)

            # Phase 4: Validation
            reasoning_trace.append({
                "phase": "validation",
                "action": "Validating result against task requirements"
            })
            validation = self._validate_result(output, task)
            success = validation["valid"]

            # Update world model with new knowledge
            self._update_world_model(task, output, reasoning_trace)

        except Exception as e:
            success = False
            reasoning_trace.append({
                "phase": "error",
                "error": str(e)
            })
            output = {"error": str(e)}

        processing_time = time.time() - start_time

        # Calculate confidence
        confidence = self._calculate_confidence(
            success, capabilities_used, task.complexity, processing_time
        )

        # Generate insights
        insights = self._generate_insights(task, output, reasoning_trace)

        result = CognitiveResult(
            task_id=task.id,
            success=success,
            output=output,
            capabilities_used=capabilities_used,
            reasoning_trace=reasoning_trace,
            confidence=confidence,
            processing_time=processing_time,
            insights=insights
        )

        # Update performance statistics
        self._update_performance_stats(result)

        self.result_history.append(result)

        print(f"[AGI] Task completed: {'SUCCESS' if success else 'FAILED'}")
        print(f"    Confidence: {confidence:.2%}")
        print(f"    Time: {processing_time:.2f}s")

        return result

    def _gather_information(self, task: CognitiveTask) -> Dict:
        """Gather relevant information for processing."""
        info = {
            "task_context": task.context,
            "input_data": task.input_data,
            "relevant_concepts": [],
            "related_structures": []
        }

        # Search world model for relevant concepts
        desc_words = set(task.description.lower().split())
        for concept_name, concept_data in self.world_model.concepts.items():
            if any(word in concept_name.lower() for word in desc_words):
                info["relevant_concepts"].append({
                    "name": concept_name,
                    "data": concept_data
                })

        # Search for related mathematical structures
        for struct_name, struct_data in self.world_model.mathematical_structures.items():
            if any(word in struct_name.lower() for word in desc_words):
                info["related_structures"].append({
                    "name": struct_name,
                    "data": struct_data
                })

        return info

    def _execute_capability(
        self,
        capability: CognitiveCapability,
        system: Any,
        task: CognitiveTask,
        info: Dict
    ) -> Any:
        """Execute a specific cognitive capability."""
        try:
            if capability == CognitiveCapability.PATTERN_LEARNING:
                # Identify patterns in the task
                if hasattr(system, 'identify_pattern'):
                    return system.identify_pattern(task.description)
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.COMPOSITIONAL_REASONING:
                # Compose solution from components
                if hasattr(system, 'compose'):
                    return system.compose(task.description, info.get("relevant_concepts", []))
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.DEEP_ABSTRACTION:
                # Find abstract structure
                if hasattr(system, 'abstract'):
                    return system.abstract(task.description)
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.FRAMEWORK_INVENTION:
                # Invent new framework if needed
                if hasattr(system, 'invent_framework') and self.llm:
                    from .framework_invention import FrameworkGap
                    gap = FrameworkGap(
                        problem_domain=task.task_type,
                        required_capabilities=[],
                        missing_tools=[],
                        existing_frameworks=[],
                        gap_severity=0.5
                    )
                    return system.invent_framework(gap, self.llm)
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.ADVANCED_SYNTHESIS:
                # Synthesize frameworks
                if hasattr(system, 'synthesize_frameworks') and self.llm:
                    # Get available frameworks
                    frameworks = list(system.inventor.frameworks.keys())[:2]
                    if len(frameworks) >= 2:
                        return system.synthesize_frameworks(frameworks, "product", self.llm)
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.PHYSICAL_GROUNDING:
                # Ground in physical reality
                if hasattr(system, 'ground_concept') and self.llm:
                    return system.ground_concept(task.description, task.description, self.llm)
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.PHYSICAL_SIMULATION:
                # Run simulation
                if hasattr(system, 'simulate_system'):
                    return system.simulate_system(
                        domain="classical_mechanics",
                        equations=[],
                        initial_conditions={"x": 1.0, "v": 0.0},
                        time_span=(0, 1)
                    )
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.MULTIMODAL_REASONING:
                # Process multimodal inputs
                if hasattr(system, 'reason_multimodally') and self.llm:
                    from .multimodal_reasoning import ModalityInput, Modality
                    inputs = [ModalityInput(
                        modality=Modality.TEXT,
                        content=task.description
                    )]
                    return system.reason_multimodally(task.description, inputs, self.llm)
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.AUTONOMOUS_AGENT:
                # Use agent for autonomous processing
                if hasattr(system, 'set_goal') and hasattr(system, 'plan'):
                    goal = system.set_goal(task.description)
                    plan = system.plan(goal.id)
                    return {"agent_plan": plan.id if plan else None}
                # No placeholder - return None if capability not available
                return None

            elif capability == CognitiveCapability.META_LEARNING:
                # Apply meta-learning
                if hasattr(system, 'adapt'):
                    return system.adapt(task.description)
                # No placeholder - return None if capability not available
                return None

        except Exception as e:
            return {"error": str(e), "capability": capability.value}

        return None

    def _integrate_output(
        self,
        current_output: Any,
        new_output: Any,
        capability: CognitiveCapability
    ) -> Dict:
        """Integrate new output with existing output."""
        if current_output is None:
            current_output = {}

        if isinstance(current_output, dict) and isinstance(new_output, dict):
            current_output[capability.value] = new_output
        else:
            current_output[capability.value] = new_output

        return current_output

    def _synthesize_results(
        self,
        outputs: Dict,
        task: CognitiveTask,
        trace: List[Dict]
    ) -> Dict:
        """Synthesize results from multiple capabilities."""
        if not self.llm:
            return outputs

        prompt = f"""Synthesize these cognitive processing results into a coherent answer.

Task: {task.description}

Results from different cognitive systems:
{json.dumps(outputs, indent=2, default=str)}

Provide a unified, coherent synthesis that addresses the task:"""

        try:
            synthesis = self.llm.generate(prompt)
            return {
                "synthesis": synthesis.strip(),
                "component_results": outputs
            }
        except Exception:
            return outputs

    def _validate_result(self, output: Any, task: CognitiveTask) -> Dict:
        """Validate the result against task requirements."""
        validation = {
            "valid": True,
            "checks": []
        }

        # Check if output exists
        if output is None:
            validation["valid"] = False
            validation["checks"].append("No output generated")
            return validation

        # Check for errors
        if isinstance(output, dict) and "error" in output:
            validation["valid"] = False
            validation["checks"].append(f"Error in output: {output['error']}")

        # Task-specific validation
        if task.task_type == "solve":
            # Should have some solution
            if isinstance(output, dict) and not any(
                k in output for k in ["solution", "answer", "result", "synthesis"]
            ):
                validation["checks"].append("Warning: No explicit solution found")

        return validation

    def _update_world_model(
        self,
        task: CognitiveTask,
        output: Any,
        trace: List[Dict]
    ):
        """Update world model with new knowledge."""
        # Extract concepts from task
        task_concepts = self._extract_concepts(task.description)

        for concept in task_concepts:
            if concept not in self.world_model.concepts:
                self.world_model.concepts[concept] = {
                    "learned_from": task.id,
                    "timestamp": datetime.now().isoformat()
                }

        # Add relationships
        if len(task_concepts) >= 2:
            self.world_model.relationships.append({
                "concepts": task_concepts,
                "type": "co-occurred",
                "task_id": task.id
            })

        self.world_model.last_updated = datetime.now().isoformat()

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple extraction - would use NLP in production
        words = text.lower().split()
        # Filter common words and keep meaningful ones
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "being", "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "must", "shall",
                    "can", "need", "dare", "ought", "used", "to", "of", "in",
                    "for", "on", "with", "at", "by", "from", "as", "into",
                    "through", "during", "before", "after", "above", "below"}

        concepts = [w for w in words if w not in stopwords and len(w) > 3]
        return concepts[:5]  # Top 5 concepts

    def _calculate_confidence(
        self,
        success: bool,
        capabilities_used: List[str],
        complexity: TaskComplexity,
        processing_time: float
    ) -> float:
        """Calculate confidence in the result."""
        base_confidence = 0.7 if success else 0.3

        # Adjust for capabilities used
        cap_factor = min(len(capabilities_used) / 5, 1.0) * 0.15
        base_confidence += cap_factor

        # Adjust for complexity (harder tasks = less confident)
        complexity_penalty = {
            TaskComplexity.TRIVIAL: 0,
            TaskComplexity.SIMPLE: 0.05,
            TaskComplexity.MODERATE: 0.10,
            TaskComplexity.COMPLEX: 0.15,
            TaskComplexity.NOVEL: 0.20
        }
        base_confidence -= complexity_penalty.get(complexity, 0.1)

        # Adjust for processing time (longer = potentially more thorough)
        if processing_time > 10:
            base_confidence += 0.05  # More processing might mean more thorough

        return max(0.1, min(0.95, base_confidence))

    def _generate_insights(
        self,
        task: CognitiveTask,
        output: Any,
        trace: List[Dict]
    ) -> List[str]:
        """Generate insights from processing."""
        insights = []

        # Capability insights
        capabilities_used = [
            t.get("capability") for t in trace
            if t.get("phase") == "capability_execution"
        ]

        if len(capabilities_used) > 3:
            insights.append(f"Complex task requiring {len(capabilities_used)} cognitive systems")

        # Error insights
        errors = [t for t in trace if t.get("phase") == "error"]
        if errors:
            insights.append("Some cognitive processes encountered errors")

        # Task type insights
        if task.complexity == TaskComplexity.NOVEL:
            insights.append("Novel task may require new frameworks or approaches")

        return insights

    def _update_performance_stats(self, result: CognitiveResult):
        """Update performance statistics."""
        for cap in result.capabilities_used:
            if cap in self.capability_performance:
                stats = self.capability_performance[cap]
                if result.success:
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

                # Update average time
                total = stats["successes"] + stats["failures"]
                stats["avg_time"] = (
                    (stats["avg_time"] * (total - 1) + result.processing_time) / total
                )

    def query(self, question: str) -> str:
        """
        Simple query interface for the AGI system.

        Args:
            question: Question to answer

        Returns:
            Answer string
        """
        result = self.process(question)

        if result.success:
            if isinstance(result.output, dict):
                if "synthesis" in result.output:
                    return result.output["synthesis"]
                return json.dumps(result.output, indent=2, default=str)
            return str(result.output)
        else:
            return f"Failed to process: {result.output}"

    def learn(self, concept: str, description: str) -> bool:
        """
        Teach the AGI a new concept.

        Args:
            concept: Concept name
            description: Concept description

        Returns:
            True if learned successfully
        """
        self.world_model.concepts[concept] = {
            "description": description,
            "learned_at": datetime.now().isoformat(),
            "source": "direct_teaching"
        }

        print(f"[AGI] Learned concept: {concept}")
        return True

    def save(self):
        """Save AGI state to disk."""
        try:
            data = {
                "world_model": {
                    "concepts": self.world_model.concepts,
                    "relationships": self.world_model.relationships,
                    "physical_laws": self.world_model.physical_laws,
                    "mathematical_structures": self.world_model.mathematical_structures,
                    "last_updated": self.world_model.last_updated
                },
                "capability_performance": self.capability_performance,
                "task_count": len(self.task_history),
                "result_count": len(self.result_history)
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[+] AGI state saved to {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to save AGI state: {e}")

    def load(self):
        """Load AGI state from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load world model
            wm = data.get("world_model", {})
            self.world_model.concepts = wm.get("concepts", {})
            self.world_model.relationships = wm.get("relationships", [])
            self.world_model.physical_laws = wm.get("physical_laws", {})
            self.world_model.mathematical_structures = wm.get("mathematical_structures", {})
            self.world_model.last_updated = wm.get("last_updated", datetime.now().isoformat())

            # Load performance stats
            self.capability_performance = data.get("capability_performance", self.capability_performance)

            print(f"[+] AGI state loaded from {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to load AGI state: {e}")

    def get_status(self) -> Dict:
        """Get current AGI system status."""
        return {
            "systems_loaded": len(self._systems),
            "systems_status": self._system_status,
            "world_model": {
                "concepts": len(self.world_model.concepts),
                "relationships": len(self.world_model.relationships),
                "physical_laws": len(self.world_model.physical_laws),
                "mathematical_structures": len(self.world_model.mathematical_structures)
            },
            "tasks_processed": len(self.task_history),
            "results_generated": len(self.result_history),
            "capability_performance": {
                cap: {
                    "success_rate": stats["successes"] / max(stats["successes"] + stats["failures"], 1),
                    "avg_time": stats["avg_time"]
                }
                for cap, stats in self.capability_performance.items()
                if stats["successes"] + stats["failures"] > 0
            }
        }

    def summarize(self) -> str:
        """Get human-readable summary."""
        status = self.get_status()
        lines = [
            "=" * 50,
            "UNIFIED AGI CONTROLLER STATUS",
            "=" * 50,
            f"Systems loaded: {status['systems_loaded']}/10",
            f"Concepts known: {status['world_model']['concepts']}",
            f"Relationships: {status['world_model']['relationships']}",
            f"Tasks processed: {status['tasks_processed']}",
            "",
            "Capability Performance:"
        ]

        for cap, perf in status.get("capability_performance", {}).items():
            lines.append(f"  {cap}: {perf['success_rate']:.1%} success, {perf['avg_time']:.2f}s avg")

        lines.append("")
        lines.append("Active Systems:")
        for sys_name, sys_status in self._system_status.items():
            lines.append(f"  {sys_name}: {sys_status}")

        return "\n".join(lines)
