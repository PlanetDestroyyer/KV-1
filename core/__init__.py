"""
KV-1 Core Module

Core components for KV-1 AGI learning system.

Phases:
- Phase 1: Pattern Learning
- Phase 2: Compositional Reasoning
- Phase 3: Deep Abstraction
- Phase 4: Framework Invention (with Advanced Synthesis)
- Phase 5: Physical Grounding (with Simulation)
- Phase 6: Multimodal Reasoning
- Phase 7: Autonomous Agent System

Plus: Unified AGI Controller for orchestration
"""

# Load environment variables using python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

# Core Infrastructure
from .llm import LLMBridge
from .web_researcher import WebResearcher
from .hybrid_memory import HybridMemory
from .neurosymbolic_gpu import NeurosymbolicGPU
from .math_connect import MathConnect
from .knowledge_validator import KnowledgeValidator

# Phase 1: Pattern Learning
from .pattern_learner import MathematicalStructureLearner, MathematicalStructure

# Phase 2: Compositional Reasoning
from .compositional_reasoner import CompositionEngine, AbstractionBuilder

# Phase 3: Deep Abstraction
from .deep_abstraction import DeepAbstractionEngine, FrameworkSelector

# Phase 4: Framework Invention + Advanced Synthesis
from .framework_invention import (
    FrameworkInventor,
    MathematicalFramework,
    FrameworkGap,
    AdvancedSynthesisEngine,
    AxiomSystem,
    ConceptFusion
)

# Phase 5: Physical Grounding + Simulation
from .physical_grounding import (
    PhysicalGroundingEngine,
    PhysicalMapping,
    PhysicalDomain,
    PhysicalSimulationEngine,
    SimulationState,
    SimulationResult,
    PhysicalExperiment
)

# Phase 6: Multimodal Reasoning
from .multimodal_reasoning import (
    MultimodalReasoningEngine,
    VisionProcessor,
    AudioProcessor,
    Modality,
    ModalityInput,
    MultimodalConcept,
    CrossModalAlignment
)

# Phase 7: Autonomous Agent System
from .autonomous_agent import (
    AutonomousAgent,
    MultiAgentSystem,
    AgentState,
    Goal,
    Action,
    Plan,
    Tool
)

# Unified AGI Controller
from .unified_agi_controller import (
    UnifiedAGIController,
    CognitiveCapability,
    CognitiveTask,
    CognitiveResult,
    WorldModel
)

# Supporting modules
from .meta_learner import MetaLearner
from .causal_reasoner import CausalReasoner
from .curiosity_engine import CuriosityEngine
from .goal_planner import GoalPlanner
from .transfer_learning import TransferLearning
from .analogical_reasoning import AnalogicalReasoner

__version__ = "0.2.0"

__all__ = [
    # Core Infrastructure
    "LLMBridge",
    "WebResearcher",
    "HybridMemory",
    "NeurosymbolicGPU",
    "MathConnect",
    "KnowledgeValidator",

    # Phase 1: Pattern Learning
    "MathematicalStructureLearner",
    "MathematicalStructure",

    # Phase 2: Compositional Reasoning
    "CompositionEngine",
    "AbstractionBuilder",

    # Phase 3: Deep Abstraction
    "DeepAbstractionEngine",
    "FrameworkSelector",

    # Phase 4: Framework Invention
    "FrameworkInventor",
    "MathematicalFramework",
    "FrameworkGap",
    "AdvancedSynthesisEngine",
    "AxiomSystem",
    "ConceptFusion",

    # Phase 5: Physical Grounding
    "PhysicalGroundingEngine",
    "PhysicalMapping",
    "PhysicalDomain",
    "PhysicalSimulationEngine",
    "SimulationState",
    "SimulationResult",
    "PhysicalExperiment",

    # Phase 6: Multimodal Reasoning
    "MultimodalReasoningEngine",
    "VisionProcessor",
    "AudioProcessor",
    "Modality",
    "ModalityInput",
    "MultimodalConcept",
    "CrossModalAlignment",

    # Phase 7: Autonomous Agent
    "AutonomousAgent",
    "MultiAgentSystem",
    "AgentState",
    "Goal",
    "Action",
    "Plan",
    "Tool",

    # Unified AGI Controller
    "UnifiedAGIController",
    "CognitiveCapability",
    "CognitiveTask",
    "CognitiveResult",
    "WorldModel",

    # Supporting modules
    "MetaLearner",
    "CausalReasoner",
    "CuriosityEngine",
    "GoalPlanner",
    "TransferLearning",
    "AnalogicalReasoner",
]
