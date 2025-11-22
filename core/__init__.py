"""
KV-1 Core Module

Core components for KV-1 learning system.
"""

from .env_loader import load_env
load_env()

from .llm import LLMBridge
from .web_researcher import WebResearcher
from .hybrid_memory import HybridMemory
from .neurosymbolic_gpu import NeurosymbolicGPU
from .math_connect import MathConnect
from .knowledge_validator import KnowledgeValidator

__version__ = "0.1.0"

__all__ = [
    "LLMBridge",
    "WebResearcher",
    "HybridMemory",
    "NeurosymbolicGPU",
    "MathConnect",
    "KnowledgeValidator",
]
