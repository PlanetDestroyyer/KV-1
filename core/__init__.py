"""
KV-1 Core Module

Core components for KV-1 learning system.
"""

# Load environment variables using python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

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
