"""
KV-1 Core Module

The brain of KV-1 OS - an AI-native operating system.
"""

from .env_loader import load_env
load_env()

from .orchestrator import KV1Orchestrator, get_kv1
from .trauma import TraumaSystem, TraumaMemory
from .user_profile import UserProfile, UserProfileManager
from .proactive_monitor import ProactiveMonitor, ProactiveTrigger
from .mcp import MCPRegistry, MCPConnector
from .llm import LLMBridge

__version__ = "0.1.0"

__all__ = [
    "KV1Orchestrator",
    "get_kv1",
    "TraumaSystem",
    "TraumaMemory",
    "UserProfile",
    "UserProfileManager",
    "ProactiveMonitor",
    "ProactiveTrigger",
    "MCPRegistry",
    "MCPConnector",
    "LLMBridge",
]
