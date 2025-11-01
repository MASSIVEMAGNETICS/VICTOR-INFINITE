"""
OmniForge Studio v1.1 - Runtime Module
Visual Runtime Architecture Standard 1.1 (VRAS-1.1)
"""

from .execution_engine import (
    ExecutionEngine,
    ExecutionMode,
    NodeState,
    NodeInstance,
    Wire,
    ExecutionSnapshot
)

__all__ = [
    'ExecutionEngine',
    'ExecutionMode',
    'NodeState',
    'NodeInstance',
    'Wire',
    'ExecutionSnapshot'
]

__version__ = "1.1.0"
