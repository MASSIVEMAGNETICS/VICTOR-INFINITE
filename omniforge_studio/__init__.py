"""
OmniForge Studio v1.1 - Main Entry Point
Visual Runtime Architecture Standard 1.1 (VRAS-1.1)
"""

__version__ = "1.1.0"
__author__ = "OmniForge Team / Massive Magnetics"
__license__ = "Proprietary"

from .backend.core import *
from .backend.runtime import *

__all__ = [
    'register_node',
    'input_port',
    'output_port',
    'node_param',
    'node_operation',
    'ExecutionEngine',
    'OmniForgeProject',
    'ASTNodeScanner'
]
