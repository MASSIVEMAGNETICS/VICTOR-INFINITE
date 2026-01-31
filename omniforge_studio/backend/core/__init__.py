"""
OmniForge Studio v1.1 - Backend Core Module
Visual Runtime Architecture Standard 1.1 (VRAS-1.1)
"""

from .node_decorators import (
    register_node,
    input_port,
    output_port,
    node_param,
    node_operation,
    PortType,
    PortDataType,
    NodePort,
    NodeMetadata,
    get_node_registry,
    clear_registry,
    scan_class_for_ports
)

from .ast_scanner import ASTNodeScanner, quick_scan

from .project_serializer import OmniForgeProject, ProjectManager

__all__ = [
    'register_node',
    'input_port',
    'output_port',
    'node_param',
    'node_operation',
    'PortType',
    'PortDataType',
    'NodePort',
    'NodeMetadata',
    'get_node_registry',
    'clear_registry',
    'scan_class_for_ports',
    'ASTNodeScanner',
    'quick_scan',
    'OmniForgeProject',
    'ProjectManager'
]

__version__ = "1.1.0"
