#!/usr/bin/env python3
"""
FILE: omniforge_studio/backend/core/node_decorators.py
VERSION: v1.1.0-VRAS
NAME: Node Decorators System
PURPOSE: Decorator-based node registration system for OmniForge Studio
AUTHOR: OmniForge Team / Massive Magnetics
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""

import functools
from typing import Any, Callable, Optional, Dict, List
from enum import Enum


class PortType(Enum):
    """Port type definitions for node I/O"""
    INPUT = "input"
    OUTPUT = "output"
    PARAM = "param"


class PortDataType(Enum):
    """Data type definitions for ports"""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    TENSOR = "tensor"
    SPIKE_TRAIN = "spike_train"
    ANY = "any"


class NodePort:
    """Represents a single I/O port on a node"""
    def __init__(self, name: str, port_type: PortType, data_type: PortDataType = PortDataType.ANY,
                 description: str = "", default: Any = None, required: bool = True):
        self.name = name
        self.port_type = port_type
        self.data_type = data_type
        self.description = description
        self.default = default
        self.required = required
        self.connected_to: List[str] = []  # List of connected node IDs

    def to_dict(self) -> Dict:
        """Serialize port to dictionary"""
        return {
            "name": self.name,
            "port_type": self.port_type.value,
            "data_type": self.data_type.value,
            "description": self.description,
            "default": self.default,
            "required": self.required,
            "connected_to": self.connected_to
        }


class NodeMetadata:
    """Stores metadata about a registered node"""
    def __init__(self, name: str, category: str = "General"):
        self.name = name
        self.category = category
        self.inputs: Dict[str, NodePort] = {}
        self.outputs: Dict[str, NodePort] = {}
        self.params: Dict[str, NodePort] = {}
        self.operations: List[str] = []
        self.description: str = ""
        self.version: str = "1.0.0"

    def add_port(self, port: NodePort):
        """Add a port to the appropriate collection"""
        if port.port_type == PortType.INPUT:
            self.inputs[port.name] = port
        elif port.port_type == PortType.OUTPUT:
            self.outputs[port.name] = port
        elif port.port_type == PortType.PARAM:
            self.params[port.name] = port

    def to_dict(self) -> Dict:
        """Serialize metadata to dictionary"""
        return {
            "name": self.name,
            "category": self.category,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "operations": self.operations,
            "description": self.description,
            "version": self.version
        }


# Global registry for node metadata
_NODE_REGISTRY: Dict[str, NodeMetadata] = {}


def register_node(name: str, category: str = "General"):
    """Class decorator to register a node"""
    def decorator(cls):
        metadata = NodeMetadata(name, category)
        metadata.description = cls.__doc__ or ""
        
        # Store metadata on class
        cls._omniforge_metadata = metadata
        _NODE_REGISTRY[name] = metadata
        
        return cls
    return decorator


def input_port(name: str, data_type: PortDataType = PortDataType.ANY,
               description: str = "", default: Any = None, required: bool = True):
    """Decorator to mark a method parameter as an input port"""
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_omniforge_inputs'):
            func._omniforge_inputs = []
        
        port = NodePort(name, PortType.INPUT, data_type, description, default, required)
        func._omniforge_inputs.append(port)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Preserve the port metadata
        wrapper._omniforge_inputs = func._omniforge_inputs
        return wrapper
    return decorator


def output_port(name: str, data_type: PortDataType = PortDataType.ANY,
                description: str = ""):
    """Decorator to mark a method return value as an output port"""
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_omniforge_outputs'):
            func._omniforge_outputs = []
        
        port = NodePort(name, PortType.OUTPUT, data_type, description)
        func._omniforge_outputs.append(port)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Preserve the port metadata
        wrapper._omniforge_outputs = func._omniforge_outputs
        return wrapper
    return decorator


def node_param(name: str, data_type: PortDataType = PortDataType.ANY,
               description: str = "", default: Any = None):
    """Decorator to mark a class attribute as a configurable parameter"""
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_omniforge_params'):
            func._omniforge_params = []
        
        port = NodePort(name, PortType.PARAM, data_type, description, default, False)
        func._omniforge_params.append(port)
        
        return func
    return decorator


def node_operation(func: Callable) -> Callable:
    """Decorator to mark a method as a node operation"""
    func._omniforge_operation = True
    return func


def get_node_registry() -> Dict[str, NodeMetadata]:
    """Get the global node registry"""
    return _NODE_REGISTRY


def clear_registry():
    """Clear the node registry (mainly for testing)"""
    _NODE_REGISTRY.clear()


def scan_class_for_ports(cls) -> NodeMetadata:
    """Scan a class for decorated methods and extract port information"""
    if hasattr(cls, '_omniforge_metadata'):
        metadata = cls._omniforge_metadata
    else:
        metadata = NodeMetadata(cls.__name__)
        metadata.description = cls.__doc__ or ""
    
    # Scan all methods for decorators
    for attr_name in dir(cls):
        if attr_name.startswith('_'):
            continue
        
        try:
            attr = getattr(cls, attr_name)
            
            # Check for input ports
            if hasattr(attr, '_omniforge_inputs'):
                for port in attr._omniforge_inputs:
                    metadata.add_port(port)
            
            # Check for output ports
            if hasattr(attr, '_omniforge_outputs'):
                for port in attr._omniforge_outputs:
                    metadata.add_port(port)
            
            # Check for parameters
            if hasattr(attr, '_omniforge_params'):
                for port in attr._omniforge_params:
                    metadata.add_port(port)
            
            # Check for operations
            if hasattr(attr, '_omniforge_operation'):
                metadata.operations.append(attr_name)
        except (AttributeError, TypeError):
            continue
    
    return metadata
