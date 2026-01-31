#!/usr/bin/env python3
"""
FILE: omniforge_studio/backend/core/ast_scanner.py
VERSION: v1.1.0-VRAS
NAME: AST Node Scanner
PURPOSE: Auto-scan Python files and extract node definitions using AST
AUTHOR: OmniForge Team / Massive Magnetics
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""

import ast
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .node_decorators import (
    NodeMetadata, NodePort, PortType, PortDataType,
    get_node_registry, scan_class_for_ports
)


class ASTNodeScanner:
    """Scans Python files using AST to extract node information"""
    
    def __init__(self):
        self.scanned_files: Dict[str, NodeMetadata] = {}
    
    def scan_file(self, file_path: str) -> List[NodeMetadata]:
        """
        Scan a Python file and extract all node definitions
        
        Args:
            file_path: Path to the Python file to scan
            
        Returns:
            List of NodeMetadata objects found in the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try to import and scan using decorators first
        nodes_from_decorators = self._scan_with_import(file_path)
        if nodes_from_decorators:
            return nodes_from_decorators
        
        # Fall back to AST scanning for non-decorated files
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
            return self._scan_ast(tree, file_path)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return []
    
    def _scan_with_import(self, file_path: Path) -> List[NodeMetadata]:
        """Try to import the module and scan for decorated classes"""
        nodes = []
        
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for registered nodes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if hasattr(obj, '_omniforge_metadata'):
                        nodes.append(obj._omniforge_metadata)
                    else:
                        # Scan for decorated methods even without class decorator
                        metadata = scan_class_for_ports(obj)
                        if metadata.inputs or metadata.outputs or metadata.operations:
                            nodes.append(metadata)
        except Exception as e:
            print(f"Could not import {file_path}: {e}")
            return []
        
        return nodes
    
    def _scan_ast(self, tree: ast.AST, file_path: Path) -> List[NodeMetadata]:
        """Scan AST tree for node definitions"""
        nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metadata = self._extract_class_metadata(node, file_path)
                if metadata:
                    nodes.append(metadata)
            elif isinstance(node, ast.FunctionDef):
                # Check for standalone functions that could be nodes
                metadata = self._extract_function_metadata(node, file_path)
                if metadata:
                    nodes.append(metadata)
        
        return nodes
    
    def _extract_class_metadata(self, node: ast.ClassDef, file_path: Path) -> Optional[NodeMetadata]:
        """Extract metadata from a class definition"""
        metadata = NodeMetadata(node.name)
        metadata.description = ast.get_docstring(node) or ""
        
        # Scan methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._extract_method_info(item, metadata)
        
        # Only return if we found something useful
        if metadata.inputs or metadata.outputs or metadata.operations:
            return metadata
        
        return None
    
    def _extract_function_metadata(self, node: ast.FunctionDef, file_path: Path) -> Optional[NodeMetadata]:
        """Extract metadata from a standalone function"""
        # Look for functions that look like node operations
        if node.name.startswith('_'):
            return None
        
        metadata = NodeMetadata(node.name)
        metadata.description = ast.get_docstring(node) or ""
        
        # Extract function signature
        self._extract_function_signature(node, metadata)
        
        # Only return if we found something useful
        if metadata.inputs or metadata.outputs:
            metadata.operations.append(node.name)
            return metadata
        
        return None
    
    def _extract_method_info(self, node: ast.FunctionDef, metadata: NodeMetadata):
        """Extract information from a method"""
        # Skip private methods
        if node.name.startswith('_') and node.name not in ['__init__', '__call__']:
            return
        
        # Extract signature
        self._extract_function_signature(node, metadata)
        
        # Mark as operation if it's a likely candidate
        if node.name in ['step', 'run', 'execute', 'process', '__call__']:
            metadata.operations.append(node.name)
    
    def _extract_function_signature(self, node: ast.FunctionDef, metadata: NodeMetadata):
        """Extract function signature and infer I/O ports"""
        # Parse arguments
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
            
            # Infer data type from annotation
            data_type = PortDataType.ANY
            if arg.annotation:
                data_type = self._infer_type_from_annotation(arg.annotation)
            
            # Create input port
            port = NodePort(
                name=arg.arg,
                port_type=PortType.INPUT,
                data_type=data_type,
                description=f"Input parameter: {arg.arg}"
            )
            metadata.add_port(port)
        
        # Check return annotation for output
        if node.returns:
            data_type = self._infer_type_from_annotation(node.returns)
            port = NodePort(
                name="output",
                port_type=PortType.OUTPUT,
                data_type=data_type,
                description=f"Return value from {node.name}"
            )
            metadata.add_port(port)
    
    def _infer_type_from_annotation(self, annotation: ast.expr) -> PortDataType:
        """Infer PortDataType from type annotation"""
        if isinstance(annotation, ast.Name):
            type_map = {
                'int': PortDataType.INT,
                'float': PortDataType.FLOAT,
                'str': PortDataType.STRING,
                'bool': PortDataType.BOOL,
                'list': PortDataType.LIST,
                'dict': PortDataType.DICT,
            }
            return type_map.get(annotation.id, PortDataType.ANY)
        elif isinstance(annotation, ast.Subscript):
            # Handle List[T], Dict[K,V], etc.
            if isinstance(annotation.value, ast.Name):
                if annotation.value.id == 'List':
                    return PortDataType.LIST
                elif annotation.value.id == 'Dict':
                    return PortDataType.DICT
        
        return PortDataType.ANY
    
    def scan_directory(self, directory: str, pattern: str = "*.py") -> Dict[str, List[NodeMetadata]]:
        """
        Scan all Python files in a directory
        
        Args:
            directory: Directory path to scan
            pattern: File pattern to match (default: *.py)
            
        Returns:
            Dictionary mapping file paths to lists of NodeMetadata
        """
        directory = Path(directory)
        results = {}
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                nodes = self.scan_file(str(file_path))
                if nodes:
                    results[str(file_path)] = nodes
        
        return results
    
    def get_node_summary(self, metadata: NodeMetadata) -> Dict[str, Any]:
        """Get a summary of a node's capabilities"""
        return {
            "name": metadata.name,
            "category": metadata.category,
            "description": metadata.description,
            "input_count": len(metadata.inputs),
            "output_count": len(metadata.outputs),
            "param_count": len(metadata.params),
            "operations": metadata.operations,
            "version": metadata.version
        }


def quick_scan(file_path: str) -> List[Dict[str, Any]]:
    """Quick scan utility function"""
    scanner = ASTNodeScanner()
    nodes = scanner.scan_file(file_path)
    return [scanner.get_node_summary(node) for node in nodes]
