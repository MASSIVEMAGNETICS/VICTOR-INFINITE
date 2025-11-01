#!/usr/bin/env python3
"""
FILE: omniforge_studio/backend/core/project_serializer.py
VERSION: v1.1.0-VRAS
NAME: Project Serialization System
PURPOSE: Save/load .omniforgeproj files with graph state
AUTHOR: OmniForge Team / Massive Magnetics
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""

import json
import pickle
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib


class OmniForgeProject:
    """Represents an OmniForge Studio project"""
    
    VERSION = "1.1.0"
    
    def __init__(self):
        self.metadata = {
            'version': self.VERSION,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'author': '',
            'description': '',
            'tags': []
        }
        
        self.graph = {
            'nodes': {},
            'wires': {},
            'node_positions': {}  # UI positions
        }
        
        self.runtime_config = {
            'tick_rate': 60,
            'max_workers': None,
            'execution_mode': 'real_time',
            'seed': None
        }
        
        self.node_code = {}  # Store source code for custom nodes
        self.binary_data = {}  # Store pickled objects (weights, etc.)
    
    def add_node(self, node_id: str, node_type: str, params: Dict = None,
                 position: tuple = (0, 0), source_code: str = None):
        """Add a node to the project"""
        self.graph['nodes'][node_id] = {
            'node_type': node_type,
            'params': params or {},
            'created': datetime.now().isoformat()
        }
        
        self.graph['node_positions'][node_id] = {
            'x': position[0],
            'y': position[1]
        }
        
        if source_code:
            self.node_code[node_id] = source_code
    
    def add_wire(self, wire_id: str, from_node: str, from_port: str,
                 to_node: str, to_port: str):
        """Add a wire to the project"""
        self.graph['wires'][wire_id] = {
            'from_node': from_node,
            'from_port': from_port,
            'to_node': to_node,
            'to_port': to_port
        }
    
    def store_binary_data(self, key: str, data: Any):
        """Store binary data (e.g., model weights)"""
        # Serialize with pickle and encode as base64
        pickled = pickle.dumps(data)
        encoded = base64.b64encode(pickled).decode('utf-8')
        self.binary_data[key] = encoded
    
    def get_binary_data(self, key: str) -> Any:
        """Retrieve binary data"""
        if key not in self.binary_data:
            return None
        
        encoded = self.binary_data[key]
        decoded = base64.b64decode(encoded.encode('utf-8'))
        return pickle.loads(decoded)
    
    def save(self, file_path: str):
        """
        Save project to a .omniforgeproj file
        
        Args:
            file_path: Path to save the project file
        """
        file_path = Path(file_path)
        
        # Ensure .omniforgeproj extension
        if file_path.suffix != '.omniforgeproj':
            file_path = file_path.with_suffix('.omniforgeproj')
        
        # Update modified timestamp
        self.metadata['modified'] = datetime.now().isoformat()
        
        # Prepare data for serialization
        project_data = {
            'metadata': self.metadata,
            'graph': self.graph,
            'runtime_config': self.runtime_config,
            'node_code': self.node_code,
            'binary_data': self.binary_data
        }
        
        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2)
        
        print(f"Project saved to: {file_path}")
        return str(file_path)
    
    @classmethod
    def load(cls, file_path: str) -> 'OmniForgeProject':
        """
        Load project from a .omniforgeproj file
        
        Args:
            file_path: Path to the project file
            
        Returns:
            OmniForgeProject instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Project file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
        
        # Create new project
        project = cls()
        
        # Load data
        project.metadata = project_data.get('metadata', {})
        project.graph = project_data.get('graph', {'nodes': {}, 'wires': {}, 'node_positions': {}})
        project.runtime_config = project_data.get('runtime_config', {})
        project.node_code = project_data.get('node_code', {})
        project.binary_data = project_data.get('binary_data', {})
        
        print(f"Project loaded from: {file_path}")
        return project
    
    def export_summary(self) -> Dict[str, Any]:
        """Export a summary of the project"""
        return {
            'metadata': self.metadata,
            'node_count': len(self.graph['nodes']),
            'wire_count': len(self.graph['wires']),
            'has_binary_data': bool(self.binary_data),
            'runtime_config': self.runtime_config
        }
    
    def calculate_checksum(self) -> str:
        """Calculate checksum of project data"""
        # Create deterministic string representation
        data_str = json.dumps({
            'graph': self.graph,
            'runtime_config': self.runtime_config
        }, sort_keys=True)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def create_version_diff(self, other: 'OmniForgeProject') -> Dict[str, Any]:
        """Create a diff between this project and another version"""
        diff = {
            'nodes_added': [],
            'nodes_removed': [],
            'nodes_modified': [],
            'wires_added': [],
            'wires_removed': []
        }
        
        # Compare nodes
        our_nodes = set(self.graph['nodes'].keys())
        their_nodes = set(other.graph['nodes'].keys())
        
        diff['nodes_added'] = list(our_nodes - their_nodes)
        diff['nodes_removed'] = list(their_nodes - our_nodes)
        
        # Check for modified nodes
        for node_id in our_nodes & their_nodes:
            if self.graph['nodes'][node_id] != other.graph['nodes'][node_id]:
                diff['nodes_modified'].append(node_id)
        
        # Compare wires
        our_wires = set(self.graph['wires'].keys())
        their_wires = set(other.graph['wires'].keys())
        
        diff['wires_added'] = list(our_wires - their_wires)
        diff['wires_removed'] = list(their_wires - our_wires)
        
        return diff


class ProjectManager:
    """Manages multiple OmniForge projects"""
    
    def __init__(self):
        self.current_project: Optional[OmniForgeProject] = None
        self.project_history: list = []
    
    def new_project(self, author: str = "", description: str = "") -> OmniForgeProject:
        """Create a new project"""
        project = OmniForgeProject()
        project.metadata['author'] = author
        project.metadata['description'] = description
        self.current_project = project
        return project
    
    def open_project(self, file_path: str) -> OmniForgeProject:
        """Open an existing project"""
        project = OmniForgeProject.load(file_path)
        self.current_project = project
        self.project_history.append(file_path)
        return project
    
    def save_project(self, file_path: str = None) -> str:
        """Save the current project"""
        if not self.current_project:
            raise ValueError("No project is currently open")
        
        if not file_path:
            raise ValueError("File path is required")
        
        return self.current_project.save(file_path)
    
    def close_project(self):
        """Close the current project"""
        self.current_project = None
    
    def get_recent_projects(self, limit: int = 10) -> list:
        """Get recently opened projects"""
        return self.project_history[-limit:]
