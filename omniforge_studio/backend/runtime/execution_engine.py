#!/usr/bin/env python3
"""
FILE: omniforge_studio/backend/runtime/execution_engine.py
VERSION: v1.1.0-VRAS
NAME: Parallel Execution Engine
PURPOSE: Multi-threaded node execution engine with state management
AUTHOR: OmniForge Team / Massive Magnetics
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""

import threading
import multiprocessing
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
import traceback


class ExecutionMode(Enum):
    """Execution mode for the runtime"""
    REAL_TIME = "real_time"
    SIMULATED = "simulated"
    ACCELERATED = "accelerated"


class NodeState(Enum):
    """State of a node during execution"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class NodeInstance:
    """Represents a single node instance in the graph"""
    node_id: str
    node_type: str
    instance: Any
    state: NodeState = NodeState.IDLE
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    last_execution_time: float = 0.0
    execution_count: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "state": self.state.value,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "params": self.params,
            "error": self.error,
            "last_execution_time": self.last_execution_time,
            "execution_count": self.execution_count
        }


@dataclass
class Wire:
    """Represents a connection between two nodes"""
    wire_id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    active: bool = True
    data_buffer: Any = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "wire_id": self.wire_id,
            "from_node": self.from_node,
            "from_port": self.from_port,
            "to_node": self.to_node,
            "to_port": self.to_port,
            "active": self.active
        }


class ExecutionSnapshot:
    """Snapshot of the execution state for time-travel debugging"""
    def __init__(self, tick: int, nodes: Dict[str, NodeInstance], timestamp: float):
        self.tick = tick
        self.timestamp = timestamp
        self.node_states = {
            node_id: {
                'state': node.state.value,
                'inputs': node.inputs.copy(),
                'outputs': node.outputs.copy(),
            }
            for node_id, node in nodes.items()
        }


class ExecutionEngine:
    """
    Multi-threaded execution engine for OmniForge nodes
    Supports parallel execution, state management, and time-travel debugging
    """
    
    def __init__(self, max_workers: int = None, tick_rate: int = 60):
        """
        Initialize the execution engine
        
        Args:
            max_workers: Maximum number of parallel workers (default: CPU count)
            tick_rate: Execution tick rate in Hz
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.tick_rate = tick_rate
        self.tick_interval = 1.0 / tick_rate
        
        # Graph state
        self.nodes: Dict[str, NodeInstance] = {}
        self.wires: Dict[str, Wire] = {}
        
        # Execution state
        self.running = False
        self.paused = False
        self.current_tick = 0
        self.execution_mode = ExecutionMode.REAL_TIME
        self.speed_multiplier = 1.0
        
        # Threading
        self.execution_thread: Optional[threading.Thread] = None
        self.worker_pool: List[threading.Thread] = []
        self.execution_queue: Queue = Queue()
        
        # State history for time-travel
        self.snapshots: List[ExecutionSnapshot] = []
        self.max_snapshots = 1000
        self.snapshot_interval = 10  # Take snapshot every N ticks
        
        # Telemetry
        self.telemetry = {
            'total_ticks': 0,
            'total_executions': 0,
            'average_tick_time': 0.0,
            'active_threads': 0,
            'errors': []
        }
        
        # Callbacks
        self.on_tick_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
    
    def add_node(self, node_id: str, node_type: str, instance: Any, params: Dict = None) -> str:
        """
        Add a node to the execution graph
        
        Args:
            node_id: Unique identifier for the node (auto-generated if None)
            node_type: Type/class name of the node
            instance: The actual node instance
            params: Node parameters
            
        Returns:
            The node_id
        """
        if not node_id:
            node_id = f"{node_type}_{uuid.uuid4().hex[:8]}"
        
        node_instance = NodeInstance(
            node_id=node_id,
            node_type=node_type,
            instance=instance,
            params=params or {}
        )
        
        self.nodes[node_id] = node_instance
        return node_id
    
    def remove_node(self, node_id: str):
        """Remove a node from the graph"""
        if node_id in self.nodes:
            # Remove all connected wires
            wires_to_remove = [
                wire_id for wire_id, wire in self.wires.items()
                if wire.from_node == node_id or wire.to_node == node_id
            ]
            for wire_id in wires_to_remove:
                del self.wires[wire_id]
            
            del self.nodes[node_id]
    
    def add_wire(self, from_node: str, from_port: str, to_node: str, to_port: str) -> str:
        """
        Connect two nodes with a wire
        
        Returns:
            The wire_id
        """
        wire_id = f"wire_{uuid.uuid4().hex[:8]}"
        wire = Wire(
            wire_id=wire_id,
            from_node=from_node,
            from_port=from_port,
            to_node=to_node,
            to_port=to_port
        )
        self.wires[wire_id] = wire
        return wire_id
    
    def remove_wire(self, wire_id: str):
        """Remove a wire from the graph"""
        if wire_id in self.wires:
            del self.wires[wire_id]
    
    def start(self):
        """Start the execution engine"""
        if self.running:
            return
        
        self.running = True
        self.paused = False
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
    
    def pause(self):
        """Pause execution"""
        self.paused = True
    
    def resume(self):
        """Resume execution"""
        self.paused = False
    
    def stop(self):
        """Stop the execution engine"""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5.0)
    
    def reset(self):
        """Reset the execution state"""
        self.stop()
        self.current_tick = 0
        self.snapshots.clear()
        self.telemetry['total_ticks'] = 0
        self.telemetry['total_executions'] = 0
        
        # Reset all node states
        for node in self.nodes.values():
            node.state = NodeState.IDLE
            node.execution_count = 0
            node.error = None
    
    def rewind_to_tick(self, tick: int) -> bool:
        """
        Rewind execution to a specific tick (time-travel debugging)
        
        Args:
            tick: The tick to rewind to
            
        Returns:
            True if successful, False otherwise
        """
        # Find the closest snapshot
        snapshot = None
        for snap in reversed(self.snapshots):
            if snap.tick <= tick:
                snapshot = snap
                break
        
        if not snapshot:
            return False
        
        # Restore state from snapshot
        self.current_tick = snapshot.tick
        for node_id, state in snapshot.node_states.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.state = NodeState[state['state'].upper()]
                node.inputs = state['inputs'].copy()
                node.outputs = state['outputs'].copy()
        
        return True
    
    def _execution_loop(self):
        """Main execution loop"""
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            tick_start = time.time()
            
            # Execute one tick
            self._execute_tick()
            
            # Take snapshot if needed
            if self.current_tick % self.snapshot_interval == 0:
                self._take_snapshot()
            
            # Call tick callbacks
            for callback in self.on_tick_callbacks:
                try:
                    callback(self.current_tick, self.get_state())
                except Exception as e:
                    print(f"Error in tick callback: {e}")
            
            # Update telemetry
            tick_time = time.time() - tick_start
            self.telemetry['average_tick_time'] = (
                self.telemetry['average_tick_time'] * self.telemetry['total_ticks'] + tick_time
            ) / (self.telemetry['total_ticks'] + 1)
            self.telemetry['total_ticks'] += 1
            
            # Sleep to maintain tick rate
            sleep_time = max(0, self.tick_interval / self.speed_multiplier - tick_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            self.current_tick += 1
    
    def _execute_tick(self):
        """Execute one tick of the simulation"""
        # Propagate data through wires
        for wire in self.wires.values():
            if not wire.active:
                continue
            
            from_node = self.nodes.get(wire.from_node)
            to_node = self.nodes.get(wire.to_node)
            
            if from_node and to_node:
                # Get output from source node
                output_data = from_node.outputs.get(wire.from_port)
                
                # Send to target node input
                if output_data is not None:
                    to_node.inputs[wire.to_port] = output_data
                    wire.data_buffer = output_data
        
        # Execute nodes that have step/run methods
        threads = []
        for node in self.nodes.values():
            if node.state == NodeState.ERROR:
                continue
            
            # Execute in parallel
            thread = threading.Thread(target=self._execute_node, args=(node,))
            thread.start()
            threads.append(thread)
            
            # Limit concurrent threads
            if len(threads) >= self.max_workers:
                for t in threads:
                    t.join()
                threads.clear()
        
        # Wait for remaining threads
        for t in threads:
            t.join()
        
        self.telemetry['active_threads'] = len(threads)
    
    def _execute_node(self, node: NodeInstance):
        """Execute a single node"""
        try:
            node.state = NodeState.RUNNING
            exec_start = time.time()
            
            instance = node.instance
            
            # Try to call step method if it exists
            if hasattr(instance, 'step'):
                result = instance.step(self.current_tick, **node.inputs)
                if result is not None:
                    node.outputs['output'] = result
            
            # Try run method
            elif hasattr(instance, 'run'):
                result = instance.run(**node.inputs)
                if result is not None:
                    node.outputs['output'] = result
            
            # Try __call__
            elif callable(instance):
                result = instance(**node.inputs)
                if result is not None:
                    node.outputs['output'] = result
            
            node.last_execution_time = time.time() - exec_start
            node.execution_count += 1
            node.state = NodeState.COMPLETED
            self.telemetry['total_executions'] += 1
            
        except Exception as e:
            node.state = NodeState.ERROR
            node.error = str(e)
            error_info = {
                'node_id': node.node_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.telemetry['errors'].append(error_info)
            
            for callback in self.on_error_callbacks:
                try:
                    callback(node.node_id, error_info)
                except Exception as cb_error:
                    print(f"Error in error callback: {cb_error}")
    
    def _take_snapshot(self):
        """Take a snapshot of the current state"""
        snapshot = ExecutionSnapshot(
            tick=self.current_tick,
            nodes=self.nodes,
            timestamp=time.time()
        )
        
        self.snapshots.append(snapshot)
        
        # Limit snapshot history
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
    
    def get_state(self) -> Dict:
        """Get the current execution state"""
        return {
            'running': self.running,
            'paused': self.paused,
            'current_tick': self.current_tick,
            'execution_mode': self.execution_mode.value,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'wires': {wire_id: wire.to_dict() for wire_id, wire in self.wires.items()},
            'telemetry': self.telemetry.copy()
        }
    
    def set_execution_mode(self, mode: ExecutionMode, speed_multiplier: float = 1.0):
        """Set the execution mode and speed"""
        self.execution_mode = mode
        self.speed_multiplier = speed_multiplier
