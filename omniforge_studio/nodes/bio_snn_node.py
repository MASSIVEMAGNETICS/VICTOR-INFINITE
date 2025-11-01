#!/usr/bin/env python3
"""
FILE: omniforge_studio/nodes/bio_snn_node.py
VERSION: v1.1.0-VRAS
NAME: BioSNN OmniForge Node
PURPOSE: Decorated BioSNN node for OmniForge Studio
AUTHOR: OmniForge Team / Massive Magnetics
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""

import random
import math
from typing import List
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from core.node_decorators import (
    register_node, input_port, output_port, node_operation, node_param,
    PortDataType
)


class Synapse:
    """Weighted connection carrying voltage & plasticity metadata."""
    def __init__(self, target: "BioNode", weight: float):
        self.target = target
        self.weight = weight
        self.last_update = 0

    def clamp(self, w_min: float = -2.0, w_max: float = 2.0):
        """Keep weights bounded."""
        self.weight = max(w_min, min(w_max, self.weight))


class BioNode:
    """Leaky-integrate-and-fire neuron with refractory + STDP."""
    def __init__(self, nid: int):
        self.id = nid
        self.v = random.uniform(0.0, 1.0)
        self.th = 2.0
        self.decay = 0.985
        self.refract = 4
        self.last_spike = -1
        self.neigh: List[Synapse] = []

    def add_neigh(self, node: "BioNode"):
        w = random.uniform(-1.2, 1.2)
        self.neigh.append(Synapse(node, w))

    def step(self, t: int) -> bool:
        """Returns True if this neuron fires on tick t."""
        if t - self.last_spike < self.refract:
            self.v *= self.decay
            return False

        if self.v > self.th:
            self.last_spike = t
            self.v = 0.0
            return True

        self.v *= self.decay
        return False

    def receive(self, signal: float):
        """Accumulate external or neighbor signals."""
        self.v += signal

    def apply_stdp(self, t: int, tau: float = 20.0, a_ltp: float = 0.01, a_ltd: float = 0.01):
        """Full STDP: LTP for pre→post, LTD for post→pre."""
        for syn in self.neigh:
            delta_t = t - syn.target.last_spike
            if delta_t > 0 and syn.target.last_spike > 0:
                syn.weight += a_ltp * math.exp(-delta_t / tau)
            elif delta_t < 0:
                syn.weight -= a_ltd * math.exp(delta_t / tau)
            syn.clamp()
            syn.last_update = t


@register_node("BioSNN", category="Neural")
class BioSNNNode:
    """
    Bio-inspired Spiking Neural Network with STDP learning.
    A plastic spiking micro-cortex for fractal cognition.
    """
    
    def __init__(self, n_neurons: int = 10):
        self.n = n_neurons
        self.nodes: List[BioNode] = [BioNode(i) for i in range(n_neurons)]
        self.tick = 0
        self.spike_history: List[List[int]] = []
        
        # Create random topology
        for node in self.nodes:
            for _ in range(random.randint(2, 5)):
                target = random.choice(self.nodes)
                if target.id != node.id:
                    node.add_neigh(target)
    
    @node_param("n_neurons", PortDataType.INT, "Number of neurons in the network", default=10)
    def set_neurons(self, n: int):
        """Configure number of neurons"""
        self.n = n
    
    @node_param("learning_rate", PortDataType.FLOAT, "STDP learning rate", default=0.01)
    def set_learning_rate(self, rate: float):
        """Configure learning rate"""
        self.learning_rate = rate
    
    @input_port("stimulus", PortDataType.FLOAT, "External stimulus intensity", default=0.0)
    @output_port("spike_count", PortDataType.INT, "Number of neurons that spiked")
    @output_port("network_activity", PortDataType.FLOAT, "Average membrane potential")
    @node_operation
    def step(self, t: int, stimulus: float = 0.0) -> dict:
        """
        Execute one simulation step
        
        Args:
            t: Current time tick
            stimulus: External stimulus to apply
            
        Returns:
            Dictionary with spike_count and network_activity
        """
        self.tick = t
        
        # Apply external stimulus to random subset
        if stimulus > 0:
            n_stim = max(1, self.n // 3)
            targets = random.sample(self.nodes, n_stim)
            for node in targets:
                node.receive(stimulus)
        
        # Propagate spikes
        spikes = []
        for node in self.nodes:
            if node.step(t):
                spikes.append(node.id)
                # Forward spike to neighbors
                for syn in node.neigh:
                    syn.target.receive(syn.weight * 0.5)
        
        # Apply STDP learning
        for node in self.nodes:
            if node.id in spikes:
                node.apply_stdp(t)
        
        # Track history
        self.spike_history.append(spikes)
        
        # Calculate metrics
        spike_count = len(spikes)
        avg_potential = sum(n.v for n in self.nodes) / self.n
        
        return {
            'spike_count': spike_count,
            'network_activity': avg_potential
        }
    
    @node_operation
    def reset(self):
        """Reset the network to initial state"""
        self.tick = 0
        self.spike_history.clear()
        for node in self.nodes:
            node.v = random.uniform(0.0, 1.0)
            node.last_spike = -1
    
    @node_operation
    def get_spike_raster(self) -> List[List[int]]:
        """Get the spike raster plot data"""
        return self.spike_history
    
    def to_dict(self) -> dict:
        """Serialize state"""
        return {
            'n_neurons': self.n,
            'tick': self.tick,
            'spike_history_length': len(self.spike_history),
            'total_spikes': sum(len(spikes) for spikes in self.spike_history)
        }


# Example of standalone function node
@input_port("x", PortDataType.FLOAT, "Input value")
@output_port("result", PortDataType.FLOAT, "Computed result")
def sigmoid_activation(x: float) -> float:
    """Sigmoid activation function node"""
    return 1.0 / (1.0 + math.exp(-x))


@register_node("StimGenerator", category="Utilities")
class StimGeneratorNode:
    """
    Generates periodic or random stimulus signals for neural networks.
    """
    
    def __init__(self, mode: str = "periodic", amplitude: float = 1.0):
        self.mode = mode  # "periodic", "random", "burst"
        self.amplitude = amplitude
        self.tick = 0
    
    @input_port("tick", PortDataType.INT, "Current time tick")
    @output_port("stimulus", PortDataType.FLOAT, "Generated stimulus value")
    @node_operation
    def step(self, tick: int) -> float:
        """Generate stimulus for current tick"""
        self.tick = tick
        
        if self.mode == "periodic":
            # Sine wave
            return self.amplitude * math.sin(tick / 10.0)
        elif self.mode == "random":
            # Random pulses
            return self.amplitude if random.random() > 0.8 else 0.0
        elif self.mode == "burst":
            # Burst pattern
            if (tick % 100) < 20:
                return self.amplitude
            return 0.0
        
        return 0.0


if __name__ == "__main__":
    # Test the node
    print("Testing BioSNN Node...")
    node = BioSNNNode(n_neurons=20)
    
    for t in range(100):
        result = node.step(t, stimulus=1.5 if t % 20 == 0 else 0.0)
        if result['spike_count'] > 0:
            print(f"Tick {t}: {result['spike_count']} spikes, activity={result['network_activity']:.3f}")
    
    print(f"\nTotal simulation ticks: {node.tick}")
    print(f"Spike history length: {len(node.spike_history)}")
