# FILE: synapse.py
# VERSION: v2.0.0-SYNAPSE-GODCORE
# NAME: Synapse
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Defines the Synapse class representing weighted connections between neurons.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

class Synapse:
    """Represents a connection from one neuron to another with a specific weight."""
    def __init__(self, target_node, weight):
        self.target = target_node  # The BioNode instance this synapse connects to
        self.weight = weight       # The strength of this synaptic connection

    def __repr__(self):
        return f"Synapse(target_id={self.target.id}, weight={self.weight:.2f})"


# FILE: bionode.py
# VERSION: v2.0.0-BIONODE-GODCORE
# NAME: BioNode
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Implements a Leaky Integrate-and-Fire neuron with dynamic threshold and spike history.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

from collections import deque
from synapse import Synapse  # adjust import if using a package

# Configuration constants (can be imported from a config module)
INITIAL_VOLTAGE_RANGE = (-0.1, 0.1)
SPIKE_THRESHOLD = 1.0
DECAY_FACTOR = 0.98
DYNAMIC_THRESHOLD_INCREASE = 0.05
DYNAMIC_THRESHOLD_DECAY = 0.99
SPIKE_HISTORY_WINDOW = 10

class BioNode:
    """
    Simulates a single biological neuron with Leaky Integrate-and-Fire (LIF) behavior.
    Includes spike history, dynamic threshold, and weighted synapses.
    """
    def __init__(self, node_id, initial_voltage=0.0, threshold=SPIKE_THRESHOLD):
        self.id = node_id
        self.voltage = initial_voltage
        self.threshold = threshold
        self.base_threshold = threshold # Store original threshold for dynamic adjustments
        self.spiked_this_step = False
        self.last_spike_time = -1      # Track the last emulation step this node spiked at
        self.spike_history = deque(maxlen=SPIKE_HISTORY_WINDOW) # For future STDP
        self.neighbors = []            # List of Synapse objects

    def add_neighbor(self, target_node, weight):
        """Adds a weighted synaptic connection to a target node."""
        self.neighbors.append(Synapse(target_node, weight))

    def integrate(self, input_current):
        """Integrates incoming current and applies decay."""
        self.voltage += input_current
        self.voltage *= DECAY_FACTOR  # Natural decay (leakiness)

    def spike(self, current_time):
        """
        Checks if the node's voltage exceeds its threshold.
        If so, it "spikes", resets its voltage, and signals its neighbors.
        Updates spike history and dynamic threshold.
        """
        self.spiked_this_step = False
        if self.voltage >= self.threshold:
            self.spiked_this_step = True
            self.voltage = 0.0  # Reset voltage after spiking
            self.last_spike_time = current_time
            self.spike_history.append(current_time) # Record spike time

            # Dynamic threshold increase (fatigue)
            self.threshold += DYNAMIC_THRESHOLD_INCREASE * (1.0 - self.threshold / (self.base_threshold * 2))
            return True
        return False

    def decay_threshold(self):
        """Allows the dynamic threshold to slowly decay back to its base level."""
        if self.threshold > self.base_threshold:
            self.threshold = self.threshold * DYNAMIC_THRESHOLD_DECAY
            if self.threshold < self.base_threshold:
                self.threshold = self.base_threshold

    def __repr__(self):
        return f"BioNode(id={self.id}, V={self.voltage:.2f}, T={self.threshold:.2f})"


# FILE: spiking_neural_emulator.py
# VERSION: v2.0.0-EMULATOR-GODCORE
# NAME: SpikingNeuralEmulator
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Simulates and manages a network of BioNodes for spiking neural dynamics.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import random
import time
import numpy as np
from bionode import BioNode
from synapse import Synapse

# Simulation parameters
STIMULUS_INTERVAL = 50
STIMULUS_STRENGTH = 2.0
EMULATOR_TICK_DELAY = 0.01

class SpikingNeuralEmulator:
    """
    Manages a collection of BioNodes and their interactions, simulating
    a basic spiking neural network in an emulated, continuous environment.
    """
    def __init__(self, num_nodes=100, connection_fanout=3):
        self.nodes = []
        self.spike_counts = {}          # node_id: total_spikes
        self.active_spikes_per_step = 0 # For logging current step's activity
        self.current_time_step = 0
        self.running = False            # Control flag for the emulator loop

        # 1. Initialize Nodes
        for i in range(num_nodes):
            initial_voltage = random.uniform(*INITIAL_VOLTAGE_RANGE)
            node = BioNode(i, initial_voltage)
            self.nodes.append(node)
            self.spike_counts[i] = 0

        # 2. Establish Connections
        for node in self.nodes:
            possible_neighbors = [n for n in self.nodes if n.id != node.id]
            chosen = random.sample(possible_neighbors, min(connection_fanout, len(possible_neighbors)))
            for neighbor in chosen:
                weight = random.uniform(*BASE_SYNAPTIC_WEIGHT_RANGE)
                node.add_neighbor(neighbor, weight)

        print(f"Initialized Spiking Neural Emulator with {num_nodes} nodes.")
        print(f"Each node connects to {connection_fanout} random neighbors.")

    def _single_emulator_step(self):
        """Performs one step of the neural network emulation."""
        self.current_time_step += 1
        self.active_spikes_per_step = 0
        incoming_currents = {node.id: 0.0 for node in self.nodes}

        # External stimulus
        if self.current_time_step % STIMULUS_INTERVAL == 0:
            stim = random.choice(self.nodes)
            incoming_currents[stim.id] += STIMULUS_STRENGTH

        # Phase 1: Integrate & Spike
        spiking_nodes = []
        for node in self.nodes:
            node.integrate(incoming_currents[node.id])
            if node.spike(self.current_time_step):
                spiking_nodes.append(node)
                self.spike_counts[node.id] += 1
                self.active_spikes_per_step += 1

        # Phase 2: Propagate
        for spiker in spiking_nodes:
            for syn in spiker.neighbors:
                syn.target.voltage += syn.weight

        # Phase 3: Decay thresholds
        for node in self.nodes:
            node.decay_threshold()

    def start_emulator(self, max_steps=None):
        """Starts the continuous emulation loop."""
        self.running = True
        print("--- Starting Spiking Neural Emulator ---")
        step = 0
        while self.running and (max_steps is None or step < max_steps):
            self._single_emulator_step()
            step += 1
            if self.current_time_step % STIMULUS_INTERVAL == 0:
                total = sum(self.spike_counts.values())
                print(f"Step {self.current_time_step}: Active Spikes={self.active_spikes_per_step}, Total Spikes={total}")
            time.sleep(EMULATOR_TICK_DELAY)
        print("--- Emulator Stopped ---")

    def stop_emulator(self):
        self.running = False

    def get_spike_activity_report(self):
        report = []
        for nid, count in sorted(self.spike_counts.items(), key=lambda x: x[1], reverse=True):
            report.append((nid, count))
        return report

    def get_network_summary(self):
        summary = {
            'num_nodes': len(self.nodes),
            'current_step': self.current_time_step,
            'total_spikes': sum(self.spike_counts.values()),
            'avg_voltage': float(np.mean([n.voltage for n in self.nodes])),
            'avg_threshold': float(np.mean([n.threshold for n in self.nodes]))
        }
        return summary

    def inject_current(self, node_id, strength):
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].voltage += strength
            print(f"Injected {strength:.2f} current into Node {node_id}.")
        else:
            print(f"Error: Node ID {node_id} out of range.")


# FILE: run_emulator.py
# VERSION: v2.0.0-RUNNER-GODCORE
# NAME: EmulatorRunner
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Entry point to launch the Spiking Neural Emulator and handle user commands.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import threading
from spiking_neural_emulator import SpikingNeuralEmulator

if __name__ == "__main__":
    emulator = SpikingNeuralEmulator(num_nodes=100, connection_fanout=3)
    t = threading.Thread(target=emulator.start_emulator)
    t.daemon = True
    t.start()

    while True:
        cmd = input("Enter command (status, stop, stimulate <node> <strength>): ").strip().lower()
        if cmd == 'stop':
            emulator.stop_emulator()
            break
        elif cmd == 'status':
            print(emulator.get_network_summary())
            print(emulator.get_spike_activity_report())
        elif cmd.startswith('stimulate '):
            parts = cmd.split()
            try:
                nid = int(parts[1]); strength = float(parts[2])
                emulator.inject_current(nid, strength)
            except:
                print("Usage: stimulate <node_id> <strength>")
        else:
            print("Unknown command.")
