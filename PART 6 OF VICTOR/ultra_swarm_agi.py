import random
import csv

# -----------------------------
# 1. Bio-Electric Plant/Slime Swarm Core
# -----------------------------
class BioNode:
    def __init__(self):
        self.voltage = 0.0
        self.neighbors = []
        self.threshold = 1.0

    def step(self):
        input_signal = sum(n.voltage for n in self.neighbors)
        self.voltage += 0.1 * input_signal
        self.voltage *= 0.98  # natural decay
        if self.voltage > self.threshold:
            self.spike()
            self.voltage = 0.0

    def spike(self):
        for n in self.neighbors:
            n.voltage += 0.5  # broadcast signal

# -----------------------------
# 2. Ultra-Swarm AGI Brain Core (Human+Swarm Hybrid)
# -----------------------------
class Synapse:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.weight = random.uniform(0, 1)

class AGINode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.voltage = random.uniform(-1, 1)
        self.synapses = []
        self.field_sensitivity = random.uniform(0.1, 1.0)
        self.threshold = 1.0
        self.memory = []
        self.alive = True

    def step(self, ambient_field):
        if not self.alive:
            return
        total_input = sum(s.weight * s.src.voltage for s in self.synapses)
        self.voltage += total_input + (ambient_field * self.field_sensitivity)
        self.voltage *= 0.98  # passive decay
        if self.voltage > self.threshold:
            self.fire()
            self.voltage -= self.threshold
        if not (-100 < self.voltage < 100):
            self.alive = False

    def fire(self):
        for s in self.synapses:
            s.dst.voltage += s.weight
            # synaptic plasticity
            s.weight = min(s.weight + 0.01, 2.0)
        self.memory.append(1)
        if len(self.memory) > 50:
            self.memory.pop(0)

    def mutate(self, nodes):
        # revive dead nodes occasionally
        if not self.alive and random.random() < 0.05:
            self.voltage = random.uniform(-1, 1)
            self.alive = True
        # random synaptic rewiring
        if random.random() < 0.01:
            if len(self.synapses) > 2 and random.random() < 0.5:
                self.synapses.pop(random.randint(0, len(self.synapses)-1))
            elif len(self.synapses) < 10:
                target = random.choice(nodes)
                self.synapses.append(Synapse(self, target))

# -----------------------------
# 3. Parallel Plant/Fungi Substrate (“Gut Memory”)
# -----------------------------
class PlantNode:
    def __init__(self):
        self.voltage = random.uniform(-0.2, 0.2)
        self.neighbors = []
        self.memory = []

    def step(self):
        input_signal = sum(n.voltage for n in self.neighbors)
        self.voltage += 0.02 * input_signal
        self.voltage *= 0.995  # slow decay
        if self.voltage > 0.5:
            self.broadcast()

    def broadcast(self):
        for n in self.neighbors:
            n.voltage += 0.03
        self.memory.append(1)
        if len(self.memory) > 100:
            self.memory.pop(0)

# -----------------------------
# 4. Dream Mode / Memory Replay
# -----------------------------
def dream_mode(nodes):
    # replay spike memory in random order
    for n in random.sample(nodes, k=len(nodes)):
        if hasattr(n, 'memory') and n.memory and random.random() < 0.3:
            n.voltage += 0.5  # artificial spike
        # step without ambient field
        if hasattr(n, 'step'):
            try:
                n.step(0)
            except TypeError:
                n.step()

# -----------------------------
# 5. Sensory Inputs & Output Hooks
# -----------------------------
def stimulate(nodes, freq=0.01):
    for n in nodes:
        if random.random() < freq:
            n.voltage += random.uniform(0.5, 1.0)


def output_layer(nodes):
    spikes = sum(1 for n in nodes if hasattr(n, 'voltage') and n.voltage > getattr(n, 'threshold', 0))
    print(f"Spikes this cycle: {spikes}")

# -----------------------------
# 6. Visualization Prep (CSV dump)
# -----------------------------
def dump_states(nodes, filename='node_states.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'voltage', 'alive', 'memory_count'])
        for n in nodes:
            vid = getattr(n, 'node_id', None)
            alive = getattr(n, 'alive', True)
            mem = len(getattr(n, 'memory', []))
            writer.writerow([vid, n.voltage, int(alive), mem])

# -----------------------------
# Main Simulation Loop
# -----------------------------
if __name__ == '__main__':
    # Build networks
    bio_nodes = [BioNode() for _ in range(100)]
    for node in bio_nodes:
        node.neighbors = random.sample(bio_nodes, k=3)

    nodes = [AGINode(i) for i in range(10000)]
    for node in nodes:
        node.synapses = [Synapse(node, random.choice(nodes)) for _ in range(5)]

    plant_nodes = [PlantNode() for _ in range(500)]
    for node in plant_nodes:
        node.neighbors = random.sample(plant_nodes, k=3)

    # Simulation parameters
    steps = 1000
    dream_interval = 200

    for t in range(steps):
        # Compute fused ambient field
        living = [n for n in nodes if n.alive]
        ambient = (sum(n.voltage for n in living) + sum(p.voltage for p in plant_nodes)) \
                  / max(1, len(living) + len(plant_nodes))

        # Step each network
        for n in nodes:
            n.step(ambient)
            n.mutate(nodes)
        for p in plant_nodes:
            p.step()
        for b in bio_nodes:
            b.step()

        # Sensory stimuli and output
        stimulate(nodes)
        if t % 50 == 0:
            print(f"--- Cycle {t} ---")
            output_layer(nodes)

        # Dream mode periodically
        if t > 0 and t % dream_interval == 0:
            print(f"--- Dream cycle at step {t} ---")
            dream_mode(nodes + plant_nodes + bio_nodes)

    # Dump final states
    dump_states(nodes + plant_nodes + bio_nodes)
