# FILE: /Victor/modules/bio_snn_v2_1.py
# VERSION: v2.1.0-STDPFULL-GODCORE
# NAME: BioSNN
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Plastic spiking micro‑cortex with full STDP (LTP+LTD), refractory lockout, adaptive thresholds, and excitatory/inhibitory synapses. Designed for plug‑n‑play inside Victor’s fractal cognition stack.
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

import random, math
from typing import List

class Synapse:
    """Weighted connection carrying voltage & plasticity metadata."""
    def __init__(self, target: "BioNode", weight: float):
        self.target = target
        self.weight = weight  # can be ±
        self.last_update = 0  # tick of last STDP change

    def clamp(self, w_min: float = -2.0, w_max: float = 2.0):
        """Keep weights bounded to avoid blow‑ups."""
        self.weight = max(w_min, min(w_max, self.weight))

class BioNode:
    """Leaky‑integrate‑and‑fire neuron with refractory + full STDP."""
    def __init__(self, nid: int):
        self.id = nid
        self.v = random.uniform(0.0, 1.0)
        self.th = 2.0
        self.decay = 0.985
        self.refract = 4          # ticks of silence after spike
        self.last_spike = -1
        self.neigh: List[Synapse] = []

    # ---- connectivity ----
    def add_neigh(self, node: "BioNode"):
        w = random.uniform(-1.2, 1.2)  # excitatory (>0) or inhibitory (<0)
        self.neigh.append(Synapse(node, w))

    # ---- per‑tick update ----
    def step(self, t: int) -> bool:
        """Returns True if this neuron fires on tick t."""
        # refractory lockout
        if t - self.last_spike < self.refract:
            self.v *= self.decay
            return False

        # check for spike
        if self.v > self.th:
            t_pre = t
            self.last_spike = t_pre
            self.v = 0.0  # reset membrane

            for syn in self.neigh:
                # propagate spike first
                syn.target.v += syn.weight

                # ---- full STDP ----
                t_post = syn.target.last_spike
                if t_post == -1:
                    continue  # target never spiked yet
                dt = t_post - t_pre  # +ve: pre before post; ‑ve: post before pre

                # LTP: causal window 0<dt<20
                if 0 < dt < 20:
                    syn.weight += 0.05 * math.exp(-dt / 10)
                # LTD: acausal window -20<dt<0
                elif -20 < dt < 0:
                    syn.weight -= 0.05 * math.exp(dt / 10)  # dt negative → exp(dt/10) in (0,1)

                syn.clamp()
                syn.last_update = t

            # threshold fatigue
            self.th += 0.1
            return True

        # leak + threshold recovery
        self.v *= self.decay
        self.th = max(1.5, self.th - 0.005)
        return False

# ---- quick driver (pure python; swap for numpy for scale) ----

def run_net(ticks: int = 500, n_nodes: int = 50, conn: int = 5,
            stim_int: int = 40, stim_amp: float = 2.5):
    """Runs the network and returns per‑node spike counts list."""
    net = [BioNode(i) for i in range(n_nodes)]
    for n in net:
        for nb in random.sample([x for x in net if x is not n], conn):
            n.add_neigh(nb)

    spike_ct = [0] * n_nodes
    for t in range(ticks):
        if t and t % stim_int == 0:
            random.choice(net).v += stim_amp
        for n in net:
            if n.step(t):
                spike_ct[n.id] += 1
    return spike_ct

if __name__ == "__main__":
    sc = run_net()
    print(f"Total spikes: {sum(sc)}  |  Active neurons: {sum(1 for x in sc if x)}")
