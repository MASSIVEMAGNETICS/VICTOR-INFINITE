# FILE: thelight_tensor_attention_fusion.py
# VERSION: v1.4.0-FRACTAL-ATTENTION-GODCORE
# NAME: TheLightTensorAttentionFusion
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Integrate OmegaTensor-based fractal attention heads into TheLight nodes for a self-organizing attention substrate
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import numpy as np
import copy
import os

from omega_tensor import OmegaTensor
from fractal_attention import FractalAttention

_rng = np.random.default_rng(int(os.getenv("LIGHT_SEED", "0")))  # Deterministic if seed set

def nan_guard(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

class TheLightTensorAttentionFusion:
    def __init__(self,
                 dimensions=3,
                 quantization=0.25,
                 radius=1.0,
                 entropy=0.0,
                 temperature=300,
                 heads=4,
                 head_dim=64):
        # Core fractal parameters
        self.dimensions = dimensions
        self.quantization = quantization
        self.radius = radius
        self.entropy = entropy
        self.temperature = temperature
        self.perimeter_points = self._generate_perimeter()
        self._age = 0.0
        self._homeo_interval = 10.0
        self._triggered = set()

        # OmegaTensor-based embedding for perimeter points
        in_dim = self.perimeter_points.shape[0]
        self.embedding = OmegaTensor(np.random.randn(in_dim, head_dim) * 0.02, requires_grad=True)

        # Fractal attention transformer: multiple mini attention heads
        self.fractal_attention = FractalAttention(
            num_heads=heads,
            head_dim=head_dim,
            tensor_engine=OmegaTensor
        )

    def _generate_perimeter(self):
        num_points = max(3, int(self.quantization * 6) + 1)
        phi = np.pi * (3. - np.sqrt(5.))
        points = []
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2
            rad = np.sqrt(1 - y * y)
            theta = phi * i
            coords = [np.cos(theta) * rad, y, np.sin(theta) * rad][:self.dimensions]
            coords = [c + _rng.normal(0, self.entropy * self.radius * 0.1) for c in coords]
            points.append(coords)
        return nan_guard(np.array(points))

    def coherence_score(self):
        d = np.linalg.norm(self.perimeter_points - self.perimeter_points.mean(axis=0), axis=1)
        score = 1 - (d.std() / (self.radius + 1e-8))
        return float(np.clip(score, 0.0, 1.0))

    def homeostasis(self):
        self.entropy = min(1.0, self.temperature / 1000.0 + 0.001)
        self.perimeter_points = self._generate_perimeter()

    def step(self, dt=1.0):
        self._age += dt
        if self._age % self._homeo_interval < dt:
            self.homeostasis()
        # fuse fractal substrate with attention outputs
        self.apply_attention()

    def apply_attention(self):
        # Embed perimeter points into OmegaTensor space
        tensor_points = OmegaTensor(self.perimeter_points)
        emb = tensor_points.matmul(self.embedding)  # shape: [num_points, head_dim]
        # Multi-head fractal attention (query, key, value all emb)
        attn_out = self.fractal_attention(emb, emb, emb)
        # Update perimeter based on attention output (reshape to original dims)
        new_points = attn_out.data.reshape(self.perimeter_points.shape)
        self.perimeter_points = nan_guard(new_points)

    def on_phase_event(self, threshold, callback, once=True):
        coh = self.coherence_score()
        key = (id(callback), threshold)
        if coh >= threshold and key not in self._triggered:
            callback(self)
            if once:
                self._triggered.add(key)

    def replicate(self):
        if self.coherence_score() >= 0.98:
            shard = copy.deepcopy(self)
            shard.radius *= 0.5
            shard.entropy = min(1.0, self.entropy + 0.05)
            return shard
        return None

    def excite(self, delta_temp=100):
        self.temperature += delta_temp
        self.entropy = min(1.0, self.temperature / 1000.0 + 0.001)
        self.perimeter_points = self._generate_perimeter()

    def cool(self, delta_temp=100):
        self.temperature = max(0, self.temperature - delta_temp)
        self.entropy = min(1.0, self.temperature / 1000.0 + 0.001)
        self.perimeter_points = self._generate_perimeter()

# --- HIVE ---

class LightHiveTensorAttention:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        required = ['dimensions', 'radius', 'perimeter_points', 'apply_attention']
        if not all(hasattr(node, a) for a in required):
            raise TypeError("Node missing fused Light-Attention protocol.")
        self.nodes.append(node)

    def mean_variance_coherence(self):
        scores = np.array([n.coherence_score() for n in self.nodes])
        return scores.mean(), scores.var()

    def synchronise(self, mode='average'):
        if not self.nodes:
            return
        if mode == 'average':
            dims = int(np.mean([n.dimensions for n in self.nodes]))
            rad = float(np.mean([n.radius for n in self.nodes]))
        elif mode == 'max_coherent':
            idx = np.argmax([n.coherence_score() for n in self.nodes])
            dims = self.nodes[idx].dimensions
            rad = self.nodes[idx].radius
        else:
            return
        for n in self.nodes:
            n.dimensions = dims
            n.radius = rad
            n.perimeter_points = n._generate_perimeter()

    def spawn(self):
        new = []
        for n in self.nodes:
            r = n.replicate()
            if r:
                new.append(r)
        self.nodes.extend(new)

    def trigger_bloom_events(self, event_key='ai_seed', threshold=0.99):
        for n in self.nodes:
            n.on_phase_event(threshold, lambda node: print(f"Bloom event '{event_key}' on node {id(node)}"))
