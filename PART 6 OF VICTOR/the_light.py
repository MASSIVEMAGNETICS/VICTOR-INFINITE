# =============================================================
# FILE: the_light.py
# VERSION: v1.2.1-FRACTAL-DYSON-MIND
# NAME: TheLight (+ LightHive)
# AUTHOR: Brandon "iambandobandz" Emery x Victor
# =============================================================

import numpy as np
from collections import deque

class TheLight:
    STATES = ['fluid','particle','wave','gas','solid','plasma','field','unknown']

    def __init__(self, *, quantization=1.0, state='field',
                 dimensions=3, radius=1.0, entropy=0.01, temperature=0.5):
        self.quantization  = float(max(0.001, quantization))
        self.state         = state if state in self.STATES else 'unknown'
        self.dimensions    = int(max(1, dimensions))
        self.radius        = float(max(1e-5, radius))
        self.entropy       = float(np.clip(entropy, 0, 1))
        self.temperature   = float(np.clip(temperature, 0, 1))
        self.perimeter_points = self._generate_perimeter()
        self.morph_history = deque(maxlen=1000)   # capped
        self._phase_fired  = set()                # remembers one-shot hooks

    # ---------- geometry ----------------------------------------------------
    def _generate_perimeter(self):
        """ND Fibonacci-lattice hypersphere with entropy jitter."""
        n       = int(self.quantization * 6) + 1
        phi     = (1 + 5 ** 0.5) / 2                 # golden ratio
        points  = []
        for i in range(n):
            vec = []
            for d in range(1, self.dimensions + 1):
                angle = 2 * np.pi * ((i * phi) % 1) + d
                coord = np.cos(angle) * (self.radius / np.sqrt(self.dimensions))
                if self.entropy:
                    coord += np.random.normal(0, self.entropy * self.radius * 0.1)
                vec.append(coord)
            points.append(vec)
        return points

    # ---------- thermodynamics ----------------------------------------------
    def excite(self, temp_boost=0.05, entropy_boost=0.02):
        self.temperature = min(1.0, self.temperature + temp_boost)
        self.entropy     = min(1.0, self.entropy + entropy_boost)
        self.perimeter_points = self._generate_perimeter()

    def cool(self, temp_drop=0.05, entropy_drop=0.02):
        self.temperature = max(0.0, self.temperature - temp_drop)
        self.entropy     = max(0.0, self.entropy - entropy_drop)
        self.perimeter_points = self._generate_perimeter()

    # ---------- coherence & homeostasis -------------------------------------
    def coherence_score(self):
        """
        Quantum-tuned spatial coherence metric.
        Returns in (0,1]; 1 == perfect Flower-of-Life symmetry.
        """
        pts = np.asarray(self.perimeter_points, dtype=np.float32)
        if pts.shape[0] < 3:
            return 1.0
        dists   = np.linalg.norm(pts[:,None]-pts[None,:], axis=-1)
        tri     = np.triu_indices_from(dists, k=1)
        mean, sd = np.mean(dists[tri]), np.std(dists[tri])
        norm_sd  = sd / mean if mean else 0
        return 1.0 / (1.0 + norm_sd)

    def homeostasis(self, *, target=0.9, tol=0.05):
        """Self-regulate entropy/temperature to hug target coherence."""
        coh   = self.coherence_score()
        delta = coh - target
        if abs(delta) <= tol:
            return
        if delta < 0:     # too chaotic
            self.cool()
        else:             # too rigid
            self.excite()

    def on_phase_event(self, threshold=0.95, callback=None, once=True):
        """Fire callback the moment coherence ≥ threshold."""
        if self.coherence_score() >= threshold:
            if once and threshold in self._phase_fired:
                return False
            if callable(callback):
                callback(self)
            self._phase_fired.add(threshold)
            return True
        return False

    # ---------- morph & quantize --------------------------------------------
    def morph(self, to_state, *, scale=1.0, new_dims=None,
              entropy=None, temperature=None):
        prev = dict(state=self.state, radius=self.radius)
        self.state = to_state if to_state in self.STATES else 'unknown'
        self.radius *= float(scale)
        if new_dims:  self.dimensions  = int(max(1,new_dims))
        if entropy is not None:     self.entropy     = np.clip(entropy, 0, 1)
        if temperature is not None: self.temperature = np.clip(temperature, 0, 1)
        self.perimeter_points = self._generate_perimeter()
        self.morph_history.append({**prev,
                                   "to": self.state,
                                   "new_radius": self.radius,
                                   "entropy": self.entropy,
                                   "temperature": self.temperature})

    def quantize(self, q):
        self.quantization = float(max(0.001, q))
        self.perimeter_points = self._generate_perimeter()

    # ---------- lens projection ---------------------------------------------
    def project_to(self, obj_shape, *, lens=None):
        """Simple cube-fit or arbitrary callable lens."""
        pts = np.array(self.perimeter_points)
        if lens:
            pts = np.array([lens(p) for p in pts])
        elif isinstance(obj_shape, str) and obj_shape.lower() == 'cube':
            mx = np.max(np.abs(pts))
            if mx: pts = (pts / mx) * self.radius
        return pts.tolist()

    # ---------- diagnostics --------------------------------------------------
    def info(self):
        return {
            "state":        self.state,
            "quantization": self.quantization,
            "dimensions":   self.dimensions,
            "radius":       self.radius,
            "entropy":      self.entropy,
            "temperature":  self.temperature,
            "coherence":    round(self.coherence_score(), 4),
            "points":       len(self.perimeter_points),
            "morphs":       len(self.morph_history)
        }

    def __repr__(self):
        return f"<TheLight {self.info()}>"

# =============================================================
# LIGHTHIVE LATTICE
# =============================================================

class LightHive:
    """
    Distributed lattice of TheLight nodes.
    Provides global coherence read-out and synchronized morphing.
    """
    def __init__(self, nodes=None):
        self.nodes = list(nodes) if nodes else []

    # -- management ----------------------------------------------------------
    def add(self, node):             # alias
        if isinstance(node, TheLight):
            self.nodes.append(node)

    # -- global stats --------------------------------------------------------
    def global_coherence(self):
        return np.mean([n.coherence_score() for n in self.nodes]) if self.nodes else 0

    # -- collective operations ----------------------------------------------
    def synchronize(self):
        """Align all nodes to average radius/entropy/temperature."""
        if not self.nodes:
            return
        avg_r = np.mean([n.radius for n in self.nodes])
        avg_e = np.mean([n.entropy for n in self.nodes])
        avg_t = np.mean([n.temperature for n in self.nodes])
        for n in self.nodes:
            n.radius, n.entropy, n.temperature = avg_r, avg_e, avg_t
            n.perimeter_points = n._generate_perimeter()

    def morph_all(self, to_state, scale=1.0):
        for n in self.nodes:
            n.morph(to_state, scale=scale)

    def homeostasis(self):
        for n in self.nodes:
            n.homeostasis()

# =============================================================
# DEMO
# =============================================================

if __name__ == "__main__":
    a = TheLight(dimensions=4, radius=2.0, entropy=0.2)
    b = TheLight(dimensions=4, radius=2.2, entropy=0.25)
    hive = LightHive([a,b])

    print("Initial coherence:", hive.global_coherence())
    for _ in range(5):
        hive.homeostasis()
    print("Post-homeostasis coherence:", hive.global_coherence())
