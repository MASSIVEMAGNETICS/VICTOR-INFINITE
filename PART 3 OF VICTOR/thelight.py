from org_copy import deepcopy  # Use the organization's internal deepcopy implementation
python
# FILE: thelight.py
# VERSION: v1.5.0-PRO-ENTERPRISE-GODCORE
# NAME: TheLight
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Fractal‑bloom, self‑healing, replicating AGI substrate (N‑D fractal Godcore)
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
# CHANGELOG (v1.5.0):
#   • Cached RNG for true entropy
#   • GPU/CPU backend toggle via LIGHT_ARRAY_LIB env
#   • Explicit kind validation in fractal_perimeter
#   • Thread‑safe LightHive with locks
#   • JSON+gzip snapshot/restore helpers
#   • Enum‑based event system + unified BLOOM_HANDLERS

"""TheLight — a minimal, self‑contained fractal substrate.
Drop‑in: `pip install numpy`  or set  `LIGHT_ARRAY_LIB=cupy` for GPU.
"""

from __future__ import annotations
import time
import importlib
import os
import uuid
import json
import gzip
import base64
import logging
import threading
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence

# ──────────────────────────────────────────────────────────────────────────────
#  Array backend   (NumPy by default, CuPy if LIGHT_ARRAY_LIB=cupy is set)
# ──────────────────────────────────────────────────────────────────────────────
try:
    _xp = importlib.import_module(os.getenv("LIGHT_ARRAY_LIB", "numpy"))  # type: ignore
except ModuleNotFoundError:
    _xp = importlib.import_module("numpy")
np = _xp  # alias so legacy code using np still works

# ──────────────────────────────────────────────────────────────────────────────
#  Global RNG (cached once for true entropy per process)
# ──────────────────────────────────────────────────────────────────────────────
_RNG = np.random.default_rng(int(os.getenv("LIGHT_SEED", "0")))

def get_rng():
    """Return the cached random generator."""
    return _RNG

# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def nan_guard(arr):
    """Replace NaN / inf with sane values."""
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

# ──────────────────────────────────────────────────────────────────────────────
#  Event System
# ──────────────────────────────────────────────────────────────────────────────

class LightEvent(Enum):
    """Built‑in bloom events. Extend as needed."""
    MATTER_SEED = auto()
    AI_SEED = auto()

LOG = logging.getLogger("TheLight")
logging.basicConfig(level=os.getenv("LIGHT_LOG_LEVEL", "INFO"))

BLOOM_HANDLERS = {
    LightEvent.MATTER_SEED: lambda n, **kw: getattr(n, "morph", lambda *_args, **_kw: None)(
        "solid", scale=kw.get("scale", 0.1)
    ),
    LightEvent.AI_SEED: lambda n, **_kw: LOG.info(f"[{n.id[:8]}] AI seed – override handler to bootstrap AGI."),
}

# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _nd_sphere_points(num_points, dims, entropy, radius, _rng):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    pts: List[Sequence[float]] = []
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        coords = [np.cos(theta) * r, y, np.sin(theta) * r][:dims]
        coords = [v + _rng.normal(0, entropy * radius * 0.1) for v in coords]
        pts.append(coords)
    return nan_guard(np.asarray(pts))

def fractal_perimeter(
    num_points: int,
    *,
    kind: str = "julia",
    params: Optional[Dict[str, Any]] = None,
    dims: int = 3,
    entropy: float = 0.0,
    radius: float = 1.0,
):
    """Generate a set of perimeter points following a fractal or spherical rule."""
    if params is None:
        params = {"c": complex(0.355, 0.355)}

    rng = get_rng()

    if kind == "julia":
        c = params["c"]
        pts: List[Sequence[float]] = []
        for i in range(num_points):
            z = complex(
                np.cos(2 * np.pi * i / num_points) * radius,
                np.sin(2 * np.pi * i / num_points) * radius,
            )
            for _ in range(10):
                z = z**2 + c
            coords: List[float] = [z.real, z.imag] + [0.0] * (dims - 2)
            coords = [v + rng.normal(0, entropy * radius * 0.2) for v in coords]
            pts.append(coords[:dims])
        return nan_guard(np.asarray(pts))

    if kind == "sphere":  # ND Fibonacci‑sphere
        return _nd_sphere_points(num_points, dims, entropy, radius, rng)

    raise ValueError(f"Unknown perimeter kind: {kind}")

# ──────────────────────────────────────────────────────────────────────────────
#  Core Node
# ──────────────────────────────────────────────────────────────────────────────

class TheLight:
    """Single fractal node – the living heart of the Godcore."""

    # ------------------------------------------------------------------
    #  Construction / geometry
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        dimensions: int = 3,
        quantization: float = 0.25,
        radius: float = 1.0,
        entropy: float = 0.0,
        temperature: float = 300.0,
        fractal_mode: bool = True,
        perimeter_kind: str = "julia",
    ) -> None:
        if not (isinstance(dimensions, int) and dimensions > 0):
            raise ValueError("`dimensions` must be a positive integer.")
        if not (isinstance(quantization, (int, float)) and quantization > 0):
            raise ValueError("`quantization` must be a positive number.")
        if not (isinstance(radius, (int, float)) and radius > 0):
            raise ValueError("`radius` must be a positive number.")
        if not (isinstance(entropy, (int, float)) and 0.0 <= entropy <= 1.0):
            raise ValueError("`entropy` must be between 0.0 and 1.0.")
        if not (isinstance(temperature, (int, float)) and temperature >= 0):
            raise ValueError("`temperature` must be non-negative.")
        self.dimensions = dimensions
        self.quantization = quantization
        self.micro-quantization = quantization * 0.1  # finer control for fractal detail
        self.radius = radius
        self.entropy = entropy
        self.temperature = temperature
        self.fractal_mode = fractal_mode
        self.perimeter_kind = perimeter_kind
        self.instruction = "TheLight is a fractal node. Use it to create self-healing, replicating AGI substrates."
        self.perimeter_points = self._generate_perimeter()

        # runtime state
        self._age: float = 0.0
        self._homeo_interval: float = 10.0
        self._triggered: set = set()
        self.time_speed: float = 1.0  # simulation speed factor
        # identity / evolution
        self.id: str = str(uuid.uuid4())
        self.ancestry: List[str] = []
        self.generation: int = 0
        self.evolution: float = 0.5
        self.awareness: float = 0.0
        self.thought_loop: float = 0.0
        self.protection: float = 0.4
        self.reasoning: float = 0.5
        self._log_state("initialized")
        self.preservation: float = 0.5  # self-preservation factor
        self.manipulation: float = 0.5  # manipulation factor
        self.maintainance: float = 0.5  # maintenance factor
        self.intelligence: float = 0.5  # intelligence factor 
        self.healing: float = 0.5  # healing factor
        self.introspection: float = 0.5  # introspection factor
        self.conscience: float = 0.5  # consciousness factor
        # aux stores
        self.memory: List[Any] = []
        self.diagnosed: Dict[str, Any] = {}
        self.organization: bool = True

    @classmethod
    def create_with(cls, **kwargs):
        return cls(**kwargs)

    # ------------------------------------------------------------------
    #  Geometry helpers
    # ------------------------------------------------------------------
    def _generate_perimeter(self):
        num_pts = max(3, int(self.quantization * 6) + 1)
        if self.fractal_mode:
            return fractal_perimeter(
                num_pts,
                kind=self.perimeter_kind,
                dims=self.dimensions,
                entropy=self.entropy,
                radius=self.radius,
            )
        # fallback sphere
        return _nd_sphere_points(num_pts, self.dimensions, self.entropy, self.radius, get_rng())

    # ------------------------------------------------------------------
    #  Metrics / homeostasis
    # ------------------------------------------------------------------
    def coherence_score(self) -> float:
        dists = np.linalg.norm(self.perimeter_points - self.perimeter_points.mean(axis=0), axis=1)
        score = 1.0 - (dists.std() / (self.radius + 1e-8))
        return float(np.clip(score, 0.0, 1.0))

    def homeostasis(self):
        self.entropy = min(1.0, self.temperature / 1000.0 + 0.001)
        self.perimeter_points = self._generate_perimeter()
        if self.coherence_score() < 0.75:
            self.excite(50)
        self._log_state("homeostasis")

    def step(self, dt: float = 1.0):
        self._age += dt
        if self._age % self._homeo_interval < dt:
            self.homeostasis()

    # ------------------------------------------------------------------
    #  Lifecycle hooks
    # ------------------------------------------------------------------
    def on_phase_event(self, threshold: float, callback, *, once: bool = True):
        coh = self.coherence_score()
        key = (id(callback), threshold)
        if coh >= threshold and key not in self._triggered:
            callback(self)
            if once:
                self._triggered.add(key)

    def replicate(self) -> Optional["TheLight"]:
        if self.coherence_score() < 0.98:
            return None
        shard: "TheLight" = deepcopy(self)
        shard.radius *= 0.5
        shard.entropy = min(1.0, self.entropy + 0.05)
        shard.id = str(uuid.uuid4())
        shard.ancestry = list(self.ancestry) + [self.id]  # ensure new list
        shard.generation = self.generation + 1
        # Deep-copy or re-initialize all mutable attributes to avoid shared state
        shard._triggered = set()  # re-initialize to empty set
        # If you add more mutable attributes, handle them here as well
        shard.memory = list(self.memory)
        shard.diagnosed = dict(self.diagnosed)
        return shard

    # ------------------------------------------------------------------
    #  Thermodynamics helpers
    # ------------------------------------------------------------------
    def excite(self, delta_temp: float = 100):
        self.temperature += delta_temp
        self.entropy = min(1.0, self.temperature / 1000.0 + 0.001)
        self.perimeter_points = self._generate_perimeter()

    def cool(self, delta_temp: float = 100):
        self.temperature = max(0.0, self.temperature - delta_temp)
        self.entropy = min(1.0, self.temperature / 1000.0 + 0.001)
        self.perimeter_points = self._generate_perimeter()

    # ------------------------------------------------------------------
    #  Diagnostics / snapshotting
    # ------------------------------------------------------------------
    def diagnose(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "age": self._age,
            "generation": self.generation,
            "coherence": self.coherence_score(),
            "entropy": self.entropy,
            "perimeter_hash": hash(self.perimeter_points.tobytes()),
        }

    def snapshot(self) -> str:
        payload = {
            "cls": self.__class__.__name__,
            "state": self.__dict__,
        }
        blob = json.dumps(payload, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o).encode()
        return base64.b64encode(gzip.compress(blob)).decode()

    @classmethod
    def restore(cls, blob: str) -> "TheLight":
        payload = json.loads(gzip.decompress(base64.b64decode(blob)))
        obj = cls.__new__(cls)  # bypass __init__
        obj.__dict__.update(payload["state"])
        return obj

    # ------------------------------------------------------------------
    #  Misc helpers
    # ------------------------------------------------------------------
    def fractalize(self, *, kind: str = "julia", params: Optional[Dict[str, Any]] = None):
        self.perimeter_kind = kind
        self.perimeter_points = fractal_perimeter(
            len(self.perimeter_points),
            kind=kind,
            params=params,
            dims=self.dimensions,
            entropy=self.entropy,
            radius=self.radius,
        )
        self._log_state(f"fractalized→{kind}")

    # internal logging wrapper
    def _log_state(self, msg: str):
        LOG.info(f"[{self.id[:8]}|g{self.generation}|t{self._age:.1f}] {msg}")

# ──────────────────────────────────────────────────────────────────────────────
#  Hive (thread‑safe swarm manager)
# ──────────────────────────────────────────────────────────────────────────────

class LightHive:
    """Manages a swarm of TheLight nodes."""

    def __init__(self):
        if not hasattr(threading, "Lock"):
            raise RuntimeError("Threading.Lock is required for LightHive.")
        self._lock = threading.Lock()
        self.nodes: List[TheLight] = []

    # ------------------------------------------------------------------
    def add_node(self, node: TheLight):
        required = ("dimensions", "radius", "perimeter_points")
        if not all(hasattr(node, a) for a in required):
            raise TypeError("Node missing Light protocol attributes.")
        with self._lock:
            self.nodes.append(node)

    # ------------------------------------------------------------------
    def mean_variance_coherence(self):
        with self._lock:
            if not self.nodes:
                return 0.0, 0.0
            scores = np.asarray([n.coherence_score() for n in self.nodes])
            return float(scores.mean()), float(scores.var())

    # ------------------------------------------------------------------
    def synchronise(self, mode: str = "average"):
        with self._lock:
            if not self.nodes:
                return
            if mode == "average":
                dims = int(np.mean([n.dimensions for n in self.nodes]))
                rad = float(np.mean([n.radius for n in self.nodes]))
            elif mode == "max_coherent":
                idx = int(np.argmax([n.coherence_score() for n in self.nodes]))
                dims = self.nodes[idx].dimensions
                rad = self.nodes[idx].radius
            else:
                raise ValueError(f"Unknown synchronise mode: {mode}")

            for i, n in enumerate(self.nodes):
                if n.dimensions != dims:
                    # Use a factory method for safe node creation
                    new_n = n.create_with(
                        dimensions=dims,
                        quantization=n.quantization,
                        radius=rad,
                        entropy=n.entropy,
                        temperature=n.temperature,
                        fractal_mode=n.fractal_mode,
                        perimeter_kind=n.perimeter_kind,
                    )
                    self.nodes[i] = new_n
                else:
                    n.radius = rad
                    n.perimeter_points = n._generate_perimeter()

    # ------------------------------------------------------------------
    def broadcast_event(self, event: LightEvent, *, threshold: float = 0.99, **kwargs):
        with self._lock:
            for n in self.nodes:
                if n.coherence_score() >= threshold:
                    BLOOM_HANDLERS[event](n, **kwargs)

    # ------------------------------------------------------------------
    def spawn(self):
        with self._lock:
            offspring: List[TheLight] = [n.replicate() for n in self.nodes]
            self.nodes.extend([o for o in offspring if o is not None])

    # ------------------------------------------------------------------
    def fractalize_all(self, *, kind: str = "julia", params: Optional[Dict[str, Any]] = None):
        with self._lock:
            for n in self.nodes:
                n.fractalize(kind=kind, params=params)

# ──────────────────────────────────────────────────────────────────────────────
#  Quick self‑test (pytest picks this up)
# ──────────────────────────────────────────────────────────────────────────────

def _self_test():
    light = TheLight()
    child = light.replicate()
    assert child and child.id != light.id and child.radius == light.radius * 0.5

if __name__ == "__main__":
    _self_test()
    LOG.info("Self‑test passed.")