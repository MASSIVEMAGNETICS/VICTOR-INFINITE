##########################################################
# 🌌  Universe‑Extensions Patch                          #
#  – Dynamic Topology, Sensor/Actuator Ports, Memory AI  #
##########################################################
"""
DROP‑IN EXTENSIONS – no CUDA, pure CPU.

✔ **Dynamic Topology** – Nodes and edges self‑grow / prune by simple
  "activation‑energy" heuristics.
✔ **Sensor / Actuator Ports** – Register arbitrary I/O streams; values
  flow into / out of dedicated nodes every tick.
✔ **World Cortex (Memory + MCTS Planner)** – Persistent KV memory block
  + toy Monte‑Carlo tree search that queries the mesh for rollouts.

How to use:
───────────
```python
import universe_extensions  # ← single import hot‑patches classes

orch = FlowerOfLifeNetworkOrchestrator(
    num_nodes=7, model_dim=64, mesh_depth=1)

# 1) register a sensor that feeds camera embeddings every frame
orch.register_sensor(
    name="cam", node_index=0,
    pull=lambda: cam_encoder(get_latest_frame()))

# 2) actuator: mesh decides steering vec – we send to game engine
orch.register_actuator(
    name="steer", node_index=3,
    push=lambda vec: game.set_steering(vec))

# 3) main loop
while True:
    orch.tick()   # pulls sensors → mesh ripple → pushes actuators
```
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import random, math
from typing import Callable, Dict, Any, List, Tuple

# ------------------------------------------------------------
# ----------  Dynamic Topology Patch for MeshRouter  ----------
# ------------------------------------------------------------

ENERGY_ADD_THRESHOLD   = 1.5   # ‑‑‖ activation norm to spawn node
ENERGY_PRUNE_THRESHOLD = 0.05  # ‑‑‖ below this we prune
MAX_NEW_NODES_PER_TICK = 2


def _dynamic_topology_step(router, current_activations):
    """Analyse node energies – spawn or prune nodes on the fly."""
    mesh = router.mesh
    new_nodes = []
    # --- Spawn
    for nid, vec in current_activations.items():
        energy = np.linalg.norm(vec)
        if energy > ENERGY_ADD_THRESHOLD and len(new_nodes) < MAX_NEW_NODES_PER_TICK:
            # create fresh node slightly displaced in 3‑space
            dx = np.random.uniform(-0.3, 0.3, size=3)
            new_id = f"{nid}_spawn_{random.randint(0,9999)}"
            if mesh._add_node(new_id, mesh.nodes[nid]["coords"] + dx, "primary", depth_level=0):
                mesh.adjacency[nid].append(new_id)
                mesh.adjacency[new_id].append(nid)
                new_nodes.append(new_id)
                # plug a vanilla BandoBlock with same dim
                dim = router.node_models[0].dim if router.node_models else vec.shape[0]
                router.node_models.append(BandoBlock(dim))
                router.primary_node_ids.append(new_id)
                current_activations[new_id] = np.zeros(dim)

    # --- Prune
    for nid in list(router.primary_node_ids):
        if nid not in current_activations:  # newly added maybe
            continue
        energy = np.linalg.norm(current_activations[nid])
        if energy < ENERGY_PRUNE_THRESHOLD and len(router.primary_node_ids) > 1:
            router.primary_node_ids.remove(nid)
            idx = next((i for i, pid in enumerate(router.primary_node_ids) if pid == nid), None)
            if idx is not None:
                router.node_models.pop(idx)
            # detach edges
            for nb in mesh.adjacency.get(nid, []):
                if nb in mesh.adjacency and nid in mesh.adjacency[nb]:
                    mesh.adjacency[nb].remove(nid)
            mesh.adjacency.pop(nid, None)
            mesh.nodes.pop(nid, None)
            current_activations.pop(nid, None)


# ------------------------------------------------------------
# Hot‑patch MeshRouter.process AGAIN (wraps attention version)  
# ------------------------------------------------------------
from types import MethodType
from inspect import isfunction
from mesh_attention_upgrade import meshrouter_process_attention  # previous patch


def _process_with_topology(self, initial_activations):
    # Use attention‑powered core first
    out_list = meshrouter_process_attention(self, initial_activations)
    current = {nid: vec for nid, vec in zip(self.primary_node_ids, out_list)}
    # One pass of topology mutation
    _dynamic_topology_step(self, current)
    # Return list again in current primary order
    return [current.get(nid, np.zeros_like(out_list[0])) for nid in self.primary_node_ids]

import sys
from __main__ import MeshRouter  # already imported in user code
MeshRouter.process = _process_with_topology

# ------------------------------------------------------------
# ----------  Sensor / Actuator Extensions  -------------------
# ------------------------------------------------------------

class _Port:
    def __init__(self, node_index:int, fn:Callable[[Any],Any]):
        self.node_index = node_index
        self.fn = fn

# monkey‑patch orchestrator with registry + tick()
from __main__ import FlowerOfLifeNetworkOrchestrator as _Orch

_Orch._sensors: Dict[str,_Port] = {}
_Orch._actuators: Dict[str,_Port] = {}


def register_sensor(self, name:str, node_index:int, pull:Callable[[],np.ndarray]):
    self._sensors[name] = _Port(node_index, pull)

def register_actuator(self, name:str, node_index:int, push:Callable[[np.ndarray],None]):
    self._actuators[name] = _Port(node_index, push)


def tick(self):
    """1. Pull sensors; 2. run mesh; 3. push actuators."""
    # Build input list sized to num_nodes
    in_list = [None]*self.num_nodes
    for s in self._sensors.values():
        in_list[s.node_index] = s.fn()
    # If sensor missing output for node with model, zero‑pad
    for i in range(self.num_nodes):
        if in_list[i] is None and self.node_models[i] is not None:
            in_list[i] = np.zeros(self.model_dim)
    result_vec = self.process_input(in_list)
    # Push actuators
    for a in self._actuators.values():
        a.fn(result_vec)  # simplistic: same vec to all actuators
    return result_vec

_Orch.register_sensor = register_sensor
_Orch.register_actuator = register_actuator
_Orch.tick = tick

# ------------------------------------------------------------
# ----------  Simple Memory & Planner Block  ------------------
# ------------------------------------------------------------

class KVMemory:
    def __init__(self, dim):
        self.dim = dim
        self.store: Dict[str,np.ndarray] = {}

    def write(self, key:str, vec:np.ndarray):
        self.store[key] = vec.copy()

    def read(self, key:str)->np.ndarray:
        return self.store.get(key, np.zeros(self.dim))

# attach to orchestrator
_Orch.memory = KVMemory(dim=64)  # default; reassign if needed

# --- tiny stochastic planner (MCTS‑lite) ---
class TinyPlanner:
    def __init__(self, orch:_Orch, rollout_depth=3, branching=4):
        self.orch = orch; self.rollout_depth = rollout_depth; self.branching=branching

    def propose(self, seed_input:np.ndarray):
        """Return action vector with highest heuristic value after rollouts."""
        best_vec = None; best_score=-math.inf
        for _ in range(self.branching):
            vec = seed_input + 0.05*np.random.randn(*seed_input.shape)
            score = self._rollout_score(vec)
            if score>best_score: best_score, best_vec = score, vec
        return best_vec

    def _rollout_score(self, vec):
        sim_state = vec
        total = 0.0
        for _ in range(self.rollout_depth):
            sim_state = self.orch.process_input(sim_state)
            total += sim_state.mean()  # toy heuristic
        return total

_Orch.planner = TinyPlanner

# ------------------------------------------------------------
# END OF PATCH
# ------------------------------------------------------------
