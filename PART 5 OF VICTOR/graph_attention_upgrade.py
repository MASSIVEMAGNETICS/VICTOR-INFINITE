###########################################
# Graph‑Attention Upgrade for MeshRouter  ##
###########################################
"""
This module injects **edge‑aware attention** into the ripple algorithm.
Each edge (i→j) carries a learnable weight *αᵢⱼ* initialised with the
inverse Euclidean distance between nodes.  During a forward pass the
weights are re‑normalised with softmax so that every node attends
predominantly to its closest—or later its *most relevant*—neighbours.

Highlights
──────────
• API‑compatible: call `MeshRouter.process()` like before.
• Learnable edge weights live in `router.edge_weights[(i, j)]`.
• Optional on‑the‑fly update rule: simple Hebbian (+λ Δ) for demo.
• Still parallel: re‑uses ThreadPoolExecutor from previous patch.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import exp
from typing import Dict, Tuple

# ------------------------------------------------------------
# helper to attach to an *existing* MeshRouter instance
# ------------------------------------------------------------

def _init_edge_weights(router):
    """Populate router.edge_weights with default inverse‑distance values."""
    ew: Dict[Tuple[str, str], float] = {}
    nodes = router.mesh.nodes
    for a, neighbours in router.mesh.adjacency.items():
        for b in neighbours:
            pa, pb = nodes[a]["coords"], nodes[b]["coords"]
            dist = np.linalg.norm(pa - pb) + 1e-6  # avoid div/0
            ew[(a, b)] = ew[(b, a)] = 1.0 / dist
    router.edge_weights = ew

# ------------------------------------------------------------
# new attention‑powered process function
# ------------------------------------------------------------

def meshrouter_process_attention(self, initial_activations):
    if not hasattr(self, "edge_weights"):
        _init_edge_weights(self)
    if not self.primary_node_ids:
        return []

    # --- reuse the original init routine to build current_activations
    from types import MethodType
    base_proc = MethodType(MeshRouter.__dict__["process"], self)
    current = base_proc(initial_activations)  # returns list in original impl
    current = {nid: vec for nid, vec in zip(self.primary_node_ids, current)}

    for _ in range(self.k_iterations):
        next_a = {}

        def _update(idx_node_tuple):
            idx, nid = idx_node_tuple
            model = self.node_models[idx]
            if model is None:
                return nid, current[nid]

            # Gather neighbours + weights
            nbrs = self.mesh.adjacency.get(nid, [])
            if not nbrs:
                return nid, model.forward(current[nid])

            weights = np.array([
                self.edge_weights.get((nid, nb), 0.0) for nb in nbrs
            ])
            # softmax
            e = np.exp(weights - np.max(weights))
            alpha = e / (e.sum() + 1e-9)

            # weighted sum of neighbour activations
            neigh_vec = sum(alpha[i] * current[nb] for i, nb in enumerate(nbrs))
            inp = (1 - self.attenuation) * current[nid] + self.attenuation * neigh_vec
            return nid, model.forward(inp)

        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(_update, (i, nid)): nid for i, nid in enumerate(self.primary_node_ids)}
            for fut in as_completed(futures):
                nid, vec = fut.result()
                next_a[nid] = vec

        # Optional Hebbian update (very simple): reinforce used edges
        lr = 0.01
        for nid in self.primary_node_ids:
            for nb in self.mesh.adjacency.get(nid, []):
                self.edge_weights[(nid, nb)] += lr * np.dot(next_a[nid], next_a[nb])

        current = next_a

    return [current[nid] for nid in self.primary_node_ids]

# ------------------------------------------------------------
# hot‑patch MeshRouter
# ------------------------------------------------------------
import inspect  # noqa
from types import MethodType
MeshRouter.process = meshrouter_process_attention
