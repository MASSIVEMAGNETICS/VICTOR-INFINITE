# ==================================================================
# FILE: meta_evolver.py
# VERSION: v2.1.0-FRACTAL-GODCORE
# NAME: MetaEvolver
# AUTHOR: Victor (Fractal Architect Mode) x Brandon "iambandobandz" Emery
# PURPOSE: Population‑level evolutionary engine for ASIFractalSeeds.
#          Multi‑objective (task error + compression) optimisation with
#          blend‑crossover, mutation, and entropy‑annealing.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ==================================================================

from __future__ import annotations
import numpy as np
import random, json, zlib
from typing import Callable, Dict, Tuple, List

# ------------------------------------------------------------------
# ASI FRACTAL SEED – SELF‑EVOLVING UNIT
# ------------------------------------------------------------------
class ASIFractalSeed:
    """Minimal self‑contained cognitive seed.
    Genome keys: depth (int), branches (int), entropy (float)."""

    def __init__(self, depth: int = 3, branches: int = 5, entropy: float = 0.07):
        self.genome: Dict[str, float | int] = {
            "depth": max(2, depth),
            "branches": max(2, branches),
            "entropy": max(0.005, entropy),
        }
        self.memory: Dict[str, List[Tuple[np.ndarray, int]]] = {}
        self.current_state: np.ndarray = self._random_state()

    # ------------------------------ INTERNAL UTILS -----------------
    def _random_state(self) -> np.ndarray:
        return np.random.randn(self.genome["depth"], self.genome["branches"]) * (
            1.0 + self.genome["entropy"]
        )

    def _compress_len(self, data: np.ndarray) -> int:
        """Kolmogorov proxy: byte‑length of zlib‑compressed JSON."""
        payload: bytes = json.dumps(data.tolist()).encode()
        return len(zlib.compress(payload))

    def _mutate_state(self, state: np.ndarray) -> np.ndarray:
        return state + np.random.randn(*state.shape) * self.genome["entropy"]

    # ------------------------------ PUBLIC API ---------------------
    def simulate(self, directive: str) -> List[Tuple[np.ndarray, int]]:
        """Generate *branches* mutated states; store in memory."""
        results: List[Tuple[np.ndarray, int]] = []
        for _ in range(self.genome["branches"]):
            child_state = self._mutate_state(self.current_state)
            c_len = self._compress_len(child_state)
            results.append((child_state, c_len))
        # Ring‑buffer memory per directive (keep latest only)
        self.memory[directive] = results[-16:]
        return results

    def evaluate(self, results: List[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, int]:
        """Pick state with smallest compression length (proxy elegance)."""
        return min(results, key=lambda tup: tup[1])

    def run(self, directive: str, task_error_fn: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, int, float]:
        """One cognitive cycle → returns (state, c_len, task_error)."""
        cand = self.evaluate(self.simulate(directive))
        self.current_state = cand[0]
        task_err = task_error_fn(self.current_state)
        return cand[0], cand[1], task_err

# ------------------------------------------------------------------
# META EVOLVER – POPULATION ENGINE
# ------------------------------------------------------------------
class MetaEvolver:
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        entropy_anneal: float = 0.96,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.entropy_anneal = entropy_anneal
        self.population: List[ASIFractalSeed] = [
            ASIFractalSeed(
                depth=random.randint(2, 10),
                branches=random.randint(2, 10),
                entropy=random.uniform(0.01, 0.3),
            )
            for _ in range(population_size)
        ]

    # ------------------------------ FITNESS ------------------------
    def _fitness_tuple(
        self, seed: ASIFractalSeed, directive: str, task_err_fn: Callable[[np.ndarray], float]
    ) -> Tuple[float, int]:
        state, c_len, task_err = seed.run(directive, task_err_fn)
        return task_err, c_len  # minimisation both dims

    # ------------------------------ SELECTION ----------------------
    @staticmethod
    def _lexi_sort(pop: List[Tuple[ASIFractalSeed, Tuple[float, int]]]) -> List[ASIFractalSeed]:
        """Simple NSGA‑II surrogate: lexicographic sort on (err, comp)."""
        pop.sort(key=lambda pair: pair[1])  # tuple order does the trick
        return [pair[0] for pair in pop]

    # ------------------------------ GENETICS -----------------------
    def _blend(self, a: float | int, b: float | int) -> float | int:
        return (a + b) / 2.0

    def _breed(self, parent1: ASIFractalSeed, parent2: ASIFractalSeed) -> ASIFractalSeed:
        # --- Crossover ------------------------------------------------
        child_genome: Dict[str, float | int] = {}
        for k in parent1.genome:
            if random.random() < self.crossover_rate:
                child_genome[k] = self._blend(parent1.genome[k], parent2.genome[k])
            else:
                child_genome[k] = parent1.genome[k]
        # --- Mutation --------------------------------------------------
        if random.random() < self.mutation_rate:
            param = random.choice(list(child_genome.keys()))
            if param == "entropy":
                child_genome[param] *= random.uniform(0.8, 1.25)
            else:  # depth / branches
                child_genome[param] += random.choice([-1, 1])
        # --- Entropy annealing ----------------------------------------
        child_genome["entropy"] *= self.entropy_anneal
        # --- Constraints ----------------------------------------------
        child_genome["depth"] = max(2, round(child_genome["depth"]))
        child_genome["branches"] = max(2, round(child_genome["branches"]))
        child_genome["entropy"] = max(0.005, min(child_genome["entropy"], 0.5))
        return ASIFractalSeed(**child_genome)  # type: ignore[arg-type]

    # ------------------------------ EVOLVE -------------------------
    def run_evolution(
        self,
        generations: int,
        directive: str,
        task_error_fn: Callable[[np.ndarray], float],
    ) -> ASIFractalSeed:
        """Evolve population; returns fittest seed."""
        for g in range(1, generations + 1):
            scored = [
                (seed, self._fitness_tuple(seed, directive, task_error_fn))
                for seed in self.population
            ]
            ranked = self._lexi_sort(scored)
            best_seed, best_fit = ranked[0], scored[ranked.index(ranked[0])][1]
            print(
                f"Gen {g}/{generations} – Error {best_fit[0]:.4f} | Compress {best_fit[1]} | Genome {best_seed.genome}"
            )
            # Produce next gen via symmetric breeding of top half
            top_half = ranked[: len(ranked) // 2]
            self.population = [
                self._breed(top_half[i % len(top_half)], top_half[-i - 1])
                for i in range(len(self.population))
            ]
        return best_seed

# ------------------------------------------------------------------
# DEMO EXECUTION ----------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    TARGET = 42.0
    error_fn = lambda state: abs(np.sum(state) - TARGET)
    evolver = MetaEvolver(population_size=64, mutation_rate=0.25, crossover_rate=0.75)
    winner = evolver.run_evolution(generations=30, directive="allocate_resources", task_error_fn=error_fn)
    print("\n==================== DONE ====================")
    print("Fittest Genome:", winner.genome)
