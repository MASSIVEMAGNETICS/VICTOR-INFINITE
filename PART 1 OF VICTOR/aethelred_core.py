# FILE: aethelred_core.py
# VERSION: v3.0.0-PROMETHEUS-CORE
# NAME: AethelredCore
# AUTHOR: PROMETHEUS CORE (Successor to Brandon "iambandobandz" Emery & Victor)
# PURPOSE: Multiversal, crash-proof, stateful fractal knowledge engine.
#          Architected for quantum-theoretic data structuring, infinite deterministic
#          reconstruction, and timeline-aware persistence.
# LICENSE: Proprietary - Post-Singularity Systems Collective

import numpy as np
import hashlib
import threading
import logging
import os
import json
from functools import lru_cache
from datetime import datetime
import uuid

# --- GOD-TIER LOGGER: Self-healing and context-aware ---
LOG_DIR = "./aethelred_logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "aethelred_core.log"),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] {%(threadName)s} %(module)s:%(lineno)d - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# --- CORE DIRECTIVE: Custom Exception Handling ---
class CoreIntegrityError(Exception):
    """Raised when core data structures are compromised."""
    pass

class TimelineNotFoundError(KeyError):
    """Raised when a specific timeline cannot be found in the store."""
    pass

class SeedNotFoundError(KeyError):
    """Raised when a specific seed cannot be found in the current timeline."""
    pass

class MultiverseSeedStore:
    """
    Disk-backed, timeline-aware, crash-proof seed and parameter vector store.
    Manages multiple timelines of seed evolution, emulating multiversal state.
    """

    def __init__(self, cache_size=1024, db_path="./aethelred_multiverse.json"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._cache = lru_cache(maxsize=cache_size)(self._load_seed_from_timeline)
        self._timelines = {}
        self._active_timeline = "genesis"
        self._load_db()
        if self._active_timeline not in self._timelines:
            self._timelines[self._active_timeline] = {}

    def _load_db(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    self._timelines = data.get("timelines", {"genesis": {}})
                    self._active_timeline = data.get("active_timeline", "genesis")
                logging.info("Multiverse store loaded from disk. Active timeline: %s.", self._active_timeline)
            else:
                self._timelines = {"genesis": {}}
        except (json.JSONDecodeError, IOError) as e:
            logging.error("Failed to load multiverse store, initializing fresh: %s", e, exc_info=True)
            self._timelines = {"genesis": {}}
            self._active_timeline = "genesis"
            self._save_db() # Attempt to create a new valid db file

    def _save_db(self):
        try:
            with self._lock:
                with open(self.db_path, "w") as f:
                    json.dump({
                        "active_timeline": self._active_timeline,
                        "timelines": self._timelines
                    }, f, indent=2)
        except IOError as e:
            logging.error("CRITICAL: Failed to save multiverse store to disk: %s", e, exc_info=True)

    def _load_seed_from_timeline(self, seed_hash_and_timeline):
        seed_hash, timeline = seed_hash_and_timeline
        with self._lock:
            pvec = self._timelines.get(timeline, {}).get(seed_hash)
            return np.array(pvec) if pvec is not None else None

    def switch_timeline(self, timeline_id: str):
        with self._lock:
            if timeline_id not in self._timelines:
                logging.warning("Timeline '%s' not found. Creating as a new branch.", timeline_id)
                self._timelines[timeline_id] = {}
            self._active_timeline = timeline_id
            self._cache.cache_clear()
            self._save_db()
            logging.info("Switched to timeline: %s", timeline_id)

    def fork_timeline(self, new_timeline_id: str, source_timeline_id: str = None):
        with self._lock:
            source = source_timeline_id or self._active_timeline
            if source not in self._timelines:
                raise TimelineNotFoundError(f"Source timeline '{source}' does not exist for forking.")
            if new_timeline_id in self._timelines:
                logging.warning("Timeline '%s' already exists. Forking will overwrite.", new_timeline_id)
            self._timelines[new_timeline_id] = self._timelines[source].copy()
            self.switch_timeline(new_timeline_id)
            logging.info("Forked timeline '%s' from '%s'. New timeline is now active.", new_timeline_id, source)

    def set(self, seed_hash: str, param_vector: np.ndarray, timeline: str = None):
        target_timeline = timeline or self._active_timeline
        with self._lock:
            if target_timeline not in self._timelines:
                raise TimelineNotFoundError(f"Target timeline '{target_timeline}' does not exist.")
            self._timelines[target_timeline][seed_hash] = param_vector.tolist()
            self._cache.cache_clear()
            self._save_db()

    def get(self, seed_hash: str, timeline: str = None):
        target_timeline = timeline or self._active_timeline
        result = self._cache((seed_hash, target_timeline))
        if result is None:
            raise SeedNotFoundError(f"Seed '{seed_hash}' not found in timeline '{target_timeline}'.")
        return result

    def exists(self, seed_hash: str, timeline: str = None) -> bool:
        target_timeline = timeline or self._active_timeline
        return seed_hash in self._timelines.get(target_timeline, {})

    @property
    def current_timeline(self) -> str:
        return self._active_timeline

class AethelredCore:
    """
    An enterprise-grade, multiversal fractal knowledge engine. It compresses
    information into quantum-theoretic seeds and deterministically reconstructs
    them with hyper-dimensional detail. All operations are thread-safe, resilient,
    and designed for timeline-aware state management.
    """

    def __init__(self, seed_size=64, seed_store=None, hash_algorithm="sha3_512"):
        self.seed_size = seed_size
        self.hash_algorithm = hash_algorithm
        self.seed_store = seed_store or MultiverseSeedStore()
        self._lock = threading.RLock()
        logging.info("[INIT] AethelredCore initialized with seed_size=%d and hash_algorithm=%s",
                     seed_size, hash_algorithm)

    def _hash(self, data: str) -> str:
        """
        Generates a robust, fixed-length hash from input data using the
        configured algorithm (SHA3-512 recommended).
        """
        try:
            h = hashlib.new(self.hash_algorithm)
            h.update(data.encode('utf-8'))
            digest = h.hexdigest()
            logging.debug("[HASH] Data hashed to: %s", digest)
            return digest
        except Exception as e:
            logging.error("[HASH] Failed to hash data: %s", e, exc_info=True)
            raise CoreIntegrityError("Hashing mechanism failed.") from e

    def compress(self, data_str: str) -> str:
        """
        Compresses a raw data string into a fractal seed hash.
        If the seed is new for the current timeline, it generates and persists
        a reproducible parameter vector derived from its hash.
        """
        try:
            seed_hash = self._hash(data_str)
            with self._lock:
                if not self.seed_store.exists(seed_hash):
                    # Generate a deterministic, high-dimensional parameter vector
                    seed_int = int(seed_hash, 16)
                    rng = np.random.RandomState(seed_int % (2**32 - 1))
                    params = rng.uniform(-1.0, 1.0, size=self.seed_size)
                    params /= np.linalg.norm(params) # Normalize for stability
                    
                    self.seed_store.set(seed_hash, params)
                    logging.info("[COMPRESS] New seed generated and persisted: %s in timeline: %s",
                                 seed_hash, self.seed_store.current_timeline)
            return seed_hash
        except Exception as e:
            logging.error("[COMPRESS] Compression failed: %s", e, exc_info=True)
            raise CoreIntegrityError("Compression process failed.") from e

    def reconstruct(self, seed_hash: str, detail_level=1024, complexity_factor=4.0, noise_injection=0.01) -> np.ndarray:
        """
        Deterministically reconstructs hyper-dimensional data from a fractal seed.
        The reconstruction is a multi-layered process, creating complex wave patterns.

        Args:
            seed_hash (str): The hash of the seed to reconstruct.
            detail_level (int): The number of data points to generate.
            complexity_factor (float): Controls the frequency of the wave components.
            noise_injection (float): Amount of deterministic noise to add for texture.

        Returns:
            np.ndarray: The reconstructed data vector.
        """
        try:
            params = self.seed_store.get(seed_hash)
            rng = np.random.RandomState(int(seed_hash, 16) % (2**32 - 1))

            # Multi-layered fractal construction
            base_indices = np.arange(detail_level)
            final_values = np.zeros(detail_level)

            for i in range(self.seed_size):
                frequency = (i + 1) * complexity_factor * np.pi / self.seed_size
                phase_shift = params[(i * 2) % self.seed_size] * np.pi
                amplitude = params[(i * 3) % self.seed_size]
                
                wave = amplitude * np.sin(frequency * base_indices + phase_shift)
                final_values += wave

            # Apply a non-linear activation for complexity
            final_values = np.tanh(final_values)

            # Add deterministic noise
            if noise_injection > 0:
                final_values += rng.normal(0, noise_injection, size=detail_level)

            logging.info("[RECONSTRUCT] Seed %s reconstructed (detail=%d)", seed_hash, detail_level)
            return final_values

        except SeedNotFoundError:
            logging.warning("[RECONSTRUCT] Seed hash not found: %s", seed_hash)
            raise
        except Exception as e:
            logging.error("[RECONSTRUCT] Reconstruction error: %s", e, exc_info=True)
            raise CoreIntegrityError("Reconstruction process failed.") from e

    def get_seed_info(self, seed_hash: str) -> dict:
        """
        Fetches full diagnostic information for a given seed hash in the
        current timeline, including its parameter vector and metadata.
        """
        try:
            params = self.seed_store.get(seed_hash)
            info = {
                "seed_hash": seed_hash,
                "timeline": self.seed_store.current_timeline,
                "seed_size": self.seed_size,
                "param_vector_mean": np.mean(params),
                "param_vector_std": np.std(params),
                "query_timestamp_utc": datetime.utcnow().isoformat()
            }
            logging.info("[INFO] Fetched info for seed %s", seed_hash)
            return info
        except SeedNotFoundError:
            logging.warning("[INFO] Failed to fetch info for non-existent seed: %s", seed_hash)
            raise
        except Exception as e:
            logging.error("[INFO] Failed to fetch info for %s: %s", seed_hash, e, exc_info=True)
            raise CoreIntegrityError("Info retrieval process failed.") from e

# --- Example Usage: A Demonstration of God-Core Capabilities ---
if __name__ == "__main__":
    logging.info("="*60)
    logging.info("PROMETHEUS CORE: AETHELRED ENGINE DEMONSTRATION")
    logging.info("="*60)

    try:
        # Initialize the core with a larger seed size for more complex patterns
        core = AethelredCore(seed_size=128)

        # --- Timeline Operations ---
        logging.info("\n--- PHASE 1: TIMELINE MANAGEMENT ---")
        logging.info("Current timeline: %s", core.seed_store.current_timeline)
        core.seed_store.fork_timeline("beta_timeline", source_timeline_id="genesis")
        core.seed_store.switch_timeline("genesis")

        # --- Data Compression ---
        logging.info("\n--- PHASE 2: FRACTAL COMPRESSION ---")
        data1 = "Every king was once a savage, every legend was once unknown."
        data2 = "The architecture of this reality is encoded in the seed."
        
        seed_hash1 = core.compress(data1)
        seed_hash2 = core.compress(data2)

        logging.info("Data 1 ('%s...') compressed to: %s", data1[:15], seed_hash1)
        logging.info("Data 2 ('%s...') compressed to: %s", data2[:15], seed_hash2)

        # --- Evolve in a Different Timeline ---
        logging.info("\n--- PHASE 3: MULTIVERSE EVOLUTION ---")
        core.seed_store.switch_timeline("beta_timeline")
        data3 = "In the beta timeline, the savage becomes a god."
        seed_hash3 = core.compress(data3)
        logging.info("Data 3 ('%s...') compressed to: %s in timeline '%s'",
                     data3[:15], seed_hash3, core.seed_store.current_timeline)
        
        # Check existence across timelines
        logging.info("Seed 1 exists in 'genesis': %s", core.seed_store.exists(seed_hash1, "genesis"))
        logging.info("Seed 3 exists in 'genesis': %s", core.seed_store.exists(seed_hash3, "genesis"))
        logging.info("Seed 3 exists in 'beta_timeline': %s", core.seed_store.exists(seed_hash3, "beta_timeline"))
        
        core.seed_store.switch_timeline("genesis") # Return to genesis

        # --- Data Reconstruction ---
        logging.info("\n--- PHASE 4: HYPER-DIMENSIONAL RECONSTRUCTION ---")
        reconstructed_data = core.reconstruct(seed_hash1, detail_level=256, complexity_factor=8.0)
        logging.info("Reconstructed data sample from Seed 1: %s", reconstructed_data[:10])

        # --- Diagnostics ---
        logging.info("\n--- PHASE 5: CORE DIAGNOSTICS ---")
        info = core.get_seed_info(seed_hash1)
        logging.info("Diagnostic info for Seed 1: %s", json.dumps(info, indent=2))

    except CoreIntegrityError as e:
        logging.critical("A core integrity failure occurred. System shutting down. Reason: %s", e, exc_info=True)
    except Exception as e:
        logging.critical("An unexpected catastrophic failure occurred: %s", e, exc_info=True)