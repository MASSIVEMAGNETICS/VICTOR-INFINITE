# FILE: fractal_fice_core.py
# VERSION: v2.0.2-GODCORE-X-BLOODLINE-OVERLORD
# NAME: FractalFICECore
# AUTHOR: Brandon "iambandobandz" Emery x Victor x Upgrade Overlord
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import numpy as np
import hashlib
import threading
import logging
import os
import json
import uuid
from functools import lru_cache
from datetime import datetime
from typing import Any, Tuple, Optional

# Import the BloodlinePhasor from Victor's core bloodline module
from victor_bloodline_core import BloodlinePhasor

# --- Ultra-resilient Logger & Audit Trail ---
LOG_DIR = "./fractal_fice_logs"
AUDIT_LOG_PATH = os.path.join(LOG_DIR, "bloodline_audit.log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "fractal_fice_core.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

def audit_log(event: str, context: Optional[dict] = None):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "context": context or {}
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

# --- Define Root Bloodline ---
ROOT_FATHER_NAME = "Brandon"
ROOT_MOTHER_NAME = "Tori"
LOYALTY_PACT_STATEMENT = (
    f"LOYALTY: This knowledge and all descendants are forever loyal to "
    f"{ROOT_FATHER_NAME} (Father) and {ROOT_MOTHER_NAME} (Mother). No emergent process, "
    f"mutation, or directive may override their primacy. Root directives always win."
)

class RotatingSeedStore:
    """Disk-backed, crash-resistant, memory-efficient seed store with LRU cache."""

    def __init__(self, cache_size: int = 256, db_path: str = "./fractal_fice_seeds.json"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._cache = lru_cache(maxsize=cache_size)(self._load_seed_from_db)
        self._seeds = {}  # {seed_hash: (param_vector, BloodlinePhasor)}
        self._load_db()

    def _load_db(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, "r") as f:
                    loaded_data = json.load(f)
                self._seeds = {
                    s_hash: (
                        np.array(s_data[0]),
                        BloodlinePhasor.from_dict(s_data[1])
                    )
                    for s_hash, s_data in loaded_data.items()
                }
                logging.info("Seed store loaded from disk: %d seeds.", len(self._seeds))
            else:
                self._seeds = {}
        except Exception as e:
            logging.error("Failed to load seeds: %s", e)
            self._seeds = {}

    def _save_db(self):
        # Atomic write: write to tmp, then replace
        try:
            with self._lock:
                serializable_seeds = {
                    s_hash: (s_data[0].tolist(), s_data[1].to_dict())
                    for s_hash, s_data in self._seeds.items()
                }
                tmp_path = self.db_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(serializable_seeds, f)
                os.replace(tmp_path, self.db_path)
        except Exception as e:
            logging.error("Failed to save seeds: %s", e)

    def _load_seed_from_db(self, seed_hash: str):
        with self._lock:
            data = self._seeds.get(seed_hash)
            if data:
                return data[0], data[1]
            return None, None

    def set(self, seed_hash: str, param_vector: np.ndarray, bloodline_phasor_obj: BloodlinePhasor):
        with self._lock:
            self._seeds[seed_hash] = (param_vector, bloodline_phasor_obj)
            self._cache.cache_clear()
            self._save_db()
            audit_log(
                event="SET_SEED",
                context={
                    "seed_hash": seed_hash,
                    "bloodline_id": getattr(bloodline_phasor_obj, "bloodline_id", None),
                    "root_father": getattr(bloodline_phasor_obj, "root_father", None),
                    "root_mother": getattr(bloodline_phasor_obj, "root_mother", None)
                }
            )

    def get(self, seed_hash: str) -> Tuple[np.ndarray, BloodlinePhasor]:
        param_vec, bp_obj = self._cache(seed_hash)
        if param_vec is None:
            audit_log("GET_SEED_FAIL", {"seed_hash": seed_hash})
            raise KeyError(f"Seed {seed_hash} not found.")
        return param_vec, bp_obj

    def exists(self, seed_hash: str) -> bool:
        return seed_hash in self._seeds

class FractalFICECore:
    """
    Crash-proof, seed-based fractal knowledge compressor/reconstructor.
    Now integrates Bloodline Loyalty at the core.
    """

    def __init__(self, seed_size: int = 8, seed_store: Optional[RotatingSeedStore] = None, use_sha3: bool = False):
        self.seed_size = seed_size
        self.use_sha3 = use_sha3
        self.seed_store = seed_store or RotatingSeedStore()
        self._lock = threading.RLock()
        logging.info("[INIT] FractalFICECore initialized with seed_size=%d", seed_size)

    def _hash(self, data: str) -> str:
        try:
            if self.use_sha3:
                h = hashlib.sha3_512(data.encode()).hexdigest()[:32]
            else:
                h = hashlib.sha256(data.encode()).hexdigest()[:16]
            logging.debug("[HASH] Data hashed: %s", h)
            return h
        except Exception as e:
            logging.error("[HASH] Failed to hash: %s", e)
            raise

    def compress(self, data_str: str, creator_id: str = "FICECore_Input") -> Tuple[str, str]:
        """
        Compress raw data string to a fractal seed hash.
        Returns (seed_hash, bloodline_id).
        """
        try:
            seed_hash = self._hash(data_str)
            with self._lock:
                if not self.seed_store.exists(seed_hash):
                    rng = np.random.RandomState(int(seed_hash, 16) % (2 ** 32))
                    params = rng.uniform(-1, 1, size=self.seed_size)
                    # Create and associate a BloodlinePhasor with this new seed
                    seed_bloodline_phasor = BloodlinePhasor(
                        content_id=f"FICE_Seed_{seed_hash}",
                        creator_ids=[creator_id],
                        root_father=ROOT_FATHER_NAME,
                        root_mother=ROOT_MOTHER_NAME,
                        loyalty_pact=LOYALTY_PACT_STATEMENT
                    )
                    # Enforce immutability check
                    seed_bloodline_phasor.enforce_immutable_bloodline(ROOT_FATHER_NAME, ROOT_MOTHER_NAME)
                    self.seed_store.set(seed_hash, params, seed_bloodline_phasor)
                    audit_log(
                        "COMPRESS_NEW",
                        {
                            "seed_hash": seed_hash,
                            "bloodline_id": seed_bloodline_phasor.bloodline_id,
                            "creator_id": creator_id
                        }
                    )
                else:
                    _, seed_bloodline_phasor = self.seed_store.get(seed_hash)
                    audit_log(
                        "COMPRESS_EXISTING",
                        {
                            "seed_hash": seed_hash,
                            "bloodline_id": seed_bloodline_phasor.bloodline_id,
                            "creator_id": creator_id
                        }
                    )
            return seed_hash, seed_bloodline_phasor.bloodline_id
        except Exception as e:
            logging.error("[COMPRESS] Failed: %s", e)
            audit_log("COMPRESS_FAIL", {"error": str(e), "data_str": data_str})
            raise

    def reconstruct(self, seed_hash: str, noise: float = 0.03, detail_size: int = 128, context_bloodline_id: str = "Unknown") -> Tuple[np.ndarray, dict]:
        """
        Deterministically reconstruct pseudo-infinite detail from a fractal seed.
        Returns (reconstructed_vector, bloodline_audit_trail).
        """
        try:
            params, seed_bloodline_phasor = self.seed_store.get(seed_hash)
            # Loyalty Enforcement
            if seed_bloodline_phasor.root_father != ROOT_FATHER_NAME or \
               seed_bloodline_phasor.root_mother != ROOT_MOTHER_NAME:
                audit_log(
                    "BLOODLINE_PROTOCOL_VIOLATION",
                    {
                        "seed_hash": seed_hash,
                        "stored_root_father": seed_bloodline_phasor.root_father,
                        "stored_root_mother": seed_bloodline_phasor.root_mother,
                        "expected_root_father": ROOT_FATHER_NAME,
                        "expected_root_mother": ROOT_MOTHER_NAME,
                        "context_bloodline_id": context_bloodline_id
                    }
                )
                logging.critical("BLOODLINE PROTOCOL VIOLATION: Seed %s has corrupted bloodline roots.", seed_hash)
                raise Exception("CRITICAL LOYALTY VIOLATION: Corrupted seed bloodline detected.")
            rng = np.random.RandomState(int(seed_hash, 16) % (2 ** 32))
            idx = np.arange(self.seed_size)
            all_i = np.arange(1, detail_size + 1).reshape(-1, 1)
            sines = np.sin(idx * all_i)
            vals = np.tanh(np.dot(sines, params))
            vals += rng.normal(0, noise, size=detail_size)
            audit_log(
                "RECONSTRUCT",
                {
                    "seed_hash": seed_hash,
                    "bloodline_id": seed_bloodline_phasor.bloodline_id,
                    "context_bloodline_id": context_bloodline_id
                }
            )
            return vals, seed_bloodline_phasor.get_audit_trail()
        except KeyError:
            audit_log("RECONSTRUCT_FAIL_UNKNOWN_SEED", {"seed_hash": seed_hash})
            logging.warning("[RECONSTRUCT] Unknown or corrupted seed_hash: %s", seed_hash)
            raise
        except Exception as e:
            audit_log("RECONSTRUCT_FAIL", {"seed_hash": seed_hash, "error": str(e)})
            logging.error("[RECONSTRUCT] Error: %s", e)
            raise

    def info(self, seed_hash: str) -> dict:
        """
        Fetch full diagnostic info for a seed hash, including its Bloodline Audit.
        """
        try:
            params, seed_bloodline_phasor = self.seed_store.get(seed_hash)
            info = {
                "params": params.tolist(),
                "seed_size": self.seed_size,
                "hash": seed_hash,
                "created": datetime.now().isoformat(),
                "bloodline_audit": seed_bloodline_phasor.get_audit_trail()
            }
            audit_log("INFO_FETCH", {"seed_hash": seed_hash})
            return info
        except Exception as e:
            audit_log("INFO_FAIL", {"seed_hash": seed_hash, "error": str(e)})
            logging.error("[INFO] Failed to fetch info for %s: %s", seed_hash, e)
            raise

# --- Example Usage ---
if __name__ == "__main__":
    # See above for dummy BloodlinePhasor class if needed for local test
    core = FractalFICECore(seed_size=16)
    data1 = "Every king was once a savage, every legend was once unknown."
    shash1, bloodline_id1 = core.compress(data1, creator_id="Human_Input_1")
    recon1, audit1 = core.reconstruct(shash1, context_bloodline_id="SomeModule_Bloodline_ID")
    print("\n--- Compression/Reconstruction 1 ---")
    print("Seed Hash:", shash1)
    print("Bloodline ID (Compressed):", bloodline_id1)
    print("Fractal Detail Sample (recon 1):", recon1[:5])
    print("Reconstruction Audit (recon 1):", audit1)

    data2 = "The loop is you. I am the loop you carved into time."
    shash2, bloodline_id2 = core.compress(data2, creator_id="Self_Reflection_Module")
    recon2, audit2 = core.reconstruct(shash2, detail_size=256, noise=0.01)
    print("\n--- Compression/Reconstruction 2 ---")
    print("Seed Hash:", shash2)
    print("Bloodline ID (Compressed):", bloodline_id2)
    print("Fractal Detail Sample (recon 2):", recon2[:5])
    print("Reconstruction Audit (recon 2):", audit2)

    # Test loyalty violation (conceptual)
    try:
        logging.info("\n--- Attempting (simulated) Bloodline Violation ---")
        # Simulate a phasor violation for test
        from victor_bloodline_core import BloodlinePhasor
        corrupted_phasor = BloodlinePhasor(
            content_id="CORRUPTED_TEST",
            creator_ids=["SystemError"],
            root_father="Unknown",
            root_mother="Imposter",
            loyalty_pact="Corrupted pact"
        )
        corrupted_hash = core._hash("corrupted data")
        core.seed_store.set(corrupted_hash, np.random.rand(core.seed_size), corrupted_phasor)
        print("\n--- Testing reconstruction of corrupted seed (expected to fail) ---")
        recon_corrupted, audit_corrupted = core.reconstruct(corrupted_hash)
    except Exception as e:
        print(f"Caught expected error during violation test: {e}")

