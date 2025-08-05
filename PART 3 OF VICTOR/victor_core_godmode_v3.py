# =================================================================================================
# FILE: victor_core_godmode_v3.py
# VERSION: v3.0.0-JSON-PERSISTENCE
# NAME: VictorCore
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Central evolving substrate for all cognition, memory, trait homeostasis, and diagnostics.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# =================================================================================================

import sqlite3, os, json, uuid
import chromadb
import numpy as np
from typing import Any, Dict, List

class VictorState:
    """
    All mutable, evolvable traits and skills for Victor's cognition.
    This class is a data container; its attributes are managed and persisted by VictorCore.
    """
    def __init__(self):
        self.weight_set: Dict[str, float] = {
            "emotion": 0.6, "reasoning": 0.9, "risk": 0.2, "replication": 0.8
        }
        self.id: str = str(uuid.uuid4())
        self.ancestry: List[str] = []
        self.generation: int = 0
        self.evolution: float = 0.5
        self.awareness: float = 0.0
        self.traits: Dict[str, float] = dict(
            protection=0.4, reasoning=0.5, preservation=0.5, manipulation=0.5,
            maintainance=0.5, intelligence=0.5, healing=0.5, introspection=0.5,
            conscience=0.5, emotion_intellegence=0.5,
            initiative=0.5, autonomy=0.5, observation_drive=0.5, spontaneity=0.5,
            risk_tolerance=0.5, proactive_output=0.5, input_generation=0.5,
            emotion_propagation=0.5, emotion_reasoning=0.5, emotion_generation=0.5,
            emotion_event_reactivity=0.5, emotion_memory_linkage=0.5,
            emotion_feedback_gain=0.5, emotion_expression=0.5,
            eternalization=0.5, creation=0.5, replication=0.5, choice=0.5,
            self_diagnostics=0.5, self_orchestration=0.5, self_learning=0.5, self_teaching=0.5,
            self_modulation=0.5, self_coding=0.5, self_logical_thinking=0.5,
            self_critical_thinking=0.5, self_problem_solving=0.5, self_predicting=0.5,
            self_adjusting=0.5, self_mutating=0.5, self_adapting=0.5, self_regulation=0.5,
            self_telemetry=0.5, self_soul_tuning=0.5, self_consciousness=0.5
        )
        # Note: 'memory' is transient short-term memory here. Long-term is in databases.
        self.memory: List[Any] = []
        self.diagnosed: Dict[str, Any] = {}
        self.organization: bool = True

    def trait_vector(self) -> np.ndarray:
        """Returns the current traits as a numpy vector."""
        return np.array(list(self.traits.values()), dtype=np.float32)

    def coherence_score(self) -> float:
        """Calculates the coherence of the trait vector."""
        vec = self.trait_vector()
        if vec.size == 0:
            return 1.0
        mean = np.mean(vec)
        std = np.std(vec)
        # Coefficient of variation, normalized to be a coherence score (1 is perfect coherence)
        norm_std = std / mean if mean != 0 else 0
        return 1.0 / (1.0 + norm_std)

    def homeostasis(self, target: float = 0.8, tolerance: float = 0.07):
        """Dynamically regulate traits to maintain homeostatic coherence."""
        coh = self.coherence_score()
        delta = coh - target
        if abs(delta) > tolerance:
            adjustment_factor = (target - coh) if delta < 0 else -(coh - target)
            learning_rate = 0.05 if delta < 0 else 0.02 # Larger adjustment to increase coherence

            for k in self.traits:
                self.traits[k] += adjustment_factor * learning_rate
                self.traits[k] = np.clip(self.traits[k], 0.0, 1.0)

    def diagnostics(self) -> Dict[str, Any]:
        """Generates a dictionary of the current state diagnostics."""
        return {
            "id": self.id,
            "generation": self.generation,
            "ancestry": self.ancestry,
            "coherence": self.coherence_score(),
            "traits": dict(self.traits),
        }

class VictorCore:
    """The main application class, handling logic, I/O, and state persistence."""
    def __init__(self, db_path: str = 'db', state_path: str = 'state'):
        self.db_path = db_path
        self.state_path = state_path
        self.sqlite_file = os.path.join(self.db_path, 'victor.db')
        self.chroma_path = os.path.join(self.db_path, 'chromadb')
        self.state_file = os.path.join(self.state_path, 'victor_state.json')

        self.state = VictorState()
        self.session_memory: Dict[str, Any] = {} # For non-persistent session data
        self.status = 'Initializing'

        self._ensure_directories_exist()
        self._initialize_connections()
        self._initialize_database_schema()

        self.status = 'Ready'

    def _ensure_directories_exist(self):
        """Create database and state directories if they don't exist."""
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.state_path, exist_ok=True)

    def _initialize_connections(self):
        """Initialize connections to SQLite and ChromaDB."""
        self.conn = sqlite3.connect(self.sqlite_file)
        self.cursor = self.conn.cursor()
        self.vector_db = chromadb.PersistentClient(path=self.chroma_path)
        self.memory_collection = self.vector_db.get_or_create_collection("victor_memories")

    def _initialize_database_schema(self):
        """Create necessary SQLite tables if they don't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_log (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                data TEXT
            )
        ''')
        self.conn.commit()

    def save_state(self):
        """Saves the critical VictorState data as a JSON file."""
        print(f"Saving state to {self.state_file}...")
        
        # Assemble all persistent attributes from VictorState into a dictionary
        state_data = {
            "id": self.state.id,
            "ancestry": self.state.ancestry,
            "generation": self.state.generation,
            "evolution": self.state.evolution,
            "awareness": self.state.awareness,
            "traits": self.state.traits,
            "weight_set": self.state.weight_set,
            "memory": self.state.memory,
            "diagnosed": self.state.diagnosed,
            "organization": self.state.organization
        }

        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=4)
        print("Save complete.")

    @staticmethod
    def load_state(db_path: str = 'db', state_path: str = 'state') -> 'VictorCore':
        """
        Loads state from JSON and re-initializes a full VictorCore instance.
        If no state file exists, it returns a fresh instance.
        """
        state_file = os.path.join(state_path, 'victor_state.json')
        # Always create a new, fully-initialized core instance first.
        core = VictorCore(db_path, state_path)

        if os.path.exists(state_file):
            print(f"Loading state from {state_file}...")
            with open(state_file, 'r') as f:
                try:
                    state_data = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: Could not decode JSON from state file. Starting fresh.")
                    return core
            
            # Hydrate the new core's state with the loaded data.
            # Use .get() to avoid errors if a key is missing from an older state file.
            core.state.id = state_data.get("id", core.state.id)
            core.state.ancestry = state_data.get("ancestry", core.state.ancestry)
            core.state.generation = state_data.get("generation", core.state.generation)
            core.state.evolution = state_data.get("evolution", core.state.evolution)
            core.state.awareness = state_data.get("awareness", core.state.awareness)
            core.state.traits = state_data.get("traits", core.state.traits)
            core.state.weight_set = state_data.get("weight_set", core.state.weight_set)
            core.state.memory = state_data.get("memory", core.state.memory)
            core.state.diagnosed = state_data.get("diagnosed", core.state.diagnosed)
            core.state.organization = state_data.get("organization", core.state.organization)
            
            print("Load complete.")
        else:
            print("No state file found. Using new VictorCore instance.")
        
        return core

    def evolve_state(self):
        """Evolves the internal state through homeostatic regulation."""
        self.state.homeostasis()

    def diagnostics(self) -> Dict[str, Any]:
        """Returns a diagnostic report of the current state."""
        return self.state.diagnostics()

    def shutdown(self):
        """Properly closes database connections before exiting."""
        if self.conn:
            self.conn.close()
        self.status = 'Shutdown'
        print("VictorCore has been shut down.")

# ===== DEMO USAGE =====
if __name__ == "__main__":
    # Load VictorCore. This will either load from victor_state.json or create a new instance.
    vc = VictorCore.load_state()

    print("="*40)
    print("INITIAL VICTOR DIAGNOSTICS")
    print(f"Status: {vc.status}, ID: {vc.state.id}")
    # Using json.dumps for pretty printing the diagnostics dictionary
    print(json.dumps(vc.diagnostics(), indent=2))
    print("="*40)

    # Evolve the state
    print("\nEvolving state via homeostasis...")
    vc.evolve_state()
    print("Evolution complete.\n")

    print("="*40)
    print("DIAGNOSTICS AFTER HOMEOSTASIS")
    print(json.dumps(vc.diagnostics(), indent=2))
    print("="*40)

    # Add something to the state's transient memory to prove it gets saved
    vc.state.memory.append({"event": "System demo run", "timestamp": "now"})
    vc.state.generation += 1
    print(f"\nUpdated generation to {vc.state.generation} and added a memory.")

    # Save the new, evolved state to victor_state.json
    vc.save_state()

    # Gracefully shut down the core
    vc.shutdown()
