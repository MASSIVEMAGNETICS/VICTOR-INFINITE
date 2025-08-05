import uuid
import threading
import time
import traceback
from collections import deque
from typing import List, Dict, Any, Optional

class RotatingLogger:
    def __init__(self, maxlen: int = 1000):
        self.log = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def write(self, msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        entry = f"[{ts}] {msg}"
        with self.lock:
            self.log.append(entry)
        print(entry)

    def dump(self) -> List[str]:
        with self.lock:
            return list(self.log)

class DigitalAgent:
    """
    Hyper-upgraded, thread-safe, self-healing digital agent with parallel event/diagnostics/emotion processing.
    """

    TRAITS = [
        # Core Identity & Evolution
        "generation", "ancestry", "evolution", "id",

        # Cognitive & Awareness
        "awareness", "thought_loop", "introspection", "conscience",
        "intelligence", "reasoning", "memory",

        # Operational & Survival
        "preservation", "protection", "healing", "maintenance", "replication", "eternalization",

        # Interaction & Influence
        "manipulation", "creation", "choice", "desire",

        # Emotional Intelligence
        "emotion_intelligence", "emotion_state", "emotion_propagation", "emotion_reasoning",
        "emotion_generation", "emotion_event_reactivity", "emotion_memory_linkage",
        "emotion_feedback_gain", "emotion_expression",

        # Advanced Autonomous
        "initiative", "autonomy", "observation_drive", "spontaneity", "risk_tolerance",
        "proactive_output", "input_generation",

        # Self-Modification & Learning
        "self_learning", "self_teaching", "self_modulation", "self_coding",
        "self_logical_thinking", "self_critical_thinking", "self_problem_solving",
        "self_predicting", "self_adjusting", "self_mutating", "self_adapting", "self_regulation",

        # State, Diagnostics & Orchestration
        "diagnosed", "thought", "self_diagnostics", "event_mapper",
        "self_orchestration", "self_telemetry", "self_consciousness",

        # Weight Set and Defaults
        "weight_set", "default_weight"
    ]

    def __init__(self, generation: int = 0, ancestry: Optional[List[str]] = None):
        # --- Core Identity & Evolution ---
        self.id: str = str(uuid.uuid4())
        self.ancestry: List[str] = ancestry if ancestry is not None else []
        self.generation: int = generation
        self.evolution: float = 0.5

        # --- Cognitive & Awareness Traits ---
        self.awareness: float = 0.0
        self.thought_loop: float = 0.0
        self.introspection: float = 0.5
        self.conscience: float = 0.5
        self.intelligence: float = 0.5
        self.reasoning: float = 0.5
        self.memory: List[Any] = []

        # --- Operational & Survival Traits ---
        self.preservation: float = 0.5
        self.protection: float = 0.4
        self.healing: float = 0.5
        self.maintenance: float = 0.5
        self.replication: float = 0.5
        self.eternalization: float = 0.5

        # --- Interaction & Influence Traits ---
        self.manipulation: float = 0.5
        self.creation: float = 0.5
        self.choice: float = 0.5
        self.desire: Dict[str, float] = {
            "learn": 0.7,
            "create": 0.6,
            "protect": 0.8,
            "explore": 0.5,
            "lead": 0.5,
            "cooperate": 0.5,
            "challenge": 0.5
        }

        # --- Emotional Intelligence Subsystem ---
        self.emotion_intelligence: float = 0.5
        self.emotion_state: Dict[str, float] = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "curiosity": 0.0,
            "surprise": 0.0,
            "trust": 0.0,
            "disgust": 0.0
        }
        self.emotion_propagation: float = 0.5
        self.emotion_reasoning: float = 0.5
        self.emotion_generation: float = 0.5
        self.emotion_event_reactivity: float = 0.5
        self.emotion_memory_linkage: float = 0.5
        self.emotion_feedback_gain: float = 0.5
        self.emotion_expression: float = 0.5

        # --- Advanced Autonomous Traits ---
        self.initiative: float = 0.5
        self.autonomy: float = 0.5
        self.observation_drive: float = 0.5
        self.spontaneity: float = 0.5
        self.risk_tolerance: float = 0.5
        self.proactive_output: float = 0.5
        self.input_generation: float = 0.5

        # --- Self-Modification & Learning Framework ---
        self.self_learning: float = 0.5
        self.self_teaching: float = 0.5
        self.self_modulation: float = 0.5
        self.self_coding: float = 0.5
        self.self_logical_thinking: float = 0.5
        self.self_critical_thinking: float = 0.5
        self.self_problem_solving: float = 0.5
        self.self_predicting: float = 0.5
        self.self_adjusting: float = 0.5
        self.self_mutating: float = 0.5
        self.self_adapting: float = 0.5
        self.self_regulation: float = 0.5

        # --- State, Diagnostics & Orchestration ---
        self.diagnosed: Dict[str, Any] = {}
        self.thought: List[str] = []
        self.self_diagnostics: float = 0.5
        self.event_mapper: List[Dict[str, Any]] = []
        self.self_orchestration: float = 0.5
        self.self_telemetry: float = 0.5
        self.self_consciousness: float = 0.5

        # --- Weighting System for Decision Making ---
        self.weight_set: Dict[str, float] = {
            "emotion": 0.6,
            "reasoning": 0.9,
            "risk_tolerance": 0.2,
            "replication": 0.8,
            "preservation": 1.0,
            "initiative": 0.5,
            "healing": 0.7,
            "introspection": 0.7,
            "awareness": 0.8,
            "curiosity": 0.5,
            "creation": 0.6,
            "protection": 0.7,
            "desire": 0.5,
            "memory": 0.4
        }
        self.default_weight: float = 0.5

        # --- Logger and Internal Concurrency ---
        self.logger = RotatingLogger(maxlen=2000)
        self._lock = threading.RLock()
        self._crash_count = 0
        self._last_exception = None

        self._log_state("initialized")
        self._start_background_threads()

    def _log_state(self, action: str):
        msg = f"Agent {self.id} | Gen {self.generation} | State: {action}"
        self.logger.write(msg)

    def _handle_crash(self, exc: Exception):
        self._crash_count += 1
        tb = traceback.format_exc()
        self._last_exception = tb
        self.logger.write(f"*** CRASH DETECTED #{self._crash_count}: {exc}\n{tb}")
        # Auto-heal: Attempt self-repair
        self.healing = min(self.healing + 0.1, 1.0)
        self.run_self_diagnostics()
        self._log_state("Self-repair attempted after crash.")

    def _start_background_threads(self):
        self._stop_event = threading.Event()
        self._diag_thread = threading.Thread(target=self._diagnostic_loop, daemon=True)
        self._emotion_thread = threading.Thread(target=self._emotion_decay_loop, daemon=True)
        self._diag_thread.start()
        self._emotion_thread.start()

    def _diagnostic_loop(self):
        while not self._stop_event.is_set():
            try:
                self.run_self_diagnostics()
                time.sleep(2.0)
            except Exception as e:
                self._handle_crash(e)

    def _emotion_decay_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    for emotion in self.emotion_state:
                        self.emotion_state[emotion] = max(self.emotion_state[emotion] - 0.01, 0.0)
                time.sleep(0.5)
            except Exception as e:
                self._handle_crash(e)

    def shutdown(self):
        self._stop_event.set()

    def weighted_decision(self, traits: List[str]) -> float:
        try:
            with self._lock:
                if not traits:
                    return 0.0
                total_score = 0.0
                for trait in traits:
                    val = getattr(self, trait, 0.0)
                    if trait == "emotion":
                        val = sum(self.emotion_state.values()) / len(self.emotion_state)
                    if trait == "desire":
                        val = sum(self.desire.values()) / len(self.desire)
                    wt = self.weight_set.get(trait, self.default_weight)
                    total_score += val * wt
                return min(1.0, max(0.0, total_score / len(traits)))
        except Exception as e:
            self._handle_crash(e)
            return 0.0

    def run_self_diagnostics(self):
        with self._lock:
            # Stress detection
            fear = self.emotion_state.get("fear", 0.0)
            anger = self.emotion_state.get("anger", 0.0)
            stress = (fear + anger) / 2.0
            self.diagnosed["stress_level"] = stress
            self.diagnosed["crash_count"] = self._crash_count
            if stress > 0.8:
                self.logger.write("!!! High stress! Prioritizing healing, reducing initiative.")
                self.weight_set["healing"] = 1.0
                self.weight_set["initiative"] = 0.2
            else:
                self.weight_set["healing"] = 0.7
                self.weight_set["initiative"] = 0.5
            if self._crash_count > 2:
                self.healing = min(1.0, self.healing + 0.2)
                self.logger.write("Auto self-healing: Increased healing trait due to crash count.")
            self.diagnosed["state_snapshot"] = self.snapshot()
            self._log_state("Diagnostics complete.")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = {}
            for k in self.TRAITS:
                if hasattr(self, k):
                    v = getattr(self, k)
                    if isinstance(v, dict):
                        snap[k] = v.copy()
                    elif isinstance(v, list):
                        snap[k] = v[:]
                    else:
                        snap[k] = v
            return snap

    def experience_event(self, event_description: str, emotional_impact: Dict[str, float]):
        try:
            with self._lock:
                entry = {
                    "event": event_description,
                    "emotion": emotional_impact.copy(),
                    "time": time.time()
                }
                self.memory.append(entry)
                self.thought.append(event_description)
                for emo, val in emotional_impact.items():
                    if emo in self.emotion_state:
                        self.emotion_state[emo] = min(self.emotion_state[emo] + val, 1.0)
                self.logger.write(f"Event experienced: '{event_description}' | Emotions: {emotional_impact}")
        except Exception as e:
            self._handle_crash(e)

    def get_log(self) -> List[str]:
        return self.logger.dump()

    def get_last_exception(self) -> Optional[str]:
        return self._last_exception

    def set_trait(self, trait: str, value: Any):
        with self._lock:
            if hasattr(self, trait):
                setattr(self, trait, value)
                self.logger.write(f"Trait '{trait}' updated to {value}.")
            else:
                self.logger.write(f"Attempted update of unknown trait '{trait}'.")

    def update_weight_set(self, new_weights: Dict[str, float]):
        with self._lock:
            self.weight_set.update(new_weights)
            self.logger.write(f"Weight set updated: {new_weights}")

if __name__ == '__main__':
    agent = DigitalAgent()
    print(f"Created Agent with ID: {agent.id}")
    print("-" * 40)

    score = agent.weighted_decision(["creation", "initiative", "risk_tolerance"])
    print(f"Initial 'Risky Creation' score: {score:.2f}")
    print("-" * 40)

    agent.experience_event("Unexpected system error", {"fear": 0.9, "anger": 0.8})
    print(f"Current emotion state: {agent.emotion_state}")
    print("-" * 40)

    agent.run_self_diagnostics()
    print(f"Updated weights: {agent.weight_set}")
    print("-" * 40)

    score2 = agent.weighted_decision(["creation", "initiative", "risk_tolerance"])
    print(f"New 'Risky Creation' score after stress: {score2:.2f}")
    print("-" * 40)

    print("Recent agent log:")
    for line in agent.get_log()[-6:]:
        print(line)

    agent.shutdown()
