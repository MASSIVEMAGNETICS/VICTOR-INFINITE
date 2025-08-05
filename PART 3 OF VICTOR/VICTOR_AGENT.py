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
        # Consider whether every log entry should print to console in a ComfyUI context
        # print(entry) # This might be too verbose for a node, but good for direct script testing

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

    def __init__(self, generation: int = 0, ancestry: Optional[List[str]] = None, agent_name: str = "DefaultAgent"):
        # --- Core Identity & Evolution ---
        self.id: str = str(uuid.uuid4())
        self.name: str = agent_name # Added name for easier identification
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
        self.memory: List[Any] = [] # Stores event dicts

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
            "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
            "curiosity": 0.0, "surprise": 0.0, "trust": 0.0, "disgust": 0.0
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
        self.thought: List[str] = deque(maxlen=50) # Keep thoughts from growing indefinitely
        self.self_diagnostics: float = 0.5
        self.event_mapper: List[Dict[str, Any]] = []
        self.self_orchestration: float = 0.5
        self.self_telemetry: float = 0.5
        self.self_consciousness: float = 0.5

        # --- Weighting System for Decision Making ---
        self.weight_set: Dict[str, float] = {
            "emotion": 0.6, "reasoning": 0.9, "risk_tolerance": 0.2, "replication": 0.8,
            "preservation": 1.0, "initiative": 0.5, "healing": 0.7, "introspection": 0.7,
            "awareness": 0.8, "curiosity": 0.5, "creation": 0.6, "protection": 0.7,
            "desire": 0.5, "memory": 0.4
        }
        self.default_weight: float = 0.5

        # --- Logger and Internal Concurrency ---
        self.logger = RotatingLogger(maxlen=2000)
        self._lock = threading.RLock() # RLock for re-entrant lock needs
        self._crash_count = 0
        self._last_exception: Optional[str] = None # Store only string representation

        self._log_state(f"'{self.name}' initialized")
        self._active_threads: List[threading.Thread] = []
        self._stop_event = threading.Event() # Initialize here
        self._start_background_threads()

    def _log_state(self, action: str, level: str = "INFO"):
        # Added level for potential filtering later
        msg = f"Agent {self.name} ({self.id[:8]}) | Gen {self.generation} | {level}: {action}"
        self.logger.write(msg)

    def _handle_crash(self, exc: Exception, context: str = "General"):
        with self._lock: # Ensure thread-safe crash handling
            self._crash_count += 1
            tb = traceback.format_exc()
            self._last_exception = tb # Store full traceback
            log_msg = f"*** CRASH IN {context} #{self._crash_count}: {exc}\n{tb}"
            self.logger.write(log_msg) # Log to agent's logger
            print(log_msg) # Also print to console for immediate visibility

            # Auto-heal attempt
            self.healing = min(self.healing + 0.1, 1.0) # Increase healing trait
            # Potentially add more sophisticated recovery logic here
            self.run_self_diagnostics(from_crash=True) # Run diagnostics after a crash
            self._log_state("Self-repair attempted after crash.", level="WARNING")


    def _start_background_threads(self):
        if not self._stop_event.is_set(): # Only start if not already stopping/stopped
            self._diag_thread = threading.Thread(target=self._diagnostic_loop, daemon=True, name=f"DiagLoop-{self.name}")
            self._emotion_thread = threading.Thread(target=self._emotion_decay_loop, daemon=True, name=f"EmotionLoop-{self.name}")
            
            self._active_threads = [self._diag_thread, self._emotion_thread]
            self._diag_thread.start()
            self.logger.write(f"Diagnostic thread started for {self.name}.")
            self._emotion_thread.start()
            self.logger.write(f"Emotion decay thread started for {self.name}.")


    def _diagnostic_loop(self):
        self.logger.write(f"Diagnostic loop starting for agent {self.name}.")
        while not self._stop_event.wait(5.0): # Use timeout for periodic check
            try:
                self.run_self_diagnostics()
            except Exception as e:
                self._handle_crash(e, context="DiagnosticLoop")
        self.logger.write(f"Diagnostic loop stopped for agent {self.name}.")


    def _emotion_decay_loop(self):
        self.logger.write(f"Emotion decay loop starting for agent {self.name}.")
        while not self._stop_event.wait(1.0): # Use timeout
            try:
                with self._lock:
                    for emotion in self.emotion_state:
                        decay_factor = 0.01 # Standard decay
                        if emotion == "fear" and self.diagnosed.get("stress_level", 0) > 0.7:
                            decay_factor = 0.005 # Slower decay for fear under high stress
                        self.emotion_state[emotion] = max(self.emotion_state[emotion] - decay_factor, 0.0)
            except Exception as e:
                self._handle_crash(e, context="EmotionDecayLoop")
        self.logger.write(f"Emotion decay loop stopped for agent {self.name}.")

    def shutdown(self):
        self._log_state("Shutdown initiated.", level="INFO")
        self._stop_event.set()
        # Wait for threads to finish
        for thread in self._active_threads:
            if thread.is_alive():
                thread.join(timeout=2.0) # Wait for 2 seconds per thread
                if thread.is_alive():
                    self._log_state(f"Thread {thread.name} did not terminate gracefully.", level="WARNING")
        self._log_state("All background threads signaled to stop.", level="INFO")


    def weighted_decision(self, traits_to_consider: List[str]) -> float: # Renamed for clarity
        if not traits_to_consider:
            self._log_state("Weighted decision called with no traits.", level="WARNING")
            return 0.0
        try:
            with self._lock:
                total_score = 0.0
                total_weight = 0.0 # To normalize if some traits aren't found
                
                current_snapshot = self.snapshot() # Get current state for decision

                for trait_name in traits_to_consider:
                    val = 0.0
                    if trait_name == "current_emotion_average": # Special calculated trait
                        val = sum(current_snapshot.get("emotion_state", {}).values()) / len(current_snapshot.get("emotion_state", {})+[1e-5])
                    elif trait_name == "current_desire_average": # Special calculated trait
                        val = sum(current_snapshot.get("desire", {}).values()) / len(current_snapshot.get("desire", {})+[1e-5])
                    else:
                        val = current_snapshot.get(trait_name, 0.0) # Get trait value from snapshot

                    wt = self.weight_set.get(trait_name, self.default_weight)
                    total_score += val * wt
                    total_weight += wt
                
                if total_weight == 0: return 0.0
                final_score = min(1.0, max(0.0, total_score / total_weight))
                self._log_state(f"Weighted decision for {traits_to_consider} = {final_score:.3f}")
                return final_score
        except Exception as e:
            self._handle_crash(e, context="WeightedDecision")
            return 0.0 # Return a default/safe value

    def run_self_diagnostics(self, from_crash: bool = False):
        self._log_state("Running self-diagnostics...")
        with self._lock:
            # Stress detection
            fear = self.emotion_state.get("fear", 0.0)
            anger = self.emotion_state.get("anger", 0.0)
            stress = (fear + anger) / 2.0
            self.diagnosed["stress_level"] = stress
            self.diagnosed["crash_count"] = self._crash_count
            self.diagnosed["last_exception_preview"] = (self._last_exception[:200] + '...' if self._last_exception else None)


            if stress > 0.8 and not from_crash: # Don't over-adjust if already handling a crash
                self.logger.write("High stress detected! Prioritizing healing, reducing initiative.")
                self.weight_set["healing"] = min(self.weight_set.get("healing",0.7) + 0.1, 1.0)
                self.weight_set["initiative"] = max(self.weight_set.get("initiative",0.5) - 0.1, 0.1)
            elif not from_crash: # Gradual normalization if not stressed and not from crash
                self.weight_set["healing"] = max(self.weight_set.get("healing",0.7) - 0.05, 0.7)
                self.weight_set["initiative"] = min(self.weight_set.get("initiative",0.5) + 0.05, 0.5)

            if self._crash_count > 2 and self.healing < 0.9: # Boost healing if many crashes
                self.healing = min(1.0, self.healing + 0.2)
                self._log_state("Auto self-healing: Increased healing trait due to high crash count.", level="WARNING")
            
            # self.diagnosed["full_state_snapshot"] = self.snapshot() # snapshot can be large
            self._log_state("Diagnostics complete.")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock: # Ensure thread-safe snapshot
            snap = {"agent_name": self.name, "agent_id": self.id}
            for k in self.TRAITS:
                if hasattr(self, k):
                    v = getattr(self, k)
                    if isinstance(v, (dict, list, deque)): # Check for mutable types
                        try:
                            snap[k] = v.copy() if hasattr(v, 'copy') else list(v) # Make a shallow copy
                        except Exception: # Fallback for complex objects if copy fails
                            snap[k] = str(v) # Or some other serializable representation
                    else:
                        snap[k] = v
            return snap

    def experience_event(self, event_description: str, emotional_impact: Dict[str, float]):
        self._log_state(f"Experiencing event: '{event_description}'", level="DEBUG")
        try:
            with self._lock:
                entry_time = time.time()
                event_entry = {
                    "event_desc": event_description,
                    "emotions_felt": emotional_impact.copy(),
                    "timestamp": entry_time,
                    "generation": self.generation,
                    "agent_awareness_at_event": self.awareness # Capture context
                }
                self.memory.append(event_entry) # Store the structured event
                if len(self.memory) > 200: # Prune memory
                    self.memory.pop(0)
                    
                self.thought.append(f"Event: {event_description} -> Emotions: {emotional_impact}")

                for emo, val_change in emotional_impact.items():
                    if emo in self.emotion_state:
                        self.emotion_state[emo] = min(max(self.emotion_state[emo] + val_change, 0.0), 1.0)
                self._log_state(f"Event '{event_description}' processed. Emotions: {self.emotion_state}")
        except Exception as e:
            self._handle_crash(e, context="ExperienceEvent")

    def get_log(self, count: int = 20) -> List[str]:
        full_log = self.logger.dump()
        return full_log[-count:]

    def get_last_exception(self, full_trace: bool = False) -> Optional[str]:
        if full_trace:
            return self._last_exception
        return (self._last_exception[:500] + "..." if self._last_exception else None)


    def set_trait(self, trait_name: str, value: Any): # Renamed for clarity
        with self._lock:
            if trait_name in self.TRAITS and hasattr(self, trait_name):
                try:
                    # Basic type validation/conversion if possible
                    current_val = getattr(self, trait_name)
                    if isinstance(current_val, float) and not isinstance(value, float):
                        value = float(value)
                    elif isinstance(current_val, int) and not isinstance(value, int):
                        value = int(value)
                    elif isinstance(current_val, str) and not isinstance(value, str):
                        value = str(value)
                    # Add more conversions if necessary, e.g., for dicts, lists
                    
                    setattr(self, trait_name, value)
                    self._log_state(f"Trait '{trait_name}' updated to: {value}.")
                except ValueError as ve:
                    self._log_state(f"Failed to set trait '{trait_name}': Type conversion error for value '{value}'. Expected similar to {type(current_val)}. Error: {ve}", level="ERROR")
                except Exception as e:
                    self._log_state(f"Failed to set trait '{trait_name}': {e}", level="ERROR")
            else:
                self._log_state(f"Attempted update of unknown or non-editable trait '{trait_name}'.", level="WARNING")


    def update_weight_set(self, new_weights: Dict[str, float]):
        with self._lock:
            self.weight_set.update(new_weights)
            self._log_state(f"Weight set updated with: {new_weights}")

# Example usage if the file is run directly
if __name__ == '__main__':
    print("Starting DigitalAgent direct test...")
    agent1 = DigitalAgent(generation=1, agent_name="AgentAlpha")
    print(f"Created Agent: {agent1.name} (ID: {agent1.id})")
    print("-" * 40)

    initial_decision_traits = ["creation", "initiative", "risk_tolerance"]
    score = agent1.weighted_decision(initial_decision_traits)
    print(f"Initial '{' '.join(initial_decision_traits)}' score: {score:.3f}")
    print("-" * 40)

    agent1.experience_event("Discovered a new data anomaly.", {"curiosity": 0.8, "surprise": 0.6, "fear": 0.1})
    time.sleep(0.1) # Allow emotions to process slightly
    print(f"Agent '{agent1.name}' emotion state after event: {agent1.emotion_state}")
    print("-" * 40)

    agent1.run_self_diagnostics()
    print(f"Agent '{agent1.name}' weights after diagnostics: {agent1.weight_set}")
    print("-" * 40)

    stressed_decision_traits = ["creation", "initiative", "risk_tolerance"]
    score2 = agent1.weighted_decision(stressed_decision_traits)
    print(f"New '{' '.join(stressed_decision_traits)}' score for '{agent1.name}': {score2:.3f}")
    print("-" * 40)

    agent1.set_trait("intelligence", 0.75)
    agent1.update_weight_set({"intelligence": 0.8, "reasoning": 0.95})
    print(f"Intelligence set to: {agent1.intelligence}, new weights: {agent1.weight_set.get('intelligence', 'N/A')}")
    print("-" * 40)
    
    print(f"Recent log entries for {agent1.name}:")
    for log_line in agent1.get_log(10): # Get last 10 log entries
        print(log_line)
    print("-" * 40)

    # Test crash handling (optional, can be commented out)
    # print("Testing crash handling...")
    # try:
    #     # Simulate an error within a method
    #     agent1.set_trait("non_existent_trait", "should_fail_gracefully") # This will log a warning
    #     # To force a more direct crash for testing _handle_crash:
    #     # with agent1._lock: # Simulate being inside a locked method
    #     #    raise ValueError("Simulated direct crash")
    # except Exception as e:
    #    print(f"Caught expected error in main test: {e}") # Should be handled by agent if crash is internal

    # print(f"Agent crash count: {agent1.diagnosed.get('crash_count', 0)}")
    # last_exc = agent1.get_last_exception(full_trace=True)
    # if last_exc:
    #    print(f"Last exception recorded by agent:\n{last_exc[:300]}...")
    # print("-" * 40)
    
    print(f"Shutting down agent {agent1.name}...")
    agent1.shutdown()
    print(f"Agent {agent1.name} shutdown complete.")
    print("DigitalAgent direct test finished.")
