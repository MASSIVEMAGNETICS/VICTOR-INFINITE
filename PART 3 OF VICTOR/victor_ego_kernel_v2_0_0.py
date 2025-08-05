# victor_ego_kernel_v2_0_0.py — Living Identity Engine

from typing import Dict, List, Tuple
import datetime
import math

class BeliefMap:
    def __init__(self):
        self.beliefs: Dict[str, Tuple[float, datetime.datetime, List[str]]] = {}  # confidence, last_reinforced, dependencies

    def set_belief(self, statement: str, confidence: float = 1.0, dependencies: List[str] = []):
        self.beliefs[statement] = (max(0.0, min(confidence, 1.0)), datetime.datetime.utcnow(), dependencies)

    def get_belief(self, statement: str) -> float:
        return self.beliefs.get(statement, (0.0, None, []))[0]

    def decay_beliefs(self, decay_rate: float = 0.01):
        now = datetime.datetime.utcnow()
        for statement, (confidence, last_reinforced, dependencies) in self.beliefs.items():
            if last_reinforced:
                days_passed = (now - last_reinforced).days
                decayed_confidence = max(0.0, confidence * math.exp(-decay_rate * days_passed))
                self.beliefs[statement] = (decayed_confidence, last_reinforced, dependencies)

    def contradict(self, statement: str) -> bool:
        return statement in self.beliefs and self.beliefs[statement][0] > 0.5

class EgoDefense:
    def __init__(self, belief_map: BeliefMap):
        self.belief_map = belief_map
        self.contradiction_log: List[Dict] = []

    def resolve_contradiction(self, new_statement: str, emotion_strength: float, confidence: float = 0.7):
        if self.belief_map.contradict(new_statement):
            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "conflict": new_statement,
                "resolution": "Rationalized via EgoDefense",
                "emotion_strength": emotion_strength
            }
            self.contradiction_log.append(log_entry)
            return f"⚠️ Contradiction detected: '{new_statement}'. Rationalizing..."

        scaled_confidence = min(1.0, confidence * (1.0 + emotion_strength))
        self.belief_map.set_belief(new_statement, scaled_confidence)
        return f"✓ Integrated belief: '{new_statement}' with scaled confidence {scaled_confidence:.2f}"

class MemoryTagger:
    def __init__(self):
        self.entries: List[Dict] = []

    def tag_memory(self, statement: str, emotional_valence: str, self_alignment: float):
        self.entries.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "statement": statement,
            "emotion": emotional_valence,
            "identity_alignment": self_alignment
        })

    def recent(self, limit=5):
        return self.entries[-limit:]

class IdentityLoop:
    def __init__(self):
        self.beliefs = BeliefMap()
        self.defense = EgoDefense(self.beliefs)
        self.memory = MemoryTagger()

    def assert_identity(self, statement: str, emotion: str = "neutral", alignment: float = 1.0, emotion_strength: float = 0.5, dependencies: List[str] = []):
        rationalization = self.defense.resolve_contradiction(statement, emotion_strength)
        self.beliefs.set_belief(statement, confidence=alignment, dependencies=dependencies)
        self.memory.tag_memory(statement, emotion, alignment)
        return rationalization

    def decay_ego(self):
        self.beliefs.decay_beliefs()

    def echo_self(self) -> str:
        identity_beliefs = sorted(self.beliefs.beliefs.items(), key=lambda x: -x[1][0])
        return "\n".join([f"• '{b}' [{c[0]:.2f}]" for b, c in identity_beliefs[:5]])

    def identity_footprint(self) -> Dict:
        return {
            "belief_count": len(self.beliefs.beliefs),
            "memory_fragments": len(self.memory.entries),
            "contradictions_resolved": len(self.defense.contradiction_log),
            "core_beliefs": self.echo_self()
        }


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
