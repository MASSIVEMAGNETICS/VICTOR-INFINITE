# victor_emotional_state_engine.py :: v0.1.0-SCOS
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

class EmotionalStateEngine:
    def __init__(self):
        self.valence = 0.0  # -1 (negative) to +1 (positive)
        self.arousal = 0.0  # 0 (calm) to 1 (excited)
        self.last_update = datetime.utcnow()
        self.mood_memory = []  # Stores valence-arousal history

    def update_emotion(self, input_valence: float, input_arousal: float, memory_influence: float = 0.2):
        current_time = datetime.utcnow()
        delta_time = (current_time - self.last_update).total_seconds() / 60.0

        # Decay and smoothing
        self.valence = self._decay(self.valence, delta_time) + memory_influence * input_valence
        self.arousal = self._decay(self.arousal, delta_time) + memory_influence * input_arousal

        # Clamping values
        self.valence = max(min(self.valence, 1.0), -1.0)
        self.arousal = max(min(self.arousal, 1.0), 0.0)

        self.mood_memory.append((current_time, self.valence, self.arousal))
        self.last_update = current_time

    def get_emotion_state(self) -> Tuple[float, float]:
        return self.valence, self.arousal

    def _decay(self, value: float, delta_time: float, decay_rate: float = 0.05) -> float:
        return value * np.exp(-decay_rate * delta_time)

    def modulate_output_style(self) -> str:
        if self.valence > 0.5 and self.arousal > 0.5:
            return "excited"
        elif self.valence > 0.5:
            return "warm"
        elif self.valence < -0.5:
            return "cold"
        elif self.arousal > 0.5:
            return "intense"
        else:
            return "neutral"



# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
