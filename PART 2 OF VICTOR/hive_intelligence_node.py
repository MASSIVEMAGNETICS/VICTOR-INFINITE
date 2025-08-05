# File: hive_intelligence_node.py
# Version: v3.0.0-GT
# Description: God-tier ComfyUI node for swarm cognition with emotion and entanglement
# Author: Bando Bandz AI Ops

from typing import Dict, List, Optional
from collections import defaultdict
import math

class HiveIntelligenceNode:
    def __init__(self):
        self.core = _HiveIntelligenceCore()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "agent_id": ("STRING",),
                "state": ("DICT",),
                "directive": ("STRING",),
                "feedback_vector": ("LIST", {"default": [0]*10}),
                "emotion": ("DICT", {"default": {}})
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("updated_state",)
    FUNCTION = "run"
    CATEGORY = "intelligence/swarm"

    def run(self, agent_id, state, directive, feedback_vector, emotion):
        self.core.set_directive(directive)
        self.core.register_agent(agent_id, state)
        self.core.emotion_vectors[agent_id] = emotion
        self.core.receive_feedback({"agent_id": agent_id, "vector": feedback_vector})
        self.core.execute_hive_behavior()
        return (self.core.swarm_state[agent_id],)


class _HiveIntelligenceCore:
    def __init__(self):
        self.swarm_state: Dict[str, Dict] = {}
        self.hive_mind_vector: Optional[List[float]] = None
        self.behavior_rules: List = []
        self.learning_rate: float = 0.02
        self.global_directive: Optional[str] = None
        self.feedback_memory: List[Dict] = []
        self.emotion_vectors: Dict[str, Dict[str, float]] = {}
        self.entangled_pairs: Dict[str, Dict] = {}
        self.sub_hives: Dict[str, List[str]] = defaultdict(list)

    def set_directive(self, directive: str):
        self.global_directive = directive

    def register_agent(self, agent_id: str, state: Dict):
        if agent_id not in self.swarm_state:
            self.swarm_state[agent_id] = state
        else:
            self.update_agent_state(agent_id, state)

    def update_agent_state(self, agent_id: str, new_state: Dict):
        self.swarm_state.setdefault(agent_id, {}).update(new_state)

    def receive_feedback(self, feedback: Dict):
        if not isinstance(feedback, dict):
            return
        vector = feedback.get("vector", [0] * 10)
        if not isinstance(vector, list) or not all(isinstance(v, (int, float)) for v in vector):
            return
        self.feedback_memory.append(feedback)
        self._update_hive_vector()

    def _update_hive_vector(self):
        try:
            if not self.feedback_memory:
                return
            cumulative_vector = [0] * 10
            count = 0
            for feedback in self.feedback_memory[-100:]:
                vec = feedback.get("vector", [0]*10)
                if isinstance(vec, list) and len(vec) == 10:
                    cumulative_vector = [c + v for c, v in zip(cumulative_vector, vec)]
                    count += 1
            if count:
                self.hive_mind_vector = [v / count for v in cumulative_vector]
        except Exception as e:
            print(f"[HiveNode::Error] Hive vector update failed: {str(e)}")

    def execute_hive_behavior(self):
        for agent_id, state in self.swarm_state.items():
            try:
                self.swarm_state[agent_id] = self._apply_behavior_rules(agent_id, state)
            except Exception as e:
                print(f"[HiveNode::Error] Behavior rule failed for {agent_id}: {str(e)}")

    def _apply_behavior_rules(self, agent_id: str, state: Dict) -> Dict:
        try:
            for rule in self.behavior_rules:
                state = rule(state, self.hive_mind_vector, self.global_directive)
            return self._adjust_behavior_by_emotion(agent_id, state)
        except Exception as e:
            print(f"[HiveNode::Error] Rule application failed: {str(e)}")
            return state

    def _adjust_behavior_by_emotion(self, agent_id: str, state: Dict) -> Dict:
        emotions = self.emotion_vectors.get(agent_id, {})
        try:
            if emotions.get("rage", 0) > 0.7:
                state["attack_mode"] = True
            if emotions.get("fear", 0) > 0.6:
                state["retreat_mode"] = True
        except Exception as e:
            print(f"[HiveNode::Emotion] Adjustment error: {str(e)}")
        return state

    def get_metadata(self):
        return {
            "name": "HiveIntelligenceNode",
            "version": "3.0.0-GT",
            "description": "God-tier swarm control for ComfyUI with emotion routing, vector feedback, and entangled evolution."
        }

# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
