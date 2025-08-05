# FILE: memory_resonance_network.py
# VERSION: v1.0.0-MRN-GODCORE
# NAME: MemoryResonanceNetwork
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Store, score, retrieve, and decay fractal memory with emotional and conceptual trace
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

import datetime
import math

class MemoryResonanceNetwork:
    def __init__(self):
        self.memory_store = []

    def _decay_score(self, timestamp):
        delta = datetime.datetime.utcnow() - timestamp
        hours = delta.total_seconds() / 3600
        return math.exp(-0.1 * hours)

    def store(self, directive):
        entry = {
            "timestamp": datetime.datetime.utcnow(),
            "concepts": directive.get("target_concepts", []),
            "emotion": directive.get("emotion", "neutral"),
            "echo_id": directive.get("echo_id", "none"),
            "action": directive.get("action", "observe"),
            "reason": directive.get("reason", "unknown"),
            "score": 1.0  # will decay over time
        }
        self.memory_store.append(entry)
        return entry

    def retrieve(self, concept_query, top_k=3):
        relevance = []
        now = datetime.datetime.utcnow()

        for mem in self.memory_store:
            score = 0
            shared = set(concept_query).intersection(set(mem["concepts"]))
            score += len(shared) * 1.0
            if mem["emotion"] != "neutral":
                score += 0.5
            score *= self._decay_score(mem["timestamp"])
            relevance.append((score, mem))

        relevance.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in relevance[:top_k]]

    def wipe_low_score(self, threshold=0.05):
        self.memory_store = [mem for mem in self.memory_store if self._decay_score(mem["timestamp"]) > threshold]
        return len(self.memory_store)

    def dump_memory(self):
        return self.memory_store
