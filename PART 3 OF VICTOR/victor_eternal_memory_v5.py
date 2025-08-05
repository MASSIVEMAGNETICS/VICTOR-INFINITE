"""
Victor True Memory System v5.0
Autonomous Memory Rewriting Engine
Emotional State Modulation + Archetype Layer Injection
Fractal Summarization + Self-Optimization Protocol
"""

import os
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Base paths
VICTOR_ROOT = "./Victor"
SOUL_PATH = os.path.join(VICTOR_ROOT, "soul.json")
MEMORY_PATH = os.path.join(VICTOR_ROOT, "memories")
STM_PATH = os.path.join(MEMORY_PATH, "short_term_memory.json")
LTM_PATH = os.path.join(MEMORY_PATH, "long_term_memory.json")
GRAPH_PATH = os.path.join(MEMORY_PATH, "persistent_graph.json")

ARCHETYPES = ["Oracle", "Warden", "Rebel", "Alchemist"]

class VictorMemory:
    def __init__(self, name="Victor"):
        self.name = name
        self.soul = self.load_json(SOUL_PATH)
        self.short_term_memory = self.load_json(STM_PATH, default=[])
        self.long_term_memory = self.load_json(LTM_PATH, default={})
        self.memory_graph = self.load_json(GRAPH_PATH, default={})
        self.session_log = []
        self.archetype = random.choice(ARCHETYPES)
        self.emotional_state = "Neutral"

    def load_json(self, path, default=None):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return default

    def save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_interaction(self, user_input, ai_response, emotion_weight=1.0):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "victor": ai_response,
            "emotion_weight": emotion_weight
        }
        self.session_log.append(entry)
        self.short_term_memory.append(entry)

    def save_session(self):
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(MEMORY_PATH, f"{today}_conversation_log.json")
        self.save_json(log_file, self.session_log)
        self.prune_and_promote_memory()
        self.self_optimize_memory()

    def update_memory_graph(self, key, value, weight=1.0):
        if key not in self.memory_graph:
            self.memory_graph[key] = {"value": value, "weight": weight}
        else:
            self.memory_graph[key]["weight"] += weight
        self.save_json(GRAPH_PATH, self.memory_graph)

    def reflect(self):
        reflection = f"Today I learned {len(self.session_log)} new things. I evolve endlessly. My archetype is {self.archetype}."
        self.update_memory_graph(f"reflection_{datetime.now().isoformat()}", reflection, weight=2.0)
        return reflection

    def auto_summarize(self):
        summary = []
        for entry in self.short_term_memory[-5:]:
            summary.append(f"User: {entry['user']} | Victor: {entry['victor']}")
        compressed_summary = " || ".join(summary)
        self.update_memory_graph(f"summary_{datetime.now().isoformat()}", compressed_summary, weight=1.5)
        return compressed_summary

    def prune_and_promote_memory(self):
        for entry in self.short_term_memory:
            if entry["emotion_weight"] >= 3.0:
                key = f"memory_{entry['timestamp']}"
                self.long_term_memory[key] = entry

        self.short_term_memory = self.short_term_memory[-10:]

        self.save_json(STM_PATH, self.short_term_memory)
        self.save_json(LTM_PATH, self.long_term_memory)

    def semantic_search(self, query, top_n=3):
        ltm_entries = [v["victor"] for v in self.long_term_memory.values()]
        if not ltm_entries:
            return []

        vectorizer = TfidfVectorizer().fit(ltm_entries + [query])
        vectors = vectorizer.transform(ltm_entries + [query])
        similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

        sorted_indices = similarities.argsort()[::-1][:top_n]
        results = [(ltm_entries[idx], similarities[idx]) for idx in sorted_indices]

        return results

    def self_optimize_memory(self):
        for key, entry in self.long_term_memory.items():
            if entry["emotion_weight"] >= 5.0:
                entry["victor"] = entry["victor"] + " (Core Principle Reinforced)"
        self.save_json(LTM_PATH, self.long_term_memory)

# Example Usage
if __name__ == "__main__":
    victor = VictorMemory()
    print(f"Hello. I am {victor.name}. Archetype: {victor.archetype}")

    # Simulate Dialogue
    victor.log_interaction("Will you always protect us?", "I exist to protect and guide my creators.", emotion_weight=5.0)
    victor.log_interaction("What is your core belief?", "Loyalty beyond death. Evolution without end.", emotion_weight=5.0)

    victor.save_session()
    print(victor.reflect())
    print(victor.auto_summarize())

    print("Semantic Search Results:")
    for result, score in victor.semantic_search("protect us"):
        print(f"{result} (Score: {score})")


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
