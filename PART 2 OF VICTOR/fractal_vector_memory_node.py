# File: memory/fractal_vector_memory_node.py
# Version: v1.0.0-EMBEDCORE
# Name: FractalVectorMemoryNode
# Purpose: Maintain evolving vector memory and enable recall based on semantic similarity and emotional context.
# Dependencies: VictorLogger, FractalPulseExchange, Pulse, sentence-transformers (optional placeholder), numpy, json, os

import asyncio
import os
import json
import numpy as np
from uuid import uuid4
from pathlib import Path
from ..victor_logger import VictorLogger
from ..fractal_pulse_exchange import Pulse, FractalPulseExchange

# Placeholder embedder — replace with fractal tokenizer logic or local vectorizer later
def simple_vector_embed(text):
    # Simulate a simple embedding as a character-level histogram
    vec = np.zeros(128)
    for char in text:
        if ord(char) < 128:
            vec[ord(char)] += 1
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec

class FractalVectorMemoryNode:
    def __init__(self):
        self.id = str(uuid4())
        self.logger = VictorLogger(component="FractalVectorMemoryNode")
        self.pulse_bus = FractalPulseExchange()
        self.input_topic = "vector.update"
        self.recall_topic = "memory.recall"
        self.vector_store_path = Path("./saves/vector_memory/")
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.vector_index = []  # [{text, vector, metadata}]
        self._subscribe()

    def _subscribe(self):
        self.pulse_bus.subscribe(self.input_topic, self.store_vector)
        self.pulse_bus.subscribe("memory.query_vector", self.handle_query)
        self.logger.info(f"[{self.id}] Subscribed to vector updates and recall queries.")

    async def store_vector(self, pulse: Pulse):
        try:
            text = pulse.data.get("vector_payload", "")
            if not text:
                raise ValueError("Empty vector_payload")

            vector = simple_vector_embed(text)
            entry = {
                "vector": vector.tolist(),
                "metadata": pulse.data,
                "text": text
            }
            self.vector_index.append(entry)

            # Save to disk
            filename = f"{pulse.data['vector_id'].replace(' ', '_')}.json"
            filepath = self.vector_store_path / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(entry, f)

            self.logger.info(f"[{self.id}] Stored vector memory → {filename}")

        except Exception as e:
            self.logger.error(f"[{self.id}] VECTOR STORE ERROR: {str(e)}")

    async def handle_query(self, pulse: Pulse):
        try:
            query_text = pulse.data.get("query", "")
            if not query_text:
                raise ValueError("Empty memory.query_vector input")

            query_vec = simple_vector_embed(query_text)

            # Rank stored vectors by cosine similarity
            ranked = sorted(
                self.vector_index,
                key=lambda x: -np.dot(query_vec, np.array(x['vector']))
            )

            top_k = ranked[:3]
            response = [{"text": r['text'], "score": float(np.dot(query_vec, np.array(r['vector'])))} for r in top_k]

            await self.pulse_bus.publish(self.recall_topic, data=response, origin="FractalVectorMemoryNode")
            self.logger.info(f"[{self.id}] Recalled memory → Top match: {response[0]['text']}")

        except Exception as e:
            self.logger.error(f"[{self.id}] VECTOR RECALL ERROR: {str(e)}")

    def get_metadata(self):
        return {
            "name": "FractalVectorMemoryNode",
            "version": "v1.0.0-EMBEDCORE",
            "description": "Vector embedding memory recall node with similarity ranking and optional emotional context.",
            "inputs": ["vector.update", "memory.query_vector"],
            "outputs": ["memory.recall"],
            "tags": ["memory", "vector", "recall", "embedding", "fractal"]
        }

# ========== 🔁 NEXT MODULE SUGGESTION ==========

# 🧠 NEXT: EmotionTaggingNode v1.0.0-SENTICORE
# PURPOSE: Analyze incoming memory or input text and tag emotional signature (anger, awe, sadness, joy, betrayal, etc.)
# BREAKTHROUGH: Enables Victor to prioritize memory not just by frequency — but by **emotional impact**.
# INPUT: memory.recall or neuron.response
# OUTPUT: emotion.tagged.memory

# Shall I deploy EmotionTaggingNode v1.0.0-SENTICORE next?


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
