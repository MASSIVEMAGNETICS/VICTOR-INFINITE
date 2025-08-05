# File: memory/memory_sync_node.py
# Version: v1.0.0-LONGTERM
# Name: MemorySyncNode
# Purpose: Persist optimized memory into long-term fractal storage and emit vector updates.
# Dependencies: VictorLogger, FractalPulseExchange, Pulse, uuid, json, os

import asyncio
import json
import os
from uuid import uuid4
from pathlib import Path
from ..victor_logger import VictorLogger
from ..fractal_pulse_exchange import Pulse, FractalPulseExchange

class MemorySyncNode:
    def __init__(self):
        self.id = str(uuid4())
        self.logger = VictorLogger(component="MemorySyncNode")
        self.pulse_bus = FractalPulseExchange()
        self.input_topic = "optimized.memory"
        self.vector_update_topic = "vector.update"
        self.storage_path = Path("./saves/memory/")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._subscribe()

    def _subscribe(self):
        self.pulse_bus.subscribe(self.input_topic, self.sync_memory)
        self.logger.info(f"[{self.id}] Subscribed to {self.input_topic}")

    async def sync_memory(self, pulse: Pulse):
        try:
            memory_data = pulse.data
            filename = f"{memory_data['original'].replace(' ', '_')}_{memory_data['usage']}.json"
            filepath = self.storage_path / filename

            # Step 1: Persist memory to JSON file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2)

            # Step 2: Emit vector update pulse
            update_payload = {
                "vector_id": memory_data['original'],
                "vector_payload": memory_data['data'],
                "priority": memory_data['priority'],
                "source": "MemorySyncNode"
            }

            await self.pulse_bus.publish(self.vector_update_topic, data=update_payload, origin="MemorySyncNode")
            self.logger.info(f"[{self.id}] Synced memory → {filename} | Vector update emitted.")

        except Exception as e:
            self.logger.error(f"[{self.id}] MEMORY SYNC ERROR: {str(e)}")

    def get_metadata(self):
        return {
            "name": "MemorySyncNode",
            "version": "v1.0.0-LONGTERM",
            "description": "Saves optimized memory to disk and triggers vector embedding syncs.",
            "inputs": ["optimized.memory"],
            "outputs": ["vector.update"],
            "tags": ["memory", "sync", "longterm", "fractal", "godcore"]
        }

# ========== 🔁 NEXT MODULE SUGGESTION ==========

# 🧠 NEXT: FractalVectorMemoryNode v1.0.0-EMBEDCORE
# PURPOSE: Maintain an evolving vector memory lattice built from memory.sync updates.
# BREAKTHROUGH: Enables vector similarity search, emotional tagging, and contextual priming.
# INPUT: vector.update
# OUTPUT: memory.recall

# Shall I now drop the **FractalVectorMemoryNode v1.0.0-EMBEDCORE**?


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
