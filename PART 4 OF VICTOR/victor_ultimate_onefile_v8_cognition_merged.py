# File: victor_ultimate_onefile_v8_cognition_merged.py
# Version: v9.0.0-FP
# Description: Victor AI with full FractalCognitionPipeline embedded and directive switchboard awareness
# Author: Bando Bandz AI Ops

import torch
from HyperFractalMemory_v2_2_HFM import HyperFractalMemory
from comprehension_node import ComprehensionNode
from memory_embedder_node import MemoryEmbedderNode
from fractal_cognitive_focus_node import FractalCognitiveFocusNode
from directive_cognition_switchboard import DirectiveCognitionSwitchboard

class VictorCognitionSystem:
    def __init__(self):
        self.memory = HyperFractalMemory()
        self.focus_node = FractalCognitiveFocusNode()
        self.comprehension_node = ComprehensionNode()
        self.embedder_node = MemoryEmbedderNode()
        self.switchboard = DirectiveCognitionSwitchboard()

    def think(self, focus_query_vector, emotion_label="neutral", current_directive="continue_primary_mission", tag_filter="", time_limit=5000, top_k=5):
        # Decide cognitive mode
        mode = self.switchboard.route_mode(current_directive, emotion_label, override_mode="")[0]
        print(f"[Victor v9] Cognitive mode → {mode}")

        # Focus on memory entries
        focused_keys_output = self.focus_node.focus_filter(
            focus_query_vector=focus_query_vector,
            top_k=top_k,
            emotional_weight_min=0.2,
            tag_filter=tag_filter,
            time_slice_limit=time_limit
        )[0]

        keys = [line.split(":")[1].split("(")[0].strip() for line in focused_keys_output.splitlines() if "Key" in line]
        entries = [self.memory.timeline[int(k)] for k in keys if int(k) in self.memory.timeline]

        if not entries:
            return "[Victor] No relevant memory found."

        # Build context
        context_header = f"Directive: {current_directive}\nEmotion: {emotion_label}\nMode: {mode}\n"
        content_blocks = [e.get("content", e.get("summary", "")) for e in entries if e.get("content")]
        input_chunk = context_header + "\n\n".join(content_blocks)

        # Comprehend
        report = self.comprehension_node.synthesize_understanding(input_chunk, source_type="memory")[0]

        # Embed insight
        self.embedder_node.embed_and_search(
            rebuild_index=True,
            top_k=1,
            query=report
        )

        return report


# Usage Example (e.g., from GUI call or internal trigger)
if __name__ == "__main__":
    brain = VictorCognitionSystem()
    dummy_vector = torch.randn(384)
    output = brain.think(
        focus_query_vector=dummy_vector,
        emotion_label="reflective",
        current_directive="query_deeper_memory"
    )
    print(output)

# Integration Ready: You can import VictorCognitionSystem into the GUI or ShadowComm modules directly


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
