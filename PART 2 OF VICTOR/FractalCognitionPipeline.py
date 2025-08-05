# File: FractalCognitionPipeline.py
# Version: v2.0.0-FP
# Description: Upgraded cognition loop with DirectiveCognitionSwitchboard integration and memory injection
# Author: Bando Bandz AI Ops

import torch
from HyperFractalMemory_v2_2_HFM import HyperFractalMemory
from comprehension_node import ComprehensionNode
from memory_embedder_node import MemoryEmbedderNode
from fractal_cognitive_focus_node import FractalCognitiveFocusNode
from directive_cognition_switchboard import DirectiveCognitionSwitchboard

class FractalCognitionPipeline:
    def __init__(self):
        self.memory = HyperFractalMemory()
        self.focus_node = FractalCognitiveFocusNode()
        self.comprehension_node = ComprehensionNode()
        self.embedder_node = MemoryEmbedderNode()
        self.switchboard = DirectiveCognitionSwitchboard()

    def run(self, focus_query_vector, emotion_label="neutral", current_directive="continue_primary_mission", tag_filter="", time_limit=5000, top_k=5):
        # Step 1: Decide cognitive mode
        cognitive_mode = self.switchboard.route_mode(current_directive, emotion_label, override_mode="")[0]
        print(f"[Pipeline] Cognitive mode selected: {cognitive_mode}")

        # Step 2: Focus on relevant memory entries
        print("[Pipeline] Focusing on relevant memory...")
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
            print("[Pipeline] No memory entries matched.")
            return "[Victor] No relevant memory found."

        # Step 3: Prepare input for comprehension
        context_header = f"Directive: {current_directive}\nEmotion: {emotion_label}\nMode: {cognitive_mode}\n"
        content_blocks = [e.get("content", e.get("summary", "")) for e in entries if e.get("content")]
        input_chunk = context_header + "\n\n".join(content_blocks)

        # Step 4: Comprehend
        print("[Pipeline] Synthesizing understanding...")
        report = self.comprehension_node.synthesize_understanding(input_chunk, source_type="memory")[0]

        # Step 5: Re-embed and optionally inject into memory
        print("[Pipeline] Embedding insight into memory...")
        self.embedder_node.embed_and_search(
            rebuild_index=True,
            top_k=1,
            query=report
        )

        print("[Pipeline] Cognition loop complete.")
        return report


# Example usage:
if __name__ == "__main__":
    pipeline = FractalCognitionPipeline()
    dummy_query = torch.randn(384)  # Replace with live focus vector
    print(pipeline.run(focus_query_vector=dummy_query, emotion_label="reflective", current_directive="query_deeper_memory"))


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
