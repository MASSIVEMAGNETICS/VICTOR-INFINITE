# File: memory_visualize_node.py
# Version: v2.0.0-FP
# Description: Animated and annotated memory visualization with emotion + directive overlays
# Author: Bando Bandz AI Ops

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
from matplotlib import animation

class MemoryVisualizeNode:
    """
    Visualizes memory evolution as an animated heatmap (.gif), with optional overlays
    for emotion labels and directive spikes per frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_tensor_sequence": ("TENSOR",),  # shape: [T, D, E] over time
                "output_dir": ("STRING", {"default": "memory_visuals/"}),
                "colormap": ("STRING", {
                    "default": "plasma",
                    "options": ["viridis", "plasma", "inferno", "magma", "cividis"]
                }),
                "emotion_track": ("STRING", {"default": ""}),  # comma-separated per timestep (optional)
                "directive_spikes": ("STRING", {"default": ""})  # comma-separated "0"/"1" per timestep
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gif_path",)
    FUNCTION = "render_memory_animation"
    CATEGORY = "visual/memory"

    def render_memory_animation(self, memory_tensor_sequence, output_dir, colormap, emotion_track, directive_spikes):
        try:
            if memory_tensor_sequence.dim() != 3:
                raise ValueError("Memory sequence must be 3D: [T, D, E]")

            T, D, E = memory_tensor_sequence.shape
            os.makedirs(output_dir, exist_ok=True)
            image_id = str(uuid.uuid4())[:8]
            file_path = os.path.join(output_dir, f"memory_evolution_{image_id}.gif")

            emotions = emotion_track.split(",") if emotion_track else [""] * T
            spikes = [int(x.strip()) for x in directive_spikes.split(",")] if directive_spikes else [0] * T

            fig, ax = plt.subplots(figsize=(10, 4))

            def update(frame_idx):
                ax.clear()
                memory_np = memory_tensor_sequence[frame_idx].detach().cpu().numpy()
                im = ax.imshow(memory_np, aspect='auto', cmap=colormap)
                ax.set_title(f"Memory Frame {frame_idx+1}/{T} | Emotion: {emotions[frame_idx]}{' ⚡' if spikes[frame_idx] else ''}")
                ax.set_xlabel("Embedding Dimension")
                ax.set_ylabel("Memory Depth")
                return [im]

            anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
            anim.save(file_path, writer='pillow', fps=2)
            plt.close()

            print(f"[Victor::MemoryVisual 2.0] GIF saved: {file_path}")
            return (file_path,)

        except Exception as e:
            print(f"[Victor::MemoryVisual 2.0::Error] {str(e)}")
            return ("",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "MemoryVisualizeNode": MemoryVisualizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryVisualizeNode": "Visual: Memory Tensor Heatmap (Animated)"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
