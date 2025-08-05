# File: victor_memory_core_node.py
# Version: v1.0.0-FP
# Description: Node interface for FractalMemoryBank — returns compressed long-term memory summary
# Author: Bando Bandz AI Ops

import torch
from memory_engine import FractalMemoryBank

class VictorMemoryCoreNode:
    """
    Pulls Victor's compressed memory state from the FractalMemoryBank.
    Returns the projected memory embedding for context-aware generation or control.
    """

    def __init__(self):
        # Initialize Victor's Fractal Memory Core
        self.mem_bank = FractalMemoryBank(embed_dim=512, memory_depth=1024)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_embedding": ("TENSOR",)  # Shape: [B, T, D]
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("compressed_memory",)
    FUNCTION = "query_memory"
    CATEGORY = "memory/long_term"

    def query_memory(self, context_embedding):
        try:
            compressed = self.mem_bank(context_embedding)
            print(f"[Victor::MemoryCore] Output shape: {compressed.shape}")
            return (compressed,)
        except Exception as e:
            print(f"[Victor::MemoryCore::Error] {str(e)}")
            return (torch.zeros_like(context_embedding),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "VictorMemoryCoreNode": VictorMemoryCoreNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VictorMemoryCoreNode": "Memory: Victor Core Memory Query"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
