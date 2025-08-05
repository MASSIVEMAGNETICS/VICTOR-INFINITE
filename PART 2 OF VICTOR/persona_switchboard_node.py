# File: persona_switchboard_node.py
# Version: v1.0.0-FP
# Description: Handles Victor's multiple voice personas, assigns .npz/.pt embeddings to persona tags
# Author: Bando Bandz AI Ops

import os
import torch
import numpy as np

class PersonaSwitchboardNode:
    """
    Switches between multiple persona voice embeddings by tag.
    Outputs the corresponding embedding for TTS or memory-routing use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "persona_name": ("STRING", {"default": "bando"}),
                "persona_library_path": ("STRING", {"default": "custom_voices/"}),
                "file_format": ("STRING", {
                    "default": "npz",
                    "options": ["npz", "pt"]
                }),
                "fallback_to_default": ("BOOLEAN", {"default": True}),
                "default_persona": ("STRING", {"default": "default"})
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("voice_embedding",)
    FUNCTION = "load_persona"
    CATEGORY = "audio/identity_switch"

    def load_persona(self, persona_name, persona_library_path, file_format, fallback_to_default, default_persona):
        try:
            persona_path = os.path.join(persona_library_path, f"{persona_name}.{file_format}")
            if not os.path.exists(persona_path):
                if fallback_to_default:
                    print(f"[Victor::Switchboard] Persona '{persona_name}' not found. Fallback engaged.")
                    persona_path = os.path.join(persona_library_path, f"{default_persona}.{file_format}")
                else:
                    raise FileNotFoundError(f"[Victor::Switchboard] Persona '{persona_name}' not found.")

            embedding = self._load_embedding(persona_path, file_format)
            if embedding is None:
                raise ValueError(f"[Victor::Switchboard] Failed to load persona embedding: {persona_name}")

            print(f"[Victor::Switchboard] Persona '{persona_name}' loaded.")
            return (embedding,)

        except Exception as e:
            print(f"[Victor::Switchboard::Error] {str(e)}")
            return (torch.zeros(1),)

    def _load_embedding(self, path, file_format):
        try:
            if file_format == "pt":
                return torch.load(path)
            elif file_format == "npz":
                data = np.load(path)
                return torch.tensor(data) if not isinstance(data, torch.Tensor) else data
            else:
                raise ValueError(f"[Victor::Switchboard::LoadError] Unsupported format: {file_format}")
        except Exception as e:
            print(f"[Victor::Switchboard::LoadError] {str(e)}")
            return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "PersonaSwitchboardNode": PersonaSwitchboardNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonaSwitchboardNode": "Audio: Persona Switchboard"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
