# Filename: InputNode.py
# Version: v1.0.0-SCOS-ComfyBridge
# Description: Multi-modal input node for ComfyUI - accepts prompt, image, audio transcription, signal
# SCOS-Eternal Integration Layer

import numpy as np
from PIL import Image
import torch
import folder_paths
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

class InputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["prompt", "audio_transcription", "visual", "signal"],),
                "text_input": ("STRING", {"default": ""}),
                "image_input": ("IMAGE",),
                "signal_input": ("JSON", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "JSON",)
    RETURN_NAMES = ("processed_text", "processed_image", "processed_signal",)
    FUNCTION = "process"
    CATEGORY = "SCOS/Bridge"

    def process(self, input_type, text_input, image_input, signal_input):
        processed_text = ""
        processed_image = None
        processed_signal = {}

        if input_type == "prompt" or input_type == "audio_transcription":
            processed_text = text_input.strip()

        elif input_type == "visual":
            image_tensor = image_input
            image_pil = Image.fromarray((image_tensor[0].cpu().numpy() * 255).astype(np.uint8))
            processed_image = image_pil

        elif input_type == "signal":
            processed_signal = signal_input

        return (processed_text, processed_image, processed_signal)


NODE_CLASS_MAPPINGS.update({
    "InputNodeSCOS": InputNode
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "InputNodeSCOS": "🧠 SCOS Input Node"
})


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
