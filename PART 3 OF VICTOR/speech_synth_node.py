# File: speech_synth_node.py
# Version: v1.1.0-FP
# Description: Future-proofed speech synthesis using voice embeddings from .npz/.pt files
# Author: Bando Bandz AI Ops

import os
import torch
import numpy as np
from bark_tts import bark_synthesize  # Replace with your custom backend or Bark model

class SpeechSynthNode:
    """
    Synthesizes speech using Victor's cloned voice profiles.
    Loads a .npz/.pt voice embedding and feeds it into a Bark-based TTS engine.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {"multiline": True}),
                "voice_embedding_path": ("STRING", {"default": "custom_voices/my_voice.npz"}),
                "sample_rate": ("INT", {"default": 24000}),
                "fallback_to_default": ("BOOLEAN", {"default": True}),
                "default_embedding_path": ("STRING", {"default": "default_voice/default.npz"})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_output",)
    FUNCTION = "synthesize"
    CATEGORY = "audio/synthesis"

    def synthesize(self, text_prompt, voice_embedding_path, sample_rate, fallback_to_default, default_embedding_path):
        try:
            embedding = None

            # Validate path and load voice embedding
            if os.path.exists(voice_embedding_path):
                embedding = self._load_embedding(voice_embedding_path)
            elif fallback_to_default and os.path.exists(default_embedding_path):
                print(f"[Victor::SpeechSynth] Voice profile not found. Falling back to default: {default_embedding_path}")
                embedding = self._load_embedding(default_embedding_path)
            else:
                raise FileNotFoundError(f"[SpeechSynth::Error] Voice profile missing: {voice_embedding_path}")

            if embedding is None:
                raise ValueError("[SpeechSynth::Error] Failed to load voice embedding (empty or invalid format)")

            # Run the TTS engine
            audio_output = bark_synthesize(text_prompt, embedding, sample_rate)
            print(f"[SpeechSynth] Generated voice for: \"{text_prompt[:40]}...\"")
            return (audio_output,)

        except Exception as e:
            print(f"[Victor::SpeechSynth::FatalError] {str(e)}")
            return (torch.zeros(1),)

    def _load_embedding(self, path):
        try:
            if path.endswith(".pt"):
                return torch.load(path)
            elif path.endswith(".npz"):
                loaded = np.load(path)
                return torch.tensor(loaded) if not isinstance(loaded, torch.Tensor) else loaded
            else:
                raise ValueError(f"[SpeechSynth::LoadError] Unsupported file type: {path}")
        except Exception as e:
            print(f"[SpeechSynth::LoadError] Could not load embedding: {str(e)}")
            return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "SpeechSynthNode": SpeechSynthNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeechSynthNode": "Audio: Speech Synthesizer (Victor)"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
