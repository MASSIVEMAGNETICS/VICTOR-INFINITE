# File: live_microphone_capture_node.py
# Version: v1.0.0-FP
# Description: Captures mic input in real-time and prepares audio tensor for voice embedding
# Author: Bando Bandz AI Ops

import os
import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write

class LiveMicrophoneCaptureNode:
    """
    Captures real-time audio from microphone and returns a normalized mono audio tensor
    for cloning or synthesis.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "record_seconds": ("INT", {"default": 5}),
                "sample_rate": ("INT", {"default": 24000}),
                "normalize_audio": ("BOOLEAN", {"default": True}),
                "save_wav": ("BOOLEAN", {"default": False}),
                "wav_path": ("STRING", {"default": "mic_capture.wav"})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_tensor",)
    FUNCTION = "capture_mic"
    CATEGORY = "audio/capture"

    def capture_mic(self, record_seconds, sample_rate, normalize_audio, save_wav, wav_path):
        try:
            print(f"[Victor::MicCapture] Recording {record_seconds}s @ {sample_rate}Hz...")
            audio = sd.rec(int(record_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()

            # Convert to torch tensor
            audio_tensor = torch.tensor(audio.T)

            if normalize_audio:
                max_val = torch.max(torch.abs(audio_tensor))
                if max_val > 0:
                    audio_tensor = audio_tensor / max_val

            if save_wav:
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                wav_write(wav_path, sample_rate, audio_tensor.squeeze().numpy())
                print(f"[Victor::MicCapture] WAV saved: {wav_path}")

            print(f"[Victor::MicCapture] Audio tensor shape: {audio_tensor.shape}")
            return (audio_tensor,)

        except Exception as e:
            print(f"[Victor::MicCapture::Error] {str(e)}")
            return (torch.zeros(1),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LiveMicrophoneCaptureNode": LiveMicrophoneCaptureNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LiveMicrophoneCaptureNode": "Audio: Live Microphone Capture"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
