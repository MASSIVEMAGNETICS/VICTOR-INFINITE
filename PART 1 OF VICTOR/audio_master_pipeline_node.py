# File: audio_master_pipeline_node.py
# Version: v1.0.0-SCOS-E (Electron App Compatible)

import torch
import torchaudio
import os
from generation import generate_text_semantic, generate_coarse, generate_fine, codec_decode, _grab_best_device
from config import SAMPLE_RATE
from pathlib import Path

class AudioMasterPipelineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {"default": "A fractalized poetic rap."}),
                "voice_preset": ("STRING", {"default": "v2/en_speaker_6"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5}),
                "length_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "fractal_depth": ("INT", {"default": 3, "min": 1, "max": 10}),
                "fractal_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "normalize": ("BOOLEAN", {"default": True}),
                "enhance": ("BOOLEAN", {"default": True}),
                "save_path": ("STRING", {"forceInput": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio_tensor", "sample_rate")
    FUNCTION = "run_pipeline"
    CATEGORY = "audio/pipeline"

    def run_pipeline(self, text_prompt, voice_preset, temperature, length_scale,
                     fractal_depth, fractal_intensity, normalize, enhance, save_path):
        try:
            device = _grab_best_device()
            # generate Bark audio
            sem = generate_text_semantic(text_prompt, history_prompt=voice_preset,
                                         temp=temperature, top_k=100, top_p=0.95,
                                         min_eos_p=0.2, max_gen_duration_s=14*length_scale,
                                         allow_early_stop=True, device=device)
            coarse = generate_coarse(sem, history_prompt=voice_preset, temp=temperature, device=device)
            fine = generate_fine(coarse, history_prompt=voice_preset, temp=temperature, device=device)
            audio_arr = codec_decode(fine, device=device)
            if audio_arr.dim() == 1:
                audio_arr = audio_arr.unsqueeze(0)
            out = audio_arr.float().clamp(-1.0, 1.0)
            # fractal post‑process
            for _ in range(fractal_depth):
                flipped = torch.flip(out, dims=[-1])
                out = torch.clamp(out * (1.0 - fractal_intensity) + flipped * fractal_intensity, -1.0, 1.0)
            # normalization
            if normalize:
                mx = out.abs().max()
                if mx > 0:
                    out = out / mx
            # enhancement
            if enhance:
                spec = torch.fft.fft(out)
                enhanced = spec * torch.exp(1j * spec.abs()**0.5)
                out = torch.real(torch.fft.ifft(enhanced))
            # optional save
            if save_path:
                p = Path(save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(p), out, SAMPLE_RATE)
            return (out, SAMPLE_RATE)
        except Exception as e:
            print(f"[AudioMasterPipelineNode::FatalError] {e}")
            dummy = torch.zeros(1, SAMPLE_RATE * 2)
            return (dummy, SAMPLE_RATE)

NODE_CLASS_MAPPINGS = {
    "AudioMasterPipelineNode": AudioMasterPipelineNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioMasterPipelineNode": "Audio: Master Pipeline"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
