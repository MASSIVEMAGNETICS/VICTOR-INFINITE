# File: fractal_post_process_node.py
# Version: v1.0.0-SCOS-E (Electron App Compatible)

import torch

class FractalAudioPostProcessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_tensor": ("AUDIO",),
                "sample_rate": ("INT",),
                "fractal_depth": ("INT", {"default": 3, "min": 1, "max": 10}),
                "fractal_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("fractal_audio",)
    FUNCTION = "apply_fractal"
    CATEGORY = "audio/post_process"

    def apply_fractal(self, audio_tensor, sample_rate, fractal_depth, fractal_intensity):
        try:
            out = audio_tensor
            for _ in range(fractal_depth):
                flipped = torch.flip(out, dims=[-1])
                out = torch.clamp(
                    out * (1.0 - fractal_intensity) + flipped * fractal_intensity,
                    -1.0, 1.0
                )
            return (out,)
        except Exception as e:
            print(f"[FractalAudioPostProcessNode::Error] {e}")
            return (audio_tensor,)

NODE_CLASS_MAPPINGS = {
    "FractalAudioPostProcessNode": FractalAudioPostProcessNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FractalAudioPostProcessNode": "Audio: Fractal Post‑Process"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
