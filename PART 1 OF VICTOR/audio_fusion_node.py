# File: audio_fusion_node.py
# Version: v1.0.2-SCOS (Electron App Compatible)

class AudioFusionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_tensor_1": ("AUDIO",),
                "audio_tensor_2": ("AUDIO",),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("blended_audio",)
    FUNCTION = "blend_audio"
    CATEGORY = "audio/blending"

    def blend_audio(self, audio_tensor_1, audio_tensor_2, blend_ratio):
        try:
            min_len = min(audio_tensor_1.shape[-1], audio_tensor_2.shape[-1])
            audio_tensor_1 = audio_tensor_1[..., :min_len]
            audio_tensor_2 = audio_tensor_2[..., :min_len]

            blended = (blend_ratio * audio_tensor_1) + ((1.0 - blend_ratio) * audio_tensor_2)

            return (blended,)
        
        except Exception as e:
            print(f"[AudioFusionNode::Error] {str(e)}")
            return (audio_tensor_1 * 0,)

NODE_CLASS_MAPPINGS = {
    "AudioFusionNode": AudioFusionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFusionNode": "Audio: Fusion Node"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
