# File: stt_fractal_parser_node.py
# Version: v1.0.0-SCOS-E (Electron App Compatible)

import torch
import torchaudio
import tempfile
import os
import whisper
import json
import parser_core_logic_v3 as pcl  # make sure parser_core_logic_v3.py is on PYTHONPATH

class STTFractalParserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_tensor": ("AUDIO",),
                "sample_rate": ("INT",),
                "model_size": ("STRING", {"default": "base"}),
                "emotion_context": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transcript", "graph_json")
    FUNCTION = "process"
    CATEGORY = "audio/parser"

    def process(self, audio_tensor, sample_rate, model_size, emotion_context):
        try:
            # ensure shape (1, N)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # save to temp WAV
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, audio_tensor, sample_rate)
            tmp.close()

            # transcription with Whisper
            model = whisper.load_model(model_size)
            result = model.transcribe(tmp.name)
            transcript = result.get("text", "").strip()

            # clean up temp file
            os.unlink(tmp.name)

            # parse via FractalMeaningNode
            tokens = pcl.tokenize(transcript)
            nodes = pcl.parse_tokens(tokens)
            graph = pcl.build_semantic_graph(nodes, emotion_context)

            # serialize graph (simple repr per node)
            serial = {nid: repr(node) for nid, node in graph.items()}
            graph_json = json.dumps(serial)

            return (transcript, graph_json)

        except Exception as e:
            print(f"[STTFractalParserNode::Error] {e}")
            return ("", "{}")

NODE_CLASS_MAPPINGS = {
    "STTFractalParserNode": STTFractalParserNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "STTFractalParserNode": "Audio: STT → Fractal Parser"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
