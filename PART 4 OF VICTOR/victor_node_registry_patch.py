# NODE REGISTRATION PATCH
# Paste into any custom node missing its NODE_CLASS_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "EchoNode": EchoNode,
    "EmotiveSyncNode": EmotiveSyncNode,
    "DirectiveNode": DirectiveNode,
    "SpeechSynthNode": SpeechSynthNode,
    "VictorNlpEngineNode": VictorNlpEngineNode,
    "VictorMemoryCoreNode": VictorMemoryCoreNode,
    "PersonaSwitchboardNode": PersonaSwitchboardNode,
    "LiveMicrophoneCaptureNode": LiveMicrophoneCaptureNode,
    "VoiceProfileManagerNode": VoiceProfileManagerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EchoNode": "Echo: Recursive Reply",
    "EmotiveSyncNode": "Emotion: Sync",
    "DirectiveNode": "Directive: Logic Core",
    "SpeechSynthNode": "TTS: Speech Synth",
    "VictorNlpEngineNode": "Language: NLP Engine",
    "VictorMemoryCoreNode": "Memory: Core Vectorizer",
    "PersonaSwitchboardNode": "Persona: Voice Selector",
    "LiveMicrophoneCaptureNode": "Mic: Live Capture",
    "VoiceProfileManagerNode": "Voices: Manager",
}

# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
