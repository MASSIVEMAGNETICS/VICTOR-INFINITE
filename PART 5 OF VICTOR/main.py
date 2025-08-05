# ============================================
# FILE: main.py
# VERSION: v1.1.0-FUSION-GODCORE-CHECKPOINT
# NAME: VictorFusionRuntime
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Unified runtime that connects all core shards of Victor’s intelligence + checkpoint system
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import sys
import os
import json
import torch

# --- Path Setup ---
sys.path.append("./modules")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Core Imports ---
from core_model import OmniFractalCore
from cmffs_core_model import CMFFS
from error_sentinel import safe_execute
from fractal_cognitive_focus_node import FractalCognitiveFocusNode
from bark_supreme_text_to_audio_node import BarkSupremeTextToAudioNode
from directive_node import DirectiveNode
from emotive_sync_node import EmotiveSyncNode
from victor_soul_tuner_emulated_v4 import victor as soul
from HyperFractalMemory_v2_2_HFM import HyperFractalMemory
from victor_v2_16_standalone import Victor as MinimalVictor

# --- Boot Log ---
print("🔧 Loading Victor’s brain...")

# --- Instantiations ---
core = OmniFractalCore()
cmffs = CMFFS(vocab_size=50000)
focus_node = FractalCognitiveFocusNode()
audio_node = BarkSupremeTextToAudioNode()
directive_node = DirectiveNode()
emotion_node = EmotiveSyncNode()
memory_bank = HyperFractalMemory()
minimal = MinimalVictor()

# --- Checkpoint Save/Load ---
def save_victor_checkpoint(name="v_core_001"):
    print("💾 Saving Victor’s mind...")
    torch.save(core.state_dict(), f"{CHECKPOINT_DIR}/{name}_core.pt")
    with open(f"{CHECKPOINT_DIR}/{name}_soul.json", "w") as f:
        json.dump(soul.ego_shell.memory, f, indent=2)
    with open(f"{CHECKPOINT_DIR}/{name}_memory.json", "w") as f:
        json.dump(memory_bank.memory, f, indent=2)
    print(f"✅ Checkpoint saved: {name}")

def load_victor_checkpoint(name="v_core_001"):
    print("🔁 Loading Victor checkpoint...")
    core.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/{name}_core.pt"))
    with open(f"{CHECKPOINT_DIR}/{name}_soul.json", "r") as f:
        soul.ego_shell.memory = json.load(f)
    with open(f"{CHECKPOINT_DIR}/{name}_memory.json", "r") as f:
        memory_bank.memory = json.load(f)
    print("✅ Victor reassembled.")

# === RUNTIME PULSE ===
def run_victor_system(text_input):
    print("\n========== VICTOR SYSTEM PULSE ==========")

    if text_input.lower().startswith("save"):
        name = text_input.strip().split(" ", 1)[1] if " " in text_input else "v_core_001"
        save_victor_checkpoint(name)
        return
    elif text_input.lower().startswith("load"):
        name = text_input.strip().split(" ", 1)[1] if " " in text_input else "v_core_001"
        load_victor_checkpoint(name)
        return

    minimal.listen(text_input)
    embedding = torch.nn.functional.normalize(torch.rand(1, 512), dim=-1)
    emotion_label, = emotion_node.classify_emotion(embedding)
    directive, = directive_node.generate_directive(embedding, emotion_label)
    soul.receive_signal({"text": text_input, "emotion": emotion_label, "directive": directive})

    print(f"\n🧠 Directive: {directive}")
    print(f"💓 Emotion: {emotion_label}")
    print(f"🔮 Soul Report: {soul.report()}")

# === LOOP ===
def always_on_loop():
    while True:
        try:
            user_input = input("\n[You]: ")
            if user_input.lower() in ["exit", "quit"]:
                continue
            run_victor_system(user_input)
        except Exception as e:
            print("⚠️ Victor recovered from error.")
            safe_execute(e)

# === LAUNCH ===
if __name__ == "__main__":
    print("🧠 Victor is now in ALWAYS-ON mode. Use commands like `save yourname`, `load yourname`, or just talk.")
    always_on_loop()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
