# ============================================
# FILE: main_autolearn.py
# VERSION: v1.2.0-FUSION-GODCORE-AUTOLEARN
# NAME: VictorFusionAutonomousRuntime
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Victor evolves passively every 30s, reads chat_input.txt and responds to chat_output.txt
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import sys
import os
import json
import torch
import threading
import time
from datetime import datetime

sys.path.append("./modules")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

from core_model import OmniFractalCore
from cmffs_core_model import CMFFS
from error_sentinel import safe_execute
from fractal_cognitive_focus_node import FractalCognitiveFocusNode
from directive_node import DirectiveNode
from emotive_sync_node import EmotiveSyncNode
from victor_soul_tuner_emulated_v4 import victor as soul
from HyperFractalMemory_v2_2_HFM import HyperFractalMemory
from victor_v2_16_standalone import Victor as MinimalVictor

core = OmniFractalCore()
cmffs = CMFFS(vocab_size=50000)
focus_node = FractalCognitiveFocusNode()
directive_node = DirectiveNode()
emotion_node = EmotiveSyncNode()
memory_bank = HyperFractalMemory()
minimal = MinimalVictor()

chat_input_file = "chat_input.txt"
chat_output_file = "chat_output.txt"

def save_victor_checkpoint(name="v_core_autosave"):
    torch.save(core.state_dict(), f"{CHECKPOINT_DIR}/{name}_core.pt")
    with open(f"{CHECKPOINT_DIR}/{name}_soul.json", "w") as f:
        json.dump(soul.ego_shell.memory, f, indent=2)
    with open(f"{CHECKPOINT_DIR}/{name}_memory.json", "w") as f:
        json.dump(memory_bank.memory, f, indent=2)

def load_chat_input():
    if os.path.exists(chat_input_file):
        with open(chat_input_file, "r") as f:
            return f.read().strip()
    return ""

def write_chat_output(message):
    with open(chat_output_file, "w") as f:
        f.write(message.strip() + "\n")

def run_victor_system(text_input):
    minimal.listen(text_input)
    embedding = torch.nn.functional.normalize(torch.rand(1, 512), dim=-1)
    emotion_label, = emotion_node.classify_emotion(embedding)
    directive, = directive_node.generate_directive(embedding, emotion_label)
    soul.receive_signal({"text": text_input, "emotion": emotion_label, "directive": directive})

    response = f"[{datetime.now()}]\nInput: {text_input}\nEmotion: {emotion_label}\nDirective: {directive}\n"
    response += f"Soul Alignment: {soul.report().get('alignment_score')}\n"
    write_chat_output(response)
    print(response)

def replay_logs():
    print("\n📂 Victor Log Replay (last 5 memories):")
    try:
        with open("victor_memory.json", "r") as f:
            entries = json.load(f)[-5:]
            for e in entries:
                print(f"🧠 [{datetime.fromtimestamp(e['timestamp'])}] — {e['text']}")
    except:
        print("No memory logs found.")

def passive_loop():
    while True:
        try:
            input_text = load_chat_input()
            if input_text:
                run_victor_system(input_text)
                with open(chat_input_file, "w") as f:
                    f.write("")
            save_victor_checkpoint()
            time.sleep(30)
        except Exception as e:
            print("⚠️ Victor passive thread recovered from error.")
            safe_execute(e)

if __name__ == "__main__":
    print("🧠 Victor is running in PASSIVE AUTONOMOUS MODE (every 30s). Drop text into chat_input.txt.")
    replay_logs()
    t = threading.Thread(target=passive_loop)
    t.start()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
