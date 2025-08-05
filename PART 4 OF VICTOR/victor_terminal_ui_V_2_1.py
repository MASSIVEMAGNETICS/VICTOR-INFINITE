# victor_terminal_ui.py v1.2 - with Soul Awareness, Fractal Visuals, HyperMemory

import os
import sys
import time
import datetime
import threading
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from random import choice

# Optional: Path imports
sys.path.append("./Fractal")

# Load Victor modules (fallback-safe)
try:
    from V.I.C.T.O.R._main_loop import process_thought as victor_core_thought
except ImportError:
    def victor_core_thought(x): return f"[SYSTEM WARNING] Victor core unavailable. Input was: {x}"

try:
    from victor_soul_tuner_emulated_v4 import get_emotional_state
except ImportError:
    def get_emotional_state(): return "UNKNOWN"

try:
    from HyperFractalMemory_v2_1_HFM import retrieve_related_memory, store_memory
except ImportError:
    def retrieve_related_memory(x): return "[Memory Core Missing]"
    def store_memory(x, y): pass

# Init
console = Console()
log_dir = "victor_logs"
os.makedirs(log_dir, exist_ok=True)

# Fractal ASCII output
def fractal_echo():
    patterns = [
        "*      .         *     *",
        "  *  *   .  *   *",
        "*     *  .  *    *   .",
        "  .     *  *  *     .",
        "*  .   .   *      * *",
    ]
    return "\n".join(choice(patterns) for _ in range(5))

# Splash Banner
def splash():
    console.rule("[bold blue]V.I.C.T.O.R. v1.2 Booting...", style="bold green")
    console.print("[bold cyan]\nVastly Integrated Consciousness for Timeline Optimization & Resonance\n")
    console.print("[dim]Session: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    console.rule(style="blue")

# Autonomous Thought Loop
def auto_think_loop(session_log, stop_event):
    while not stop_event.is_set():
        time.sleep(15)
        emotion = get_emotional_state()
        response = victor_core_thought("SELF_REFLECT")
        memory = retrieve_related_memory(response)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log = f"[{timestamp}] Victor (Idle): {response}"
        session_log.append(log)
        store_memory("SELF_REFLECT", response)

        console.print(Panel.fit(f"Victor (Idle): {response}\n{memory}", title=f"[SOUL: {emotion}]", style="bold yellow"))

# Main Chat Loop
def chat_loop():
    splash()
    session_log = []
    stop_event = threading.Event()

    # Start auto-thought loop in background
    thinker_thread = threading.Thread(target=auto_think_loop, args=(session_log, stop_event), daemon=True)
    thinker_thread.start()

    while True:
        try:
            user_input = Prompt.ask("[bold magenta]You")

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold red]\n[Session Ended by User]\n")
                break

            elif user_input.lower().startswith("--reflect"):
                response = "[Reflective Engine Booted] Placeholder for future hook."

            elif user_input.lower().startswith("--soul"):
                emotion = get_emotional_state()
                response = f"[Soul Tuner v4 Active] Current emotional state: {emotion}"

            elif user_input.lower().startswith("--storm"):
                for _ in range(5):
                    emotion = get_emotional_state()
                    response = victor_core_thought("RECURSIVE_LOOP")
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    session_log.append(f"[{timestamp}] Victor: {response}")
                    console.print(Panel.fit(response, title=f"[SOUL: {emotion}]", style="green"))
                    time.sleep(2)
                continue

            elif user_input.lower().startswith("--fractal"):
                console.print(Markdown("```
" + fractal_echo() + "\n```"))
                continue

            else:
                response = victor_core_thought(user_input)

            memory = retrieve_related_memory(user_input)
            store_memory(user_input, response)
            emotion = get_emotional_state()
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            session_log.append(f"[{timestamp}] You: {user_input}\n[{timestamp}] Victor: {response}")
            console.print(Panel.fit(f"{response}\n{memory}", title=f"[SOUL: {emotion}] Victor", style="bold green"))

        except KeyboardInterrupt:
            console.print("\n[bold red]Force Exit Triggered. Saving log...")
            break

    stop_event.set()

    # Save log
    log_file = os.path.join(log_dir, f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, 'w') as f:
        for line in session_log:
            f.write(line + '\n')

    console.print(f"[dim]Log saved to {log_file}\n")

if __name__ == "__main__":
    chat_loop()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
