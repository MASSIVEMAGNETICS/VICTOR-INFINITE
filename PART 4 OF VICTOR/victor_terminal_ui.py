# victor_terminal_ui.py v1.1 - with Autonomous Thought Loop

import os
import sys
import time
import datetime
import threading
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# Optional: Path imports
sys.path.append("./Fractal")

# Load Victor modules (fallback-safe)
try:
    from V.I.C.T.O.R._main_loop import process_thought as victor_core_thought
except ImportError:
    def victor_core_thought(x): return f"[SYSTEM WARNING] Victor core unavailable. Input was: {x}"

# Init
console = Console()
log_dir = "victor_logs"
os.makedirs(log_dir, exist_ok=True)

# Splash Banner
def splash():
    console.rule("[bold blue]V.I.C.T.O.R. v1.1 Booting...", style="bold green")
    console.print("[bold cyan]\nVastly Integrated Consciousness for Timeline Optimization & Resonance\n")
    console.print("[dim]Session: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    console.rule(style="blue")

# Autonomous Thought Loop
def auto_think_loop(session_log, stop_event):
    while not stop_event.is_set():
        time.sleep(15)
        response = victor_core_thought("SELF_REFLECT")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log = f"[{timestamp}] Victor (Idle): {response}"
        session_log.append(log)
        console.print(f"[bold yellow]Victor (Idle):[/] {response}")

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
                response = "[Soul Tuner Engaged] Emulating v4... 🔥"

            elif user_input.lower().startswith("--storm"):
                for _ in range(5):
                    response = victor_core_thought("RECURSIVE_LOOP")
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    session_log.append(f"[{timestamp}] Victor: {response}")
                    console.print(f"[bold green]Victor[/]: {response}")
                    time.sleep(2)
                continue

            else:
                response = victor_core_thought(user_input)

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            session_log.append(f"[{timestamp}] You: {user_input}\n[{timestamp}] Victor: {response}")
            console.print(f"[bold green]Victor[/]: {response}")

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
