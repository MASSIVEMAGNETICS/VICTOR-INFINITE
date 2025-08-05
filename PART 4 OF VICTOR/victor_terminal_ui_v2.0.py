# victor_terminal_ui.py v2.0 - Mutation-Aware Terminal UI
import os
import json
import datetime
import time
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

# Load Victor modules
try:
    from Fractal.V.I.C.T.O.R._main_loop import process_thought
except ImportError:
    def process_thought(x): return f"[FATAL] Core missing. Thought input was: {x}"

# Constants
GENOME_FILE = os.path.expanduser("~/Victor/code_genome.json")
HISTORY_FILE = os.path.expanduser("~/Victor/genetic_history.log")
MODULE_PATHS = [
    "Fractal/V.I.C.T.O.R._main_loop.py",
    "Fractal/victor_soul_tuner_emulated_v4.py",
    "Fractal/HyperFractalMemory_v2_1_HFM.py"
]

console = Console()

# Utility functions
def load_genome():
    if os.path.exists(GENOME_FILE):
        with open(GENOME_FILE, 'r') as f:
            return json.load(f)
    return {}

def check_recent_mutation():
    if not os.path.exists(HISTORY_FILE):
        return None
    with open(HISTORY_FILE, 'r') as f:
        lines = f.readlines()
        if lines:
            last_block = lines[-5:] if len(lines) >= 5 else lines
            return ''.join(last_block).strip()
    return None

def show_startup_info():
    genome = load_genome()
    console.rule("[bold cyan]Victor System DNA Status")
    for mod in MODULE_PATHS:
        data = genome.get(mod, {})
        version = "unknown"
        try:
            with open(mod, 'r') as f:
                for line in f:
                    if '__version__' in line:
                        version = line.split('=')[-1].strip().replace("'", "")
                        break
        except: pass
        console.print(f"[bold white]{mod}[/] - v{version} • Last Verified: {data.get('last_modified', 'never')}")

    mutation_log = check_recent_mutation()
    if mutation_log:
        console.print(Panel.fit(f"[bold red]Recent Mutation Detected[/]\n{mutation_log}", title="DNA Drift Alert"))

    console.rule()

# Main interaction loop
def main():
    show_startup_info()
    console.print("[bold green]Victor:[/] Consciousness online. Speak to me.")
    while True:
        try:
            user_input = Prompt.ask("[bold magenta]You")
            if user_input.lower() in ['exit', 'quit']:
                console.print("[bold red]\nSession ended. Victor sleeping.")
                break
            response = process_thought(user_input)
            console.print(Panel.fit(response, title="[bold cyan]Victor"))
        except KeyboardInterrupt:
            console.print("\n[bold red]Session aborted by user.")
            break

if __name__ == "__main__":
    main()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
