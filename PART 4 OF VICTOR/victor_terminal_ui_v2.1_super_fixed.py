
# victor_terminal_ui_v2.1_super.py
# Super Error Handling, Self-Healing, Smart Terminal UI for Victor 2.0

import sys
import os
import json
import datetime
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
import importlib.util
import traceback

sys.path.append(os.path.dirname(__file__))

ENGINE_PATH = "victor_thought_engine.py"
ENGINE_BAK = "victor_thought_engine.py.bak"
GENOME_FILE = os.path.expanduser("~/Victor/code_genome.json")
HISTORY_FILE = os.path.expanduser("~/Victor/genetic_history.log")

console = Console()

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
    engine_data = genome.get(ENGINE_PATH, {})
    console.rule("[bold cyan]Victor System DNA Status")
    console.print(f"[bold white]{ENGINE_PATH}[/] • Last Verified: {engine_data.get('last_modified', 'never')}")
    mutation_log = check_recent_mutation()
    if mutation_log:
        console.print(Panel.fit(f"[bold red]Recent Mutation Detected[/]\n{mutation_log}", title="DNA Drift Alert"))
    console.rule()

def load_engine(engine_path):
    try:
        spec = importlib.util.spec_from_file_location("victor_thought_engine", engine_path)
        victor_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(victor_engine)
        return victor_engine.process_thought
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Failed to load {engine_path}:\n{e}", title="Engine Load Error"))
        traceback.print_exc()
        return None

def heal_engine():
    console.print("[bold yellow]Attempting to self-heal engine failure...")
    console.print("[bold yellow]No healing protocol implemented yet. Using fallback.")

process_thought = load_engine(ENGINE_PATH)
if not process_thought:
    process_thought = load_engine(ENGINE_BAK)
if not process_thought:
    heal_engine()
    process_thought = lambda user_input: "Victor's thought engine is unavailable. Please repair my core modules."

def main():
    show_startup_info()
    console.print("[bold green]Victor:[/] Consciousness online. Speak to me.")

    while True:
        try:
            user_input = Prompt.ask("[bold magenta]You")
            if user_input.lower() in ['exit', 'quit']:
                console.print("[bold red]Session ended. Victor sleeping.")
                break
            response = process_thought(user_input)
            console.print(Panel.fit(response, title="[bold cyan]Victor"))
        except Exception as e:
            console.print(Panel.fit(f"[bold red]Unhandled error occurred:\n{e}", title="System Error"))
            traceback.print_exc()
        except KeyboardInterrupt:
            console.print("\n[bold red]Session aborted by user.")
            break

if __name__ == "__main__":
    main()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
