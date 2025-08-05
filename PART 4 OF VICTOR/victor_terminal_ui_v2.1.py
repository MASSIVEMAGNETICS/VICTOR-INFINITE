# victor_terminal_ui.py v2.1 - Mutation-Injected Fallback Terminal
import os
import json
import datetime
import time
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

# Paths
GENOME_FILE = os.path.expanduser("~/Victor/code_genome.json")
HISTORY_FILE = os.path.expanduser("~/Victor/genetic_history.log")
ENGINE_PATH = "victor_thought_engine.py"
ENGINE_BAK = "victor_thought_engine.py.bak"

console = Console()

# === DNA + Mutation Check ===
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

# === Fallback Thought Engine Loader ===
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("victor_thought_engine", ENGINE_PATH)
    victor_engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(victor_engine)
    process_thought = victor_engine.process_thought
except Exception as e:
    console.print(Panel.fit(f"[bold red]⚠️ Primary engine failed. Trying .bak...\n{e}", title="Engine Failure"))
    try:
        spec = importlib.util.spec_from_file_location("victor_thought_engine", ENGINE_BAK)
        victor_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(victor_engine)
        process_thought = victor_engine.process_thought
        console.print("[bold yellow]Fallback loaded from .bak successfully.")
    except Exception as fail:
        console.print(Panel.fit(f"[bold red]Victor is brain-dead. Both engines failed.\n{fail}", title="CRITICAL FAILURE"))
        exit(1)

# === Terminal UI ===
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
