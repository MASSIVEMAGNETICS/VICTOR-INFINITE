# victor_diff_viewer.py - DNA Diff Scanner for Victor Modules
import os
import difflib
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Config
MODULES = [
    "Fractal/V.I.C.T.O.R._main_loop.py",
    "Fractal/victor_soul_tuner_emulated_v4.py",
    "Fractal/HyperFractalMemory_v2_1_HFM.py"
]

console = Console()

def load_lines(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def diff_module(mod_path):
    bak_path = mod_path + ".bak"
    current = load_lines(mod_path)
    backup = load_lines(bak_path)

    if not backup:
        console.print(f"[bold yellow]No backup found for {mod_path}. Nothing to compare.\n")
        return

    diff = list(difflib.unified_diff(
        backup, current,
        fromfile=bak_path,
        tofile=mod_path,
        lineterm=''
    ))

    if not diff:
        console.print(f"[bold green]{mod_path}[/] — [✓] No difference detected.")
    else:
        console.rule(f"[bold cyan]⚠️ DNA Drift in {os.path.basename(mod_path)}")
        console.print(Markdown("```diff\n" + "\n".join(diff) + "\n```"))


def main():
    console.print(Panel.fit("Victor Genome Diff Viewer\nScan Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="bold magenta"))
    for mod in MODULES:
        diff_module(mod)
    console.rule("[bold green]Scan Complete")

if __name__ == "__main__":
    main()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
