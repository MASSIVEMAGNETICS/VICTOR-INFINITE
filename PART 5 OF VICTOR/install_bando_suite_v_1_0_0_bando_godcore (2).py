"""
FILE: /installer/install_bando_suite_v1.1.0-BANDO-GODCORE.py
VERSION: v1.1.0-BANDO-GODCORE
NAME: BandoSuiteInstaller
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: One‑shot installer that bootstraps a self‑contained virtual env, installs all deps for
         Bando Copilot + Dataset Trainer GUI, drops handy launch scripts, **and can auto‑launch**
         the Trainer GUI for true one‑click start‑up.
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

CHANGELOG v1.1.0:
  • Added `--auto-launch / -l` flag – spawns the Trainer GUI instantly after install.
  • Generates extra top‑level launcher  `start_bando_trainer.(bat|sh)` alongside ./launchers/ variants.
  • Minor code clean‑ups + inline README_INSTALL scaffold.

USAGE (shell):
    python install_bando_suite_v1.1.0-BANDO-GODCORE.py [--venv DIR] [--no-shortcuts] [--auto-launch]
"""
import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# === CONFIG ===
DEFAULT_VENV = Path("bando_env")
REQUIREMENTS = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "cryptography>=42",
    "torch>=2.3; platform_system!='Darwin'",  # macOS M‑series wheels messy – manual later.
    "tqdm>=4.66",
    "sentencepiece>=0.2.0",
]
COPILOT_ENTRY = "bando_copilot_core_v1.0.0-BANDO-GODCORE.py"
TRAINER_ENTRY = "bando_dataset_trainer_gui_v1.0.0-BANDO-GODCORE.py"
README_TEMPLATE = """# Bando Suite – Quick Start\n\n## Activate environment\n```bash\nsource {venv}/bin/activate  # Windows: {venv}\\Scripts\\activate.bat\n```\n\n## Launch tools\n```bash\n./launchers/bando-copilot   # FastAPI copilot\n./launchers/bando-trainer   # Tkinter trainer GUI\n```\n"""

# === UTILS ===

def run(cmd, env=None):
    print(f"[RUN] {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd, env=env or os.environ.copy())


def create_venv(venv_dir: Path):
    if venv_dir.exists():
        print(f"[INFO] Virtual env already exists at {venv_dir}")
        return
    print(f"[INFO] Creating virtual environment → {venv_dir}")
    run([sys.executable, "-m", "venv", str(venv_dir)])


def pip_install(venv_dir: Path, pkgs):
    pip = venv_dir / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    run([str(pip), "install", "--upgrade", "pip"])
    run([str(pip), "install", *pkgs])


def write_requirements():
    req_txt = Path("requirements.txt")
    req_txt.write_text("\n".join(REQUIREMENTS) + "\n")
    print(f"[INFO] Wrote requirements → {req_txt}")


def write_readme(venv_dir: Path):
    readme = Path("README_INSTALL.md")
    readme.write_text(README_TEMPLATE.format(venv=venv_dir))
    print(f"[INFO] Wrote README → {readme}")


def write_launcher(name: str, target: str, venv_dir: Path, root_alias: bool = False):
    launch_dir = Path("launchers")
    launch_dir.mkdir(exist_ok=True)

    if os.name == "nt":
        content = f"@echo off\ncall {venv_dir}\\Scripts\\activate.bat\npython {target} %*\n"
        ext = ".bat"
    else:
        content = f"#!/usr/bin/env bash\nsource {venv_dir}/bin/activate\npython {target} \"$@\"\n"
        ext = ".sh"

    script_path = launch_dir / f"{name}{ext}"
    script_path.write_text(content)
    script_path.chmod(0o755)
    print(f"[INFO] Launcher created → {script_path}")

    if root_alias:
        alias_path = Path(f"start_{name}{ext}")
        shutil.copy2(script_path, alias_path)
        alias_path.chmod(0o755)
        print(f"[INFO] Root‑level alias → {alias_path}")


def launch_trainer_gui(venv_dir: Path):
    """Spawn the Trainer GUI in a separate process (non‑blocking)."""
    python_exec = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print("[INFO] Auto‑launching Trainer GUI …")
    subprocess.Popen([str(python_exec), TRAINER_ENTRY], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# === MAIN SCRIPT ===

def main():
    parser = argparse.ArgumentParser(description="Install Bando Copilot + Trainer suite.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Virtual env directory")
    parser.add_argument("--no-shortcuts", action="store_true", help="Skip launcher creation")
    parser.add_argument("--auto-launch", "-l", action="store_true", help="Launch Trainer GUI immediately after install")
    args = parser.parse_args()

    venv_dir: Path = args.venv.resolve()

    create_venv(venv_dir)
    pip_install(venv_dir, REQUIREMENTS)
    write_requirements()
    write_readme(venv_dir)

    if not args.no_shortcuts:
        write_launcher("bando-copilot", COPILOT_ENTRY, venv_dir)
        write_launcher("bando-trainer", TRAINER_ENTRY, venv_dir, root_alias=True)  # root alias for one‑click

    if args.auto_launch:
        launch_trainer_gui(venv_dir)

    print("[DONE] Installation complete. Happy hustling, Bando! ✨")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e.cmd}")
        sys.exit(e.returncode)
    except Exception as ex:
        print(f"[FATAL] {ex}")
        sys.exit(1)
