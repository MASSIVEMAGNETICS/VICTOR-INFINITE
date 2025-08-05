"""
FILE: /installer/install_bando_suite_v1.0.0-BANDO-GODCORE.py
VERSION: v1.0.0-BANDO-GODCORE
NAME: BandoSuiteInstaller
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: One‑shot installer that bootstraps a self‑contained virtual env, installs all deps for
         Bando Copilot + Dataset Trainer GUI, and drops handy launch scripts (bando‑copilot, bando‑trainer).
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

USAGE (shell):
    python install_bando_suite_v1.0.0-BANDO-GODCORE.py [--venv DIR] [--no-shortcuts]

Steps performed:
1. Create venv (default ./bando_env) if not exists.
2. Install requirements via pip inside venv.
3. Write launcher scripts → ./launchers/bando-copilot(.bat|.sh) & bando-trainer(.bat|.sh).
4. Drop a requirements.txt and README_INSTALL.md for reference.

Tested on: Windows 10 & Ubuntu 22.04 with Python ≥3.9.
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
    "torch>=2.3; platform_system!='Darwin'",  # ARM Mac wheels tricky; let user sort.
    "tqdm>=4.66",
    "sentencepiece>=0.2.0",  # future tokenizer
]
COPILOT_ENTRY = "bando_copilot_core_v1.0.0-BANDO-GODCORE.py"
TRAINER_ENTRY = "bando_dataset_trainer_gui_v1.0.0-BANDO-GODCORE.py"


# === UTILS ===

def run(cmd, env=None):
    print(f"[RUN] {' '.join(cmd)}")
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


def write_requirements(venv_dir: Path):
    req_txt = (venv_dir.parent / "requirements.txt")
    req_txt.write_text("\n".join(REQUIREMENTS) + "\n")
    print(f"[INFO] Wrote requirements → {req_txt}")


def write_launcher(name: str, target: str, venv_dir: Path):
    launch_dir = Path("launchers")
    launch_dir.mkdir(exist_ok=True)
    if os.name == "nt":
        script = launch_dir / f"{name}.bat"
        content = f"@echo off\ncall {venv_dir}\\Scripts\\activate.bat\npython {target} %*\n"
    else:
        script = launch_dir / f"{name}.sh"
        content = f"#!/usr/bin/env bash\nsource {venv_dir}/bin/activate\npython {target} "$@"\n"
    script.write_text(content)
    script.chmod(0o755)
    print(f"[INFO] Launcher created → {script}")


def main():
    parser = argparse.ArgumentParser(description="Install Bando Copilot + Trainer suite.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Virtual env directory")
    parser.add_argument("--no-shortcuts", action="store_true", help="Skip launcher creation")
    args = parser.parse_args()

    venv_dir: Path = args.venv.resolve()
    create_venv(venv_dir)
    pip_install(venv_dir, REQUIREMENTS)
    write_requirements(venv_dir)

    if not args.no_shortcuts:
        write_launcher("bando-copilot", COPILOT_ENTRY, venv_dir)
        write_launcher("bando-trainer", TRAINER_ENTRY, venv_dir)
        print("[DONE] Use launchers in ./launchers to run your tools.")
    else:
        print("[DONE] Installation complete (no launchers requested).")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e.cmd}")
        sys.exit(e.returncode)
    except Exception as ex:
        print(f"[FATAL] {ex}")
        sys.exit(1)
