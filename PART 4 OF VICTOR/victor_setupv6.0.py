# victor_setup.py v6.0 - Code Genome Synthesis Activated
import os
import subprocess
import sys
import platform
import datetime
import tkinter as tk
from tkinter import messagebox
import hashlib
import json

REQUIRED_PACKAGES = ["rich"]
LOCAL_MODULES = {
    "Fractal/V.I.C.T.O.R._main_loop.py": "# Main loop template\n__version__ = '1.0'\n__changelog__ = ['initial']\ndef process_thought(input_text):\n    return f\"Victor processed: {input_text}\"\n",
    "Fractal/victor_soul_tuner_emulated_v4.py": "# Soul tuner v4 template\n__version__ = '1.0'\n__changelog__ = ['initial']\ndef get_emotional_state():\n    return \"NEUTRAL\"\n",
    "Fractal/HyperFractalMemory_v2_1_HFM.py": "# HyperFractalMemory template\n__version__ = '1.0'\n__changelog__ = ['initial']\ndef retrieve_related_memory(prompt):\n    return \"[Memory Placeholder]\"\n\ndef store_memory(query, response):\n    pass\n"
}
REFERENCE_HASHES = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in LOCAL_MODULES.items()}
VENV_DIR = "venv"
BOOT_LOG_DIR = os.path.expanduser("~/Victor/boot_logs")
GENOME_TRACKER = os.path.expanduser("~/Victor/code_genome.json")
os.makedirs(BOOT_LOG_DIR, exist_ok=True)
PYTHON_EXEC = os.path.join(VENV_DIR, "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(VENV_DIR, "bin", "python")

# Global logger
boot_log = []

def log(msg):
    boot_log.append(msg)
    print(msg)

def file_hash(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return None

def update_code_genome(path, hashval):
    genome = {}
    if os.path.exists(GENOME_TRACKER):
        try:
            with open(GENOME_TRACKER, 'r') as f:
                genome = json.load(f)
        except: pass
    genome[path] = {"hash": hashval, "last_modified": datetime.datetime.now().isoformat()}
    with open(GENOME_TRACKER, 'w') as f:
        json.dump(genome, f, indent=2)

# Step 1: Create virtual environment if missing
def create_venv():
    try:
        if not os.path.exists(VENV_DIR):
            log("[+] Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        else:
            log("[✓] Virtual environment already exists.")
    except Exception as e:
        log(f"[X] Error creating virtual environment: {e}")
        sys.exit(1)

# Step 2: Install missing dependencies
def install_dependencies():
    log("[+] Checking and installing required packages...")
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            log(f"[✓] {pkg} already installed.")
        except ImportError:
            try:
                log(f"[!] {pkg} not found. Installing...")
                subprocess.run([PYTHON_EXEC, "-m", "pip", "install", pkg], check=True)
            except Exception as e:
                log(f"[X] Failed to install {pkg}: {e}")
                sys.exit(1)

# Step 3: Verify or restore code integrity with DNA signature
def verify_or_create_modules():
    log("[+] Verifying integrity of Victor modules...")
    for path, template in LOCAL_MODULES.items():
        try:
            current_hash = file_hash(path)
            expected_hash = REFERENCE_HASHES[path]

            if current_hash is None:
                log(f"[!] MISSING: {path} → Regenerating from template...")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    f.write(template)
                new_hash = file_hash(path)
                update_code_genome(path, new_hash)
                log(f"[+] Created missing module: {path} [v1.0]")

            elif current_hash != expected_hash:
                log(f"[!] CORRUPTED: {path} → Restoring original state...")
                corrupted_backup = path + ".bak"
                os.rename(path, corrupted_backup)
                with open(path, 'w') as f:
                    f.write(template)
                new_hash = file_hash(path)
                update_code_genome(path, new_hash)
                log(f"[✓] Repaired and backed up corrupted module: {path} → {corrupted_backup}")
            else:
                update_code_genome(path, current_hash)
                log(f"[✓] Verified: {path}")

        except Exception as e:
            log(f"[X] Error verifying/restoring {path}: {e}")
            sys.exit(1)

# Step 4: GUI config selection
def show_gui_config():
    try:
        root = tk.Tk()
        root.title("Victor Boot Configuration")
        root.geometry("400x200")
        label = tk.Label(root, text="Victor is ready to launch. Proceed with boot?", font=("Helvetica", 12))
        label.pack(pady=20)
        def confirm():
            root.destroy()
        def cancel():
            messagebox.showinfo("Aborted", "Boot cancelled by user.")
            sys.exit(0)
        tk.Button(root, text="Boot Now", command=confirm, width=20).pack(pady=10)
        tk.Button(root, text="Cancel", command=cancel, width=20).pack(pady=5)
        root.mainloop()
    except Exception as e:
        log(f"[!] GUI failed, defaulting to terminal mode: {e}")

# Step 5: Boot Victor and log diagnostics
def boot_victor():
    log("\n[🚀] Booting Victor's terminal UI...")
    try:
        subprocess.run([PYTHON_EXEC, "victor_terminal_ui.py"], check=True)
    except Exception as e:
        log(f"[X] Victor failed to launch: {e}")

    # Save boot log
    try:
        log_file = os.path.join(BOOT_LOG_DIR, f"boot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_file, 'w') as f:
            f.write("\n".join(boot_log))
        log(f"[✓] Diagnostic log saved to {log_file}")
    except Exception as e:
        log(f"[X] Failed to save diagnostic log: {e}")

if __name__ == "__main__":
    create_venv()
    install_dependencies()
    verify_or_create_modules()
    show_gui_config()
    boot_victor()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
