# victor_setup.py v3.0 - God-tier Error Handling, Self-Healing Logic
import os
import subprocess
import sys
import platform
import datetime
import tkinter as tk
from tkinter import messagebox

REQUIRED_PACKAGES = ["rich"]
LOCAL_MODULES = {
    "Fractal/V.I.C.T.O.R._main_loop.py": "# Main loop template\ndef process_thought(input_text):\n    return f\"Victor processed: {input_text}\"\n",
    "Fractal/victor_soul_tuner_emulated_v4.py": "# Soul tuner v4 template\ndef get_emotional_state():\n    return \"NEUTRAL\"\n",
    "Fractal/HyperFractalMemory_v2_1_HFM.py": "# HyperFractalMemory template\ndef retrieve_related_memory(prompt):\n    return \"[Memory Placeholder]\"\n\ndef store_memory(query, response):\n    pass\n"
}
VENV_DIR = "venv"
BOOT_LOG_DIR = os.path.expanduser("~/Victor/boot_logs")
os.makedirs(BOOT_LOG_DIR, exist_ok=True)
PYTHON_EXEC = os.path.join(VENV_DIR, "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(VENV_DIR, "bin", "python")

# Global logger
boot_log = []

def log(msg):
    boot_log.append(msg)
    print(msg)

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

# Step 3: Verify or create core modules
def verify_or_create_modules():
    log("[+] Checking proprietary Victor modules...")
    for path, template in LOCAL_MODULES.items():
        try:
            if os.path.exists(path):
                log(f"[✓] Found: {path}")
            else:
                log(f"[!] MISSING: {path} → Generating from template...")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    f.write(template)
                log(f"[+] Created missing module: {path}")
        except Exception as e:
            log(f"[X] Failed to verify/create {path}: {e}")
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
