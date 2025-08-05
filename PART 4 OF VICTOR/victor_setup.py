# victor_setup.py - Auto-env, dependency bootstrapper, and brain boot
import os
import subprocess
import sys
import platform

REQUIRED_PACKAGES = ["rich"]
LOCAL_MODULES = [
    "Fractal/V.I.C.T.O.R._main_loop.py",
    "Fractal/victor_soul_tuner_emulated_v4.py",
    "Fractal/HyperFractalMemory_v2_1_HFM.py"
]
VENV_DIR = "venv"
PYTHON_EXEC = os.path.join(VENV_DIR, "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(VENV_DIR, "bin", "python")

# Step 1: Create virtual environment if missing
def create_venv():
    if not os.path.exists(VENV_DIR):
        print("[+] Creating virtual environment...\n")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("[✓] Virtual environment already exists.")

# Step 2: Install missing dependencies
def install_dependencies():
    print("[+] Checking and installing required packages...\n")
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            print(f"[✓] {pkg} already installed.")
        except ImportError:
            print(f"[!] {pkg} not found. Installing...")
            subprocess.run([PYTHON_EXEC, "-m", "pip", "install", pkg])

# Step 3: Check local core modules exist
def verify_local_modules():
    print("[+] Checking Victor's core proprietary modules...\n")
    for path in LOCAL_MODULES:
        if os.path.exists(path):
            print(f"[✓] Found: {path}")
        else:
            print(f"[X] MISSING MODULE: {path}")

# Step 4: Boot-to-brain after setup
def boot_victor():
    print("\n[🚀] Booting Victor's terminal brain...")
    subprocess.run([PYTHON_EXEC, "victor_terminal_ui.py"])

if __name__ == "__main__":
    create_venv()
    install_dependencies()
    verify_local_modules()
    boot_victor()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
