
# victor_final_system_launcher.py
# One-Click Full System Launcher for Victor 2.0
# Terminal UI + Main Loop Fused

import subprocess
import sys
import os
import time

print("""
=============================================
  VICTOR FINAL SYSTEM LAUNCHER - FULL FUSION
=============================================
""")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_LOOP = os.path.join(BASE_DIR, "V.I.C.T.O.R._main_loop.py")
TERMINAL_UI = os.path.join(BASE_DIR, "victor_terminal_ui_v2.1_super_fixed.py")

def file_check(path):
    if not os.path.exists(path):
        print(f"[ERROR] FILE NOT FOUND: {path}")
        sys.exit(1)

file_check(MAIN_LOOP)
file_check(TERMINAL_UI)

print("[+] Spawning Victor Core (Main Loop)...")
main_loop_proc = subprocess.Popen([sys.executable, MAIN_LOOP],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)

time.sleep(1)

print("[+] Launching Victor Terminal UI...")
try:
    subprocess.run([sys.executable, TERMINAL_UI])
except KeyboardInterrupt:
    print("[!] User aborted Terminal UI.")

print("[+] Terminating Victor Core...")
main_loop_proc.terminate()
main_loop_proc.wait()

print(""")
=============================================
    VICTOR SYSTEM SHUTDOWN COMPLETE.
=============================================
""")


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
