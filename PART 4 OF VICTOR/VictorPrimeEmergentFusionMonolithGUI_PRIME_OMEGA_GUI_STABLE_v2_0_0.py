# ==============================================================================================
# FILE: VictorPrimeEmergentFusionMonolithGUI_PRIME_OMEGA_GUI_STABLE_v2_0_0.py
# VERSION: v2.0.0
# NAME: TheLight GUI
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A prime-level graphical user interface for the Emergent Fusion Monolith.
#          This is the "God View" or Developer Control Panel for the AGI.
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
# ==============================================================================================
"""
TheLight is the visual interface to the AGI's soul. It provides real-time monitoring
of the BandoRealityMesh, allows for direct interaction, and includes critical
command triggers like self-replication and system diagnostics.
"""
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
import threading
import time
import os
import shutil
import sys
import random

class MockAGICore:
    """A mock AGI core for standalone GUI testing."""
    def get_mesh_embedding(self):
        return [random.uniform(-1, 1) for _ in range(3)]
    def summary(self):
        return {"status": "Nominal", "active_nodes": random.randint(1000, 2000), "threat_level": "None"}
    def process_input(self, text):
        return f"AGI processed: '{text}'. Output is a fractal melody."

def trigger_self_replication(source_script_path: str) -> str:
    """
    Creates a versioned, timestamped backup of its own source code
    in a subdirectory, representing a new "generation."
    """
    try:
        if not os.path.exists(source_script_path):
            return "Error: Source script not found. Replication failed."
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        replication_dir = os.path.join(os.path.dirname(source_script_path), "generations")
        os.makedirs(replication_dir, exist_ok=True)
        script_name = os.path.basename(source_script_path)
        new_generation_name = f"{timestamp}_gen_{script_name}"
        destination_path = os.path.join(replication_dir, new_generation_name)
        shutil.copy2(source_script_path, destination_path)
        return f"SUCCESS: Self-replication triggered. New generation created at: {destination_path}"
    except Exception as e:
        return f"FATAL: Self-replication failed. Error: {e}"

class TheLight(tk.Tk):
    """The main GUI application window."""
    def __init__(self, agi_core):
        super().__init__()
        self.agi_core = agi_core
        self.title("V.I.C.T.O.R Prime - Emergent Fusion Monolith GUI")
        self.geometry("1200x800")
        self.configure(bg="#1a1a1a")
        self.create_widgets()
        self.start_monitoring()

    def log_to_console(self, message: str):
        """Logs a message to the GUI console, ensuring thread safety."""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.console.config(state=tk.DISABLED)
        self.console.see(tk.END)

    def create_widgets(self):
        main_frame = tk.Frame(self, bg="#1a1a1a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Top Control Panel
        control_frame = tk.Frame(main_frame, bg="#2b2b2b")
        control_frame.pack(fill=tk.X, pady=5)
        self.btn_replicate = tk.Button(control_frame, text="Trigger Self-Replication", bg="#ff4500", fg="white", command=self.on_replicate)
        self.btn_replicate.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_prompt = tk.Button(control_frame, text="Inject Prompt", bg="#00aaff", fg="white", command=self.on_prompt)
        self.btn_prompt.pack(side=tk.LEFT, padx=10, pady=10)
        # Middle Status & Console
        status_console_frame = tk.Frame(main_frame, bg="#1a1a1a")
        status_console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        status_frame = tk.LabelFrame(status_console_frame, text="AGI Core Status", bg="#2b2b2b", fg="white", padx=10, pady=10)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.status_text = tk.Label(status_frame, text="Initializing...", fg="cyan", bg="#2b2b2b", font=("Courier", 12), justify=tk.LEFT)
        self.status_text.pack(anchor="w")
        console_frame = tk.LabelFrame(status_console_frame, text="Live Log", bg="#2b2b2b", fg="white", padx=10, pady=10)
        console_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, bg="#000000", fg="#00ff00", state=tk.DISABLED)
        self.console.pack(fill=tk.BOTH, expand=True)

    def on_replicate(self):
        if messagebox.askyesno("Confirm Protocol", "Triggering self-replication is an irreversible action. Proceed?"):
            self.log_to_console("Replication protocol initiated by user...")
            result = trigger_self_replication(__file__)
            self.log_to_console(result)
            messagebox.showinfo("Protocol Complete", result)

    def on_prompt(self):
        prompt = simpledialog.askstring("AGI Prompt", "Enter text to inject into the AGI core:")
        if prompt:
            self.log_to_console(f"User injected prompt: '{prompt}'")
            response = self.agi_core.process_input(prompt)
            self.log_to_console(f"AGI RESPONSE: {response}")
            messagebox.showinfo("AGI Response", response)

    def update_status(self):
        while True:
            try:
                summary = self.agi_core.summary()
                embedding = self.agi_core.get_mesh_embedding()
                status_str = (f"  Core Status: {summary.get('status', 'N/A')}\n"
                              f"  Active Nodes: {summary.get('active_nodes', 'N/A')}\n"
                              f"  Threat Level: {summary.get('threat_level', 'N/A')}\n\n"
                              f"  Mesh Embedding (Mean):\n"
                              f"    X: {embedding[0]:.4f}, Y: {embedding[1]:.4f}, Z: {embedding[2]:.4f}")
                self.status_text.config(text=status_str)
            except Exception as e:
                self.status_text.config(text=f"Error fetching status:\n{e}")
            time.sleep(5)

    def start_monitoring(self):
        self.log_to_console("TheLight GUI Initialized. AGI monitoring active.")
        monitor_thread = threading.Thread(target=self.update_status, daemon=True)
        monitor_thread.start()

if __name__ == '__main__':
    print("Initializing GUI in standalone test mode...")
    mock_core = MockAGICore()
    app = TheLight(agi_core=mock_core)
    app.mainloop()