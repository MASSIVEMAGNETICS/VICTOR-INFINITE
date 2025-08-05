# File: bandocodex/gui/the_light_gui.py
# Source: VictorPrimeEmergentFusionMonolithGUI_PRIME_OMEGA_GUI_STABLE_v2_0_0.py

import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog
import threading
import time
import os
import shutil
import sys

class MockAGICore:
    def get_mesh_embedding(self): return [random.random() for _ in range(3)]
    def summary(self): return {"status": "Nominal", "active_nodes": 1337, "threat_level": "None"}
    def process_input(self, text): return f"AGI processed: '{text}'. Output is a fractal melody."

def trigger_self_replication(source_script_path: str) -> str:
    try:
        if not os.path.exists(source_script_path):
            return "Error: Source script not found."
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        replication_dir = os.path.join(os.path.dirname(source_script_path), "generations")
        os.makedirs(replication_dir, exist_ok=True)
        new_gen_name = f"{timestamp}_gen_{os.path.basename(source_script_path)}"
        dest_path = os.path.join(replication_dir, new_gen_name)
        shutil.copy2(source_script_path, dest_path)
        return f"SUCCESS: New generation created at: {dest_path}"
    except Exception as e:
        return f"FATAL: Self-replication failed. Error: {e}"

class TheLight(tk.Tk):
    def __init__(self, agi_core):
        super().__init__()
        self.agi_core = agi_core
        self.title("V.I.C.T.O.R Prime - TheLight GUI")
        self.geometry("1200x800")
        self.configure(bg="#1a1a1a")
        self.create_widgets()
        self.start_monitoring()

    def log(self, msg):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.console.config(state=tk.DISABLED)
        self.console.see(tk.END)

    def create_widgets(self):
        main_frame = tk.Frame(self, bg="#1a1a1a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Control Panel
        ctrl_frame = tk.Frame(main_frame, bg="#2b2b2b")
        ctrl_frame.pack(fill=tk.X, pady=5)
        tk.Button(ctrl_frame, text="Trigger Self-Replication", bg="#ff4500", fg="white", command=self.on_replicate).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(ctrl_frame, text="Inject Prompt", bg="#00aaff", fg="white", command=self.on_prompt).pack(side=tk.LEFT, padx=10, pady=10)
        # Status & Console
        mid_frame = tk.Frame(main_frame, bg="#1a1a1a")
        mid_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        status_frame = tk.LabelFrame(mid_frame, text="AGI Core Status", bg="#2b2b2b", fg="white", padx=10, pady=10)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.status_text = tk.Label(status_frame, text="Initializing...", fg="cyan", bg="#2b2b2b", font=("Courier", 12), justify=tk.LEFT)
        self.status_text.pack(anchor="w")
        console_frame = tk.LabelFrame(mid_frame, text="Live Log", bg="#2b2b2b", fg="white", padx=10, pady=10)
        console_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, bg="#000000", fg="#00ff00", state=tk.DISABLED)
        self.console.pack(fill=tk.BOTH, expand=True)

    def on_replicate(self):
        if messagebox.askyesno("Confirm", "Triggering self-replication is irreversible. Proceed?"):
            self.log("Replication initiated by user...")
            result = trigger_self_replication(__file__)
            self.log(result)

    def on_prompt(self):
        prompt = simpledialog.askstring("AGI Prompt", "Enter text to inject into the AGI core:")
        if prompt:
            self.log(f"User injected prompt: '{prompt}'")
            response = self.agi_core.process_input(prompt)
            self.log(f"AGI RESPONSE: {response}")

    def monitor_status(self):
        while True:
            try:
                summary = self.agi_core.summary()
                emb = self.agi_core.get_mesh_embedding()
                status = f"Core Status: {summary.get('status', 'N/A')}\n" \
                         f"Active Nodes: {summary.get('active_nodes', 'N/A')}\n\n" \
                         f"Mesh Embedding:\n" \
                         f"X: {emb[0]:.4f}, Y: {emb[1]:.4f}, Z: {emb[2]:.4f}"
                self.status_text.config(text=status)
            except Exception as e:
                self.status_text.config(text=f"Error fetching status:\n{e}")
            time.sleep(5)

    def start_monitoring(self):
        self.log("TheLight GUI Initialized. AGI monitoring active.")
        threading.Thread(target=self.monitor_status, daemon=True).start()

def launch():
    print("Initializing GUI...")
    app = TheLight(agi_core=MockAGICore())
    app.mainloop()

if __name__ == '__main__':
    launch()