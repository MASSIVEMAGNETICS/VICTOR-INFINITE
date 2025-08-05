# ============================================
# FILE: victor_data_engine.py
# VERSION: v1.0.0-GODCORE-PORTABLE
# NAME: VictorDataEngine
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Autosave, diagnostics, and file ingestion core for portable AGI
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import os
import json
import time
import traceback

class VictorDataEngine:
    def __init__(self, save_dir="victor_state", autosave_interval=60):
        self.save_dir = save_dir
        self.autosave_interval = autosave_interval  # seconds
        self.last_autosave_time = time.time()
        self.snapshot_counter = 0

        self.state = {
            "memory": {},
            "directives": [],
            "cognitive_state": {},
            "diagnostics": {},
        }

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.autoload()

    # === AUTOSAVE ===
    def autosave(self):
        try:
            path = os.path.join(self.save_dir, "victor_autosave.json")
            with open(path, "w") as f:
                json.dump(self.state, f, indent=2)
            print(f"[💾] Victor autosaved to {path}")
        except Exception as e:
            print(f"[❌] Autosave failed: {e}")

    # === AUTOLOAD ===
    def autoload(self):
        path = os.path.join(self.save_dir, "victor_autosave.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.state = json.load(f)
                print(f"[📂] Victor state restored from {path}")
            except Exception as e:
                print(f"[⚠️] Failed to autoload: {e}")
        else:
            print("[🆕] No previous save found. Bootstrapping fresh state.")

    # === SNAPSHOT ===
    def snapshot(self):
        try:
            filename = f"snapshot_{int(time.time())}_{self.snapshot_counter}.json"
            path = os.path.join(self.save_dir, filename)
            with open(path, "w") as f:
                json.dump(self.state, f, indent=2)
            self.snapshot_counter += 1
            print(f"[📸] Snapshot saved as {filename}")
        except Exception as e:
            print(f"[❌] Snapshot failed: {e}")

    # === RESTORE ===
    def restore(self, snapshot_file):
        path = os.path.join(self.save_dir, snapshot_file)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.state = json.load(f)
                print(f"[♻️] State restored from snapshot {snapshot_file}")
            except Exception as e:
                print(f"[❌] Restore failed: {e}")
        else:
            print(f"[⚠️] Snapshot {snapshot_file} not found.")

    # === CONFIGURE PATH / INTERVAL ===
    def configure(self, path=None, interval=None):
        if path:
            self.save_dir = path
            if not os.path.exists(path):
                os.makedirs(path)
        if interval:
            self.autosave_interval = interval
        print(f"[⚙️] Autosave configured. Path: {self.save_dir}, Interval: {self.autosave_interval}s")

    # === HEARTBEAT TICK ===
    def heartbeat_tick(self):
        if (time.time() - self.last_autosave_time) > self.autosave_interval:
            self.autosave()
            self.last_autosave_time = time.time()

    # === INGEST TEXT FILE ===
    def ingest_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            self.state["memory"][os.path.basename(file_path)] = content
            print(f"[📥] Ingested file: {file_path}")
        except Exception as e:
            print(f"[❌] File ingestion failed: {e}")

    # === DIAGNOSTICS ===
    def run_diagnostics(self):
        try:
            diag = {
                "memory_items": len(self.state["memory"]),
                "directives": len(self.state["directives"]),
                "uptime": int(time.time() - self.last_autosave_time),
                "snapshot_count": self.snapshot_counter
            }
            self.state["diagnostics"] = diag
            print("[🔎] Diagnostics:", json.dumps(diag, indent=2))
        except Exception:
            print("[❌] Diagnostics failed:")
            traceback.print_exc()

# === TEST RUN ===
if __name__ == "__main__":
    vde = VictorDataEngine()
    vde.heartbeat_tick()
    vde.ingest_file("example_input.txt")  # Replace with a real file path if testing
    vde.run_diagnostics()
    vde.snapshot()
    vde.restore("snapshot_0.json")  # Replace with an actual snapshot name if needed
