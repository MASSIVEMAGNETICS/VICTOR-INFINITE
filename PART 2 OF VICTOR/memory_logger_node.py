# File: memory_logger_node.py
# Version: v2.0.0-FP
# Description: Logs events into Victor’s memory with redundancy, healing, and recovery
# Author: Bando Bandz AI Ops

import os
import json
import uuid
import time
from shutil import copyfile

class MemoryLoggerNode:
    """
    Stores structured memory entries with unique IDs and timestamp.
    Includes self-healing, regenerating backups and replay-safe persistent history.
    """

    def __init__(self):
        self.base_dir = "victor_memory_log/"
        self.primary_log = os.path.join(self.base_dir, "memory_log.json")
        self.backup_log = os.path.join(self.base_dir, "memory_log_backup.json")
        self.regen_log = os.path.join(self.base_dir, "memory_log_regen.json")
        self.memory = self._load_memory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "entry_type": ("STRING", {"default": "evaluation"}),  # e.g., directive, emotion, narrative, trend, action
                "content": ("STRING", {"default": ""}),
                "tag": ("STRING", {"default": ""})  # optional context tag
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("entry_id",)
    FUNCTION = "log_memory"
    CATEGORY = "memory/logging"

    def log_memory(self, entry_type, content, tag):
        try:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": entry_type,
                "tag": tag,
                "content": content.strip()
            }

            self.memory.append(entry)
            self._save_all()

            print(f"[Victor::MemoryLogger] Entry saved: {entry['id']}")
            return (entry["id"],)

        except Exception as e:
            print(f"[Victor::MemoryLogger::Error] {str(e)}")
            return ("error_entry",)

    def _save_all(self):
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            with open(self.primary_log, "w") as f:
                json.dump(self.memory, f, indent=2)
            copyfile(self.primary_log, self.backup_log)
            copyfile(self.primary_log, self.regen_log)
        except Exception as e:
            print(f"[Victor::MemoryLogger::SaveError] {str(e)}")

    def _load_memory(self):
        try:
            if os.path.exists(self.primary_log):
                with open(self.primary_log, "r") as f:
                    return json.load(f)
            elif os.path.exists(self.backup_log):
                print("[Victor::MemoryLogger] Restoring from backup...")
                with open(self.backup_log, "r") as f:
                    return json.load(f)
            elif os.path.exists(self.regen_log):
                print("[Victor::MemoryLogger] Attempting regeneration...")
                return self._clean_partial_log(self.regen_log)
            return []
        except Exception as e:
            print(f"[Victor::MemoryLogger::LoadError] {str(e)}")
            return []

    def _clean_partial_log(self, path):
        try:
            with open(path, "r") as f:
                raw = f.read()
            lines = raw.split("}\n")
            recovered = []
            for line in lines:
                if "content" in line and "timestamp" in line:
                    try:
                        recovered.append(json.loads(line + "}"))
                    except:
                        continue
            print(f"[Victor::MemoryLogger] Recovered {len(recovered)} entries.")
            return recovered
        except Exception as e:
            print(f"[Victor::MemoryLogger::RegenError] {str(e)}")
            return []

# Node registration
NODE_CLASS_MAPPINGS = {
    "MemoryLoggerNode": MemoryLoggerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryLoggerNode": "Memory: Logger (Resilient v2)"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
