# node_loader.py
# Version: 1.2.0 — Victor Plugin Loader (with logging, state, persistence, error handling)
# Drop into ComfyUI/custom_nodes/
# Requires: victor_logger.py in same dir or importable path

import os
import importlib.util
import sys
import uuid
import traceback
from pathlib import Path
from typing import Any

from nodes import Node, NodeInput, NodeOutput, NodeCategory, ComfyUIControl

# --- Import Logger ---
from victor_logger import log_event

STATE_STORE = {}

class PluginLoaderNode(Node):
    def __init__(self):
        super().__init__()
        self.module_path = None
        self.module_class = None
        self.instance = None
        self.memory_slot = None
        self.stateful = False
        self.has_error = False

        self.name = "Victor Plugin Loader"
        self.category = NodeCategory.CUSTOM
        self.description = "Loads a VictorModule class from .py, autowires ports, logs all actions, supports state"

        self.add_control("Load Python File", "file", filetypes=[".py"], callback=self.load_module)
        self.add_control("Stateful?", "checkbox", default=False, callback=self.toggle_state)
        self.add_control("Memory Slot ID", "text", default="default_slot", callback=self.set_memory_slot)

    def set_memory_slot(self, val):
        self.memory_slot = val
        log_event(f"[MemorySlot] Renamed to: {val}")

    def toggle_state(self, value: bool):
        self.stateful = value
        log_event(f"[ToggleStateful] Set to: {value}")

    def load_module(self, path: str):
        try:
            if not os.path.exists(path):
                log_event(f"[LoadModule][ERROR] File does not exist: {path}")
                self.description = "[ERROR] Invalid path"
                self.has_error = True
                return

            self.module_path = path
            module_name = f"plugin_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)

            if not hasattr(mod, "VictorModule"):
                log_event(f"[LoadModule][ERROR] No VictorModule class found in {path}")
                self.description = "[ERROR] No VictorModule class"
                self.has_error = True
                return

            self.module_class = mod.VictorModule
            self.instance = self.module_class()

            metadata = self.instance.get_metadata()
            self.title = metadata.get("title", "Victor Module")
            self.description = metadata.get("description", "")
            self.memory_slot = self.title.lower().replace(" ", "_")

            self.inputs.clear()
            self.outputs.clear()
            for input_name in metadata.get("inputs", []):
                self.add_input(input_name, Any)
            for output_name in metadata.get("outputs", []):
                self.add_output(output_name, Any)

            log_event(f"[LoadModule] Loaded '{self.title}' | Inputs: {metadata['inputs']} | Outputs: {metadata['outputs']}")
            self.has_error = False

        except Exception as e:
            log_event(f"[LoadModule][EXCEPTION] {e}")
            self.description = f"[ERROR] {e}"
            self.has_error = True
            traceback.print_exc()

    def execute(self, *args):
        try:
            if self.instance is None:
                log_event("[Execute][ERROR] No module loaded")
                return [None] * len(self.outputs)

            input_data = [arg for arg in args]
            log_event(f"[Execute] Running '{self.title}' | Input: {input_data} | Stateful: {self.stateful}")

            if self.stateful:
                memory = STATE_STORE.get(self.memory_slot, {})
                result = self.instance.forward(input_data, memory)
                STATE_STORE[self.memory_slot] = memory
            else:
                result = self.instance.forward(input_data)

            if not isinstance(result, (list, tuple)):
                result = [result]

            log_event(f"[Execute] Result: {result}")
            return result

        except Exception as e:
            log_event(f"[Execute][EXCEPTION] {e}")
            self.description = f"[ERROR during execute] {e}"
            traceback.print_exc()
            return [None] * len(self.outputs)

    def serialize(self):
        log_event(f"[Serialize] path={self.module_path} stateful={self.stateful} slot={self.memory_slot}")
        return {
            "module_path": self.module_path,
            "stateful": self.stateful,
            "memory_slot": self.memory_slot
        }

    def deserialize(self, data):
        self.stateful = data.get("stateful", False)
        self.memory_slot = data.get("memory_slot", "default_slot")
        log_event(f"[Deserialize] Restoring path={data.get('module_path')} stateful={self.stateful} slot={self.memory_slot}")
        if "module_path" in data:
            self.load_module(data["module_path"])


NODE_CLASS_MAPPINGS = {
    "VictorPluginLoader": PluginLoaderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VictorPluginLoader": "🧠 Victor Plugin Loader"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
