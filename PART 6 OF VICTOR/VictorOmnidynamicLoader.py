
# VictorOmnidynamicLoader.py — God-Tier Loader with Strength & Weight
import os
import re
import traceback
import importlib.util
import sys

MODULES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules"))

class VictorOmnidynamicLoader:
    @classmethod
    def IS_CUSTOM_NODE(cls): return True

    @classmethod
    def INPUT_TYPES(cls):
        try:
            files = [f for f in os.listdir(MODULES_DIR) if f.endswith(".py")]
            cls.file_map = {f: os.path.join(MODULES_DIR, f) for f in files}
            file_choices = list(cls.file_map.keys())

            return {
                "required": {
                    "module_file": (file_choices, {"default": file_choices[0] if file_choices else ""}),
                    "symbol_name": ("STRING", {"default": ""}),
                    "input_1": ("STRING", {"default": "Hello"}),
                    "input_2": ("STRING", {"default": "World"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                    "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1})
                },
                "optional": {
                    "reload": ("BOOLEAN", {"default": False})
                }
            }
        except Exception as e:
            print(f"[VictorNode][ERROR] INPUT_TYPES failed: {e}")
            return {"required": {"input_1": ("STRING",), "input_2": ("STRING",)}}

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "run"
    CATEGORY = "Victor/Omnidynamic"

    @classmethod
    def NODE_NAME(cls): return "🧠 Victor Omnidynamic Loader"

    def __init__(self):
        self.cache = {}

    def run(self, module_file, symbol_name, input_1, input_2, strength=1.0, weight=1.0, reload=False):
        try:
            path = os.path.join(MODULES_DIR, module_file)
            if not os.path.exists(path):
                return ("[Victor] File not found", "ERROR")

            if reload or path not in self.cache:
                self.cache[path] = self._load_symbols(path)

            symbols = self.cache[path]
            if symbol_name not in symbols:
                return (f"[Victor] Symbol not found: {symbol_name}", "ERROR")

            target = symbols[symbol_name]

            # Class
            if isinstance(target, type):
                instance = target()
                if hasattr(instance, "forward"):
                    result = instance.forward([input_1, input_2, strength, weight])
                elif hasattr(instance, "main"):
                    result = instance.main(input_1, input_2, strength, weight)
                else:
                    return (f"[Victor] No usable method on {symbol_name}", "ERROR")
            # Function
            elif callable(target):
                try:
                    result = target(input_1, input_2, strength=strength, weight=weight)
                except TypeError:
                    result = target(input_1, input_2)
            else:
                result = str(target)

            if not isinstance(result, tuple):
                result = (str(result), "")
            return result

        except Exception as e:
            traceback.print_exc()
            return (f"[Victor] CRASHED: {e}", "🔥💀")

    def _load_symbols(self, path):
        module_name = f"victor_dyn_{os.path.basename(path).replace('.py','')}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[VictorNode][ERROR] Could not exec {path}: {e}")
            return {}

        return {name: getattr(mod, name) for name in dir(mod)
                if not name.startswith("__") and not name.endswith("__")}

    @classmethod
    def WIDGETS(cls):
        return [
            {
                "name": "module_file",
                "type": "combo",
                "label": "📂 Choose Module",
            },
            {
                "name": "symbol_name",
                "type": "text",
                "label": "🔍 Symbol Name (function/class/var to run)"
            },
            {
                "name": "reload",
                "type": "checkbox",
                "label": "♻️ Reload File on Execute"
            }
        ]

NODE_CLASS_MAPPINGS = {
    "VictorOmnidynamicLoader": VictorOmnidynamicLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VictorOmnidynamicLoader": "🧠 Victor Omnidynamic Loader"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
