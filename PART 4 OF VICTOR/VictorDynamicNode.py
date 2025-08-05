# victor_nodes/VictorDynamicNode.py
import os
import importlib.util

class VictorDynamicNode:
    @classmethod
    def IS_CUSTOM_NODE(cls):
        return True  # So Comfy lets us use buttons

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_1": ("STRING", {"default": "Hello"}),
                "input_2": ("STRING", {"default": "World"}),
            },
            "optional": {
                "file_path": ("STRING", {"default": "modules/sample_plugin.pi"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "run_dynamic"
    CATEGORY = "Victor/Dynamic"

    @classmethod
    def NODE_NAME(cls):
        return "🧠 Victor Dynamic Loader"

    def __init__(self):
        self.logic = None
        self.last_loaded = ""

    def run_dynamic(self, input_1, input_2, file_path=None):
        if file_path and file_path != self.last_loaded:
            self.logic = self.load_logic(file_path)
            self.last_loaded = file_path

        if self.logic:
            try:
                result = self.logic(input_1, input_2)
                return result if isinstance(result, tuple) else (str(result), "")
            except Exception as e:
                return (f"[Victor] Logic failed: {e}", "ERROR")
        return ("[Victor] No logic loaded", "ERROR")

    def load_logic(self, file_path):
        if not os.path.exists(file_path):
            print(f"[Victor] File not found: {file_path}")
            return None

        if file_path.endswith(".py") or file_path.endswith(".pi"):
            # Load as dynamic Python logic
            local_vars = {}
            with open(file_path, "r") as f:
                try:
                    code = f.read()
                    exec(code, {}, local_vars)
                    return local_vars.get("main", None)
                except Exception as e:
                    print(f"[Victor] Load failed: {e}")
        elif file_path.endswith(".json"):
            # JSON-driven logic structure? Add support here
            pass

        return None

    @classmethod
    def WIDGETS(cls):
        return [
            {
                "name": "file_path",
                "type": "file",
                "label": "📂 Load Logic File",
                "filetypes": [".pi", ".py", ".json"],
                "default": "modules/sample_plugin.pi"
            }
        ]


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
