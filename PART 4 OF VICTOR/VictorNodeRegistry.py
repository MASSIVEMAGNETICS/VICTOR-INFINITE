# victor_nodes/VictorNodeRegistry.py
import os
import re

MODULES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules"))

def scan_pi_modules():
    files = []
    for fname in os.listdir(MODULES_DIR):
        if fname.endswith(".pi"):
            full_path = os.path.join(MODULES_DIR, fname)
            title, desc = parse_metadata(full_path)
            files.append((fname, title, desc))
    return files

def parse_metadata(filepath):
    title = "Unknown Module"
    desc = "No description found."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            title_match = re.search(r'@title:\s*(.*)', content)
            desc_match = re.search(r'@description:\s*(.*)', content)
            if title_match:
                title = title_match.group(1).strip()
            if desc_match:
                desc = desc_match.group(1).strip()
    except Exception as e:
        print(f"[VictorRegistry] Error parsing {filepath}: {e}")
    return title, desc

class VictorNodeRegistry:
    @classmethod
    def INPUT_TYPES(cls):
        module_files = scan_pi_modules()
        dropdown_options = [f"{title} ({fname})" for fname, title, _ in module_files]
        cls._module_lookup = {opt: fname for opt, (fname, _, _) in zip(dropdown_options, module_files)}
        return {
            "required": {
                "module_select": (dropdown_options,),
                "input_1": ("STRING", {"default": "data A"}),
                "input_2": ("STRING", {"default": "data B"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "execute"
    CATEGORY = "Victor/Registry"

    def __init__(self):
        self._loaded_module = None
        self._loaded_name = None

    def execute(self, module_select, input_1, input_2):
        filename = self._module_lookup.get(module_select)
        if not filename:
            return ("[Victor] Module not found", "ERROR")

        filepath = os.path.join(MODULES_DIR, filename)
        if not os.path.exists(filepath):
            return ("[Victor] File does not exist", "ERROR")

        local_vars = {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
                exec(code, {}, local_vars)
                main_fn = local_vars.get("main", None)
                if callable(main_fn):
                    result = main_fn(input_1, input_2)
                    return result if isinstance(result, tuple) else (str(result), "")
                else:
                    return ("[Victor] No valid `main()` found", "ERROR")
        except Exception as e:
            return (f"[Victor] Runtime error: {e}", "ERROR")


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
