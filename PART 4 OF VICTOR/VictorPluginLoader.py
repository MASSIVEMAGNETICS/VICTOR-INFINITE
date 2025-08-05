# victor_nodes/VictorPluginLoader.py
import os
import importlib.util
import sys
import json

PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules"))
VICTOR_PLUGIN_REGISTRY = {}

def load_victor_plugins():
    if not os.path.exists(PLUGIN_DIR):
        os.makedirs(PLUGIN_DIR)

    for filename in os.listdir(PLUGIN_DIR):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            file_path = os.path.join(PLUGIN_DIR, filename)

            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module

                manifest = load_manifest(module_name)
                VICTOR_PLUGIN_REGISTRY[module_name] = {
                    "module": module,
                    "manifest": manifest
                }

                print(f"[VictorLoader] Plugin Loaded: {module_name}")
            except Exception as e:
                print(f"[VictorLoader] Failed to load {module_name}: {e}")

def load_manifest(module_name):
    manifest_path = os.path.join(PLUGIN_DIR, module_name + ".manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except:
            print(f"[VictorLoader] Invalid manifest for {module_name}")
    return {}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
