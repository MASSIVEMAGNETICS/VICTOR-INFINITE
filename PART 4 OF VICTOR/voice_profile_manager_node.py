# File: voice_profile_manager_node.py
# Version: v1.0.0
# Description: Node for listing, renaming, deleting, and backing up .npz voice profiles
# Author: Bando Bandz AI Ops

import os
import shutil

class VoiceProfileManagerNode:
    """
    Manage Victor's voice profile (.npz) library.
    List, rename, delete, and back up embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (
                    "STRING",
                    {
                        "default": "list",
                        "options": ["list", "rename", "delete", "backup"]
                    }
                ),
                "profile_name": ("STRING", {"default": ""}),
                "new_name": ("STRING", {"default": ""}),
                "directory": ("STRING", {"default": "custom_voices/"}),
                "backup_directory": ("STRING", {"default": "custom_voices/backup/"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "manage_profiles"
    CATEGORY = "audio/voice_management"

    def manage_profiles(self, action, profile_name, new_name, directory, backup_directory):
        try:
            os.makedirs(directory, exist_ok=True)
            if action == "list":
                profiles = [
                    f for f in os.listdir(directory)
                    if f.endswith(".npz") and os.path.isfile(os.path.join(directory, f))
                ]
                return ("\n".join(profiles),)

            path = os.path.join(directory, f"{profile_name}.npz")

            if action == "rename":
                if not os.path.exists(path):
                    return (f"[Error] Profile '{profile_name}' not found.",)
                new_path = os.path.join(directory, f"{new_name}.npz")
                os.rename(path, new_path)
                return (f"[Success] Renamed to '{new_name}.npz'",)

            elif action == "delete":
                if not os.path.exists(path):
                    return (f"[Error] Profile '{profile_name}' not found.",)
                os.remove(path)
                return (f"[Success] Deleted '{profile_name}.npz'",)

            elif action == "backup":
                if not os.path.exists(path):
                    return (f"[Error] Profile '{profile_name}' not found.",)
                os.makedirs(backup_directory, exist_ok=True)
                shutil.copy2(path, os.path.join(backup_directory, f"{profile_name}.npz"))
                return (f"[Success] Backed up to '{backup_directory}'",)

            else:
                return (f"[Error] Unknown action: {action}",)

        except Exception as e:
            return (f"[VoiceProfileManager::Error] {str(e)}",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "VoiceProfileManagerNode": VoiceProfileManagerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoiceProfileManagerNode": "Audio: Voice Profile Manager"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
