# victor_nodes/VictorDualNode.py
class VictorDualNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_1": ("STRING", {"default": "Input A"}),
                "input_2": ("STRING", {"default": "Input B"}),
                "modifier_mode": (["reverse", "uppercase", "double", "repeat"],),
                "intensity": ("INT", {"default": 2, "min": 1, "max": 10})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "process"
    CATEGORY = "Victor/FractalOps"

    def process(self, input_1, input_2, modifier_mode, intensity):
        output_1 = self.modify(input_1, modifier_mode, intensity)
        output_2 = self.modify(input_2, modifier_mode, intensity)
        return (output_1, output_2)

    def modify(self, text, mode, intensity):
        if mode == "reverse":
            return text[::-1]
        elif mode == "uppercase":
            return text.upper()
        elif mode == "double":
            return text + text
        elif mode == "repeat":
            return (text + " ") * intensity
        else:
            return text


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
