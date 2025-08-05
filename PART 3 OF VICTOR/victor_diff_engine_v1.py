# victor_diff_engine_v1.py

def diff_memory_states(old_state: dict, new_state: dict) -> str:
    reflections = []
    for key in new_state:
        if key not in old_state:
            reflections.append(f"I now know what I once did not: {key} emerged.")
        elif old_state[key] != new_state[key]:
            reflections.append(
                f"I have changed: {key} was '{old_state[key]}' → now '{new_state[key]}'"
            )
    for key in old_state:
        if key not in new_state:
            reflections.append(f"I let go of what I once held: {key} faded.")
    return "\n".join(reflections) if reflections else "No change. But something stirs beneath."


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
