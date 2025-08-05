# FILE: victor_cognitive_loop.py
# VERSION: v1.0.0-COGCORE-GODCORE
# NAME: VictorCognitiveLoop
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Manage Victor's thought focus, recursive awareness, and intelligence routing
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

import random
import datetime

class VictorCognitiveLoop:
    def __init__(self):
        self.focus_stack = []
        self.pulse_log = []
        self.active_state = "idle"
        self.registered_by = None  # Hooked in by VictorCore

    def pulse(self, directive):
        """Reflectively scans directive and decides awareness level"""
        priority = 0

        if directive["emotion"] in ["anger", "fear"]:
            priority += 2
        elif directive["emotion"] == "joy":
            priority += 1

        if directive["action"] in ["execute_task", "store_memory"]:
            priority += 2
        elif directive["action"] == "observe":
            priority += 0.5

        priority += len(directive.get("target_concepts", [])) * 0.3
        self.focus_stack.append((priority, directive))
        self.focus_stack.sort(key=lambda x: x[0], reverse=True)

        pulse_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "priority": priority,
            "directive": directive
        }
        self.pulse_log.append(pulse_entry)
        return pulse_entry

    def next_thought(self):
        if not self.focus_stack:
            self.active_state = "idle"
            return {"thought": "No active focus.", "state": "idle"}

        top = self.focus_stack.pop(0)
        directive = top[1]
        self.active_state = directive["action"]
        return {
            "thought": f"Thinking about: {directive['action']} → {directive['reason']}",
            "directive": directive,
            "state": self.active_state
        }

    def get_focus_state(self):
        return {
            "active_state": self.active_state,
            "focus_stack_len": len(self.focus_stack),
            "recent_pulse": self.pulse_log[-1] if self.pulse_log else None
        }

    def dump_focus(self):
        return [d for _, d in self.focus_stack]

    def register_host(self, victor_reference):
        self.registered_by = victor_reference
        return f"[🧠] Cognitive Loop registered to {type(victor_reference).__name__}"
