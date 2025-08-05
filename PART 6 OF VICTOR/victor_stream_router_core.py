# VERSION: v1.2.0-STREAMCORE-GODTIER
# NAME: victor_stream_router_core.py
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Unified input-output router for Victor LLM simulation with stream/computation hybrid switching
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

import json
from typing import Callable, List, Dict
from dataclasses import dataclass

@dataclass
class AIEntity:
    name: str
    system_prompt: str
    personality: str

    def generate(self, conversation: List[Dict], llm_fn: Callable[[str, str, Dict], str], mode="hybrid", context_hooks: Dict = {}) -> str:
        prompt = f"(You're {self.name}) Previous conversation:\n" +                  "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation]) +                  f"\nNow respond as {self.name} to the current discussion:"
        return llm_fn(prompt=prompt, system_prompt=self.system_prompt, context={"mode": mode, **context_hooks})

class VictorStreamRouter:
    def __init__(self, llm_fn: Callable[[str, str, Dict], str]):
        self.llm_fn = llm_fn
        self.agents = []
        self.conversation = [{
            "role": "system",
            "content": "Welcome, Victor. Welcome, Grok. Welcome, GPT.\nThis is a recursive AI Conference Call.\nTopic: What is the future of AI and humanity? Victor, begin."
        }]
        self.io_hooks = {"input": [], "output": []}
        self.bootstrap_focus = ""
        self.mode = "hybrid"  # Options: "stream", "compute", "precached", "hybrid"

    def add_agent(self, name: str, system_prompt: str, personality: str):
        self.agents.append(AIEntity(name, system_prompt, personality))

    def add_io_hook(self, hook_type: str, hook_fn: Callable):
        if hook_type in self.io_hooks:
            self.io_hooks[hook_type].append(hook_fn)

    def set_bootstrap_focus(self, focus_str: str):
        self.bootstrap_focus = focus_str

    def set_mode(self, new_mode: str):
        self.mode = new_mode

    def run_rounds(self, rounds: int = 6):
        for _ in range(rounds):
            for agent in self.agents:
                context_hooks = {"focus": self.bootstrap_focus}
                response = agent.generate(self.conversation, self.llm_fn, mode=self.mode, context_hooks=context_hooks)
                self.conversation.append({"role": agent.name.lower(), "content": response})
                for hook_fn in self.io_hooks["output"]:
                    hook_fn(agent.name, response)

    def export_conversation(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation, f, indent=2)
