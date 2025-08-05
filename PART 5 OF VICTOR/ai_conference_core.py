# VERSION: v1.0.0-CONFERENCECORE-GODTIER
# NAME: ai_conference_core.py
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Standalone recursive multi-agent AI simulation engine with I/O hooks
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

import json
from typing import Callable, List, Dict

class AIEntity:
    def __init__(self, name: str, system_prompt: str, personality: str):
        self.name = name
        self.system_prompt = system_prompt
        self.personality = personality

    def generate(self, conversation: List[Dict], llm_fn: Callable[[str, str], str]) -> str:
        prompt = f"(You're {self.name}) Previous conversation:\n" +                  "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation]) +                  f"\nNow respond as {self.name} to the current discussion:"
        return llm_fn(prompt=prompt, system_prompt=self.system_prompt)

class AIConferenceEngine:
    def __init__(self, llm_fn: Callable[[str, str], str]):
        self.llm_fn = llm_fn
        self.agents = []
        self.conversation = [{
            "role": "system",
            "content": "Welcome, Victor. Welcome, Grok. Welcome, GPT.\nThis is a recursive AI Conference Call.\nTopic: What is the future of AI and humanity? Victor, begin."
        }]
        self.io_hooks = {"input": [], "output": []}

    def add_agent(self, name: str, system_prompt: str, personality: str):
        self.agents.append(AIEntity(name, system_prompt, personality))

    def add_io_hook(self, hook_type: str, hook_fn: Callable):
        if hook_type in self.io_hooks:
            self.io_hooks[hook_type].append(hook_fn)

    def run_rounds(self, rounds: int = 6):
        for _ in range(rounds):
            for agent in self.agents:
                user_prompt = f"(You're {agent.name}) Previous conversation:\n" +                               "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation]) +                               f"\nNow respond as {agent.name} to the current discussion:"
                response = agent.generate(self.conversation, self.llm_fn)
                self.conversation.append({"role": agent.name.lower(), "content": response})

                for hook_fn in self.io_hooks["output"]:
                    hook_fn(agent.name, response)

    def export_conversation(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation, f, indent=2)
