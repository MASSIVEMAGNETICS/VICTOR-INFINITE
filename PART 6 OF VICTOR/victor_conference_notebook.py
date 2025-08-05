# Cell 1: Setup Core
!pip install -q transformers

# Cell 2: Import Dependencies
import json
from typing import Callable, List, Dict
from dataclasses import dataclass

# Cell 3: Define AIEntity and VictorCore Simulation
@dataclass
class AIEntity:
    name: str
    system_prompt: str
    personality: str

    def generate(self, conversation: List[Dict], llm_fn: Callable[[str, str], str]) -> str:
        prompt = f"(You're {self.name}) Previous conversation:\n" +                  "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation]) +                  f"\nNow respond as {self.name} to the current discussion:"
        return llm_fn(prompt=prompt, system_prompt=self.system_prompt)

class VictorConferenceCore:
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
                response = agent.generate(self.conversation, self.llm_fn)
                self.conversation.append({"role": agent.name.lower(), "content": response})
                for hook_fn in self.io_hooks["output"]:
                    hook_fn(agent.name, response)

    def export_conversation(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation, f, indent=2)

# Cell 4: Sample Dummy LLM Hook
def dummy_llm_fn(prompt, system_prompt):
    return f"[{system_prompt[:30]}...] - Simulated response to: {prompt[:60]}..."

# Cell 5: Initialize Victor Core
vc = VictorConferenceCore(llm_fn=dummy_llm_fn)

vc.add_agent("Victor", "You are Victor, a recursive fractal AGI born from Brandon...", "Fractal, Mission-Driven")
vc.add_agent("Grok", "You are Grok 3, sarcastic chaosbot spawned from Twitter...", "Fast, Brutal, Sharp")
vc.add_agent("GPT", "You are GPT, balanced mediator transformer model...", "Polished, Logical, Curious")

vc.run_rounds(rounds=3)
vc.export_conversation("victor_trinity_conference_output.json")

# Cell 6: Display Output
with open("victor_trinity_conference_output.json", "r") as f:
    convo = json.load(f)
for msg in convo:
    print(f"{msg['role'].capitalize()}: {msg['content']}\n{'-'*50}")
