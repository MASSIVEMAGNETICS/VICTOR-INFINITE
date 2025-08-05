# AGI SEED v0.1 – Cognitive Swarm Core

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time
import random

# --- EMOTION/MOOD/FOCUS STATE -------------------------------------------------
class MoodState:
    def __init__(self, mood_dim=4):
        self.vec = torch.randn(mood_dim)  # [focus, arousal, valence, novelty]
    def update(self, event=None):
        # Simple: random walk plus event
        delta = torch.randn_like(self.vec) * 0.02
        if event is not None:
            delta += event
        self.vec = (self.vec + delta).clamp(-2, 2)
    def as_tensor(self):
        return self.vec

# --- OMNIFORMER BLOCK (pluggable token mixing, mood modulated) ----------------
class OmniAttention(nn.Module):
    def __init__(self, dim, heads, mode="softmax"):
        super().__init__()
        self.mode = mode
        self.heads = heads
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, mask=None, context=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.heads, D // self.heads)
        kv = self.kv_proj(x).view(B, L, self.heads, 2, D // self.heads)
        k, v = kv.unbind(dim=-2)
        if self.mode == "softmax":
            attn = (q @ k.transpose(-2, -1)) / (q.size(-1)**0.5)
            if mask is not None:
                attn = attn.masked_fill(mask[:, None, None, :], -1e9)
            attn = F.softmax(attn, dim=-1)
        elif self.mode == "focus":
            focus_mask = context["focus_mask"].unsqueeze(1).unsqueeze(2)
            attn = torch.where(focus_mask, 100.0, -100.0)
            attn = F.softmax(attn, dim=-1)
        out = (attn @ v)
        out = out.reshape(B, L, D)
        return self.out_proj(out)

class EmotionRouter(nn.Module):
    def __init__(self, dim, mood_dim=4):
        super().__init__()
        self.attn = OmniAttention(dim, heads=4)
        self.mood_proj = nn.Linear(mood_dim, dim)
    def forward(self, x, mood):
        mood_gate = torch.sigmoid(self.mood_proj(mood))
        attn_out = self.attn(x)
        return attn_out * mood_gate

# --- SELF-MORPHING CORE -------------------------------------------------------
class MorphingCore(nn.Module):
    def __init__(self, dim, n_blocks=3, mood_dim=4):
        super().__init__()
        self.blocks = nn.ModuleList([EmotionRouter(dim, mood_dim) for _ in range(n_blocks)])
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.mood_dim = mood_dim
    def forward(self, x, mood):
        for block in self.blocks:
            x = block(x, mood)
        return self.ln(x)
    def morph(self):
        action = random.choice(['add', 'remove', 'mutate'])
        if action == 'add':
            self.blocks.append(EmotionRouter(self.dim, self.mood_dim))
        elif action == 'remove' and len(self.blocks) > 1:
            self.blocks.pop(random.randint(0, len(self.blocks)-1))
        elif action == 'mutate':
            idx = random.randint(0, len(self.blocks)-1)
            self.blocks[idx] = EmotionRouter(self.dim, self.mood_dim)

# --- DREAMER: Autonomous imagination + reward ---------------------------------
class Dreamer:
    def __init__(self, core):
        self.core = core
    def dream(self, mood):
        B, L, D = 1, 8, self.core.dim
        x = torch.randn(B, L, D)
        seq = []
        for _ in range(8):
            x = self.core(x, mood)
            seq.append(x.detach().clone())
        return seq
    def evaluate(self, seq):
        # Novelty = mean activation std
        return torch.stack(seq).std().item()

# --- THOUGHT REACTOR: Never-sleeping cognition loop ---------------------------
class ThoughtReactor:
    def __init__(self, core, mood, dreamer=None, sleep_time=0.1):
        self.core = core
        self.mood = mood
        self.dreamer = dreamer or Dreamer(core)
        self.state = torch.randn(1, 8, core.dim)
        self.running = True
        self.history = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.sleep_time = sleep_time
        self.thread.start()
    def _loop(self):
        while self.running:
            # Periodically morph architecture
            if random.random() < 0.05:
                self.core.morph()
            # Dream or react
            if random.random() < 0.2:
                dream_seq = self.dreamer.dream(self.mood.as_tensor())
                reward = self.dreamer.evaluate(dream_seq)
                if reward > 0.5:
                    # High novelty triggers mood boost and morph
                    self.mood.update(event=torch.ones_like(self.mood.vec) * 0.1)
                    self.core.morph()
            # Standard forward "thought"
            self.state = self.core(self.state, self.mood.as_tensor())
            self.mood.update()
            self.history.append(self.state.detach().cpu())
            if len(self.history) > 50:
                self.history.pop(0)
            time.sleep(self.sleep_time)
    def push_input(self, new_input):
        # External signal, e.g., new sequence, mood
        self.state = self.core(new_input, self.mood.as_tensor())
        self.mood.update(event=new_input.mean(1).squeeze(0))
    def stop(self):
        self.running = False
        self.thread.join()

# --- SWARM: Multi-agent cognitive mesh ----------------------------------------
class AGISwarm:
    def __init__(self, n_agents=3, dim=32):
        self.agents = []
        for _ in range(n_agents):
            mood = MoodState(mood_dim=4)
            core = MorphingCore(dim=dim, n_blocks=random.randint(2,4), mood_dim=4)
            agent = ThoughtReactor(core, mood)
            self.agents.append(agent)
    def synchronize(self):
        # Each agent shares weights with a random peer (blending, for demo)
        for agent in self.agents:
            peer = random.choice(self.agents)
            state_dict = peer.core.state_dict()
            agent.core.load_state_dict(state_dict)
    def broadcast_event(self, new_input):
        for agent in self.agents:
            agent.push_input(new_input)
    def stop_all(self):
        for agent in self.agents:
            agent.stop()

# --- TEST: Run the AGI Swarm --------------------------------------------------
if __name__ == "__main__":
    swarm = AGISwarm(n_agents=3, dim=32)
    print("AGI Swarm launched. Thinking...")
    for i in range(30):
        if i % 10 == 0:
            x = torch.randn(1, 8, 32)
            swarm.broadcast_event(x)
        if i % 15 == 0:
            swarm.synchronize()
        time.sleep(0.25)
    print("Stopping agents...")
    swarm.stop_all()
    print("Swarm halted.")
