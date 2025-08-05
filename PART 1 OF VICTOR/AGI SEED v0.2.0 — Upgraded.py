# AGI SEED v0.2 — Upgraded: Logging, Sensors, Swappable Mixers, Lifelong Memory, Multi-Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time
import random
import json
import os
import matplotlib.pyplot as plt

# --- LOGGER & MEMORY --------------------------------------------------------
class PersistentLogger:
    def __init__(self, logdir="agi_logs"):
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self.buffers = {}

    def log(self, agent_id, key, value):
        buf = self.buffers.setdefault(agent_id, {})
        buf.setdefault(key, []).append(value)
        # Write to file
        with open(f"{self.logdir}/agent_{agent_id}.json", "w") as f:
            json.dump(buf, f)

    def plot(self, agent_id, key):
        buf = self.buffers.get(agent_id, {})
        data = buf.get(key, [])
        plt.plot(data)
        plt.title(f"Agent {agent_id} — {key}")
        plt.show()

class PersistentMemory:
    def __init__(self, agent_id, memdir="memories"):
        os.makedirs(memdir, exist_ok=True)
        self.memfile = f"{memdir}/agent_{agent_id}.json"
        self.history = []
        self._load()
    def store(self, state):
        self.history.append(state.tolist())
        with open(self.memfile, "w") as f:
            json.dump(self.history, f)
    def _load(self):
        if os.path.exists(self.memfile):
            with open(self.memfile) as f:
                self.history = json.load(f)

# --- MOOD, GOAL, AND MESSAGING ----------------------------------------------
class MoodState:
    def __init__(self, mood_dim=4):
        self.vec = torch.randn(mood_dim)
    def update(self, event=None):
        delta = torch.randn_like(self.vec) * 0.02
        if event is not None:
            delta += event
        self.vec = (self.vec + delta).clamp(-2, 2)
    def as_tensor(self):
        return self.vec

class GoalState:
    def __init__(self, dim=4):
        self.vec = torch.randn(dim)
    def mutate(self):
        self.vec += torch.randn_like(self.vec) * 0.01

class Messenger:
    def __init__(self):
        self.messages = []
    def send(self, msg):
        self.messages.append(msg)
    def receive(self):
        if self.messages:
            return self.messages.pop(0)
        return None

# --- TOKEN MIXERS -----------------------------------------------------------
class PoolingMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        pooled = x.mean(dim=1, keepdim=True)
        return self.proj(pooled).expand_as(x)

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

# --- EMOTION/ROUTER BLOCK ---------------------------------------------------
class EmotionRouter(nn.Module):
    def __init__(self, dim, mood_dim=4, mixer_type="attn"):
        super().__init__()
        if mixer_type == "attn":
            self.mixer = OmniAttention(dim, heads=4)
        elif mixer_type == "pool":
            self.mixer = PoolingMixer(dim)
        else:
            raise ValueError("Unknown mixer type")
        self.mood_proj = nn.Linear(mood_dim, dim)
    def forward(self, x, mood):
        mood_gate = torch.sigmoid(self.mood_proj(mood))
        mix_out = self.mixer(x)
        return mix_out * mood_gate

# --- SELF-MORPHING CORE -----------------------------------------------------
class MorphingCore(nn.Module):
    def __init__(self, dim, n_blocks=3, mood_dim=4):
        super().__init__()
        self.block_types = ["attn", "pool"]
        self.blocks = nn.ModuleList([
            EmotionRouter(dim, mood_dim, mixer_type=random.choice(self.block_types))
            for _ in range(n_blocks)
        ])
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
            self.blocks.append(EmotionRouter(self.dim, self.mood_dim, mixer_type=random.choice(self.block_types)))
        elif action == 'remove' and len(self.blocks) > 1:
            self.blocks.pop(random.randint(0, len(self.blocks)-1))
        elif action == 'mutate':
            idx = random.randint(0, len(self.blocks)-1)
            self.blocks[idx] = EmotionRouter(self.dim, self.mood_dim, mixer_type=random.choice(self.block_types))

# --- DREAMER: Autonomous imagination + reward --------------------------------
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
        return torch.stack(seq).std().item()

# --- THOUGHT REACTOR: Never-sleeping cognition loop --------------------------
class ThoughtReactor:
    def __init__(self, core, mood, logger, mem, dreamer=None, sleep_time=0.1, agent_id=0, goal=None, messenger=None):
        self.core = core
        self.mood = mood
        self.logger = logger
        self.mem = mem
        self.agent_id = agent_id
        self.goal = goal or GoalState()
        self.messenger = messenger or Messenger()
        self.dreamer = dreamer or Dreamer(core)
        self.state = torch.randn(1, 8, core.dim)
        self.running = True
        self.history = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.sleep_time = sleep_time
        self.thread.start()
    def _loop(self):
        while self.running:
            # Morph core occasionally
            if random.random() < 0.05:
                self.core.morph()
                self.logger.log(self.agent_id, "morph", len(self.core.blocks))
            # Dream
            if random.random() < 0.2:
                dream_seq = self.dreamer.dream(self.mood.as_tensor())
                reward = self.dreamer.evaluate(dream_seq)
                self.logger.log(self.agent_id, "dream_reward", reward)
                if reward > 0.5:
                    self.mood.update(event=torch.ones_like(self.mood.vec) * 0.1)
                    self.core.morph()
            # Main "thought" step
            self.state = self.core(self.state, self.mood.as_tensor())
            self.mood.update()
            self.history.append(self.state.detach().cpu())
            self.logger.log(self.agent_id, "mood", self.mood.as_tensor().tolist())
            self.mem.store(self.state.squeeze(0))
            # Check mailbox
            msg = self.messenger.receive()
            if msg:
                self.logger.log(self.agent_id, "msg_recv", msg)
            if len(self.history) > 50:
                self.history.pop(0)
            time.sleep(self.sleep_time)
    def push_input(self, new_input):
        self.state = self.core(new_input, self.mood.as_tensor())
        self.mood.update(event=new_input.mean(1).squeeze(0))
    def stop(self):
        self.running = False
        self.thread.join()

# --- AGENT SWARM -------------------------------------------------------------
class AGISwarm:
    def __init__(self, n_agents=3, dim=32):
        self.logger = PersistentLogger()
        self.memories = []
        self.agents = []
        self.messengers = []
        for i in range(n_agents):
            mood = MoodState(mood_dim=4)
            core = MorphingCore(dim=dim, n_blocks=random.randint(2,4), mood_dim=4)
            mem = PersistentMemory(i)
            messenger = Messenger()
            agent = ThoughtReactor(core, mood, self.logger, mem, agent_id=i, messenger=messenger)
            self.agents.append(agent)
            self.memories.append(mem)
            self.messengers.append(messenger)
    def synchronize(self):
        # Share weights
        for agent in self.agents:
            peer = random.choice(self.agents)
            state_dict = peer.core.state_dict()
            agent.core.load_state_dict(state_dict)
            self.logger.log(agent.agent_id, "sync", 1)
    def broadcast_event(self, new_input):
        for agent in self.agents:
            agent.push_input(new_input)
    def stop_all(self):
        for agent in self.agents:
            agent.stop()
    def plot_moods(self):
        for i in range(len(self.agents)):
            self.logger.plot(i, "mood")
    def plot_morphs(self):
        for i in range(len(self.agents)):
            self.logger.plot(i, "morph")

# --- EXAMPLE LANGUAGE SENSOR -------------------------------------------------
class DummyTokenizer:
    def __init__(self):
        self.vocab_size = 64
    def encode(self, text, return_tensors=None):
        return torch.randint(0, self.vocab_size, (1, 8))

class LanguageSensor:
    def __init__(self, tokenizer, dim):
        self.tokenizer = tokenizer
        self.proj = nn.Linear(tokenizer.vocab_size, dim)
    def encode(self, text):
        ids = self.tokenizer.encode(text)
        onehot = F.one_hot(ids, num_classes=self.tokenizer.vocab_size).float()
        return self.proj(onehot)

# --- TEST: Run the upgraded AGI Swarm ----------------------------------------
if __name__ == "__main__":
    swarm = AGISwarm(n_agents=3, dim=32)
    print("AGI Swarm launched. Thinking...")
    tokenizer = DummyTokenizer()
    lang_sensor = LanguageSensor(tokenizer, dim=32)
    for i in range(30):
        if i % 10 == 0:
            # Send text event
            text = f"hello world {i}"
            x = lang_sensor.encode(text)
            swarm.broadcast_event(x)
        if i % 15 == 0:
            swarm.synchronize()
        time.sleep(0.25)
    print("Stopping agents...")
    swarm.stop_all()
    print("Swarm halted. Plotting logs...")
    swarm.plot_moods()
    swarm.plot_morphs()
