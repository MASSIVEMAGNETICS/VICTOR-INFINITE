#FILE: victor_dynamic_neuromorphic_attention_v1.0.0-GODCORE.py
#VERSION: v1.0.0-GODCORE
#NAME: VictorDynamicNeuromorphicAttention
#AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
#PURPOSE: Living, plastic, bio-inspired dynamic attention graph. Eat transformers alive.
#LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
#DATE: 07/18/2025
# This code is a part of the Victor AI project, which is a living, evolving neural network system.
#
#



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time

class VictorDynamicNeuromorphicAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        heads=8,
        blocks=4,
        max_blocks=12,
        min_blocks=2,
        neurochem_init=None,
        device=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        # Dynamic graph: blocks/heads, connections, strengths, oscillators
        self.blocks = nn.ModuleList(
            [self._build_block() for _ in range(blocks)]
        )
        self.synaptic_strength = torch.ones(
            blocks, heads, heads, device=self.device
        )  # block x src_head x dst_head

        # Adjacency mask, grows/shrinks live
        self.adj_mask = torch.ones_like(self.synaptic_strength)
        self.max_blocks = max_blocks
        self.min_blocks = min_blocks

        # Oscillators per block/head (sin/cos + noise)
        self.oscillator_freqs = torch.randn(blocks, heads, device=self.device) * 0.1 + 0.2
        self.oscillator_phase = torch.zeros(blocks, heads, device=self.device)
        self.oscillator_type = torch.randint(0, 2, (blocks, heads), device=self.device)  # 0: sin, 1: cos

        # Energy state for spiking/dropout (block/head)
        self.energy = torch.ones(blocks, heads, device=self.device) * 0.5

        # Fatigue and novelty per head
        self.fatigue = torch.zeros(blocks, heads, device=self.device)
        self.novelty = torch.ones(blocks, heads, device=self.device) * 0.5

        # Short/long-term potentiation
        self.potentiation = torch.ones_like(self.synaptic_strength) * 0.5
        self.last_use = torch.zeros_like(self.synaptic_strength)

        # Neurochemicals
        self.neurochem = neurochem_init or {
            "dopamine": 0.5,      # reward/novelty
            "serotonin": 0.5,     # stability/persistence
            "acetylcholine": 0.5, # focus/gating
            "cortisol": 0.0       # stress/alertness
        }

        self.growth_threshold = 0.95
        self.prune_threshold = 0.05

        self.block_names = [f"Block_{i}" for i in range(blocks)]
        self._debug("Init complete.")

    def _build_block(self):
        # Each block = multi-head self-attn sublayer, but weights can mutate
        return nn.ModuleDict({
            "query": nn.Linear(self.input_dim, self.hidden_dim),
            "key": nn.Linear(self.input_dim, self.hidden_dim),
            "value": nn.Linear(self.input_dim, self.hidden_dim),
            "out": nn.Linear(self.hidden_dim, self.input_dim)
        })

    def _oscillator_mask(self, t):
        # sin/cos + noise, per block/head, shape = (blocks, heads)
        freqs = self.oscillator_freqs
        phase = self.oscillator_phase
        kind = self.oscillator_type
        base = t * freqs + phase
        mask = torch.where(
            kind == 0, torch.sin(base), torch.cos(base)
        )
        mask += torch.randn_like(mask) * 0.07  # add noise
        mask = (mask > 0).float()  # only open if mask > 0
        return mask

    def _spike_mask(self, energy):
        # If energy > random threshold, head spikes open; else, drop
        rand = torch.rand_like(energy)
        return (energy > rand).float()

    def _update_oscillators(self):
        # Advance phase per step (could learn this, or random-walk)
        self.oscillator_phase += self.oscillator_freqs * (random.random() + 0.7)

    def _plasticity_update(self, use_matrix, reward):
        # Hebbian: what fires together wires together
        # + global decay, + reward
        delta = use_matrix * (reward + 0.1) - 0.02 * self.potentiation
        self.potentiation = (self.potentiation + delta).clamp(0.0, 1.0)
        # Update synaptic strength
        self.synaptic_strength = (self.synaptic_strength * 0.96 + self.potentiation * 0.04).clamp(0, 1)

    def _modulate_neurochem(self, region_usage):
        # Neurochemical updates based on region usage
        # Example: more novelty = more dopamine
        self.neurochem["dopamine"] = float(region_usage.mean().item()) * 0.7 + random.random() * 0.3
        self.neurochem["serotonin"] = float(1.0 - region_usage.std().item()) * 0.6
        self.neurochem["acetylcholine"] = float(region_usage.max().item())
        self.neurochem["cortisol"] = float(region_usage.std().item()) * 0.8
        # Clamp all
        for k in self.neurochem:
            self.neurochem[k] = min(max(self.neurochem[k], 0), 1)

    def _topology_morph(self):
        # Grows/prunes blocks/heads based on mean potentiation/activity
        usage = self.potentiation.mean(dim=(1, 2))  # mean by block
        # Grow
        if usage.max() > self.growth_threshold and len(self.blocks) < self.max_blocks:
            self.blocks.append(self._build_block())
            self._grow_graph()
            self.block_names.append(f"Block_{len(self.blocks) - 1}")
            self._debug(f"BLOCK GROW: New block spawned. Total: {len(self.blocks)}")
        # Prune
        if usage.min() < self.prune_threshold and len(self.blocks) > self.min_blocks:
            idx = usage.argmin().item()
            del self.blocks[idx]
            self._prune_graph(idx)
            del self.block_names[idx]
            self._debug(f"BLOCK PRUNE: Block {idx} removed. Total: {len(self.blocks)}")

    def _grow_graph(self):
        # Expand all dynamic tensors to add new block (zero-initialized)
        n = len(self.blocks)
        def grow(x): return torch.cat(
            [x, torch.zeros(1, *x.shape[1:], device=self.device)], dim=0)
        self.synaptic_strength = grow(self.synaptic_strength)
        self.potentiation = grow(self.potentiation)
        self.adj_mask = grow(self.adj_mask)
        self.energy = torch.cat([self.energy, torch.ones(1, self.heads, device=self.device) * 0.5], dim=0)
        self.oscillator_freqs = torch.cat([self.oscillator_freqs, torch.randn(1, self.heads, device=self.device) * 0.1 + 0.2], dim=0)
        self.oscillator_phase = torch.cat([self.oscillator_phase, torch.zeros(1, self.heads, device=self.device)], dim=0)
        self.oscillator_type = torch.cat([self.oscillator_type, torch.randint(0, 2, (1, self.heads), device=self.device)], dim=0)
        self.fatigue = torch.cat([self.fatigue, torch.zeros(1, self.heads, device=self.device)], dim=0)
        self.novelty = torch.cat([self.novelty, torch.ones(1, self.heads, device=self.device) * 0.5], dim=0)
        self.last_use = grow(self.last_use)

    def _prune_graph(self, idx):
        # Remove tensors at index
        def prune(x): return torch.cat([x[:idx], x[idx+1:]], dim=0)
        self.synaptic_strength = prune(self.synaptic_strength)
        self.potentiation = prune(self.potentiation)
        self.adj_mask = prune(self.adj_mask)
        self.energy = prune(self.energy)
        self.oscillator_freqs = prune(self.oscillator_freqs)
        self.oscillator_phase = prune(self.oscillator_phase)
        self.oscillator_type = prune(self.oscillator_type)
        self.fatigue = prune(self.fatigue)
        self.novelty = prune(self.novelty)
        self.last_use = prune(self.last_use)

    def forward(self, x, t=None, reward=0.0):
        """
        x: [batch, seq, input_dim]
        t: timestep, int or float
        reward: external scalar for neurochem/plasticity update
        """
        t = t or time.time()
        batch, seq, dim = x.shape
        n_blocks = len(self.blocks)

        # Update oscillators and spiking states
        self._update_oscillators()
        osc_mask = self._oscillator_mask(t)
        spike_mask = self._spike_mask(self.energy)

        # Mask heads: open if oscillation/spike passes, else zeroed
        block_outs = []
        use_matrix = torch.zeros_like(self.synaptic_strength)
        for b, block in enumerate(self.blocks):
            head_outs = []
            for h in range(self.heads):
                # Mask: only process if open
                if osc_mask[b, h] * spike_mask[b, h] > 0.5:
                    q = block["query"](x)
                    k = block["key"](x)
                    v = block["value"](x)
                    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
                    # Synaptic gating
                    attn_scores = attn_scores * self.synaptic_strength[b, h, h].clamp(0.01, 1.0)
                    attn_probs = F.softmax(attn_scores, dim=-1)
                    out = torch.matmul(attn_probs, v)
                    head_outs.append(out)
                    use_matrix[b, h, h] = 1  # Used this round
                    # Update fatigue/novelty
                    self.fatigue[b, h] += 0.1
                    self.novelty[b, h] = max(0, 1 - self.fatigue[b, h])
                else:
                    head_outs.append(torch.zeros_like(x))
                    self.fatigue[b, h] = max(0, self.fatigue[b, h] - 0.05)
                    self.novelty[b, h] = min(1, self.novelty[b, h] + 0.02)
            # Sum heads
            block_out = torch.stack(head_outs, dim=0).sum(dim=0)
            block_out = block["out"](block_out)
            block_outs.append(block_out)
        # Merge all blocks (sum or concat)
        out = torch.stack(block_outs, dim=0).sum(dim=0)
        # Apply global neurochem modulation
        mod = (
            self.neurochem["dopamine"] * 1.1 +
            self.neurochem["serotonin"] * 0.7 +
            self.neurochem["acetylcholine"] * 1.3 -
            self.neurochem["cortisol"] * 1.2
        )
        out = out * mod
        # Plasticity and topology updates
        self._plasticity_update(use_matrix, reward)
        self._modulate_neurochem(use_matrix)
        self._topology_morph()
        return out

    def _debug(self, msg):
        print(f"[VictorDNA Debug] {msg}")

    def visualize_connectome(self):
        # Simple: just print out current block/connection strengths
        print("\n[Connectome]")
        for i, name in enumerate(self.block_names):
            print(f"{name}: Strength Mean={self.synaptic_strength[i].mean():.2f}, Heads={self.heads}")
        print("[/Connectome]")

# EXAMPLE USAGE:
if __name__ == "__main__":
    # Set up Victor's cortex
    model = VictorDynamicNeuromorphicAttention(
        input_dim=128, hidden_dim=256, heads=8, blocks=4, max_blocks=10, min_blocks=2
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Fake input: [batch, seq, dim]
    x = torch.randn(2, 16, 128).to(model.device)

    for step in range(100):
        out = model(x, t=step, reward=random.random())
        if step % 10 == 0:
            model.visualize_connectome()
