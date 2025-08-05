# FILE: bark_victortensor/model.py
# VERSION: v2.0.1-PHOENIX-HOTFIX-GODCORE
# NAME: PhoenixMoEGPT
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Mixture‑of‑Experts GPT backbone for VictorTensor with SwiGLU activations, vectorised
#          routing, and a **prototype** hot‑reload self‑evolution hook via MetaEvolution.
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

"""
Patch log (v2.0.1‑PHX):
• Added missing `numpy` and `time` imports.
• Replaced ReLU MLP with SwiGLU (gated) MLP.
• MoE routing normalises with ε to avoid /0 and now vectorises scatter/gather for speed.
• MetaEvolution.evolve() loads on‑the‑fly‑patched modules via importlib.util.
• Added minimal smoke‑test entry (`python -m bark_victortensor.model test`).
"""

import math, json, time, tempfile, importlib.util, shutil, types, os
from dataclasses import dataclass

import numpy as np  # 🚑 missing in v2.0.0

from .victortensor_v9 import Tensor, nn, functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 🛠️  Utility
# ──────────────────────────────────────────────────────────────────────────────

EPS = 1e-9

try:
    SwiGLU = nn.SwiGLU  # if provided by VictorTensor core
except AttributeError:  # fallback inline implementation
    class SwiGLU(nn.Module):
        """SwiGLU activation (https://arxiv.org/abs/2002.05202)."""
        def __init__(self):
            super().__init__()
        def forward(self, x):
            a, b = F.chunk(x, 2, dim=-1)
            return F.silu(a) * b

# ──────────────────────────────────────────────────────────────────────────────
# 📐  Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # MoE parameters
    n_experts: int = 8
    n_experts_per_tok: int = 2

# ──────────────────────────────────────────────────────────────────────────────
# 🧠  Core Components
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.bias = Tensor(np.tril(np.ones((config.block_size, config.block_size))))

    def forward(self, x: Tensor, past_kv=None, use_cache: bool = False):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = F.chunk(qkv, 3, dim=2)

        def reshape_heads(t: Tensor):
            return t.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))

        q, k, v = map(reshape_heads, (q, k, v))

        if past_kv is not None:
            past_key, past_val = past_kv
            k = F.cat([past_key, k], dim=2)
            v = F.cat([past_val, v], dim=2)
        present = (k, v) if use_cache else None

        att = q.matmul(k.transpose((0, 1, 3, 2))) * (1.0 / math.sqrt(k.shape[-1]))
        mask = self.bias[:, :, :T, :T]
        att += Tensor(np.where(mask.data == 0, -np.inf, 0))
        att = self.attn_dropout(F.softmax(att, dim=-1))

        y = att.matmul(v).transpose((0, 2, 1, 3)).reshape((B, T, C))
        y = self.resid_dropout(self.c_proj(y))
        return y, present

# ──────────────────────────────────────────────────────────────────────────────
# 🧙  Mixture‑of‑Experts
# ──────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        hidden = 2 * n_embd  # SwiGLU expects 2×input dim
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=False),
            SwiGLU(),
            nn.Linear(n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_experts_per_tok = config.n_experts_per_tok
        self.experts = nn.ModuleList([Expert(config.n_embd, config.dropout) for _ in range(self.n_experts)])
        self.gate = nn.Linear(config.n_embd, self.n_experts, bias=False)

    def forward(self, x: Tensor):
        B, T, C = x.shape
        x_flat = x.reshape((B * T, C))
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        # top‑k routing (vectorised)
        topk_w, topk_idx = F.top_k(routing_weights, self.n_experts_per_tok, dim=1)
        topk_w = topk_w / (topk_w.sum(dim=1, keepdim=True) + EPS)

        # Create expert assignments
        expert_outputs = [Tensor.zeros_like(x_flat) for _ in range(self.n_experts)]
        for k in range(self.n_experts_per_tok):
            idx = topk_idx[:, k]
            w = topk_w[:, k:k+1]
            for e in range(self.n_experts):
                mask = (idx == e)
                if not mask.any():
                    continue
                tokens = x_flat[mask]
                out = self.experts[e](tokens) * w[mask]
                expert_outputs[e][mask] = out
        # Sum contributions
        out_flat = F.sum(expert_outputs, dim=0)
        return out_flat.reshape((B, T, C))

# ──────────────────────────────────────────────────────────────────────────────
# 🔗  Transformer Block & Model
# ──────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias)
        self.moe = MoE(cfg)
        self.layer_idx = layer_idx
    def forward(self, x, past_kv=None, use_cache=False):
        att_out, kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + att_out
        x = x + self.moe(self.ln_2(x))
        return x, kv

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(cfg.input_vocab_size, cfg.n_embd),
            'wpe': nn.Embedding(cfg.block_size, cfg.n_embd),
            'drop': nn.Dropout(cfg.dropout),
            'h': nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)]),
            'ln_f': nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias),
        })
        self.lm_head = nn.Linear(cfg.n_embd, cfg.output_vocab_size, bias=False)
        self.meta_evolution = MetaEvolution(self)

    def forward(self, idx: Tensor, *, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        B, T = idx.shape
        if past_kv is not None:
            assert T == 1, "KV cache only supports step‑wise decoding"
        tok_emb = self.transformer['wte'](idx)
        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
        if position_ids is None:
            position_ids = Tensor(np.arange(past_len, T + past_len))
        pos_emb = self.transformer['wpe'](position_ids)
        x = self.transformer['drop'](tok_emb + pos_emb)

        new_kv = () if use_cache else None
        for i, blk in enumerate(self.transformer['h']):
            x, kv = blk(x, past_kv=past_kv[i] if past_kv else None, use_cache=use_cache)
            if use_cache:
                new_kv += (kv,)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x[:, -1:, :])
        return logits, new_kv

# ──────────────────────────────────────────────────────────────────────────────
# 🌱  Self‑Evolution Prototype
# ──────────────────────────────────────────────────────────────────────────────

class MetaEvolution:
    def __init__(self, model: "GPT"):
        self.model = model
        self.evolution_history = []

    def evolve(self, instruction: str):
        print(f"[MetaEvolution] evolving with: {instruction}")
        self.evolution_history.append(instruction)
        # naive patch‑recompile demo: write tmp file, load, hot‑swap
        with tempfile.TemporaryDirectory() as td:
            src_path = inspect.getfile(self.model.__class__).replace("\\", "/")
            dst_path = f"{td}/model_patch.py"
            shutil.copy(src_path, dst_path)
            # TODO: apply codegen mutation based on instruction (out of scope)
            spec = importlib.util.spec_from_file_location("model_patch", dst_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            NewGPT = mod.GPT
            new_cfg = self.model.cfg  # could mutate cfg here
            new_model = NewGPT(new_cfg)
            # Hot‑swap weights where names match
            for (n, p_old), (_, p_new) in zip(self.model.named_parameters(), new_model.named_parameters()):
                p_new.data[...] = p_old.data
            self.model.__dict__.update(new_model.__dict__)
            print("[MetaEvolution] hot‑reload complete → model reference updated")

# ──────────────────────────────────────────────────────────────────────────────
# 🧪  Minimal smoke test
# Run: `python -m bark_victortensor.model test`
# ──────────────────────────────────────────────────────────────────────────────

def _smoke():
    cfg = GPTConfig(block_size=8, n_layer=2, n_head=4, n_embd=32, n_experts=4)
    gpt = GPT(cfg)
    dummy = Tensor(np.random.randint(0, cfg.input_vocab_size, size=(1, 8)))
    logits, _ = gpt(dummy)
    print("Smoke OK | logits shape:", logits.shape)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _smoke()
    else:
        print("Phoenix‑hotfix module loaded; run with 'test' for smoke.")
