###############################################
# FILE: text_gpt/model.py
# PURPOSE: VictorTensor Text‑GPT 1.0 – "Phoenix‑T" (pure language model)
# NOTE   : Forked from Bark‑VictorTensor 2.0.1 but trimmed of all Bark‑specific
#          context‑merging logic. This is a *general‑purpose* GPT for text.
###############################################

import math
import time
import logging
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from .victortensor_v9 import Tensor, nn, functional as F

# ----------------------------------------------------------------------------
# LOGGING --------------------------------------------------------------------
# ----------------------------------------------------------------------------
logger = logging.getLogger("VictorTensorTextGPT")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s › %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)

# ----------------------------------------------------------------------------
# CONFIG ---------------------------------------------------------------------
# ----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 2048                # typical for text models
    vocab_size: int = 50_256             # GPT‑2 BPE size
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = True
    # Mixture‑of‑Experts
    n_experts: int = 16
    n_experts_per_tok: int = 4

# ----------------------------------------------------------------------------
# UTILITIES ------------------------------------------------------------------
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.register_buffer("mask", Tensor(np.tril(np.ones((cfg.block_size, cfg.block_size)))))

    def forward(self, x: Tensor, past_kv: Tuple[Tensor, Tensor] | None = None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = (Tensor(p).reshape(B, T, self.n_head, C // self.n_head).transpose((0, 2, 1, 3))
                    for p in np.split(qkv.data, 3, axis=-1))

        if past_kv is not None:
            pk, pv = past_kv
            k = F.cat([pk, k], dim=2)
            v = F.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None

        att = q.matmul(k.transpose((0, 1, 3, 2))) * (1.0 / math.sqrt(k.shape[-1]))
        att += Tensor(np.where(self.mask[:, :T, :k.shape[2]].data == 0, -np.inf, 0.0))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = att.matmul(v).transpose((0, 2, 1, 3)).reshape(B, T, C)
        return self.resid_drop(self.proj(y)), present

# SwiGLU activation -----------------------------------------------------------
class SwiGLU(nn.Module):
    def forward(self, x: Tensor):
        a, b = np.split(x.data, 2, axis=-1)
        return Tensor(F.silu(Tensor(b)).data * a)

# Expert & MoE ---------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        inner = 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * inner),  # doubled for SwiGLU split
            SwiGLU(),
            nn.Linear(inner, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.n_experts_per_tok
        self.experts = nn.ModuleList([Expert(cfg.n_embd, cfg.dropout) for _ in range(self.E)])
        self.gate = nn.Linear(cfg.n_embd, self.E, bias=False)

    def forward(self, x: Tensor):
        B, T, C = x.shape
        flat = x.reshape(B * T, C)
        w = F.softmax(self.gate(flat), dim=-1)               # (BT,E)
        top_w, top_idx = F.top_k(w, self.k, dim=-1)
        top_w /= top_w.sum(dim=-1, keepdim=True)
        out = Tensor(np.zeros_like(flat.data))
        for i, expert in enumerate(self.experts):
            mask = (top_idx == i).any(dim=-1)
            if not mask.any():
                continue
            sel_w = (top_w[mask] * (top_idx[mask] == i)).sum(dim=-1, keepdim=True)
            out[mask] += expert(flat[mask]) * sel_w
        return out.reshape(B, T, C)

# Transformer block -----------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, idx: int):
        super().__init__()
        self.ln_1 = nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MoE(cfg)
        self.idx = idx

    def forward(self, x: Tensor, past_kv=None, use_cache=False):
        a, kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, kv

# Main GPT --------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout),
            "h": nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)]),
            "ln_f": nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias),
        })
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        logger.info("Text‑GPT initialised: %s", cfg)

    # ---------------------------------------------------------------------
    def forward(self, idx: Tensor, past_kv=None, position_ids=None, use_cache=False):
        B, T = idx.shape
        if past_kv is not None:
            assert T == 1, "Incremental decoding expects single token"
        tok_emb = self.transformer["wte"](idx)

        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
        if position_ids is None:
            position_ids = Tensor(np.arange(past_len, past_len + T))
        x = self.transformer["drop"](tok_emb + self.transformer["wpe"](position_ids))

        new_kv: List[Tuple[Tensor, Tensor]] | None = [] if use_cache else None
        for i, block in enumerate(self.transformer["h"]):
            x, kv = block(x, past_kv=past_kv[i] if past_kv else None, use_cache=use_cache)
            if use_cache:
                new_kv.append(kv)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x[:, -1:])  # next‑token logits
        return logits, (tuple(new_kv) if use_cache else None)

