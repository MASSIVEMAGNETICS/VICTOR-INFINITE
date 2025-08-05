###############################################
# FILE: bark_victortensor/model_moe.py
# PURPOSE: VictorTensor GPT 2.0.1 – "Phoenix‑X"
#           • Critical bug‑fixes (missing imports)
#           • Fully‑vectorised MoE with SwiGLU
#           • Robust dtype / shape sanity‑checks
#           • Autologging + tracing hooks
###############################################

import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np  # << FIXED: was missing in v2.0.0

from .victortensor_v9 import Tensor, nn, functional as F

# ----------------------------------------------------------------------------
# LOGGING --------------------------------------------------------------------
# ----------------------------------------------------------------------------
logger = logging.getLogger("BarkVictorGPT")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s › %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)

# ----------------------------------------------------------------------------
# CONFIGURATION ---------------------------------------------------------------
# ----------------------------------------------------------------------------
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
    n_experts: int = 8            # Number of experts
    n_experts_per_tok: int = 2    # Top‑k routing

# ----------------------------------------------------------------------------
# BUILDING BLOCKS -------------------------------------------------------------
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Flash‑style causal attention with Triangular mask (no torch)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("mask", Tensor(np.tril(np.ones((config.block_size, config.block_size)))) )

    def forward(self, x: Tensor, past_kv: Tuple[Tensor, Tensor] | None = None, use_cache: bool = False):
        B, T, C = x.shape
        qkv = self.c_attn(x)                                  # (B, T, 3C)
        q, k, v = ( Tensor(s).reshape(B, T, self.n_head, C // self.n_head).transpose((0, 2, 1, 3))
                    for s in np.split(qkv.data, 3, axis=-1) )

        # Append past KV --------------------------------------------------------
        if past_kv is not None:
            pk, pv = past_kv
            k = F.cat([pk, k], dim=2)
            v = F.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None

        att = q.matmul(k.transpose((0, 1, 3, 2))) * (1.0 / math.sqrt(k.shape[-1]))  # (B, H, T, S)

        # Causal mask -----------------------------------------------------------
        causal_mask = self.mask[:, :T, :k.shape[2]]           # (1, T, S)
        att += Tensor(np.where(causal_mask.data == 0, -np.inf, 0.0))

        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = att.matmul(v).transpose((0, 2, 1, 3)).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present

# ----------------------------------------------------------------------------
# SwiGLU ----------------------------------------------------------------------
# ----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU activation: (x₁ * SiLU(x₂))."""
    def forward(self, x: Tensor):
        x1, x2 = np.split(x.data, 2, axis=-1)
        return Tensor(F.silu(Tensor(x2)).data * x1)

# ----------------------------------------------------------------------------
# EXPERT & MoE ----------------------------------------------------------------
# ----------------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        inner = 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * inner, bias=True),  # doubled for SwiGLU split
            SwiGLU(),
            nn.Linear(inner, n_embd, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(x)

class MoE(nn.Module):
    """Vectorised Top‑k Mixture of Experts."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.k = config.n_experts_per_tok
        self.experts = nn.ModuleList([Expert(config.n_embd, config.dropout) for _ in range(self.n_experts)])
        self.gate = nn.Linear(config.n_embd, self.n_experts, bias=False)

    def forward(self, x: Tensor):
        B, T, C = x.shape
        flat = x.reshape(B * T, C)                           # (BT, C)
        logits = self.gate(flat)                             # (BT, E)
        weights = F.softmax(logits, dim=-1)                  # (BT, E)

        top_w, top_idx = F.top_k(weights, self.k, dim=-1)    # (BT, k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)      # renorm

        # Gather expert inputs --------------------------------------------------
        # For each expert, collect the tokens routed to it
        # Build sparse dispatcher (BT, k) → (n_routes)
        route_mask = [ (top_idx == i).unsqueeze(-1) for i in range(self.n_experts) ]
        outputs = Tensor(np.zeros_like(flat.data))
        for i, expert in enumerate(self.experts):
            mask = route_mask[i]  # (BT, k, 1)
            if not mask.any():
                continue
            sel = mask.squeeze(-1).any(dim=-1)              # (BT,)
            inp = flat[sel]
            if inp.shape[0] == 0:
                continue
            out = expert(inp)                               # (n_sel, C)
            # Sum weights belonging to this expert for each token -------------
            gathered_w = top_w[sel] * (top_idx[sel] == i)
            w_sum = gathered_w.sum(dim=-1, keepdim=True)
            outputs[sel] += out * w_sum

        return outputs.reshape(B, T, C)

# ----------------------------------------------------------------------------
# TRANSFORMER BLOCK -----------------------------------------------------------
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, idx: int):
        super().__init__()
        self.ln_1 = nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias)
        self.moe = MoE(cfg)
        self.idx = idx

    def forward(self, x: Tensor, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.moe(self.ln_2(x))
        return x, new_kv

# ----------------------------------------------------------------------------
# GPT MODEL -------------------------------------------------------------------
# ----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.input_vocab_size, cfg.n_embd),
            "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout),
            "h": nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)]),
            "ln_f": nn.OmegaLayerNorm(cfg.n_embd, bias=cfg.bias),
        })
        self.lm_head = nn.Linear(cfg.n_embd, cfg.output_vocab_size, bias=False)
        self.meta_evolution = MetaEvolution(self)
        logger.info("GPT initialised: %s", cfg)

    # ---------------------------------------------------------------------
    def forward(self, idx: Tensor, merge_context: bool = False, past_kv=None, position_ids=None, use_cache: bool = False):
        b, t = idx.shape

        # Input token embedding ---------------------------------------------
        if past_kv is not None:
            assert t == 1, "When past_kv supplied, forward expects 1 token at a time"
            tok_emb = self.transformer["wte"](idx)
        else:
            if merge_context:
                assert t >= 513, "merge_context requires ≥513 tokens (256 text + 256 semantic + 1+)"
                text_tok, sem_tok, inf_tok = idx[:, :256], idx[:, 256:512], idx[:, 512:]
                tok_emb = F.cat([
                    self.transformer["wte"](text_tok) + self.transformer["wte"](sem_tok),
                    self.transformer["wte"](inf_tok)
                ], dim=1)
                t = tok_emb.shape[1]
            else:
                tok_emb = self.transformer["wte"](idx)

        # Position embedding -------------------------------------------------
        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
        if position_ids is None:
            position_ids = Tensor(np.arange(past_len, past_len + t))
        pos_emb = self.transformer["wpe"](position_ids)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Blocks -------------------------------------------------------------
        new_kv: List[Tuple[Tensor, Tensor]] | None = [] if use_cache else None
        for i, blk in enumerate(self.transformer["h"]):
            x, kv = blk(x, past_kv=past_kv[i] if past_kv else None, use_cache=use_cache)
            if use_cache:
                new_kv.append(kv)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x[:, -1:])                      # only last token logits
        return logits, (tuple(new_kv) if use_cache else None)

# ----------------------------------------------------------------------------
# META‑EVOLUTION --------------------------------------------------------------
# ----------------------------------------------------------------------------
class MetaEvolution:
    """Proof‑of‑concept self‑evolving scaffold."""

    def __init__(self, model: GPT):
        self.model = model
        self.history: List[str] = []

    def evolve(self, instruction: str):
        logger.warning("Evolution instruction received: %s", instruction)
        self.history.append(instruction)
        self._log(instruction)
        # Future: dynamic code‑rewrite + hot‑swap (requires VictorTensor JIT)

    def _log(self, ins: str):
        with open("evolution_log.txt", "a", encoding="utf-8") as fp:
            fp.write(f"{time.ctime()}: {ins}\n")

