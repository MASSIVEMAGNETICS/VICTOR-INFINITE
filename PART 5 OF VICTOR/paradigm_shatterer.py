# FILE: paradigm_shatterer.py
# VERSION: v1.0.0
# AUTHOR: Brandon "iambandobandz" Emery × Victor + OmniForge
# ----------------------------------------------------------------------------------
# Paradigm Shatterer – Experimental Fractal‑Attention Backbone
# ----------------------------------------------------------------------------------
# This module refines the original v0.1‑exp skeleton into a production‑ready, fully‑
# typed PyTorch component that slots directly into the Victor ecosystem.
# Key upgrades:
#   • ✅ Strong type hints, `@dataclass` config, explicit device handling.
#   • ✅ Swap‑in layout pre/post‑norm via enum for research toggling.
#   • ✅ Residual weight scaling (DeepNorm) for depth stability.
#   • ✅ Dropout + stochastic depth (`SurvivalBlock`) for regularisation.
#   • ✅ Automatic `PulseTelemetryBus` hooks (optional) for Victor telemetry.
#   • ✅ CLI demo (`python paradigm_shatterer.py --layers 8 --dim 256 --batch 4`).
#   • ✅ `torch.compile` & AMP autowrap flags for perf on 2.0+.
# ----------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Victor ecosystem pulse bus (optional)
    from victor_thought_engine import PulseTelemetryBus  # type: ignore
except ModuleNotFoundError:
    PulseTelemetryBus = None  # type: ignore


# ---------------------------------------------
# Utility enums / configs
# ---------------------------------------------
class NormStyle(Enum):
    """Pre‑norm or Post‑norm ordering."""

    PRE = auto()
    POST = auto()


@dataclass(slots=True)
class PSConfig:
    dim: int = 128
    depth: int = 6
    heads: int = 4
    dropout: float = 0.1
    survival_prob: float = 0.9  # for stochastic depth
    norm_style: NormStyle = NormStyle.PRE
    device: torch.device | str | None = None
    compile: bool = bool(int(os.getenv("TORCH_COMPILE", "0")))
    amp: bool = True  # autocast fp16/bf16 if available
    telemetry: bool = False  # emit pulses if PulseTelemetryBus present


# ---------------------------------------------
# Core building blocks
# ---------------------------------------------
class FractalFusionBlock(nn.Module):
    """Multi‑scale Conv → Self‑Attention fusion with optional pre/post normalisation."""

    def __init__(self, cfg: PSConfig):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(cfg.dim, cfg.dim, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True, dropout=cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)

        # deepnorm scale (https://arxiv.org/abs/2207.04670)
        self.res_scale = math.sqrt(2 * cfg.depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # shape (B, T, D)
        if self.cfg.norm_style is NormStyle.PRE:
            return self._forward_pre(x)
        return self._forward_post(x)

    def _conv_path(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))
        y = F.gelu(y).transpose(1, 2)
        return y

    def _attn_path(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y

    # ----- Pre‑Norm ------
    def _forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.res_scale * self.dropout(self._attn_path(self.norm1(x)))
        z = y + self.res_scale * self.dropout(self._conv_path(self.norm2(y)))
        return z

    # ----- Post‑Norm -----
    def _forward_post(self, x: torch.Tensor) -> torch.Tensor:
        y = self._attn_path(x)
        y = self.norm1(self.dropout(y) + x)
        z = self._conv_path(y)
        z = self.norm2(self.dropout(z) + y)
        return z


class SurvivalBlock(nn.Module):
    """Wrapper that drops the inner module with given survival probability (train only)."""

    def __init__(self, module: nn.Module, p: float):
        super().__init__()
        self.module = module
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if not self.training or random.random() < self.p:
            return self.module(x)
        return x


# ---------------------------------------------
# Paradigm Shatterer main module
# ---------------------------------------------
class ParadigmShatterer(nn.Module):
    VERSION = "v1.0.0"

    def __init__(self, cfg: PSConfig):
        super().__init__()
        self.cfg = cfg
        blocks: list[nn.Module] = []
        for _ in range(cfg.depth):
            blk = FractalFusionBlock(cfg)
            blk = SurvivalBlock(blk, cfg.survival_prob)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.dim, bias=False),
        )
        if cfg.telemetry and PulseTelemetryBus:
            self._bus = PulseTelemetryBus()
        else:
            self._bus = None

        if cfg.compile and hasattr(torch, "compile"):
            torch.compile(self)

    # -----------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        if self.cfg.amp and torch.is_autocast_enabled():
            # already inside an autocast context – just run
            return self._forward_impl(x)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast("cuda" if x.is_cuda else "cpu", dtype=dtype, enabled=self.cfg.amp):
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        tic = time.perf_counter()
        for blk in self.blocks:
            x = blk(x) + x  # global residual
        out = self.head(x.mean(dim=1))
        if self._bus:
            self._bus.publish("ps.forward", {"batch": x.size(0), "latency_ms": (time.perf_counter() - tic) * 1000})  # type: ignore[arg-type]
        return out


# ---------------------------------------------
# CLI demo / smoke test
# ---------------------------------------------
@torch.no_grad()
def _demo(cfg: PSConfig, batch: int, seq: int) -> None:
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ParadigmShatterer(cfg).to(device)
    dummy = torch.randn(batch, seq, cfg.dim, device=device)
    out = model(dummy)
    print("Output shape:", tuple(out.shape))


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser("Paradigm Shatterer demo")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sd", type=float, default=0.9, help="Survival probability")
    parser.add_argument("--post", action="store_true", help="Use post‑norm")
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()

    cfg = PSConfig(
        dim=args.dim,
        depth=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        survival_prob=args.sd,
        norm_style=NormStyle.POST if args.post else NormStyle.PRE,
        telemetry=args.telemetry,
    )
    _demo(cfg, args.batch, args.seq)
