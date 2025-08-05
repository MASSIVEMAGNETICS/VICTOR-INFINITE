# =================================================================================================
# FILE: tokenformer_manager.py
# VERSION: v1.2.0-X10-HWCMP
# NAME: TokenformerManager
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Code Mode)
# PURPOSE: 10× fusion **plus on‑device compute optimization** — automatic mixed‑precision, dynamic
#          device placement, TorchInductor compile, and optional CPU‑offload when VRAM is scarce.
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
# CHANGELOG:
#   • v1.2.0 – *Hardware‑Compute Upgrade*
#       ◦ Autocast FP16/BF16 on CUDA; FP32 fallback on CPU.
#       ◦ `torch.compile` (Inductor) wraps Tokenformer forward for JIT speed‑up (PyTorch ≥2.1).
#       ◦ Device manager decides per‑Tokenformer placement (GPU if <occupancy‑thresh, else CPU).
#       ◦ Unified `.compute()` entry that returns both fused embedding and per‑modal logits
#         (useful for downstream routing).
#       ◦ Environment variables:
#           • `TOKENFORMER_GPU_LIMIT_MB` – soft VRAM budget (default 12000 MB).
#           • `TOKENFORMER_FORCE_CPU`    – set to "1" to disable GPU entirely.
# =================================================================================================

from __future__ import annotations
import os, psutil, gc, numpy as np, torch
from torch import nn
from typing import List

from tokenformers import (
    SemanticTokenformer,
    EmotionTokenformer,
    SymbolicTokenformer,
    PredictiveTokenformer,
    ContextTokenformer,
)
from OmegaTensor import OmegaTensor

# ------------------------------ Helper ------------------------------------------
GPU_LIMIT_MB = int(os.getenv("TOKENFORMER_GPU_LIMIT_MB", "12000"))
FORCE_CPU = os.getenv("TOKENFORMER_FORCE_CPU", "0") == "1"

USE_CUDA = (torch.cuda.is_available() and not FORCE_CPU)
DEVICE_GPU = torch.device("cuda") if USE_CUDA else torch.device("cpu")
DEVICE_CPU = torch.device("cpu")
D_COMMON = 512


def ensure_tensor(x: OmegaTensor | np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, OmegaTensor):
        x = x.to_numpy()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device)


def vram_used_mb() -> int:
    return torch.cuda.memory_allocated() // 1_000_000 if USE_CUDA else 0

# --------------------------- TokenformerManager ---------------------------------
class TokenformerManager(nn.Module):
    """Fusion + hardware‑aware compute."""

    def __init__(self, d_common: int = D_COMMON, compile_tf: bool = True):
        super().__init__()
        self.d_common = d_common
        self.compile_tf = compile_tf and hasattr(torch, "compile")

        # Initialise Tokenformers with potential device split
        self.tfs: List[nn.Module] = []
        self.proj = nn.ModuleList()
        for TF in [SemanticTokenformer, EmotionTokenformer, SymbolicTokenformer, PredictiveTokenformer, ContextTokenformer]:
            dev = self._pick_device()
            tf = TF().to(dev).eval()
            if self.compile_tf:
                tf.hidden = torch.compile(tf.hidden, mode="default", fullgraph=False)  # type: ignore
            self.tfs.append(tf)
            self.proj.append(nn.Linear(tf.hidden_size, d_common, bias=False, device=dev))
            print(f"[TF‑MANAGER] {TF.__name__} on {dev}")

        # learnable softmax gate (always on CPU for shared access)
        self.gate_logits = nn.Parameter(torch.zeros(len(self.tfs), device=DEVICE_CPU))

    # ---------------- Device picker --------------------------------------------
    def _pick_device(self) -> torch.device:
        if not USE_CUDA:
            return DEVICE_CPU
        if vram_used_mb() + 1500 < GPU_LIMIT_MB:  # heuristic 1.5 GB per Tokenformer
            return DEVICE_GPU
        return DEVICE_CPU

    # ---------------- Forward ---------------------------------------------------
    @torch.no_grad()
    def compute(self, tokens: OmegaTensor, mask: OmegaTensor) -> OmegaTensor:
        """Hardware‑aware, mixed‑precision forward."""
        hidden_list = []
        logits_list = []

        for tf, proj in zip(self.tfs, self.proj):
            dev = next(tf.parameters()).device
            tok = ensure_tensor(tokens, dev)
            attn = ensure_tensor(mask, dev)

            autocast = torch.cuda.amp.autocast if dev.type == "cuda" else torch.cpu.amp.autocast  # type: ignore
            dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
            with autocast(dtype=dtype):
                h, logits = tf.hidden(tok, attn, return_logits=True)  # Tokenformer API extended
                hidden_list.append(proj(h))
                logits_list.append(logits)  # keep per‑modal logits

        stacked = torch.stack(hidden_list, dim=0)  # (M,B,S,D)
        gate = torch.softmax(self.gate_logits.to(stacked.device), dim=0).view(-1, 1, 1, 1)
        fused = (stacked * gate).sum(dim=0)

        return OmegaTensor(fused.cpu().float().numpy(), name="fused_tokenformer_output"), logits_list

    # expose few trainable params
    def trainable_parameters(self):
        for p in self.proj.parameters(): yield p
        yield self.gate_logits

# =================================================================================================
# DEMO
# =================================================================================================
if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(42)
    mgr = TokenformerManager()

    B, S, V = 4, 32, 32000
    tok_np = np.random.randint(0, V, (B, S), dtype=np.int32)
    mask_np = np.triu(np.full((S, S), -1e9, dtype=np.float32), k=1)[None, None, :, :]

    tok_ot = OmegaTensor(tok_np, name="tok")
    mask_ot = OmegaTensor(mask_np, name="mask", requires_grad=False)

    print("[DEMO] compute…")
    fused, modal_logits = mgr.compute(tok_ot, mask_ot)
    print("[DEMO] fused shape", fused.shape, ", per‑modal logits", len(modal_logits))
    print("VRAM used ~", vram_used_mb(), "MB")
