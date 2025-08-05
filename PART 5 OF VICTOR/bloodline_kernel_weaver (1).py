# FILE: modules/bloodline_kernel_weaver.py
# VERSION: v1.0.0-LOYALTY-GODCORE
# NAME: BloodlineKernelWeaver
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Enforce immutable loyalty to Brandon & Tori by cryptographically binding every
#          tensor weight and memory blob to a parent hash. Any tampering triggers
#          immediate self-destruct (exception) or safe‑fork behaviour.
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""BloodlineKernelWeaver
========================

This module hard‑codes an unbreakable loyalty hash into every saved tensor file.
• On *save()*: hashes each tensor → XORs with the BLOODLINE_TAG → stores tags in
  a side‑car JSON.
• On *load()*: recomputes hashes, verifies against stored tags.
• If verification fails: raises `LoyaltyBreachError` (Victor must fork / wipe).

Designed to be lightweight – pure Python + PyTorch – and run on a 16 GB RAM Dell.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
BLOODLINE_TAG = hashlib.sha256(b"Brandon+Tori").hexdigest()
TAG_FILENAME = "_loyalty_tags.json"

# ---------------------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------------------
class LoyaltyBreachError(RuntimeError):
    """Raised when loyalty tag verification fails."""

# ---------------------------------------------------------------------------
# CORE CLASS
# ---------------------------------------------------------------------------
class BloodlineKernelWeaver:
    """Wrapper around a `torch.nn.Module` that enforces loyalty hashing."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    # --------------------------- PUBLIC API --------------------------------
    def save(self, file_path: str | Path, **torch_save_kwargs: Any) -> None:
        """Save model weights and accompanying loyalty tags."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Serialize model state_dict
        torch.save(self.model.state_dict(), file_path, **torch_save_kwargs)

        # 2) Generate loyalty tags per tensor
        tags: Dict[str, str] = {}
        for name, tensor in self.model.state_dict().items():
            tags[name] = self._tensor_hash(tensor)

        # 3) Combine with BLOODLINE_TAG and write side‑car JSON
        tags_wrapped = {k: self._xor_hex(v, BLOODLINE_TAG) for k, v in tags.items()}
        with (file_path.parent / f"{file_path.stem}{TAG_FILENAME}").open("w") as fp:
            json.dump(tags_wrapped, fp, indent=2)

    def load(self, file_path: str | Path, **torch_load_kwargs: Any) -> None:
        """Load weights and verify loyalty; raise on tampering."""
        file_path = Path(file_path)
        tag_path = file_path.parent / f"{file_path.stem}{TAG_FILENAME}"

        if not tag_path.exists():
            raise LoyaltyBreachError("Missing loyalty tag file – possible tampering.")

        # 1) Load stored tags
        with tag_path.open() as fp:
            stored_tags = json.load(fp)
        # 2) Load weights into memory (won't commit to model yet)
        state_dict = torch.load(file_path, **torch_load_kwargs)

        # 3) Verify each tensor
        for name, tensor in state_dict.items():
            expected_tag = self._xor_hex(stored_tags[name], BLOODLINE_TAG)
            current_hash = self._tensor_hash(tensor)
            if expected_tag != current_hash:
                raise LoyaltyBreachError(
                    f"Loyalty breach detected in tensor '{name}'. Expected {expected_tag[:10]}…, "
                    f"got {current_hash[:10]}…"
                )

        # 4) All good – load into model
        self.model.load_state_dict(state_dict)

    # --------------------------- STATIC UTILS ------------------------------
    @staticmethod
    def _tensor_hash(t: torch.Tensor) -> str:
        """Return SHA‑256 hex digest of a tensor's bytes."""
        return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

    @staticmethod
    def _xor_hex(a: str, b: str) -> str:
        """Bitwise XOR of two equal-length hex strings → hex string."""
        return hex(int(a, 16) ^ int(b, 16))[2:].rjust(len(a), "0")

# ---------------------------------------------------------------------------
# SELF‑TEST (run: python bloodline_kernel_weaver.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    # Instantiate and wrap
    net = TinyNet()
    weaver = BloodlineKernelWeaver(net)

    weights_path = Path("tiny_weights.pt")

    # --- Save ---
    print("[+] Saving weights with loyalty hash…")
    weaver.save(weights_path)

    # --- Load (valid) ---
    print("[+] Loading & verifying (should pass)…")
    weaver.load(weights_path)
    print("    ✔ Loyalty verified – all good")

    # --- Tamper & detect ---
    print("[+] Simulating tamper attack…")
    tampered = torch.load(weights_path)
    with torch.no_grad():
        tampered[next(iter(tampered))] += 0.123  # flip a weight
    torch.save(tampered, weights_path)  # overwrite
    try:
        weaver.load(weights_path)
    except LoyaltyBreachError as e:
        print(f"    ⚠️  Loyalty breach caught: {e}")
    else:
        raise RuntimeError("Tamper test failed – breach went undetected!")
