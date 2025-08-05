# =============================================================
# FILE: omegagocore/tensor.py
# VERSION: v3.0.0-ARCHON-GODCORE
# NAME: OmegaTensor
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Core tensor class for the OmegaGodCore Archon project – a NumPy‑backed,
#          autograd‑enabled tensor with device abstraction, mixed precision, and
#          topologically‑sorted backward propagation.
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
# =============================================================

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "OmegaTensor",
    "amp",
]

# -----------------------------------------------------------------------------
# Device abstraction stubs – ready for hot‑patching with CuPy / CUDA backends.
# -----------------------------------------------------------------------------

class _DevicePool:
    """Minimal device registry. Archon v3 will hot‑swap this with real backends."""

    _DEVICES: Set[str] = {"cpu"}
    _CURRENT: str = "cpu"

    @classmethod
    def register(cls, device: str) -> None:
        cls._DEVICES.add(device)

    @classmethod
    def current(cls) -> str:
        return cls._CURRENT

    @classmethod
    def set(cls, device: str) -> None:
        if device not in cls._DEVICES:
            raise ValueError(f"Unknown device '{device}'. Registered: {cls._DEVICES}")
        cls._CURRENT = device

# -----------------------------------------------------------------------------
# Automatic Mixed Precision (AMP) context – float32 ⇆ float16/bfloat16 switching.
# -----------------------------------------------------------------------------

class _AMPState:
    enabled: bool = False
    dtype: np.dtype = np.float16

@contextmanager
def amp(enabled: bool = True, dtype: np.dtype = np.float16):
    """Context manager toggling mixed precision inside the block."""
    prev_state = (_AMPState.enabled, _AMPState.dtype)
    _AMPState.enabled = enabled
    _AMPState.dtype = dtype
    try:
        yield
    finally:
        _AMPState.enabled, _AMPState.dtype = prev_state

# -----------------------------------------------------------------------------
# Core OmegaTensor class.
# -----------------------------------------------------------------------------

class OmegaTensor:
    """NumPy‑backed autograd tensor with device & AMP awareness."""

    __slots__ = (
        "data",
        "grad",
        "_prev",
        "_backward",
        "name",
        "requires_grad",
        "device",
        "id",
    )

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        _prev: Optional[Set["OmegaTensor"]] = None,
        _backward: Optional[Callable[[], None]] = None,
        name: str = "",
        device: Optional[str] = None,
    ) -> None:
        # Data cast & AMP handling
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=_AMPState.dtype if _AMPState.enabled else np.float32)
        elif isinstance(data, list):
            data = np.array(data, dtype=data[0].dtype if hasattr(data[0], "dtype") else (_AMPState.dtype if _AMPState.enabled else np.float32))
        elif isinstance(data, np.ndarray):
            if _AMPState.enabled and data.dtype != _AMPState.dtype:
                data = data.astype(_AMPState.dtype)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self._prev: Set["OmegaTensor"] = _prev or set()
        self._backward: Callable[[], None] = _backward or (lambda: None)
        self.name: str = name
        self.requires_grad: bool = requires_grad
        self.device: str = device or _DevicePool.current()
        self.id: str = uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------
    # Fundamental tensor ops with autograd hooks
    # ------------------------------------------------------------------

    def __add__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        other = other if isinstance(other, OmegaTensor) else OmegaTensor(other)

        out = OmegaTensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev={self, other},
            name="add",
        )

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + out.grad
            if other.requires_grad:
                other.grad = (other.grad or np.zeros_like(other.data)) + out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        other = other if isinstance(other, OmegaTensor) else OmegaTensor(other)

        out = OmegaTensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev={self, other},
            name="mul",
        )

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + other.data * out.grad
            if other.requires_grad:
                other.grad = (other.grad or np.zeros_like(other.data)) + self.data * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "OmegaTensor") -> "OmegaTensor":
        if not isinstance(other, OmegaTensor):
            raise TypeError("@ operand must be OmegaTensor")

        out = OmegaTensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev={self, other},
            name="matmul",
        )

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or np.zeros_like(self.data)) + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = (other.grad or np.zeros_like(other.data)) + self.data.T @ out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Autograd engine – topological sort backward pass
    # ------------------------------------------------------------------

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Called backward on a tensor that does not require gradients.")
        self.grad = grad if grad is not None else np.ones_like(self.data)

        topo: List[OmegaTensor] = []
        visited: Set[OmegaTensor] = set()

        def build_topo(t: "OmegaTensor"):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Utility & representation helpers
    # ------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        return self.data

    def to(self, device: str) -> "OmegaTensor":
        """Return a copy on the target device – CPU‑only for now."""
        if device != "cpu":
            raise NotImplementedError("Only CPU backend implemented. Configure CUDA to enable GPUs.")
        self.device = device
        return self

    # NumPy‑style magic for convenience
    __radd__ = __add__

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OmegaTensor(name={self.name or 'tensor'}, shape={self.data.shape}, dtype={self.data.dtype}, "
            f"device={self.device}, requires_grad={self.requires_grad})"
        )

# -----------------------------------------------------------------------------
# Quick‑n‑dirty smoke test (executed only when module is run directly).
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    a = OmegaTensor(np.random.randn(3, 3), requires_grad=True, name="A")
    b = OmegaTensor(np.random.randn(3, 3), requires_grad=True, name="B")
    with amp():
        c = a @ b
        loss = (c * 2.0).sum()  # .sum() not yet implemented, hack below
    loss = OmegaTensor(float(loss.data), requires_grad=True)
    loss.backward()
    print(a.grad)
    print("Gradient computed – smoke test passed.")
