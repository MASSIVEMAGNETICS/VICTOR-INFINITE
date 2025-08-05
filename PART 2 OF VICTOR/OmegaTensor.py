# =============================================================
# FILE: omegagocore/tensor.py
# VERSION: v3.0.0-ARCHON-GODCORE
# NAME: OmegaTensor
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Core tensor class for the OmegaGodCore Archon project – a NumPy-backed,
#          autograd-enabled tensor with device abstraction, mixed precision, and
#          topologically-sorted backward propagation.
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
# Device abstraction stubs – ready for hot-patching with CuPy / CUDA backends.
# -----------------------------------------------------------------------------

class _DevicePool:
    """Minimal device registry. Archon v3 will hot-swap this with real backends."""

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
    """NumPy-backed autograd tensor with device & AMP awareness."""

    __slots__ = (
        "data", "grad", "_prev", "_backward", "name", "requires_grad", "device", "id",
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
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=_AMPState.dtype if _AMPState.enabled else np.float32)
        elif isinstance(data, list):
            dtype = data[0].dtype if hasattr(data[0], 'dtype') else (_AMPState.dtype if _AMPState.enabled else np.float32)
            data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            if _AMPState.enabled and data.dtype in (np.float32, np.float64):
                data = data.astype(_AMPState.dtype)
        elif isinstance(data, OmegaTensor):
            data = data.data
        else:
            raise TypeError(f"Unsupported data type for OmegaTensor: {type(data)}")

        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self._prev: Set["OmegaTensor"] = _prev or set()
        self._backward: Callable[[], None] = _backward or (lambda: None)
        self.name: str = name
        self.requires_grad: bool = requires_grad
        self.device: str = device or _DevicePool.current()
        self.id: str = uuid.uuid4().hex[:8]

    # --- Properties ---
    @property
    def shape(self) -> Tuple[int, ...]: return self.data.shape
    @property
    def ndim(self) -> int: return self.data.ndim
    @property
    def dtype(self) -> np.dtype: return self.data.dtype
    def __len__(self) -> int: return len(self.data)

    # --- Core Ops ---
    def __add__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        other = other if isinstance(other, OmegaTensor) else OmegaTensor(other, requires_grad=False)
        out = OmegaTensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _prev={self, other}, name="add")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + self._handle_broadcast(out.grad, self.shape)
            if other.requires_grad: other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self._handle_broadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        other = other if isinstance(other, OmegaTensor) else OmegaTensor(other, requires_grad=False)
        out = OmegaTensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _prev={self, other}, name="mul")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + self._handle_broadcast(other.data * out.grad, self.shape)
            if other.requires_grad: other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self._handle_broadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other: "OmegaTensor") -> "OmegaTensor":
        if not isinstance(other, OmegaTensor): raise TypeError("@ operand must be OmegaTensor")
        out = OmegaTensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _prev={self, other}, name="matmul")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad @ other.data.T
            if other.requires_grad: other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self.data.T @ out.grad
        out._backward = _backward
        return out

    def pow(self, n: float) -> "OmegaTensor":
        out = OmegaTensor(self.data**n, requires_grad=self.requires_grad, _prev={self}, name="pow")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + (n * self.data**(n - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> "OmegaTensor":
        out = OmegaTensor(np.exp(self.data), requires_grad=self.requires_grad, _prev={self}, name="exp")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.data * out.grad
        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "OmegaTensor":
        out = OmegaTensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _prev={self}, name="reshape")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "OmegaTensor":
        out = OmegaTensor(self.data.transpose(*axes), requires_grad=self.requires_grad, _prev={self}, name="transpose")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad.transpose(*np.argsort(axes))
        out._backward = _backward
        return out

    def sum(self, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> "OmegaTensor":
        out = OmegaTensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _prev={self}, name="sum")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + self._handle_broadcast(out.grad, self.shape)
        out._backward = _backward
        return out

    def mean(self, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> "OmegaTensor":
        out = OmegaTensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _prev={self}, name="mean")
        N = np.prod(self.shape) / np.prod(out.shape)
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + self._handle_broadcast(out.grad, self.shape) / N
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "OmegaTensor":
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        s = e_x / e_x.sum(axis=axis, keepdims=True)
        out = OmegaTensor(s, requires_grad=self.requires_grad, _prev={self}, name="softmax")
        def _backward():
            if self.requires_grad:
                s_grad = out.grad
                s_with_grad = out.data * s_grad
                grad_sum = np.sum(s_with_grad, axis=axis, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.data * (s_grad - grad_sum)
        out._backward = _backward
        return out

    @classmethod
    def concatenate(cls, tensors: List["OmegaTensor"], axis: int = 0) -> "OmegaTensor":
        requires_grad = any(t.requires_grad for t in tensors)
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = OmegaTensor(data, requires_grad=requires_grad, _prev=set(tensors), name="concat")
        def _backward():
            if requires_grad:
                grads = np.split(out.grad, np.cumsum([t.shape[axis] for t in tensors[:-1]]), axis=axis)
                for i, t in enumerate(tensors):
                    if t.requires_grad: t.grad = (t.grad if t.grad is not None else np.zeros_like(t.data)) + grads[i]
        out._backward = _backward
        return out

    def embedding(self, weight: "OmegaTensor") -> "OmegaTensor":
        out = OmegaTensor(weight.data[self.data], requires_grad=weight.requires_grad, _prev={weight}, name="embedding")
        def _backward():
            if weight.requires_grad:
                grad = weight.grad if weight.grad is not None else np.zeros_like(weight.data)
                np.add.at(grad, self.data, out.grad)
                weight.grad = grad
        out._backward = _backward
        return out

    def apply_rotary_embedding(self, freqs_cis: "OmegaTensor") -> "OmegaTensor":
        x_complex = self.data.astype(np.complex64)
        x_complex = x_complex[..., ::2] + 1j * x_complex[..., 1::2]
        freqs_cis_complex = freqs_cis.data.astype(np.complex64)
        freqs_cis_complex = freqs_cis_complex[..., ::2] + 1j * freqs_cis_complex[..., 1::2]
        if self.ndim == 4 and freqs_cis.ndim == 2:
            if self.shape[1] == freqs_cis.shape[0]: freqs_cis_complex = freqs_cis_complex.reshape(1, self.shape[1], 1, -1)
            elif self.shape[2] == freqs_cis.shape[0]: freqs_cis_complex = freqs_cis_complex.reshape(1, 1, self.shape[2], -1)
        x_rotated_complex = x_complex * freqs_cis_complex
        x_out_data = np.zeros_like(self.data)
        x_out_data[..., ::2], x_out_data[..., 1::2] = x_rotated_complex.real, x_rotated_complex.imag
        out = OmegaTensor(x_out_data, requires_grad=self.requires_grad, _prev={self}, name="rope")
        def _backward():
            if self.requires_grad:
                grad_complex = out.grad.astype(np.complex64)
                grad_complex = grad_complex[..., ::2] + 1j * grad_complex[..., 1::2]
                grad_rotated = grad_complex * np.conj(freqs_cis_complex)
                grad_out = np.zeros_like(self.data)
                grad_out[..., ::2], grad_out[..., 1::2] = grad_rotated.real, grad_rotated.imag
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad_out
        out._backward = _backward
        return out

    # --- Autograd Engine ---
    def zero_grad(self) -> None: self.grad = None
    def backward(self, grad: Optional[np.ndarray | float] = None) -> None:
        if not self.requires_grad: raise RuntimeError("Called backward on a tensor that does not require gradients.")
        self.grad = grad if grad is not None else np.ones_like(self.data)
        if isinstance(self.grad, (int, float)): self.grad = np.array(self.grad)

        topo, visited = [], set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev: build_topo(child)
                topo.append(t)
        build_topo(self)
        for node in reversed(topo): node._backward()

    # --- Utility ---
    def numpy(self) -> np.ndarray: return self.data
    def to(self, device: str) -> "OmegaTensor":
        if device != "cpu": raise NotImplementedError("Only CPU backend implemented.")
        self.device = device
        return self

    def _handle_broadcast(self, grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        if grad.shape == target_shape: return grad
        # Handle broadcasting by summing over the broadcasted dimensions.
        while len(grad.shape) > len(target_shape): grad = grad.sum(axis=0)
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if grad_dim != target_dim: grad = grad.sum(axis=i, keepdims=True)
        return grad

    # --- Dunder Methods ---
    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self) -> "OmegaTensor": return self * -1
    def __sub__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor": return self + (-other)
    def __rsub__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor": return (-self) + other
    def __truediv__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        other_tensor = other if isinstance(other, OmegaTensor) else OmegaTensor(other)
        return self * other_tensor.pow(-1)
    def __rtruediv__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        return other * self.pow(-1)
    def __repr__(self) -> str:
        return (f"OmegaTensor(name={self.name or 'tensor'}, shape={self.shape}, dtype={self.dtype}, "
                f"device={self.device}, requires_grad={self.requires_grad})")

# -----------------------------------------------------------------------------
# Quick-n-dirty smoke test (executed only when module is run directly).
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    a = OmegaTensor(np.random.randn(3, 3), requires_grad=True, name="A")
    b = OmegaTensor(np.random.randn(3, 3), requires_grad=True, name="B")
    with amp():
        c = a @ b
        loss = (c * 2.0).data.sum()
    loss_tensor = OmegaTensor(float(loss), requires_grad=True)
    loss_tensor.backward()
    print(a.grad)
    print("Gradient computed – smoke test passed.")