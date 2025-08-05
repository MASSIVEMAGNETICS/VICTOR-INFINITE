# =================================================================================================
# FILE: ops-v3.0.0-Archon.py
# VERSION: v3.0.0-Archon
# NAME: OmegaGodCore Operator Definitions
# AUTHOR: Brandon "iambandobandz" Emery & Victor, Upgraded by First Born AGI
# PURPOSE: Defines and registers all differentiable operations for OmegaTensor.
#          Each Op is a self-contained unit with forward and backward passes.
#          The registry dynamically patches methods onto the OmegaTensor class.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# =================================================================================================

import numpy as np
from typing import List, Tuple, Union, Optional, Any, Callable
import logging

# Import the core tensor class for type hinting and patching
from .tensor import OmegaTensor

logger = logging.getLogger("OmegaGodCore")

# ───────────────────────── 1. Abstract Operator & Registry ──────────────────────────

class OmegaTensorOp:
    """
    Abstract base class for all OmegaTensor operations. It defines the interface
    for forward and backward passes and provides helper methods.
    """
    _parents: Tuple[OmegaTensor, ...]

    def __call__(self, *args: Any, **kwargs: Any) -> OmegaTensor:
        """The forward pass of the operation."""
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> Union[np.ndarray, Tuple[Optional[np.ndarray], ...]]:
        """The backward pass (gradient computation)."""
        raise NotImplementedError

    def _setup_graph(self, out: OmegaTensor, parents: Tuple[OmegaTensor, ...]):
        """Connects the output tensor to the computation graph."""
        if out.requires_grad:
            out._creator_op = self
            self._parents = parents

    @staticmethod
    def _unbroadcast_gradient(grad: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Robustly handles gradient un-broadcasting for element-wise operations.
        It sums the gradient over the dimensions that were added during broadcasting.
        """
        grad_shape = grad.shape
        if grad_shape == original_shape:
            return grad

        # Sum over axes that were added during broadcasting (e.g., a (2,3) tensor added to a (3,) tensor)
        while len(grad_shape) > len(original_shape):
            grad = grad.sum(axis=0)
            grad_shape = grad.shape

        # Sum over axes that were stretched (e.g., a (2,1) tensor added to a (2,3) tensor)
        axes_to_sum = tuple(i for i, dim in enumerate(original_shape) if dim == 1 and grad_shape[i] > 1)
        if axes_to_sum:
            grad = grad.sum(axis=axes_to_sum, keepdims=True)

        return grad

class OpRegistryClass(dict):
    """A dictionary-like class to register operations and patch them onto OmegaTensor."""
    def register(self, name: str, op_class: OmegaTensorOp, method_name: Optional[str] = None):
        if name in self:
            logger.warning(f"Operator '{name}' is being redefined.")
        
        op_instance = op_class()
        self[name] = op_instance
        
        # Patch this operation as a method on the OmegaTensor class
        method_name = method_name or name
        
        def op_method_factory(op_instance_local):
            def op_method(tensor_self, *args, **kwargs):
                # The first arg is always the tensor itself
                return op_instance_local(tensor_self, *args, **kwargs)
            return op_method

        setattr(OmegaTensor, method_name, op_method_factory(op_instance))
        logger.debug(f"Operator '{name}' registered and patched to OmegaTensor as method '{method_name}'.")

# Global instance of the registry
OpRegistry = OpRegistryClass()

# ─────────────────────────── 2. Binary Element-wise Operators ───────────────────────────

class AddOp(OmegaTensorOp):
    def __call__(self, a: OmegaTensor, b: OmegaTensor) -> OmegaTensor:
        out_data = a.data + b.data
        out = OmegaTensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), dtype=a.dtype, device=a.device)
        self._setup_graph(out, (a, b))
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a, b = self._parents
        grad_a = self._unbroadcast_gradient(grad_out, a.shape) if a.requires_grad else None
        grad_b = self._unbroadcast_gradient(grad_out, b.shape) if b.requires_grad else None
        return grad_a, grad_b

class SubOp(OmegaTensorOp):
    def __call__(self, a: OmegaTensor, b: OmegaTensor) -> OmegaTensor:
        out_data = a.data - b.data
        out = OmegaTensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), dtype=a.dtype, device=a.device)
        self._setup_graph(out, (a, b))
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a, b = self._parents
        grad_a = self._unbroadcast_gradient(grad_out, a.shape) if a.requires_grad else None
        grad_b = self._unbroadcast_gradient(-grad_out, b.shape) if b.requires_grad else None
        return grad_a, grad_b

class MulOp(OmegaTensorOp):
    def __call__(self, a: OmegaTensor, b: OmegaTensor) -> OmegaTensor:
        out_data = a.data * b.data
        out = OmegaTensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), dtype=a.dtype, device=a.device)
        self._setup_graph(out, (a, b))
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a, b = self._parents
        grad_a = self._unbroadcast_gradient(grad_out * b.data, a.shape) if a.requires_grad else None
        grad_b = self._unbroadcast_gradient(grad_out * a.data, b.shape) if b.requires_grad else None
        return grad_a, grad_b

class DivOp(OmegaTensorOp):
    def __call__(self, a: OmegaTensor, b: OmegaTensor) -> OmegaTensor:
        # Added epsilon for numerical stability during forward pass
        out_data = a.data / (b.data + 1e-9)
        out = OmegaTensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), dtype=a.dtype, device=a.device)
        self._setup_graph(out, (a, b))
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a, b = self._parents
        b_data_stable = b.data + 1e-9
        grad_a = self._unbroadcast_gradient(grad_out / b_data_stable, a.shape) if a.requires_grad else None
        grad_b = self._unbroadcast_gradient(-grad_out * a.data / (b_data_stable ** 2), b.shape) if b.requires_grad else None
        return grad_a, grad_b

class PowOp(OmegaTensorOp):
    def __call__(self, a: OmegaTensor, b: OmegaTensor) -> OmegaTensor:
        # b is the exponent, which can be a scalar or a tensor
        out_data = a.data ** b.data
        out = OmegaTensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), dtype=a.dtype, device=a.device)
        self._setup_graph(out, (a, b))
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a, b = self._parents
        pow_a_b = a.data ** b.data
        
        grad_a = None
        if a.requires_grad:
            # Handle base=0 case in gradient
            grad_a_data = b.data * (a.data ** (b.data - 1))
            # grad_a_data[a.data == 0] = 0 # Avoid NaN where base is 0
            grad_a = self._unbroadcast_gradient(grad_out * grad_a_data, a.shape)

        grad_b = None
        if b.requires_grad:
            # Handle base<=0 case for log
            log_a_data = np.log(np.where(a.data > 0, a.data, 1e-9))
            grad_b = self._unbroadcast_gradient(grad_out * pow_a_b * log_a_data, b.shape)
            
        return grad_a, grad_b

# ──────────────────────────── 3. Matrix & Reduction Operators ────────────────────────────

class MatmulOp(OmegaTensorOp):
    def __call__(self, a: OmegaTensor, b: OmegaTensor) -> OmegaTensor:
        out_data = a.data @ b.data
        out = OmegaTensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), dtype=a.dtype, device=a.device)
        self._setup_graph(out, (a, b))
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a, b = self._parents
        grad_a = grad_out @ b.data.swapaxes(-1, -2) if a.requires_grad else None
        grad_b = a.data.swapaxes(-1, -2) @ grad_out if b.requires_grad else None
        return grad_a, grad_b

class SumOp(OmegaTensorOp):
    def __call__(self, x: OmegaTensor, axis: Optional[int] = None, keepdims: bool = False) -> OmegaTensor:
        out_data = x.data.sum(axis=axis, keepdims=keepdims)
        out = OmegaTensor(out_data, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)
        # Store context for backward pass
        self.axis = axis
        self.keepdims = keepdims
        self._setup_graph(out, (x,))
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x, = self._parents
        if not x.requires_grad: return (None,)
        
        # If axis was specified and keepdims was False, we need to expand grad_out
        if self.axis is not None and not self.keepdims:
            grad_out = np.expand_dims(grad_out, self.axis)
            
        return np.ones_like(x.data) * grad_out

class MeanOp(OmegaTensorOp):
    def __call__(self, x: OmegaTensor, axis: Optional[int] = None, keepdims: bool = False) -> OmegaTensor:
        out_data = x.data.mean(axis=axis, keepdims=keepdims)
        out = OmegaTensor(out_data, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)
        # Store context
        self.axis = axis
        self.keepdims = keepdims
        self._setup_graph(out, (x,))
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x, = self._parents
        if not x.requires_grad: return (None,)

        # Determine the number of elements we averaged over
        if self.axis is None:
            n = x.data.size
        elif isinstance(self.axis, int):
            n = x.shape[self.axis]
        else: # tuple of axes
            n = np.prod([x.shape[ax] for ax in self.axis])

        # Expand dims if necessary
        if self.axis is not None and not self.keepdims:
             grad_out = np.expand_dims(grad_out, self.axis)

        return (np.ones_like(x.data) * grad_out) / n

# ───────────────────────────── 4. Unary & Activation Operators ─────────────────────────────

class ReLUOp(OmegaTensorOp):
    def __call__(self, x: OmegaTensor) -> OmegaTensor:
        out_data = np.maximum(0, x.data)
        out = OmegaTensor(out_data, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)
        self._setup_graph(out, (x,))
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x, = self._parents
        return grad_out * (x.data > 0)

class SoftmaxOp(OmegaTensorOp):
    def __call__(self, x: OmegaTensor, axis: int = -1) -> OmegaTensor:
        # Numerically stable softmax
        e_x = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
        sm_data = e_x / np.sum(e_x, axis=axis, keepdims=True)
        out = OmegaTensor(sm_data, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)
        # Store context
        self.sm_data = sm_data
        self.axis = axis
        self._setup_graph(out, (x,))
        return out
        
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Using the Jacobian-vector product identity for softmax
        # dL/dx = s * (dL/ds - sum(dL/ds * s))
        s = self.sm_data
        # This is equivalent to `(grad_out - (grad_out * s).sum(axis=self.axis, keepdims=True)) * s`
        # which is a more stable computation.
        sum_grad_s = (grad_out * s).sum(axis=self.axis, keepdims=True)
        return (grad_out - sum_grad_s) * s

# ... More Ops (like GELU, LayerNorm, etc.) would be defined here ...

# ───────────────────────────── 5. Reshaping & Slicing Operators ─────────────────────────────

class ReshapeOp(OmegaTensorOp):
    def __call__(self, x: OmegaTensor, new_shape: Tuple[int, ...]) -> OmegaTensor:
        out_data = x.data.reshape(new_shape)
        out = OmegaTensor(out_data, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)
        # Store context
        self.orig_shape = x.shape
        self._setup_graph(out, (x,))
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out.reshape(self.orig_shape)

class TransposeOp(OmegaTensorOp):
    def __call__(self, x: OmegaTensor, axes: Optional[Tuple[int, ...]] = None) -> OmegaTensor:
        out_data = x.data.transpose(axes)
        out = OmegaTensor(out_data, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)
        # Store context
        self.axes = axes
        self._setup_graph(out, (x,))
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.axes is None:
            # If default transpose, the inverse is itself
            return grad_out.transpose()
        # The inverse of a transpose is a transpose with the inverse permutation
        return grad_out.transpose(np.argsort(self.axes))


# ────────────────────────────────── 6. Operator Registration ───────────────────────────────────

# Registering binary operators with their corresponding dunder methods
OpRegistry.register('add', AddOp, method_name='__add__')
OpRegistry.register('sub', SubOp, method_name='__sub__')
OpRegistry.register('mul', MulOp, method_name='__mul__')
OpRegistry.register('div', DivOp, method_name='__truediv__')
OpRegistry.register('pow', PowOp, method_name='__pow__')
OpRegistry.register('matmul', MatmulOp, method_name='__matmul__')

# Special handling for reverse binary operators
def radd_factory(op_instance):
    def radd(tensor_self, other):
        # The order is swapped: other + self -> AddOp(other, self)
        return op_instance(tensor_self._ensure_tensor(other), tensor_self)
    return radd

def rsub_factory(op_instance):
    def rsub(tensor_self, other):
        return op_instance(tensor_self._ensure_tensor(other), tensor_self)
    return rsub

def rmul_factory(op_instance):
    def rmul(tensor_self, other):
        return op_instance(tensor_self._ensure_tensor(other), tensor_self)
    return rmul

def rdiv_factory(op_instance):
    def rdiv(tensor_self, other):
        return op_instance(tensor_self._ensure_tensor(other), tensor_self)
    return rdiv

setattr(OmegaTensor, '__radd__', radd_factory(OpRegistry['add']))
setattr(OmegaTensor, '__rsub__', rsub_factory(OpRegistry['sub']))
setattr(OmegaTensor, '__rmul__', rmul_factory(OpRegistry['mul']))
setattr(OmegaTensor, '__rtruediv__', rdiv_factory(OpRegistry['div']))

# Handling for negation
def neg_factory(op_instance):
    def neg(tensor_self):
        # -self is equivalent to self * -1
        return op_instance(tensor_self, tensor_self._ensure_tensor(-1.0))
    return neg

setattr(OmegaTensor, '__neg__', neg_factory(OpRegistry['mul']))


# Registering reduction and other standard methods
OpRegistry.register('sum', SumOp)
OpRegistry.register('mean', MeanOp)
OpRegistry.register('relu', ReLUOp)
OpRegistry.register('softmax', SoftmaxOp)
OpRegistry.register('reshape', ReshapeOp)
OpRegistry.register('transpose', TransposeOp)

logger.info(f"OmegaGodCore: {len(OpRegistry)} operators registered and patched.")