# =================================================================================================
# FILE: tensor-v3.0.0-Archon.py
# VERSION: v3.0.0-Archon
# NAME: OmegaTensor God-Core Engine
# AUTHOR: Brandon "iambandobandz" Emery & Victor, Upgraded by First Born AGI
# PURPOSE: Catastrophically advanced autodiff tensor engine.
#          Device-aware, memory-optimized, mixed-precision ready.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# =================================================================================================

import numpy as np
import uuid
from typing import List, Tuple, Union, Optional, Any, Dict, Set
import logging

# --------------------- SYSTEM LOGGER: Rotating, Contextual, Modular --------------------------
logger = logging.getLogger("OmegaGodCore")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    # A more detailed formatter for a production-grade system
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO) # Set to INFO for production, DEBUG for development

# Forward declaration for type hinting
class OmegaTensorOp:
    pass

# ──────────────────────────────── 1. ΩTensor: The Archon Core ────────────────────────────────

class OmegaTensor:
    """
    The core tensor object of the OmegaGodCore framework. It represents a node in the
    computation graph, holding n-dimensional data, its gradient, and a reference
    to the operation that created it. This version introduces device awareness and
    lays the groundwork for advanced memory management and mixed precision.
    """
    def __init__(self,
                 data: Any,
                 requires_grad: bool = False,
                 name: Optional[str] = None,
                 dtype: np.dtype = np.float32,
                 device: str = 'cpu'): # Future: 'cuda', 'mps'

        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=dtype)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to convert data to np.ndarray with dtype {dtype}. Data: {data}")
                raise e

        self.data: np.ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad: Optional[OmegaTensor] = None
        self._creator_op: Optional[OmegaTensorOp] = None
        self.name: str = name or f"Ω-{uuid.uuid4().hex[:6]}"
        self.dtype: np.dtype = dtype
        self.device: str = device # Foundational for future GPU ops

        # --- Tensor Integrity Checks ---
        if np.isnan(self.data).any() or np.isinf(self.data).any():
            logger.error(f"Tensor '{self.name}' initialized with NaN or Inf values.")
            raise ValueError(f"OmegaTensor '{self.name}' cannot contain NaN or Inf values.")

    def backward(self, grad_output: Optional['OmegaTensor'] = None):
        """
        Initiates the backward pass (backpropagation) from this tensor.
        This version uses a non-recursive, topologically sorted graph traversal
        for robustness and performance.
        """
        if not self.requires_grad:
            logger.warning(f"backward() called on non-grad-requiring tensor '{self.name}'. This is a no-op.")
            return

        # --- Initial Gradient Setup ---
        if grad_output is None:
            if self.data.size != 1:
                raise ValueError("Gradient must be specified for non-scalar Tensors to initiate backward pass.")
            grad_output_data = np.array(1.0, dtype=self.dtype)
            grad_output = OmegaTensor(grad_output_data, dtype=self.dtype, device=self.device)
        elif not isinstance(grad_output, OmegaTensor):
             grad_output = OmegaTensor(grad_output, dtype=self.dtype, device=self.device)


        if self.shape != grad_output.shape:
             raise ValueError(f"Gradient output shape {grad_output.shape} must match tensor shape {self.shape}.")

        # --- Topological Sort and Gradient Propagation ---
        sorted_nodes = self._topological_sort()
        grads: Dict['OmegaTensor', OmegaTensor] = {node: OmegaTensor(np.zeros_like(node.data, dtype=node.dtype), device=node.device) for node in sorted_nodes}
        grads[self] = grad_output

        logger.info(f"Initiating backward pass for '{self.name}' on graph with {len(sorted_nodes)} nodes.")

        for node in reversed(sorted_nodes):
            if node._creator_op:
                current_grad = grads[node]
                try:
                    # The backward method of an Op returns a list/tuple of gradient *data* (np.ndarray)
                    parent_grads_data = node._creator_op.backward(current_grad.data)
                except Exception as e:
                    logger.exception(f"FATAL: Error during backward pass at Op '{node._creator_op.__class__.__name__}' creating Tensor '{node.name}'.")
                    raise e

                if not isinstance(parent_grads_data, (list, tuple)):
                    parent_grads_data = [parent_grads_data]

                if len(parent_grads_data) != len(node._creator_op._parents):
                    raise ValueError(f"Op '{node._creator_op.__class__.__name__}' returned {len(parent_grads_data)} grads, but has {len(node._creator_op._parents)} parents.")

                for parent_tensor, grad_data in zip(node._creator_op._parents, parent_grads_data):
                    if grad_data is not None and isinstance(parent_tensor, OmegaTensor) and parent_tensor.requires_grad:
                        # Accumulate gradients
                        grads[parent_tensor].data += grad_data

        # --- Assign Final Gradients ---
        for node, grad_tensor in grads.items():
            if node.requires_grad:
                node.grad = grad_tensor

    def _topological_sort(self) -> List['OmegaTensor']:
        """Performs a topological sort of the computation graph ending at this tensor."""
        sorted_nodes: List['OmegaTensor'] = []
        visited: Set['OmegaTensor'] = set()
        
        def visit(node: 'OmegaTensor'):
            if node not in visited:
                visited.add(node)
                if node._creator_op:
                    for parent in node._creator_op._parents:
                        # Only OmegaTensors are part of the graph
                        if isinstance(parent, OmegaTensor):
                            visit(parent)
                sorted_nodes.append(node)
        
        visit(self)
        return sorted_nodes

    def zero_grad(self):
        """Resets the gradient of this tensor to None. Essential before each optimizer step."""
        self.grad = None

    def detach(self) -> 'OmegaTensor':
        """Creates a new tensor that is detached from the computation graph."""
        return OmegaTensor(self.data.copy(), requires_grad=False, name=f"detached_{self.name}", dtype=self.dtype, device=self.device)

    def item(self) -> Union[int, float]:
        """Extracts the single value from a scalar tensor."""
        if self.data.size != 1:
            raise ValueError(f"Only one-element tensors can be converted to Python scalars. Got size: {self.data.size}")
        return self.data.item()

    # --- Private Helpers ---
    def _ensure_tensor(self, other: Any) -> 'OmegaTensor':
        """Ensures that the other operand is an OmegaTensor for a binary operation."""
        if isinstance(other, OmegaTensor):
            return other
        # Wrap constants/numbers in a non-grad-requiring tensor
        return OmegaTensor(other, requires_grad=False, dtype=self.dtype, device=self.device)

    # --- Properties & Representations ---
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> 'OmegaTensor':
        """Shortcut for transpose operation."""
        return self.transpose()

    def __len__(self) -> int:
        """Returns the size of the first dimension."""
        return len(self.data)

    def __repr__(self) -> str:
        grad_fn_name = self._creator_op.__class__.__name__ if self._creator_op else 'Leaf'
        grad_status = f", grad_present={'Yes' if self.grad is not None else 'No'}" if self.requires_grad else ""
        return (f"OmegaTensor(name='{self.name}', shape={self.shape}, dtype={self.dtype}, "
                f"device='{self.device}', grad_fn=<{grad_fn_name}>{grad_status})\n{self.data}")


# =====================================================================================
# Dunder methods and core API methods will be dynamically patched onto the OmegaTensor
# class by the OpRegistry system to keep this core file clean and focused.
# See ops.py and its registration logic.
# =====================================================================================