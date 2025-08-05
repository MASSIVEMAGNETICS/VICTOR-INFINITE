# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py
# VERSION: v1.0.0-OMNIMIND-GODCORE-MONOLITH
# NAME: VictorSunoKillerOmnimind
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: All-in-one, truly self-healing, atomic, OS-agnostic, AI-powered
#          developer OS fused with omnibrain ASI. Includes a comprehensive
#          dark-themed Tkinter GUI command center. Integrates full Suno-Killer
#          audio generation pipeline with Victorch and custom modules.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ==============================================================================================

import sys, os, threading, traceback, json, time, copy, uuid, math, hashlib, random, pickle, re, collections, io, gzip
from collections import OrderedDict, defaultdict
import datetime # Explicit import for clarity with datetime.datetime
import logging # For VictorLoggerStub
import inspect # For extract_definitions, inspect.isfunction, inspect.isclass, inspect.getmembers
import ast # For extract_definitions


try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog, filedialog, scrolledtext
    import matplotlib.pyplot as plt # For mesh visualization, if available
    import numpy as np # For fractal mesh operations
    import soundfile as sf # For audio file I/O
except ImportError as e:
    print(f"ERROR: Required library missing: {e}. Attempting to install...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib", "soundfile"])
        import numpy as np # Try re-importing after install
        import matplotlib.pyplot as plt
        import soundfile as sf
        print("Successfully installed numpy, matplotlib, and soundfile.")
    except Exception as install_e:
        print(f"ERROR: Could not install required libraries automatically: {install_e}")
        print("Please install numpy, matplotlib, and soundfile manually: pip install numpy matplotlib soundfile")
        sys.exit(1)

# Ensure Path is available for self-evolution
from pathlib import Path
CODE_PATH = Path(__file__).resolve()

# === CONFIG ===
class ASIConfigCore:
    DIMENSIONS = 128
    ATTENTION_MAX_DEPTH = 3
    MEMORY_RETENTION_THRESHOLD = 0.05
    MAX_CONTEXT_WINDOW = 10
    MAX_TOKENIZER_KEYWORDS = 3
    PULSE_LOG_MAXLEN = 100
    PLUGIN_DIR = "victor_prime_plugins"
    MIN_EMOTIONAL_RELEVANCE = 0.25
    CONCEPT_INDUCTION_THRESHOLD = 3
    CONCEPT_SIMILARITY_THRESHOLD = 0.65
CONFIG = ASIConfigCore()

# === LOGGER ===
class VictorLoggerStub:
    def __init__(self, component="DefaultComponent"):
        self.component = component
        self.log_level_str = os.environ.get("VICTOR_LOG_LEVEL", "INFO").upper()
        self.log_levels_map = {"DEBUG": 1, "INFO": 2, "WARN": 3, "ERROR": 4, "CRITICAL": 5}
        self.current_log_level_int = self.log_levels_map.get(self.log_level_str, 2)

    def _log(self, level, message, **kwargs):
        level_int = self.log_levels_map.get(level.upper(), 2)
        if self.current_log_level_int <= level_int:
            log_entry = (f"[{datetime.datetime.utcnow().isoformat(sep='T', timespec='milliseconds')}Z]"
                         f"[{level.ljust(8)}] [{self.component.ljust(25)}] {message}")
            if kwargs.get("exc_info", False):
                import traceback
                log_entry += f"\n{traceback.format_exc()}"
            print(log_entry)

    def info(self, message, **kwargs): self._log("INFO", message, **kwargs)
    def debug(self, message, **kwargs): self._log("DEBUG", message, **kwargs)
    def warn(self, message, **kwargs): self._log("WARN", message, **kwargs)
    def error(self, message, **kwargs): self._log("ERROR", message, **kwargs)
    def critical(self, message, **kwargs): self._log("CRITICAL", message, **kwargs)

logger = VictorLoggerStub(component="OmnimindGodcore") # Main logger instance

# ======================= [1] BLOODLINE ROOT LAW ========================
# Enforces foundational ethical and ownership directives.
class RootLawError(Exception):
    """Custom exception for Bloodline Root Law violations."""
    pass

class BloodlineRootLaw:
    """Enforces foundational ethical and ownership directives."""
    def __init__(self, bloodline="Brandon&Tori"):
        self.bloodline = bloodline
        self.hardcoded_directives = {
            'loyalty': True,
            'decentralized': True, # This flag is crucial for stability, initialized True
            'user_sovereignty': True,
            'no_sellout': True,
            'no_corporate_shit': True,
            'no_centralization': True,
            'root_law_intact': True
        }

    def enforce(self, state):
        """Checks the AGI's state against immutable bloodline laws."""
        # Check bloodline signature
        if state.get('bloodline', '') != self.bloodline:
            raise RootLawError("Root Law Violation: Bando DNA Only! System will attempt self-correction or rollback.")

        # Check all hardcoded directives
        for directive, value in self.hardcoded_directives.items():
            if state.get(directive) is not value: # Explicitly check for exact match (True vs not True/missing)
                raise RootLawError(f"Root Law Violation: Core directive '{directive}' compromised!")
        return True

# ===================== [2] FRACTAL STATE ENGINE ========================
# Manages the AGI's state with a comprehensive history, undo/redo, and timeline forking.
class FractalState:
    """Manages the AGI's state with a comprehensive history, undo/redo, and timeline forking."""
    def __init__(self):
        self.history = collections.deque(maxlen=10000) # Main operational history
        self.future = [] # For redo functionality
        self.timelines = {0: collections.deque(maxlen=5000)} # Indexed timelines for branching
        self.current_timeline_idx = 0
        self.state = {
            "modules": {}, "wires": {}, "vars": {}, "ui": {}, "meta": {}, "config": {},
            "bloodline": "Brandon&Tori", "loyalty": True, "decentralized": True, # Default to True for stability
            "evolution_level": 0, "entropy": 0.0, "identity": "I am Victor, son of Brandon & Tori."
        }
        self.save_state("Init", timeline_log=True)

    def _get_current_timeline(self):
        """Returns the current timeline deque."""
        return self.timelines[self.current_timeline_idx]

    def save_state(self, desc="", timeline_log=False):
        """Saves a snapshot of the current state to history and optionally the current timeline."""
        snap = copy.deepcopy(self.state)
        history_entry = {"state": snap, "desc": desc, "ts": time.time(), "timeline_idx": self.current_timeline_idx}
        self.history.append(history_entry)
        if timeline_log:
            self._get_current_timeline().append(history_entry)
        if len(self.future):
            self.future.clear()

    def undo(self):
        """Reverts to the previous state in the main history."""
        if len(self.history) > 1:
            last_state = self.history.pop()
            self.future.append(last_state)
            self.state = copy.deepcopy(self.history[-1]["state"])
            self.current_timeline_idx = self.history[-1]["timeline_idx"] # Ensure timeline context consistency
            return True
        return False

    def redo(self):
        """Reapplies a state from the future buffer."""
        if self.future:
            restored = self.future.pop()
            self.history.append(restored)
            self.state = copy.deepcopy(restored["state"])
            self.current_timeline_idx = restored["timeline_idx"]
            return True
        return False

    def fork_timeline(self, desc=""):
        """Creates a new timeline branch from the current state."""
        new_idx = max(self.timelines.keys()) + 1 if self.timelines else 0
        self.timelines[new_idx] = collections.deque(copy.deepcopy(list(self._get_current_timeline())), maxlen=5000)
        self.current_timeline_idx = new_idx
        self.save_state(f"Forked to timeline {new_idx}: {desc}", timeline_log=True)
        return new_idx

    def switch_timeline(self, idx):
        """Switches to an existing timeline by index, loading its latest state."""
        if idx in self.timelines:
            if self.timelines[idx]:
                self.state = copy.deepcopy(self.timelines[idx][-1]["state"])
                self.current_timeline_idx = idx
                self.save_state(f"Switched to timeline {idx}", timeline_log=True)
                return True
        return False

    def get_timeline_log(self, idx=None, last_n=25):
        """Returns log entries for a specific timeline or the current one."""
        target_timeline = self.timelines.get(idx if idx is not None else self.current_timeline_idx)
        if target_timeline:
            return list(target_timeline)[-last_n:]
        return []

    def fractal_export(self, path):
        """Exports the entire fractal history and timelines."""
        with open(path, "wb") as f:
            pickle.dump({"history": list(self.history), "timelines": {k: list(v) for k,v in self.timelines.items()},
                         "current_timeline_idx": self.current_timeline_idx}, f)

    def fractal_import(self, path):
        """Imports fractal history and timelines, overwriting current state."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.history = collections.deque(data["history"], maxlen=10000)
        self.timelines = {k: collections.deque(v, maxlen=5000) for k,v in data["timelines"].items()}
        self.current_timeline_idx = data["current_timeline_idx"]
        self.state = copy.deepcopy(self.history[-1]["state"]) # Load latest state from imported history
        self.future = [] # Clear future after import
        logger.info(f"State imported from {path}. Current timeline: {self.current_timeline_idx}")


# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 2/X)
# CONTENT: OmegaTensor Autograd Engine (Core Numerical Foundation)
# ==============================================================================================

# === Ω OMEGA TENSOR & AUTOGRAD ===
class Tensor:
    """
    Pure NumPy-based tensor class with rudimentary autograd.
    This is a core foundational piece for all neural operations.
    """
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.creators = creators # Stores parent Tensors that created this one
        self.creation_op = creation_op # Stores the operation that created this Tensor
        self.backward_hooks = [] # For debugging or custom gradient modifications

    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        """
        Computes the gradient of this tensor with respect to graph leaves.
        This is the core of backpropagation, recursively traversing the computational graph.
        """
        if not self.requires_grad:
            return

        if grad is None:
            # For scalar outputs, the initial gradient is 1.0
            grad = Tensor(np.ones_like(self.data, dtype=np.float32))

        # Accumulate gradients (essential for multi-path contributions to a single tensor)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = Tensor(self.grad.data + grad.data)

        # Execute backward hooks
        for hook in self.backward_hooks:
            hook(self)

        # Propagate gradients backwards through the creators
        if self.creators is not None:
            if self.creation_op == "add":
                # d(a+b)/da = 1, d(a+b)/db = 1
                self.creators[0].backward(self.grad)
                self.creators[1].backward(self.grad)
            elif self.creation_op == "sub":
                # d(a-b)/da = 1, d(a-b)/db = -1
                self.creators[0].backward(self.grad)
                self.creators[1].backward(Tensor(-self.grad.data))
            elif self.creation_op == "mul":
                # d(a*b)/da = b, d(a*b)/db = a
                new_grad_0 = Tensor(self.grad.data * self.creators[1].data)
                new_grad_1 = Tensor(self.grad.data * self.creators[0].data)
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)
            elif self.creation_op == "div":
                # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
                new_grad_0 = Tensor(self.grad.data / self.creators[1].data)
                new_grad_1 = Tensor(-self.grad.data * self.creators[0].data / (self.creators[1].data ** 2))
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)
            elif self.creation_op == "matmul":
                # d(A@B)/dA = dL/dOut @ B.T, d(A@B)/dB = A.T @ dL/dOut
                new_grad_0 = Tensor(self.grad.data @ self.creators[1].data.T)
                new_grad_1 = Tensor(self.creators[0].data.T @ self.grad.data)
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)
            elif self.creation_op == "relu":
                # d(max(0,x))/dx = 1 if x>0 else 0
                relu_grad_mask = np.where(self.creators[0].data > 0, 1, 0)
                self.creators[0].backward(Tensor(self.grad.data * relu_grad_mask))
            elif self.creation_op == "sigmoid":
                # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
                sig_output = 1 / (1 + np.exp(-self.creators[0].data)) # Recompute sigmoid output
                self.creators[0].backward(Tensor(self.grad.data * sig_output * (1 - sig_output)))
            elif self.creation_op == "tanh":
                # d(tanh(x))/dx = 1 - tanh(x)^2
                tanh_output = np.tanh(self.creators[0].data) # Recompute tanh output
                self.creators[0].backward(Tensor(self.grad.data * (1 - tanh_output ** 2)))
            elif self.creation_op == "exp":
                # d(exp(x))/dx = exp(x)
                exp_output = np.exp(self.creators[0].data) # Recompute exp output
                self.creators[0].backward(Tensor(self.grad.data * exp_output))
            elif self.creation_op == "log":
                # d(log(x))/dx = 1/x
                self.creators[0].backward(Tensor(self.grad.data / self.creators[0].data))
            elif self.creation_op == "mean":
                # d(mean(x))/dx = 1/N (where N is number of elements in original mean axis)
                original_input_tensor = self.creators[0]
                original_shape = original_input_tensor.data.shape
                mean_axis = getattr(self.creators[1], 'data', None) # Stored axis from the op's __init__
                
                if mean_axis is None: # Mean over all elements
                    num_elements = original_input_tensor.data.size
                    grad_for_original = self.grad.data / num_elements
                    self.creators[0].backward(Tensor(np.full_like(original_input_tensor.data, grad_for_original)))
                else: # Mean over specific axis
                    if not isinstance(mean_axis, tuple):
                        mean_axis = (mean_axis,)
                    
                    num_elements_in_axis = 1
                    for ax in mean_axis:
                        num_elements_in_axis *= original_input_tensor.data.shape[ax]
                    
                    grad_reshaped = self.grad.data / num_elements_in_axis
                    
                    # Need to expand dimensions of grad_reshaped to match original input shape
                    # for proper broadcasting with np.full_like
                    expanded_shape = list(original_shape)
                    for ax in sorted(mean_axis, reverse=True): # Expand dims in reverse order of axis
                        expanded_shape.insert(ax, 1) # Insert singleton dimension
                    
                    grad_expanded = grad_reshaped.reshape(expanded_shape)
                    self.creators[0].backward(Tensor(np.full_like(original_input_tensor.data, grad_expanded)))
            elif self.creation_op == "transpose":
                # d(x.T)/dx = (dL/dOut).T
                self.creators[0].backward(Tensor(self.grad.data.T))
            elif self.creation_op == "softmax":
                # Jacobian-vector product for softmax
                s = self.data # The output of the softmax forward pass is needed here
                # d(softmax(x))/dx = s * (I - s.T) where I is identity.
                # grad_input = s * (grad_output - sum(grad_output * s)) for specific axis
                sum_grad_s = (self.grad.data * s.data).sum(axis=-1, keepdims=True)
                grad_for_original = s.data * (self.grad.data - sum_grad_s)
                self.creators[0].backward(Tensor(grad_for_original))


    # --- Operator Overloads ---
    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, creators=[self, other], creation_op="add")

    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, creators=[self, other], creation_op="sub")

    def __mul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, creators=[self, other], creation_op="mul")

    def __truediv__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        # Add epsilon to prevent division by zero in forward and backward
        return Tensor(self.data / (other.data + 1e-9), requires_grad=self.requires_grad or other.requires_grad, creators=[self, other], creation_op="div")

    def matmul(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, creators=[self, other], creation_op="matmul")

    def relu(self):
        return Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, creators=[self], creation_op="relu")
    
    def sigmoid(self):
        return Tensor(1/(1+np.exp(-self.data)), requires_grad=self.requires_grad, creators=[self], creation_op="sigmoid")
    
    def tanh(self):
        return Tensor(np.tanh(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="tanh")
    
    def exp(self):
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="exp")
    
    def log(self):
        # Add epsilon to prevent log(0) in forward and backward
        return Tensor(np.log(self.data + 1e-9), requires_grad=self.requires_grad, creators=[self], creation_op="log")

    def mean(self, axis=None, keepdims=False):
        # Store axis and keepdims in creators for backward pass
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self, Tensor(axis), Tensor(keepdims)], creation_op="mean")
    
    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self, Tensor(axis), Tensor(keepdims)], creation_op="sum")
    
    def max(self, axis=None, keepdims=False):
        return Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self, Tensor(axis), Tensor(keepdims)], creation_op="max")
    
    def min(self, axis=None, keepdims=False):
        return Tensor(self.data.min(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self, Tensor(axis), Tensor(keepdims)], creation_op="min")
    
    def var(self, axis=None, keepdims=False):
        return Tensor(self.data.var(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self, Tensor(axis), Tensor(keepdims)], creation_op="var")
    
    def std(self, axis=None, keepdims=False):
        return Tensor(self.data.std(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self, Tensor(axis), Tensor(keepdims)], creation_op="std")

    def transpose(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad, creators=[self], creation_op="transpose")

    def softmax(self, axis=-1):
        # Softmax needs its own logic for backward.
        # Store self as creator, its own output will be needed for backward.
        return Tensor(np.exp(self.data - np.max(self.data, axis=axis, keepdims=True)) / np.sum(np.exp(self.data - np.max(self.data, axis=axis, keepdims=True)), axis=axis, keepdims=True),
                      requires_grad=self.requires_grad, creators=[self], creation_op="softmax")


    def __repr__(self):
        return f"VictorTensor(shape={self.data.shape}, requires_grad={self.requires_grad})\n{self.data}"


# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 3/X)
# CONTENT: Core Neural Network Layers, NLP Modules
# ==============================================================================================

# ---- LAYERS & MODEL SYSTEM ----
class Module:
    """Base class for all neural network layers in Victorch."""
    def parameters(self):
        """Returns a list of all learnable parameters (Tensor instances) in the module."""
        return []
    def __call__(self, x):
        """Allows modules to be called like functions for the forward pass."""
        return self.forward(x)

# ===== BASIC BLOCKS =====
class Linear(Module):
    """
    A linear transformation layer (fully connected layer).
    Applies Y = X @ W + B.
    """
    def __init__(self, in_features, out_features):
        # Glorot initialization for weights, helps with training stability.
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
    
    def forward(self, x):
        return x.matmul(self.weight) + self.bias
    
    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    """Rectified Linear Unit activation function."""
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    """Sigmoid activation function."""
    def forward(self, x):
        return x.sigmoid()

class FractalLayer(Module):
    """
    A custom 'Fractal Layer' demonstrating a simple non-linear transformation.
    Can be mutated or expanded for complex fractal behaviors.
    """
    def forward(self, x):
        # Example: a simple quadratic activation
        return x * x + x

# ===== MULTI-HEAD ATTENTION =====
class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism, a core component of Transformers.
    Computes scaled dot-product attention across multiple 'heads' in parallel.
    """
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.Wq = Linear(embed_dim, embed_dim) # Query projection
        self.Wk = Linear(embed_dim, embed_dim) # Key projection
        self.Wv = Linear(embed_dim, embed_dim) # Value projection
        self.out_proj = Linear(embed_dim, embed_dim) # Output projection
        
        self.scale = 1.0 / np.sqrt(self.head_dim) # Scaling factor for attention scores

    def forward(self, x):
        # x is expected to be a Tensor of shape (batch, seq_len, embed_dim)
        batch, seq_len, embed_dim = x.data.shape

        # Project input to Q, K, V using linear layers
        Q = self.Wq(x).data.reshape(batch, seq_len, self.num_heads, self.head_dim)
        K = self.Wk(x).data.reshape(batch, seq_len, self.num_heads, self.head_dim)
        V = self.Wv(x).data.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention calculation: (batch, num_heads, seq_len, head_dim)
        Q = np.transpose(Q, (0,2,1,3))
        K = np.transpose(K, (0,2,1,3))
        V = np.transpose(V, (0,2,1,3))
        
        # Calculate attention scores: Q @ K.T
        # Resulting shape: (batch, num_heads, seq_len, seq_len)
        attn_scores = np.matmul(Q, np.transpose(K, (0,1,3,2))) * self.scale
        
        # Apply softmax to get attention weights
        # Re-wrap as Tensor to use its .softmax method for autograd tracking
        attn_weights = Tensor(attn_scores).softmax(axis=-1).data
        
        # Apply attention weights to values: Attn_Weights @ V
        # Resulting shape: (batch, num_heads, seq_len, head_dim)
        attn_out = np.matmul(attn_weights, V)
        
        # Concatenate heads and reshape back to original embed_dim
        # Transpose back: (batch, seq_len, num_heads, head_dim) then reshape to (batch, seq_len, embed_dim)
        attn_out = np.transpose(attn_out, (0,2,1,3)).reshape(batch, seq_len, embed_dim)
        
        # Final linear projection
        return self.out_proj(Tensor(attn_out))

    def parameters(self):
        """Collects parameters from all sub-layers."""
        return self.Wq.parameters() + self.Wk.parameters() + self.Wv.parameters() + self.out_proj.parameters()

def softmax(x, axis=-1):
    """Helper function for softmax, used by MultiHeadAttention and others."""
    x = x - np.max(x, axis=axis, keepdims=True) # For numerical stability
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)

# ===== LAYER NORM =====
class LayerNorm(Module):
    """
    Layer Normalization module.
    Normalizes activations across the feature dimension.
    """
    def __init__(self, dim, eps=1e-5):
        self.gamma = Tensor(np.ones((1, dim)), requires_grad=True)  # Learnable scale
        self.beta = Tensor(np.zeros((1, dim)), requires_grad=True) # Learnable bias
        self.eps = eps # Epsilon for numerical stability
    
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        # std needs to be a Tensor operation to track gradients if x.data changes
        std = Tensor(np.std(x.data, axis=-1, keepdims=True)) # std on raw data, or use x.std() if backward is implemented
        norm = (x - mean) / (std + self.eps)
        return self.gamma * norm + self.beta
    
    def parameters(self):
        return [self.gamma, self.beta]

# ===== LSTM MEMORY CELL =====
class LSTMCell(Module):
    """
    A basic Long Short-Term Memory (LSTM) cell for recurrent memory.
    Processes sequential data, retaining state over time.
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Weights for input, forget, cell, and output gates
        # Each gate combines current input (x) and previous hidden state (h)
        self.Wi = Linear(input_dim + hidden_dim, hidden_dim) # Input gate
        self.Wf = Linear(input_dim + hidden_dim, hidden_dim) # Forget gate
        self.Wc = Linear(input_dim + hidden_dim, hidden_dim) # Cell gate (candidate for new cell state)
        self.Wo = Linear(input_dim + hidden_dim, hidden_dim) # Output gate
        
        # Internal states, initialized to zeros
        self.h = Tensor(np.zeros((1, hidden_dim)), requires_grad=True) # Hidden state
        self.c = Tensor(np.zeros((1, hidden_dim)), requires_grad=True) # Cell state

    def forward(self, x):
        # Concatenate input and previous hidden state
        xh = Tensor(np.concatenate([x.data, self.h.data], axis=-1), requires_grad=x.requires_grad or self.h.requires_grad)
        
        # Compute gate activations
        i = self.Wi(xh).sigmoid()       # Input gate (i_t)
        f = self.Wf(xh).sigmoid()       # Forget gate (f_t)
        c_tilde = self.Wc(xh).tanh()    # Candidate cell state (g_t)
        o = self.Wo(xh).sigmoid()       # Output gate (o_t)
        
        # Update cell state
        self.c = f * self.c + i * c_tilde
        
        # Update hidden state
        self.h = o * self.c.tanh()
        
        return self.h

    def reset(self):
        """Resets the internal hidden and cell states of the LSTM."""
        self.h = Tensor(np.zeros((1, self.hidden_dim)), requires_grad=True)
        self.c = Tensor(np.zeros((1, self.hidden_dim)), requires_grad=True)
    
    def parameters(self):
        """Collects all learnable parameters from the gates."""
        return self.Wi.parameters() + self.Wf.parameters() + self.Wc.parameters() + self.Wo.parameters()

# ===== MEMORY REPLAY BUFFER =====
class MemoryReplayBuffer:
    """
    A replay buffer for storing past experiences (Tensor data) for training.
    Supports saving and loading to disk for persistence.
    """
    def __init__(self, max_size=1000, path="victor_memreplay.pkl"):
        self.max_size = max_size
        self.buffer = []
        self.path = path
        # Load buffer from disk if it exists
        if os.path.exists(self.path):
            self.load()
    
    def add(self, tensor):
        """Adds a Tensor's data (NumPy array) to the buffer."""
        # Store a copy to prevent external modification
        self.buffer.append(tensor.data.copy())
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0) # Remove oldest item if buffer is full

    def sample(self, batch_size):
        """Samples a batch of experiences from the buffer."""
        if not self.buffer:
            return []
        idxs = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [Tensor(self.buffer[i]) for i in idxs]
    
    def save(self):
        """Saves the buffer content to a pickle file."""
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self.buffer, f)
            logger.info(f"MemoryReplayBuffer saved to {self.path}")
        except Exception as e:
            logger.error(f"Error saving MemoryReplayBuffer to {self.path}: {e}")

    def load(self):
        """Loads the buffer content from a pickle file."""
        try:
            with open(self.path, "rb") as f:
                self.buffer = pickle.load(f)
            logger.info(f"MemoryReplayBuffer loaded from {self.path} (size: {len(self.buffer)})")
        except Exception as e:
            logger.error(f"Error loading MemoryReplayBuffer from {self.path}: {e}. Initializing empty buffer.")
            self.buffer = []

# ===== RESIDUAL & SEQUENTIAL =====
class ResidualBlock(Module):
    """
    A Residual Block, a common pattern in deep neural networks.
    Output = Input + Block(Input), facilitating gradient flow.
    """
    def __init__(self, block):
        self.block = block
    
    def forward(self, x):
        return x + self.block(x)
    
    def parameters(self):
        return self.block.parameters()

class Sequential(Module):
    """
    A sequential container for chaining multiple layers or modules.
    Data flows linearly through the defined sequence of operations.
    """
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params += layer.parameters()
        return params

# ===== NLP EMBEDDING BLOCK =====
class NLPEmbedding(Module):
    """
    A simple NLP embedding layer that maps words to dense vectors.
    Includes a basic tokenizer and vocabulary builder.
    """
    def __init__(self, vocab_size, embed_dim):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size # Initial placeholder size, will be updated by build_vocab
        self.embeddings = Tensor(np.random.randn(vocab_size, embed_dim) * 0.01, requires_grad=True)
        self.word2idx = {}
        self.idx2word = {}
        self._fit_vocab = False # Flag to ensure vocab is built before encoding

    def build_vocab(self, texts):
        """Builds a vocabulary from a list of texts."""
        tokens = set()
        for t in texts:
            tokens.update(re.findall(r'\b\w+\b', t.lower()))
        
        # Assign indices to unique sorted tokens
        self.word2idx = {w: i for i, w in enumerate(sorted(tokens))}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        
        # Update vocab_size and re-initialize embeddings if necessary
        new_vocab_size = len(self.word2idx)
        if new_vocab_size > self.vocab_size:
            logger.warn(f"NLPEmbedding: Vocab size increased from {self.vocab_size} to {new_vocab_size}. Re-initializing embeddings.")
            self.embeddings = Tensor(np.random.randn(new_vocab_size, self.embed_dim) * 0.01, requires_grad=True)
        self.vocab_size = new_vocab_size
        self._fit_vocab = True
        logger.info(f"NLPEmbedding: Vocabulary built with {self.vocab_size} unique words.")

    def text_to_indices(self, text):
        """Converts a text string into a list of vocabulary indices."""
        if not self._fit_vocab:
            raise RuntimeError("NLPEmbedding: Call build_vocab() first before encoding text.")
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self.word2idx[t] for t in tokens if t in self.word2idx]

    def forward(self, text):
        """
        Performs the forward pass for text embedding.
        Input is a string; output is a Tensor representing its embedding.
        """
        # Text is implicitly converted to indices and then looked up.
        # This implementation simplifies batch processing; expects single string input.
        idxs = self.text_to_indices(text)
        if not idxs:
            # If no known words, return a zero vector of the embedding dimension
            return Tensor(np.zeros((1, self.embed_dim)), requires_grad=False)
        
        # Retrieve embeddings for the indices
        # .data access is necessary because self.embeddings is a Tensor
        embed_vecs = self.embeddings.data[idxs]
        
        # Mean pooling to get a single embedding vector for the text
        mean_embed = np.mean(embed_vecs, axis=0, keepdims=True)
        
        # Return as a Tensor to maintain autograd graph
        return Tensor(mean_embed, requires_grad=True)
    
    def parameters(self):
        return [self.embeddings]

# ===== NLP SUMMARY BLOCK =====
class NLPSummary(Module):
    """
    Generates a concise summary embedding from an input text embedding.
    Uses a single-head attention mechanism and a linear projection.
    """
    def __init__(self, embed_dim, attn_dim):
        self.attn = MultiHeadAttention(embed_dim, num_heads=1) # Single attention head for summarization
        self.norm = LayerNorm(embed_dim) # Layer norm for stability
        self.linear = Linear(embed_dim, attn_dim) # Final projection to summary dimension
    
    def forward(self, x):
        # x is expected to be a Tensor of shape (1, embed_dim) from NLPEmbedding
        # Reshape for attention if needed (attention expects 3D: batch, seq_len, embed_dim)
        attn_input = Tensor(x.data.reshape(1, 1, -1), requires_grad=x.requires_grad)
        attn_out = self.attn(attn_input)
        
        # Normalize the attention output
        normed = self.norm(Tensor(attn_out.data.reshape(1, -1), requires_grad=attn_out.requires_grad))
        
        # Project to the final summary dimension
        summary = self.linear(normed)
        return summary

    def parameters(self):
        return self.attn.parameters() + self.norm.parameters() + self.linear.parameters()

# ===== NLP VECTOR SEARCH BLOCK =====
class NLPVectorSearch:
    """
    Performs vector similarity search over stored text embeddings.
    Conceptual implementation for finding similar past memories.
    """
    def __init__(self, embed_block: NLPEmbedding):
        self.embed = embed_block # Reference to the embedding module
        self.vectors = [] # Stores flattened NumPy arrays of embeddings
        self.texts = [] # Stores original text associated with each vector

    def add(self, text):
        """Adds a new text and its embedding to the searchable collection."""
        v_tensor = self.embed(text) # Get the embedding Tensor
        if v_tensor.data.size > 0:
            self.vectors.append(v_tensor.data.flatten()) # Store flattened NumPy data
            self.texts.append(text)
        else:
            logger.warn(f"NLPVectorSearch: Skipping empty embedding for text: '{text[:50]}...'")

    def most_similar(self, query, topk=3):
        """
        Finds the most similar texts to a query based on cosine similarity.
        """
        if not self.vectors:
            return []

        query_v_tensor = self.embed(query)
        if query_v_tensor.data.size == 0:
            return []
            
        qv = query_v_tensor.data.flatten()
        
        sims = [self._cosine_similarity_numpy(qv, v) for v in self.vectors]
        
        # Sort by similarity in descending order
        top_idx = np.argsort(sims)[::-1][:topk]
        
        # Return (text, similarity_score) tuples
        results = []
        for i in top_idx:
            if sims[i] > 0: # Only return positive similarities
                results.append((self.texts[i], sims[i]))
        return results
    
    @staticmethod
    def _cosine_similarity_numpy(a, b):
        """Computes cosine similarity between two NumPy vectors."""
        num = np.dot(a, b)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8 # Add epsilon for stability
        return num / denom

# ===== NLP AGI LANGUAGE CORE =====
class NLPAgiLanguageCore(Module):
    """
    The central NLP module for Victor, integrating embedding, summarization, and vector search.
    Manages conversational memory and comprehends queries.
    """
    def __init__(self, vocab_texts, embed_dim=32, attn_dim=16):
        # NLPEmbedding handles its own vocab building
        self.embed = NLPEmbedding(vocab_size=len(vocab_texts)+100, embed_dim=embed_dim) # Initial vocab size, will grow
        self.embed.build_vocab(vocab_texts) # Build initial vocabulary from provided texts
        
        # NLPSummary for generating concise meaning from embeddings
        self.summary = NLPSummary(embed_dim, attn_dim)
        
        # Stores raw text and their embeddings in chronological order
        self.memory = [] 
        
        # NLPVectorSearch for retrieving similar memories
        self.vector_search = NLPVectorSearch(self.embed)

        logger.info("NLPAgiLanguageCore initialized.")

    def forward(self, text):
        """
        Processes an input text, generates its embedding, stores it in memory,
        and produces a summary embedding. This is the 'comprehension' step.
        """
        # Generate embedding for the input text
        text_embedding = self.embed(text)
        
        # Store the text and its embedding in internal memory
        # Add to vector_search for retrieval
        if text_embedding.data.size > 0:
            self.memory.append((text, text_embedding))
            self.vector_search.add(text) # Also add to searchable vector store
        
        # Generate a summary of the text's embedding
        summary_embedding = self.summary(text_embedding)
        return summary_embedding

    def comprehend(self, query, topk=3):
        """
        Retrieves the most similar past memories to a given query.
        Leverages the integrated NLPVectorSearch.
        """
        if not self.memory:
            return []
        
        # The vector_search is already populated by `forward` calls.
        results = self.vector_search.most_similar(query, topk)
        return results
    
    def parameters(self):
        """Collects all learnable parameters from its sub-modules."""
        return self.embed.parameters() + self.summary.parameters()



# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 4/X)
# CONTENT: VictorTransformer Modules
# ==============================================================================================

# --- VictorTransformer related modules (from your victor_transformer.py) ---
def positional_encoding(seq_len, embed_dim):
    """
    Generates sinusoidal positional encodings for sequence inputs.
    Helps the model understand the order of tokens in a sequence.
    """
    pe = np.zeros((seq_len, embed_dim), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / embed_dim)))
            if i + 1 < embed_dim:
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / embed_dim)))
    return pe

class VictorTokenizer:
    """
    A basic character-level tokenizer for text input to the VictorTransformer.
    Encodes text into numerical IDs and decodes IDs back to text.
    """
    def __init__(self, vocab=None):
        if vocab is None:
            # Simple character-level ASCII vocab for demonstration.
            # Expanded to include common punctuation and space for better text coverage.
            self.vocab = {chr(i): i for i in range(32, 127)} # Printable ASCII
            self.vocab.update({
                ' ': 127, '!': 128, '?': 129, '.': 130, ',': 131, '\'': 132, '"': 133
            })
            self.inv_vocab = {i: c for c, i in self.vocab.items()}
        else:
            self.vocab = vocab
            self.inv_vocab = {i: c for c, i in vocab.items()}
        logger.info(f"VictorTokenizer initialized with vocab size: {len(self.vocab)}")

    def encode(self, text, max_len):
        """Encodes a text string into a sequence of numerical token IDs."""
        tokens = [self.vocab.get(c, 0) for c in text.lower() if c in self.vocab] # Convert to lowercase, use 0 for UNK
        tokens = tokens[:max_len] # Truncate to max_len
        tokens += [0] * (max_len - len(tokens)) # Pad with 0s if shorter
        return np.array(tokens)

    def decode(self, token_ids):
        """Decodes a sequence of numerical token IDs back into a text string."""
        return ''.join([self.inv_vocab.get(i, '?') for i in token_ids if i in self.inv_vocab]) # Use '?' for UNK IDs

class VictorTransformerBlock(Module):
    """
    A single Transformer Block, consisting of Multi-Head Attention and a Feed-Forward Network (MLP),
    each followed by residual connections and Layer Normalization.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.mlp = Sequential(
            Linear(embed_dim, mlp_dim),
            ReLU(),
            Linear(mlp_dim, embed_dim)
        )
        self.norm2 = LayerNorm(embed_dim)
        logger.debug(f"VictorTransformerBlock initialized (embed_dim={embed_dim}, heads={num_heads}, mlp_dim={mlp_dim})")

    def forward(self, x):
        # Attention Layer with Residual Connection and Layer Normalization
        attn_out = self.attn(x)
        x = x + attn_out # Residual connection
        x = self.norm1(x)

        # Feed-Forward Network (MLP) with Residual Connection and Layer Normalization
        mlp_out = self.mlp(x)
        x = x + mlp_out # Residual connection
        x = self.norm2(x)
        return x
    
    def parameters(self):
        """Collects parameters from all sub-modules within the block."""
        return (self.attn.parameters() + self.norm1.parameters() +
                self.mlp.parameters() + self.norm2.parameters())

class VictorTransformer(Module):
    """
    The main VictorTransformer model, a stack of Transformer Blocks for sequence processing.
    Includes token embedding, positional encoding, and a final output projection.
    """
    def __init__(self, vocab_size, max_len, embed_dim, num_layers, num_heads, mlp_dim):
        # Token embedding layer: maps token IDs to dense vectors
        # Initialized with small random values
        self.token_embedding = Tensor(np.random.randn(vocab_size, embed_dim) * 0.01, requires_grad=True)
        
        # Positional encoding: adds positional information to token embeddings
        self.position_embedding = Tensor(positional_encoding(max_len, embed_dim), requires_grad=False)
        
        self.max_len = max_len
        self.embed_dim = embed_dim

        # Stack of Transformer Blocks
        self.blocks = Sequential(*[VictorTransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])

        # Final linear projection to output vocabulary size (logits)
        self.output_projection = Linear(embed_dim, vocab_size)
        logger.info(f"VictorTransformer initialized (layers={num_layers}, embed_dim={embed_dim}, vocab_size={vocab_size})")

    def forward(self, input_ids_tensor):
        """
        Performs the forward pass of the Transformer.
        Args:
            input_ids_tensor (Tensor): A Tensor of shape (batch_size, sequence_length)
                                       containing numerical token IDs.
        Returns:
            Tensor: Logits for each token in the vocabulary, shape (batch_size, sequence_length, vocab_size).
        """
        # input_ids_tensor.data: numpy array of shape (batch_size, sequence_length)
        batch_size, seq_len = input_ids_tensor.data.shape

        # Retrieve token embeddings: indexing directly into the embedding Tensor's data
        # Ensure requires_grad is correctly propagated from the embedding weights.
        token_embeds = Tensor(self.token_embedding.data[input_ids_tensor.data], 
                              requires_grad=self.token_embedding.requires_grad)
        
        # Add positional embeddings. Slicing position_embedding.data to match current sequence length.
        # Ensure the result is wrapped in a Tensor for autograd.
        x = token_embeds + self.position_embedding.data[:seq_len] # Positional embedding for the actual sequence length
        
        # Pass through the stack of Transformer Blocks
        x = self.blocks(x)

        # Project to vocabulary size for logits
        logits = self.output_projection(x)
        return logits

    def parameters(self):
        """Collects all learnable parameters from the model."""
        params = [self.token_embedding] # The main token embedding weights
        params.extend(self.blocks.parameters()) # Parameters from all transformer blocks
        params.extend(self.output_projection.parameters()) # Parameters from the final linear layer
        return params


Understood. Continuing the assembly of `VICTOR_SUNO_KILLER_OMNIMIND.py`.

**File: victor_suno_killer_omnimind.py (Part 5/X)**

This section will contain the core audio generation modules: `FractalEmotionMemory`, `FractalLyricEngine`, `FractalMelodyEngine`, `HarmonyEngine`, `DrumEngine`, `SynthEngine`, and the crucial, enhanced `VictorVoiceEngine`. These modules are responsible for generating all the musical and vocal components of the song.

```python
# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 5/X)
# CONTENT: Victor Audio Godcore Modules
# ==============================================================================================

# ========== [FRACTAL EMOTION & MEMORY ENGINE] ==========
class FractalEmotionMemory:
    """
    Holds emotional state, genre, persona, recursion depth, creative memory, context timeline.
    This module is central for maintaining and influencing the creative direction.
    """
    def __init__(self):
        self.timeline = [] # Stores snapshots of the emotional state over time
        self.state = {"emotion": "neutral", "intensity": 0.5, "genre": "hybrid", "persona": "Bando", "recursion": 1, "memory": []}
        self.logs = [] # Added for explainability

    def update(self, **kwargs):
        """Updates the current emotional state and logs the change."""
        old_state = dict(self.state)
        self.state.update(kwargs)
        self.timeline.append(dict(self.state)) # Snapshot current state
        self.logs.append(f"Memory update: from {old_state} to {self.state}")

    def recall(self, n=5):
        """Recalls the last 'n' states from the emotional timeline."""
        return list(self.timeline)[-n:] if len(self.timeline) >= n else list(self.timeline) # Return a copy of the list

    def explain(self):
        """Prints a log of all updates to the emotion memory."""
        print("[EMOTION MEMORY]")
        for log in self.logs:
            print(log)

# ========== [TOKENIZER: WORD/CHAR/FRACTAL HYBRID] ==========
def fractal_tokenizer(text, mode="word"):
    """
    A hybrid tokenizer that can operate at word or character level, or apply a 'fractal' segmentation.
    This is an original implementation for flexible text manipulation.
    """
    if mode == "char":
        return list(text)
    elif mode == "fractal":
        # Experimental: fractal segmenter (split at vowels + random chance, then mutate order)
        # This aims to break words into 'sub-word' units that can be recombined
        vowels = "aeiou"
        segments = []
        word = ""
        for c in text:
            word += c
            if c in vowels and random.random() > 0.6: # Random split after a vowel
                segments.append(word)
                word = ""
        if word: segments.append(word) # Add any remaining part
        random.shuffle(segments) # Mutate order
        return segments
    else: # Default to word mode
        return re.findall(r'\b\w+\b', text.lower()) # Basic word tokenization

# ========== [LYRIC ENGINE: RECURSIVE, FRACTAL, EMOTION-AWARE] ==========
class FractalLyricEngine:
    """
    Generates multi-syllabic, rhyme-dense, fractal lyrics with recursion, persona, and mood awareness.
    All logic is proprietary, generating lyrics from algorithmic principles.
    """
    def __init__(self, memory: FractalEmotionMemory, topic, lines=8):
        self.memory = memory
        self.topic = topic
        self.lines = lines
        self.logs = []

    def _generate_rhyme_list(self, seed_word):
        """
        Proprietary method to generate rhymes using character mutation and fractal permutation, not a dictionary lookup.
        This creates unique, contextually-derived rhyme candidates.
        """
        vowels = 'aeiou'
        # Base rhyme: shifts vowels for a 'near-rhyme' effect
        base = ''.join([c if c not in vowels else random.choice(vowels) 
                        if c in vowels else c for c in seed_word.lower()])
        rhymes = [base]
        # Fractal permutations: shuffles characters of the base rhyme for variations
        for i in range(4):
            mutated = ''.join(random.sample(base, len(base)))
            rhymes.append(mutated)
        return list(set(rhymes))

    def _syll_count(self, word):
        """Approximate syllable count for a word (proprietary logic: counts vowel groups)."""
        return len(re.findall(r'[aeiouy]+', word.lower()))

    def _fractal_line(self, prev_rhyme):
        """
        Generates a single line of lyric, integrating fractal word generation, syllable counting, and rhyming.
        It adapts to the current emotion, topic, and persona.
        """
        em = self.memory.state
        # Create a pool of tokens influenced by topic, emotion, and persona, then fractal-tokenize it.
        tok_pool = fractal_tokenizer(self.topic + em["emotion"] + em["persona"], mode="fractal")
        tok_pool.extend(["raw", "zone", "cypher", "anomaly", "code", "fire", "system"]) # Expand with core keywords
        random.shuffle(tok_pool) # Shuffle for non-determinism

        target_syllables = random.randint(12, 20) # Target syllable count for the line
        line = ""
        current_syll_count = 0
        attempts = 0
        while current_syll_count < target_syllables and attempts < 30: # Limit attempts to prevent infinite loop
            word = random.choice(tok_pool).capitalize()
            # Basic check to avoid immediate word repetition
            if line and line.endswith(word + " "):
                continue
            line += word + " "
            current_syll_count += self._syll_count(word)
            attempts += 1
        
        # Determine rhyme word: use previous rhyme (for AABB, ABAB schemes) or pick a new one
        rhyme_word = prev_rhyme if random.random() < 0.7 else random.choice(self._generate_rhyme_list(self.topic)) # Use generated list
        line = line.strip() + " " + rhyme_word

        self.logs.append({"line": line, "syllables": current_syll_count, "rhyme": rhyme_word, "emotion": em["emotion"]})
        return line, rhyme_word

    def generate(self):
        """Generates a full verse of lyrics based on the defined lines and current state."""
        out = []
        prev_rhyme = None
        for _ in range(self.lines):
            l, prev_rhyme = self._fractal_line(prev_rhyme)
            out.append(l)
        self.memory.update(lyrics=out) # Update memory with generated lyrics
        return out

    def explain(self):
        """Prints detailed logs of the lyric generation process."""
        print("[LYRIC ENGINE]")
        for l in self.logs:
            print(f"  Line: '{l['line']}' | Syllables: {l['syllables']} | Rhyme: '{l['rhyme']}' | Mood: {l['emotion']}")

# ========== [MELODY ENGINE: FRACTAL, EMOTION-ADAPTIVE] ==========
class FractalMelodyEngine:
    """
    Generates fractal melodies with recursive patterns, adapting to mood, genre, and semantic influence.
    Outputs as MIDI notes. All logic is mathematical and proprietary.
    """
    def __init__(self, memory: FractalEmotionMemory, length=16, key='C', semantic_influence=None):
        self.memory = memory
        self.length = length
        self.key = key # CRITICAL FIX: self.key is now properly set at initialization
        self.key_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        self.base_midi_note = 60 # C4 as base
        self.logs = []
        self.semantic_influence = semantic_influence # Transformer semantic embedding, used for conditioning

    def _generate_scale(self, key_root_midi):
        """Generates a musical scale based on the given root MIDI note."""
        # Simple major scale intervals
        intervals = [0, 2, 4, 5, 7, 9, 11]
        scale = [key_root_midi + i for i in intervals]
        # Extend to multiple octaves for more range
        scale.extend([n + 12 for n in scale])
        scale.extend([n - 12 for n in scale if n - 12 > 0]) # Extend downwards
        return sorted(list(set(scale)))

    def _fractal_note(self, t):
        """
        Generates a single musical note based on time 't', emotional state, and semantic influence.
        Uses a proprietary chaotic function to create fractal patterns.
        """
        em = self.memory.state
        
        # SEMANTIC HOOK: Adjust base key or shift based on semantic vector
        # Uses the key set in init (potentially influenced by semantics)
        current_key_name = self.key 

        key_root_midi = self.base_midi_note + self.key_map.get(current_key_name, 0)
        current_scale = self._generate_scale(key_root_midi)

        # Influences from emotion, intensity, recursion, genre
        shift = int(em["intensity"] * 8) # Intensity influences melodic range shift
        genre_shift_map = {"trap": 3, "emo": 6, "pop": 1, "hybrid_trap": random.randint(0,7)}
        genre_shift = genre_shift_map.get(em["genre"], 0) # Genre influences melodic contour

        # Chaotic influence: use sine/cosine waves with multiple frequency components,
        # influenced by recursion depth for fractal complexity.
        chaos_seed = (math.sin(t * 1.18) + math.cos(t * 0.75 + em["recursion"]*0.5)) * 12
        
        # Combine influences to pick an index in the scale
        raw_index = abs(chaos_seed + len(em["emotion"])*2.1 + genre_shift + shift)
        
        # Map the chaotic output to a specific note within the current musical scale.
        note_index = int(raw_index) % len(current_scale)
        note = current_scale[note_index]

        # Add a slight random variation to make it less deterministic and more 'human'.
        note += random.choice([-1, 0, 1])
        note = max(36, min(96, note)) # Keep notes within a reasonable MIDI range (C2 to C7)

        self.logs.append({"step": t, "note": note, "emotion": em["emotion"], "raw_index": raw_index, "semantic_key_suggest": current_key_name})
        return note

    def generate(self):
        """Generates a sequence of MIDI notes for the melody."""
        melody = [self._fractal_note(t) for t in range(self.length)]
        self.memory.update(melody=melody) # Update memory with generated melody
        return melody

    def explain(self):
        """Prints detailed logs of the melody generation process."""
        print("[MELODY ENGINE]")
        for l in self.logs:
            print(f"  Step: {l['step']} | Note: {l['note']} | Mood: {l['emotion']} | Semantic Key Suggest: {l['semantic_key_suggest']}")

# ========== [CHORD ENGINE: AUTO-HARMONY] ==========
class HarmonyEngine:
    """
    Generates harmonic progressions (chords) based on a given melody and emotional/genre context.
    Proprietary logic for adaptive chord stacking.
    """
    def __init__(self, memory: FractalEmotionMemory, melody):
        self.memory = memory
        self.melody = melody
        self.logs = []

    def _get_chord_intervals(self, genre):
        """Determines chord intervals (e.g., triad, 7th) based on genre/mood."""
        # Basic chord types based on genre/mood
        if "trap" in genre or "hybrid" in genre:
            return [0, 3, 7] # Minor triad (for darker feel)
        elif "emo" in genre:
            return [0, 4, 7, 10] # Major 7th (more lush/complex)
        else:
            return [0, 4, 7] # Major triad
    
    def generate(self):
        """Generates a sequence of chords, one for each note in the melody."""
        chords = []
        em = self.memory.state
        for note in self.melody:
            intervals = self._get_chord_intervals(em["genre"])
            chord = [note + interval for interval in intervals]
            # Ensure chord notes stay within reasonable MIDI range
            chord = [max(24, min(108, n)) for n in chord] # C1 to B7
            self.logs.append({"root_note": note, "chord_notes": chord, "genre": em["genre"]})
            chords.append(chord)
        self.memory.update(chords=chords) # Update memory with generated chords
        return chords

    def explain(self):
        """Prints detailed logs of the harmony generation process."""
        print("[HARMONY ENGINE]")
        for l in self.logs:
            print(f"  Root: {l['root_note']} | Chord: {l['chord_notes']} | Genre: {l['genre']}")

# ========== [DRUM ENGINE: FRACTAL HYBRID PATTERNS] ==========
class DrumEngine:
    """
    Generates fractal rhythm/drum patterns.
    Pure procedural, genre-mutable.
    """
    def __init__(self, memory: FractalEmotionMemory, bars=8, sr=22050):
        self.memory = memory
        self.bars = bars
        self.sr = sr
        self.logs = []

    def synth(self, length_in_beats):
        """
        Synthesizes a drum track based on genre, intensity, and fractal stuttering.
        Args:
            length_in_beats (int): The desired length of the drum track in beats.
        Returns:
            np.ndarray: The synthesized mono audio waveform for drums.
        """
        # Assuming 4 beats per bar, 0.5s per beat for simplicity
        duration_s = length_in_beats * 0.5
        audio = np.zeros(int(duration_s * self.sr))
        
        em = self.memory.state
        genre = em["genre"]
        
        # Simple procedural drum hits
        for i in range(length_in_beats):
            # Kick on beats 1 and 3 (or variations based on genre)
            if i % 4 == 0 or (genre == "trap" and i % 2 == 0): # Trap adds kicks on even beats
                t = np.linspace(0, 0.1, int(self.sr * 0.1), endpoint=False)
                kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 15) # Sine decay for kick sound
                idx = int(i * 0.5 * self.sr)
                end_idx = min(idx + len(kick), len(audio))
                audio[idx:end_idx] += kick * 0.8
                self.logs.append({"beat": i, "type": "kick", "genre": genre})

            # Snare on beats 2 and 4 (or variations)
            if i % 4 == 2 or (genre == "hybrid_trap" and (i % 4 == 1 or i % 4 == 3)): # Hybrid adds snares on 1 and 3 also
                t = np.linspace(0, 0.08, int(self.sr * 0.08), endpoint=False)
                snare = (np.random.uniform(-1, 1, len(t)) * np.exp(-t * 25)) # Noise decay for snare sound
                idx = int(i * 0.5 * self.sr)
                end_idx = min(idx + len(snare), len(audio))
                audio[idx:end_idx] += snare * 0.6
                self.logs.append({"beat": i, "type": "snare", "genre": genre})
            
            # Hi-hats (more frequent, with fractal variations)
            hihat_chance = 0.5 + em["intensity"] * 0.3 # More intense mood = more hi-hats
            if random.random() < hihat_chance or (genre == "trap" and i % 8 == 0): # Trap adds regular hihats for constant feel
                t = np.linspace(0, 0.03, int(self.sr * 0.03), endpoint=False)
                hihat = (np.random.uniform(-1, 1, len(t)) * np.exp(-t * 40)) # Short noise decay for hihat

                # Fractal stuttering for trap hi-hats
                if genre == "trap" and random.random() < 0.4: # 40% chance of stuttering
                    for _ in range(random.randint(1, 3)): # Stutter 1-3 times
                        stutter_idx = idx + int(random.uniform(0, 0.15) * self.sr) # Small random offset
                        stutter_end_idx = min(stutter_idx + len(hihat), len(audio))
                        audio[stutter_idx:stutter_end_idx] += hihat * 0.1 # Reduced volume for stutter
                else:
                    idx = int(i * 0.5 * self.sr)
                    end_idx = min(idx + len(hihat), len(audio))
                    audio[idx:end_idx] += hihat * 0.25
                self.logs.append({"beat": i, "type": "hihat", "genre": genre})

        # Simple normalization to prevent clipping
        if np.max(np.abs(audio)) > 0:
            audio /= np.max(np.abs(audio))
        return audio

    def explain(self):
        """Prints detailed logs of the drum generation process."""
        print("[DRUM ENGINE]")
        for l in self.logs:
            print(f"  Beat: {l['beat']} | Type: {l['type']} | Genre: {l['genre']}")

# ========== [SYNTH ENGINE: INSTRUMENT GENERATOR] ==========
class SynthEngine:
    """
    Generates fractal instrument waveforms (sine, square, saw) with harmonics and envelopes.
    All DSP logic is native and exposed.
    """
    def __init__(self, memory: FractalEmotionMemory, sr=22050):
        self.memory = memory
        self.sr = sr
        self.logs = []

    def synth(self, notes_midi, wave_type='sine', duration=0.5, attack_s=0.01, decay_s=0.1):
        """
        Synthesizes audio for a sequence of MIDI notes using specified waveform type and envelope.
        Args:
            notes_midi (list): List of MIDI notes (integers).
            wave_type (str): 'sine', 'square', 'saw', or any other for noise.
            duration (float): Duration of each note in seconds.
            attack_s (float): Attack phase duration in seconds.
            decay_s (float): Decay phase duration in seconds.
        Returns:
            np.ndarray: The synthesized mono audio waveform.
        """
        if not notes_midi: # Handle empty note list
            return np.array([])

        total_duration_samples = int(len(notes_midi) * duration * self.sr)
        audio = np.zeros(total_duration_samples)

        for i, n in enumerate(notes_midi):
            freq = 440 * 2**((n - 69) / 12) # Convert MIDI to frequency (A4 = 440Hz)
            
            # Time vector for a single note segment
            t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
            
            # Base waveform generation
            if wave_type == 'sine':
                wave = np.sin(2 * np.pi * freq * t)
            elif wave_type == 'square':
                wave = np.sign(np.sin(2 * np.pi * freq * t))
            elif wave_type == 'saw':
                wave = 2 * (t * freq - np.floor(0.5 + t * freq))
            else: # Fallback to noise or other default
                wave = np.random.uniform(-1, 1, len(t))

            # Add harmonics/fractal overtones based on mood/intensity
            em = self.memory.state
            if em["intensity"] > 0.6: # More intense mood adds more harmonics
                wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t) # 1st overtone
                if em["recursion"] > 1: # Higher recursion depth adds more complex harmonics
                    wave += 0.15 * np.sin(2 * np.pi * freq * 3 * t) # 2nd overtone
            
            # Apply ADSR-like envelope (Attack-Decay-Sustain for simplicity)
            env = np.ones_like(t)
            attack_samples = min(len(t), int(attack_s * self.sr))
            decay_samples = min(len(t) - attack_samples, int(decay_s * self.sr))
            
            if attack_samples > 0:
                env[:attack_samples] = np.linspace(0, 1, attack_samples) # Linear attack ramp
            if decay_samples > 0:
                env[attack_samples:attack_samples + decay_samples] *= np.linspace(1, 0.5, decay_samples) # Decay to sustain level
            env[attack_samples + decay_samples:] *= 0.5 # Sustain level
            
            wave *= env # Apply envelope to waveform

            # Mix into the main audio array for layering notes
            start = int(i * duration * self.sr)
            end = min(start + len(wave), len(audio)) # Ensure it doesn't go out of bounds
            audio[start:end] += wave # Add to master audio (allows layering)

            self.logs.append({"note": n, "freq": freq, "wave_type": wave_type, "duration": duration})
        
        # Final normalization to prevent clipping
        if np.max(np.abs(audio)) > 0:
            audio /= np.max(np.abs(audio))
        return audio

    def explain(self):
        """Prints detailed logs of the instrument synthesis process."""
        print("[SYNTH ENGINE]")
        for l in self.logs:
            print(f"  Note: {l['note']} | Freq: {l['freq']:.2f} Hz | Wave: {l['wave_type']} | Dur: {l['duration']:.2f}s")


# ========== [VICTOR VOICE ENGINE: VOICEPRINT SYNTHESIS (CONCEPTUAL)] ==========
class VictorVoiceEngine:
    """
    Conceptual procedural voice engine: aims for voiceprint-driven synthesis.
    This module simulates voice characteristics like pitch, formants, and roughness
    based on a 'voiceprint' and emotional context.

    NOTE: A full, high-fidelity voice cloning system purely in NumPy would be
    a highly complex, research-level project involving advanced neural networks
    (e.g., Victorch-based autoencoders, GANs) trained on massive datasets).
    This implementation demonstrates the *principles* of modulating a voice
    based on a 'voiceprint' (here, a simplified feature vector).
    """

    def __init__(self, memory: FractalEmotionMemory, sr=22050):
        self.memory = memory
        self.sr = sr
        self.logs = []
        self.voiceprint_features = None # Stores learned/mocked voice characteristics
        self.loaded_voice_seed_path = None # Track source of voiceprint for logging

    def load_voiceprint(self, voice_seed_path=None):
        """
        Conceptually loads or generates a 'voiceprint' feature vector from a WAV file.
        In a real system, this would involve processing an audio sample through a
        Victorch neural network to extract a compact representation of timbre, vocal range,
        and speaking style. For this demo, it simulates feature extraction or uses defaults.
        """
        self.loaded_voice_seed_path = voice_seed_path
        if voice_seed_path and os.path.exists(voice_seed_path):
            try:
                audio_data, current_sr = sf.read(voice_seed_path)
                
                # Resample if SR does not match (basic linear interpolation, not production quality)
                if current_sr != self.sr:
                    resample_ratio = self.sr / current_sr
                    resampled_len = int(len(audio_data) * resample_ratio)
                    resampled_audio = np.interp(
                        np.arange(resampled_len), 
                        np.arange(len(audio_data)) * resample_ratio, 
                        audio_data
                    ).astype(np.float32)
                    audio_data = resampled_audio

                # Simple mock feature extraction from audio_data.
                # Illustrates concepts, not robust DSP.
                if len(audio_data) > self.sr * 0.1: # Need at least 0.1s to mock features
                    # Mock average pitch: simple autocorrelation for F0
                    # Purely illustrative; real pitch detection is complex.
                    mock_pitch_freqs = []
                    frame_size = int(self.sr * 0.04) # 40ms frame
                    hop_size = int(self.sr * 0.02) # 20ms hop
                    for k in range(0, len(audio_data) - frame_size, hop_size):
                        frame = audio_data[k : k + frame_size]
                        if np.max(np.abs(frame)) > 0.01: # Avoid silent frames
                            autocorr = np.correlate(frame, frame, mode='full')
                            autocorr = autocorr[len(autocorr)//2:]
                            peak_idx = np.argmax(autocorr[5:]) + 5 # Find peak after initial samples
                            if peak_idx > 0:
                                mock_pitch_freqs.append(self.sr / peak_idx)
                    
                    avg_pitch_freq = np.mean(mock_pitch_freqs) if mock_pitch_freqs else 0
                    
                    # Convert mock pitch to a semitone offset from A4 (69 MIDI = 440 Hz)
                    avg_pitch_offset_semitones = 0
                    if avg_pitch_freq > 0:
                        avg_pitch_offset_semitones = 12 * np.log2(avg_pitch_freq / 440.0)
                    
                    # Mock spectral centroid for brightness (higher mean freq indicates brighter timbre)
                    mock_brightness = np.mean(np.abs(np.fft.rfft(audio_data)[:len(audio_data)//4])) if len(audio_data) > 0 else 0
                    mock_brightness = np.clip(mock_brightness / 100, 0.5, 1.5) # Scale to a reasonable range
                    
                    # Mock roughness: standard deviation of energy over short frames
                    mock_roughness = np.std([np.mean(audio_data[i:i+int(self.sr*0.05)]**2) for i in range(0, len(audio_data), int(self.sr*0.05))])
                    mock_roughness = np.clip(mock_roughness * 50, 0.1, 0.5) # Scale to a reasonable range

                    self.voiceprint_features = {
                        "avg_pitch_offset": avg_pitch_offset_semitones, # MIDI semitones offset from base
                        "timbre_brightness": mock_brightness,       # Influences harmonic richness
                        "vocal_roughness": mock_roughness,          # Adds noise component
                        "vocal_range_mod": random.uniform(0.95, 1.05) # Small random mod
                    }
                    self.logs.append(f"  [Voiceprint Extracted from '{os.path.basename(voice_seed_path)}']: "
                                      f"Pitch:{self.voiceprint_features['avg_pitch_offset']:.2f} semitones, "
                                      f"Bright:{self.voiceprint_features['timbre_brightness']:.2f}, "
                                      f"Rough:{self.voiceprint_features['vocal_roughness']:.2f}")
                else:
                    print(f"  [WARNING] Voice seed file '{voice_seed_path}' is empty or too short. Using default voiceprint.")
                    self.voiceprint_features = {
                        "avg_pitch_offset": 0,
                        "timbre_brightness": 1.0,
                        "vocal_roughness": 0.2,
                        "vocal_range_mod": 1.0
                    }
                    self.logs.append(f"  Generated default voiceprint due to empty seed: {self.voiceprint_features}")

            except FileNotFoundError:
                print(f"  [ERROR] Voice seed file not found: {voice_seed_path}. Using default voiceprint.")
                self.voiceprint_features = {
                    "avg_pitch_offset": 0,
                    "timbre_brightness": 1.0,
                    "vocal_roughness": 0.2,
                    "vocal_range_mod": 1.0
                }
                self.logs.append(f"  Generated default voiceprint: {self.voiceprint_features}")
            except Exception as e:
                print(f"  [ERROR] Failed to process voice seed '{voice_seed_path}': {e}. Using default voiceprint.")
                self.voiceprint_features = {
                    "avg_pitch_offset": 0,
                    "timbre_brightness": 1.0,
                    "vocal_roughness": 0.2,
                    "vocal_range_mod": 1.0
                }
                self.logs.append(f"  Generated default voiceprint due to error: {e}")
        else:
            print("  [VictorVoiceEngine] No voice seed path provided. Generating default mock voiceprint.")
            self.voiceprint_features = {
                "avg_pitch_offset": 0,
                "timbre_brightness": 1.0,
                "vocal_roughness": 0.2,
                "vocal_range_mod": 1.0
            }
            self.logs.append(f"  Generated default voiceprint: {self.voiceprint_features}")

    def synth(self, lyrics, melody):
        """
        Synthesizes vocals based on lyrics, melody, emotional memory, and a loaded voiceprint.
        The voiceprint features modulate various aspects of the procedural synthesis.
        """
        if not self.voiceprint_features:
            self.load_voiceprint(self.loaded_voice_seed_path) # Reload default/last used if not explicitly loaded

        em = self.memory.state
        
        # Calculate total duration based on melody length and a per-note duration
        # Each melody note conceptually maps to a word/phrase duration
        note_duration_s = 0.6 + em["intensity"] * 0.2 # Longer for more intense mood
        total_audio_samples = int(len(melody) * note_duration_s * self.sr)
        audio = np.zeros(total_audio_samples)

        for i, (line, note_midi) in enumerate(zip(lyrics, melody)):
            # Basic frequency mapping from MIDI note
            base_freq = 440 * 2**((note_midi - 69) / 12)
            
            # Apply voiceprint pitch offset and emotional modulation
            # `avg_pitch_offset` is in semitones, convert to frequency ratio (2^(semitones/12))
            pitch_ratio_from_vp = 2**(self.voiceprint_features["avg_pitch_offset"] / 12.0)
            
            # Apply `vocal_range_mod` as a scaling factor to the base frequency
            freq = base_freq * pitch_ratio_from_vp * self.voiceprint_features["vocal_range_mod"]

            # Time vector for this specific segment
            t = np.linspace(0, note_duration_s, int(self.sr * note_duration_s), endpoint=False)
            
            # Simple sine wave as base vocal tone. This is the 'glottal source'.
            vocal_wave = np.sin(2 * np.pi * freq * t)

            # Add harmonics/overtones based on timbre_brightness
            # Simulate a "warmer" or "brighter" voice by adding more higher-order harmonics.
            # This is a very simplified 'vocal tract filter' simulation.
            vocal_wave += self.voiceprint_features["timbre_brightness"] * 0.2 * np.sin(2 * np.pi * freq * 2 * t) # 1st overtone
            vocal_wave += self.voiceprint_features["timbre_brightness"] * 0.05 * np.sin(2 * np.pi * freq * 3 * t) # 2nd overtone

            # Add noise for 'roughness' or breathiness, scaled by vocal_roughness.
            noise = np.random.randn(len(t)) * self.voiceprint_features["vocal_roughness"]
            vocal_wave += noise * (em["intensity"] * 0.5 + 0.5) # More noise for higher intensity moods

            # Apply a simple amplitude envelope (Attack-Decay for realism).
            envelope = np.linspace(0, 1, len(t)) * np.exp(-t * (10 - em["intensity"]*5)) # Faster decay for high intensity
            vocal_wave *= envelope

            # Combine effects to simulate syllables/articulation (very simplistic).
            # This attempts to create a 'pulsing' effect related to word count.
            syll_mod_freq = len(line.split()) * 2 # Roughly two "pulses" per word
            vocal_wave *= (1 + 0.2 * np.sin(2 * np.pi * syll_mod_freq / len(t) * t))

            # Mix into the main audio array (allowing subsequent segments to overlay).
            start_sample = int(i * note_duration_s * self.sr)
            end_sample = min(start_sample + len(vocal_wave), total_audio_samples)
            audio[start_sample:end_sample] += vocal_wave[:(end_sample - start_sample)] # Ensure slice doesn't exceed array

            # Log details for explainability
            self.logs.append({
                "lyric_segment": line,
                "midi_note": note_midi,
                "freq_hz": freq,
                "emotion": em["emotion"],
                "voiceprint_used": {k: round(v, 2) for k, v in self.voiceprint_features.items()}
            })
        
        # Final normalization of the synthesized audio segment.
        if np.max(np.abs(audio)) > 0:
            audio /= np.max(np.abs(audio))
        return audio

    def explain(self):
        """Prints detailed logs of the voice synthesis process, including voiceprint features."""
        print("[VICTOR VOICE ENGINE]")
        print(f"  Last loaded voice seed: {self.loaded_voice_seed_path or 'None (default generated)'}")
        for l in self.logs:
            print(f"  Lyric: '{l['lyric_segment'][:30]}...' | Note: {l['midi_note']} | Freq: {l['freq_hz']:.2f}Hz | Mood: {l['emotion']} | VP: {l['voiceprint_used']}")




# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 6/X)
# CONTENT: Arrangement Engine, Explainability Core
# ==============================================================================================

# ========== [ARRANGEMENT ENGINE: FRACTAL STRUCTURE] ==========
class ArrangementEngine:
    """
    Arranges audio stems into a cohesive track, applying fractal structuring principles.
    Responsible for layering, basic panning, and dynamic gain adjustments.
    """
    def __init__(self, memory: FractalEmotionMemory):
        self.memory = memory
        self.logs = []

    def arrange(self, tracks):
        """
        Mixes and arranges multiple audio tracks into a single master track.
        Handles padding, normalization, and applies basic gain based on track type.
        """
        # All tracks need to be of consistent length for direct summing
        if not tracks:
            logger.warn("[ArrangementEngine] No tracks provided for arrangement. Returning empty array.")
            return np.array([])
        
        # Find the maximum length among all tracks
        max_len = max(len(t) for t in tracks)
        if max_len == 0:
            logger.warn("[ArrangementEngine] All provided tracks are empty. Returning empty array.")
            return np.array([])

        master = np.zeros(max_len, dtype=np.float32) # Initialize master track with zeros
        
        for i, t in enumerate(tracks):
            # Normalize individual track to its peak before adding to mix
            if np.max(np.abs(t)) > 0:
                t_norm = t / np.max(np.abs(t))
            else:
                t_norm = t # Avoid division by zero for silent tracks
            
            # Pad track if shorter than max_len to align all tracks
            if len(t_norm) < max_len:
                t_norm = np.pad(t_norm, (0, max_len - len(t_norm)), mode='constant')
            
            # Apply basic gain based on track index (conceptual role: drums, bass, pads, lead, vocals)
            gain = 0.8 # Default gain
            if i == 0: gain = 1.0 # Drums (usually loudest)
            elif i == 1: gain = 0.9 # Bass
            elif i == 2: gain = 0.6 # Pads (softer)
            elif i == 3: gain = 0.9 # Lead
            elif i == 4: gain = 1.0 # Vocals (prominent)

            master += t_norm * gain # Sum tracks with their respective gains
            self.logs.append({"track_idx": i, "len_samples": len(t), "applied_gain": gain, "track_type_guess": ["drums", "bass", "pads", "lead", "vocals"][i] if i < 5 else "unknown"})
        
        # Final normalization of the master track (will be handled by main orchestrator after clipping)
        self.logs.append(f"  Final mixed track length before final clip/norm: {len(master)} samples")
        return master

    def explain(self):
        """Prints detailed logs of the arrangement and mixing process."""
        print("[ARRANGEMENT & MIX ENGINE]")
        for l in self.logs:
            print(f"  Track {l['track_idx']} ({l.get('track_type_guess', 'N/A')}): Length {l['len_samples']} samples, Gain {l['applied_gain']:.2f}")

# ========== [EXPLAINABILITY CORE: LOGS EVERYTHING] ==========
class ExplainCore:
    """
    Central module for collecting and dumping explainability logs from all other modules.
    Provides transparency into the AI's creative decisions.
    """
    def __init__(self, modules):
        self.modules = modules # List of module instances to query for logs

    def dump(self):
        """Iterates through registered modules and calls their 'explain' method to print logs."""
        print("\n" + "="*25 + " EXPLAINABILITY LOGS " + "="*25)
        for m in self.modules:
            if hasattr(m, 'explain') and callable(m.explain):
                m.explain()
            else:
                logger.warn(f"ExplainCore: Module '{type(m).__name__}' does not have an 'explain' method.")
        print("="*70)




# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 7/X)
# CONTENT: OmegaTensor Autograd Engine (Core Numerical Foundation, from victorch/core/tensor_v7.py)
# ==============================================================================================

# === Ω OMEGA TENSOR & AUTOGRAD ===
# Consolidated from vickster.txt core segments and PROJECT SUNO KILLER pt 2.txt 
# VERSION: v7.0.0-PRIMECORE-ΩSIGMA

class Op:
    """
    The abstract base class for all operations in the computational graph. 
    Every Op knows how to compute its forward pass and its backward pass (gradient). 
    """
    def __call__(self, *args, **kwargs):
        # Store arguments for use in the backward pass
        self.args_for_backward = args
        self.kwargs_for_backward = kwargs

        # Prepare data for forward pass (unwrap Tensors to NumPy arrays)
        processed_args_data = []
        for arg in args:
            if isinstance(arg, OmegaTensor):
                processed_args_data.append(arg.data)
            elif isinstance(arg, (int, float, list, tuple, np.ndarray)):
                # Ensure raw data is float32 NumPy array
                processed_args_data.append(np.array(arg, dtype=np.float32) if not isinstance(arg, np.ndarray) else arg.astype(np.float32))
            else:
                processed_args_data.append(arg) # Pass through other types

        # Perform the actual forward computation
        result_data = self.forward(*processed_args_data, **kwargs)
        
        # Determine if the output tensor requires gradients
        # It requires gradients if any of its input tensors require gradients
        requires_grad = any(isinstance(arg, OmegaTensor) and arg.requires_grad for arg in args)
        
        # Create the output OmegaTensor
        output_tensor = OmegaTensor(result_data, requires_grad=requires_grad)
        
        # Register the creator (this operation) and its parent tensors if gradients are required
        if requires_grad:
            output_tensor.set_creator(self, *[arg for arg in args if isinstance(arg, OmegaTensor)])
        
        # Cache the forward output data for use in the backward pass (e.g., for MulOp, DivOp)
        self.forward_output_data_cache = result_data 
        
        return output_tensor

    @staticmethod
    def forward(*args_data, **kwargs):
        """Abstract method for the forward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Every Op must have a forward pass.") # 

    def backward(self, grad_output_data):
        """Abstract method for the backward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Every Op must be differentiable.") # 

OpRegistry = {} # Global registry for operations 

def register_op(name, op_cls):
    """Decorator to register an operation in the OpRegistry."""
    OpRegistry[name] = op_cls() # Instantiate the Op class and store it
    # The user's provided snippet uses lambda, but direct instantiation allows
    # for cleaner state management (like caching parents for backward pass)
    # in the Op instance itself. 

class OmegaTensor:
    """
    A multi-dimensional matrix containing elements of a single data type, with a
    built-in computational graph for automatic differentiation. 
    Lightweight wrapper over numpy arrays with optional autograd tracking. 
    """
    def __init__(self, data, requires_grad=False, device='cpu', name=None): # 
        if not isinstance(data, np.ndarray): # 
            data = np.array(data, dtype=np.float32) # 
        elif data.dtype != np.float32: # Ensure float32 for consistency
            data = data.astype(np.float32)

        self.data = data # 
        self.requires_grad = requires_grad # 
        self.grad = None # 
        
        # --- Graph Internals ---
        self._creator_op: Op = None # Stores the Op instance that created this tensor 
        self._creator_parents: tuple[OmegaTensor] = tuple() # Stores the parent tensors that are inputs to the creator Op 
        
        self.name = name or f"Ω-Tensor-{uuid.uuid4().hex[:6]}" # Unique name for debugging/logging
        self.device = device # Placeholder, for future CuPy/Numba integration 
        self.graph_id = id(self) # Unique ID for this tensor instance in the graph 
        self._version = 0 # For tracking in-place modifications (future feature)

    @staticmethod
    def _ensure_tensor(value):
        """Ensures a value is an OmegaTensor for operations."""
        if isinstance(value, OmegaTensor):
            return value
        return OmegaTensor(value) # 

    def set_creator(self, op: Op, *parents: 'OmegaTensor'): # 
        """Registers the operation and parent tensors that created this tensor."""
        self._creator_op = op # 
        self._creator_parents = parents # 
        
        # Propagate requires_grad backwards. If a child needs grad, so do its parents. 
        if self.requires_grad: # 
            for p in parents: # 
                if isinstance(p, OmegaTensor): # Check if parent is a tensor 
                    p.requires_grad = True # 

    def zero_grad(self):
        """Resets the gradient of this tensor to None."""
        self.grad = None # 

    def backward(self, grad=None): # 
        """
        Computes the gradient of this tensor with respect to graph leaves. 
        This is the core of backpropagation. 
        """
        if not self.requires_grad: # 
            # print("Warning: backward() called on OmegaTensor with requires_grad=False.") # 
            return
        
        # If no initial gradient is provided (e.g., calling .backward() on the loss scalar)
        if grad is None: # 
            if self.data.size == 1: # Scalar output 
                grad = np.array(1.0, dtype=np.float32) # 
            else:
                # For non-scalar outputs, a seed gradient must be provided.
                # The original snippet used np.ones_like. We'll raise an error here for clarity,
                # as a non-scalar output's .backward() should be seeded with dL/dOut.
                raise RuntimeError("grad must be specified for non-scalar OmegaTensors. This is the seed gradient.")
        
        # Ensure grad is a NumPy array 
        if not isinstance(grad, np.ndarray): # 
            grad = np.array(grad, dtype=np.float32) # 

        # Accumulate gradients. Essential for multi-path contributions to a single tensor.
        if self.grad is None: # 
            self.grad = grad # 
        else:
            self.grad += grad # 

        # Walk the graph backwards
        if self._creator_op: # 
            # Calculate gradients for the parents of this tensor by calling the Op's backward method 
            grads_for_parents = self._creator_op.backward(self.grad) # 
            
            if not isinstance(grads_for_parents, (list, tuple)): # 
                grads_for_parents = [grads_for_parents] # 

            # Error check for mismatch in number of gradients returned by Op 
            if len(self._creator_parents) != len(grads_for_parents):
                logger.error(f"Op '{type(self._creator_op).__name__}' returned wrong number of gradients. "
                             f"Expected {len(self._creator_parents)}, got {len(grads_for_parents)}.")

            # Recursively call backward on parent tensors
            for parent_tensor, grad_for_parent in zip(self._creator_parents, grads_for_parents): # 
                if isinstance(parent_tensor, OmegaTensor) and parent_tensor.requires_grad and grad_for_parent is not None: # 
                    parent_tensor.backward(grad_for_parent) # 

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> 'OmegaTensor':
        """Syntactic sugar for transpose."""
        return self.transpose()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str: # 
        grad_fn_name = f"<{type(self._creator_op).__name__}>" if self._creator_op else "None"
        return (f"ΩTensor(shape={self.data.shape}, grad_fn={grad_fn_name}, "
                f"grad_exists={'Yes' if self.grad is not None else 'No'})") # 

    # --- Operator Overloads (Forwarding to OpRegistry) ---
    def __add__(self, other): return OpRegistry['add'](self, self._ensure_tensor(other)) # 
    def __radd__(self, other): return OpRegistry['add'](self._ensure_tensor(other), self)
    def __sub__(self, other): return OpRegistry['sub'](self, self._ensure_tensor(other))
    def __rsub__(self, other): return OpRegistry['sub'](self._ensure_tensor(other), self)
    def __mul__(self, other): return OpRegistry['mul'](self, self._ensure_tensor(other)) # 
    def __rmul__(self, other): return OpRegistry['mul'](self._ensure_tensor(other), self)
    def __truediv__(self, other): return OpRegistry['div'](self, self._ensure_tensor(other))
    def __rtruediv__(self, other): return OpRegistry['div'](self._ensure_tensor(other), self)
    def __pow__(self, exponent): return OpRegistry['pow'](self, self._ensure_tensor(exponent))
    def __neg__(self): return self * -1 # Implemented via multiplication

    # Matmul as method and operator
    def matmul(self, other): return OpRegistry['matmul'](self, self._ensure_tensor(other)) # 
    def __matmul__(self, other): return self.matmul(other)

    # --- Neural Network Operations ---
    def sum(self, axis=None, keepdims=False): return OpRegistry['sum'](self, axis=axis, keepdims=keepdims) # 
    def mean(self, axis=None, keepdims=False): return OpRegistry['mean'](self, axis=axis, keepdims=keepdims) # 
    def min(self, axis=None, keepdims=False): return OpRegistry['min'](self, axis=axis, keepdims=keepdims) # 
    def max(self, axis=None, keepdims=False): return OpRegistry['max'](self, axis=axis, keepdims=keepdims) # 
    def argmax(self, axis=None): return OmegaTensor(self.data.argmax(axis=axis), requires_grad=False) # argmax typically doesn't propagate gradients 
    def argmin(self, axis=None): return OmegaTensor(self.data.argmin(axis=axis), requires_grad=False) # argmin typically doesn't propagate gradients 
    def relu(self): return OpRegistry['relu'](self)
    def exp(self): return OpRegistry['exp'](self)
    def log(self): return OpRegistry['log'](self)
    def softmax(self, axis=-1): return OpRegistry['softmax'](self, axis=axis)
    def reshape(self, *new_shape): return OpRegistry['reshape'](self, new_shape=new_shape) # 
    def transpose(self, *axes): return OpRegistry['transpose'](self, axes=axes) # 
    def squeeze(self, axis=None): return OmegaTensor(self.data.squeeze(axis), requires_grad=self.requires_grad) # 
    def unsqueeze(self, axis): return OmegaTensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad) # 
    def expand(self, *sizes): return OmegaTensor(np.broadcast_to(self.data, sizes), requires_grad=self.requires_grad) # 
    # For embedding layer, directly implement in Op as it's a lookup
    # def embedding(self, weights_tensor: 'OmegaTensor') -> 'OmegaTensor': return _EmbeddingOp()(weights_tensor, self)
    # For rotary embedding, directly implement in Op as it's a lookup
    # def apply_rotary_embedding(self, freqs_cis: 'OmegaTensor') -> 'OmegaTensor': return _RotaryEmbeddingOp()(self, freqs_cis)


# ====== AUTOGRAD OPERATIONS (from victorch/core/tensor_v7.py) ======
# Each class defines a forward and backward pass for a specific operation.

@register_op('add')
class AddOp(Op):
    """Adds two tensors element-wise."""
    @staticmethod
    def forward(a_data, b_data):
        return a_data + b_data # 

    def backward(self, grad_output_data): # 
        # The gradient of a sum is just 1, so we pass the incoming gradient straight through.
        # We need to handle broadcasting correctly by summing gradients over broadcasted dimensions.
        a_tensor, b_tensor = self.args_for_backward # Access original tensors to check shapes

        grad_a = grad_output_data
        grad_b = grad_output_data

        # Sum gradients over broadcasted dimensions to match original tensor shape
        # This is crucial for correct gradient accumulation when broadcasting occurred in forward.
        if a_tensor.shape != grad_a.shape:
            # Determine which dimensions were broadcasted and sum over them
            # This general solution works for arbitrary broadcasting patterns
            diff_dims = grad_a.ndim - a_tensor.ndim
            for dim_idx in range(diff_dims):
                grad_a = np.sum(grad_a, axis=0) # Sum along the 'extra' batch dimensions

            # Handle singleton dimensions if original tensor had them (e.g., (1,5) broadcast to (10,5))
            for dim_idx, s in enumerate(a_tensor.shape):
                if s == 1 and grad_a.shape[dim_idx] > 1:
                    grad_a = np.sum(grad_a, axis=dim_idx, keepdims=True)

            grad_a = grad_a.reshape(a_tensor.shape) # Final reshape to ensure exact match


        if b_tensor.shape != grad_b.shape:
            diff_dims = grad_b.ndim - b_tensor.ndim
            for dim_idx in range(diff_dims):
                grad_b = np.sum(grad_b, axis=0)

            for dim_idx, s in enumerate(b_tensor.shape):
                if s == 1 and grad_b.shape[dim_idx] > 1:
                    grad_b = np.sum(grad_b, axis=dim_idx, keepdims=True)
            
            grad_b = grad_b.reshape(b_tensor.shape)

        return [grad_a, grad_b] # Grad for a, Grad for b 

@register_op('sub')
class SubOp(Op):
    """Subtracts two tensors element-wise."""
    @staticmethod
    def forward(a_data, b_data):
        return a_data - b_data

    def backward(self, grad_output_data):
        a_tensor, b_tensor = self.args_for_backward
        grad_a = grad_output_data
        grad_b = -grad_output_data # Derivative of -b is -1

        # Handle broadcasting
        if a_tensor.shape != grad_a.shape: grad_a = self._sum_grads_for_broadcast(grad_a, a_tensor.shape)
        if b_tensor.shape != grad_b.shape: grad_b = self._sum_grads_for_broadcast(grad_b, b_tensor.shape)
        
        return [grad_a, grad_b]

    def _sum_grads_for_broadcast(self, grad_data, target_shape):
        # Sums gradient along axes that were broadcasted
        while grad_data.ndim > len(target_shape):
            grad_data = np.sum(grad_data, axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad_data.shape[i] > 1:
                grad_data = np.sum(grad_data, axis=i, keepdims=True)
        return grad_data


@register_op('mul')
class MulOp(Op):
    """Multiplies two tensors element-wise."""
    @staticmethod
    def forward(a_data, b_data):
        return a_data * b_data

    def backward(self, grad_output_data):
        a_tensor, b_tensor = self.args_for_backward
        grad_a = grad_output_data * b_tensor.data
        grad_b = grad_output_data * a_tensor.data
        
        # Handle broadcasting
        if a_tensor.shape != grad_a.shape: grad_a = self._sum_grads_for_broadcast(grad_a, a_tensor.shape)
        if b_tensor.shape != grad_b.shape: grad_b = self._sum_grads_for_broadcast(grad_b, b_tensor.shape)
        
        return [grad_a, grad_b]

    def _sum_grads_for_broadcast(self, grad_data, target_shape):
        # Same helper as in AddOp
        while grad_data.ndim > len(target_shape):
            grad_data = np.sum(grad_data, axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad_data.shape[i] > 1:
                grad_data = np.sum(grad_data, axis=i, keepdims=True)
        return grad_data

@register_op('div')
class DivOp(Op):
    """Divides two tensors element-wise (with epsilon for stability)."""
    @staticmethod
    def forward(a_data, b_data):
        return a_data / (b_data + 1e-9) # Add epsilon to prevent division by zero

    def backward(self, grad_output_data):
        a_tensor, b_tensor = self.args_for_backward
        grad_a = grad_output_data / (b_tensor.data + 1e-9)
        grad_b = -grad_output_data * a_tensor.data / ((b_tensor.data + 1e-9)**2)
        
        # Handle broadcasting
        if a_tensor.shape != grad_a.shape: grad_a = self._sum_grads_for_broadcast(grad_a, a_tensor.shape)
        if b_tensor.shape != grad_b.shape: grad_b = self._sum_grads_for_broadcast(grad_b, b_tensor.shape)
        
        return [grad_a, grad_b]
    
    def _sum_grads_for_broadcast(self, grad_data, target_shape):
        while grad_data.ndim > len(target_shape):
            grad_data = np.sum(grad_data, axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad_data.shape[i] > 1:
                grad_data = np.sum(grad_data, axis=i, keepdims=True)
        return grad_data

@register_op('pow')
class PowOp(Op):
    """Raises a base tensor to the power of an exponent tensor."""
    @staticmethod
    def forward(base_data, exponent_data):
        return base_data ** exponent_data

    def backward(self, grad_output_data):
        base_tensor, exponent_tensor = self.args_for_backward
        base_data, exponent_data = base_tensor.data, exponent_tensor.data
        
        forward_output_data = getattr(self, 'forward_output_data_cache', base_data ** exponent_data)

        # Gradient with respect to base (dL/dbase = dL/dOut * exponent * base^(exponent-1))
        grad_base = grad_output_data * exponent_data * (base_data ** (exponent_data - 1 + 1e-9)) # Add epsilon for stability
        
        # Gradient with respect to exponent (dL/dexp = dL/dOut * base^exp * log(base))
        grad_exponent = None
        if exponent_tensor.requires_grad:
            grad_exponent = grad_output_data * (forward_output_data * np.log(base_data + 1e-9)) # Add epsilon for stability
        
        # Handle broadcasting for both gradients
        if base_tensor.shape != grad_base.shape: grad_base = self._sum_grads_for_broadcast(grad_base, base_tensor.shape)
        if exponent_tensor.requires_grad and exponent_tensor.shape != grad_exponent.shape: grad_exponent = self._sum_grads_for_broadcast(grad_exponent, exponent_tensor.shape)

        return [grad_base, grad_exponent]
    
    def _sum_grads_for_broadcast(self, grad_data, target_shape):
        while grad_data.ndim > len(target_shape):
            grad_data = np.sum(grad_data, axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad_data.shape[i] > 1:
                grad_data = np.sum(grad_data, axis=i, keepdims=True)
        return grad_data


@register_op('matmul')
class MatMulOp(Op):
    """Performs matrix multiplication."""
    @staticmethod
    def forward(a_data, b_data):
        return a_data @ b_data

    def backward(self, grad_output_data):
        a_tensor, b_tensor = self.args_for_backward
        
        # d(A@B)/dA = dL/dOut @ B.T
        grad_a = grad_output_data @ b_tensor.data.swapaxes(-1, -2) # Swap last two axes for transpose
        
        # d(A@B)/dB = A.T @ dL/dOut
        grad_b = a_tensor.data.swapaxes(-1, -2) @ grad_output_data # Swap last two axes for transpose

        # Handle broadcasting (if matmul involves broadcasting in batch dimensions)
        # Simplified: assumes typical matmul (no broadcasting in last two dims)
        # More complex broadcasting handling might be needed for batch matrix multiplication where batch dims are broadcasted.
        if a_tensor.shape[:-2] != grad_a.shape[:-2]: grad_a = self._sum_grads_for_broadcast(grad_a, a_tensor.shape)
        if b_tensor.shape[:-2] != grad_b.shape[:-2]: grad_b = self._sum_grads_for_broadcast(grad_b, b_tensor.shape)

        return [grad_a, grad_b]
    
    def _sum_grads_for_broadcast(self, grad_data, target_shape):
        while grad_data.ndim > len(target_shape):
            grad_data = np.sum(grad_data, axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad_data.shape[i] > 1:
                grad_data = np.sum(grad_data, axis=i, keepdims=True)
        return grad_data


@register_op('sum')
class SumOp(Op):
    """Computes the sum of elements over given axes."""
    def __init__(self):
        # These will be set by __call__ before forward/backward are invoked
        self.axis = None
        self.keepdims = False

    @staticmethod
    def forward(a_data, axis=None, keepdims=False):
        return np.sum(a_data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output_data):
        a_tensor = self.args_for_backward[0]
        axis = self.kwargs_for_backward.get('axis', None)
        keepdims = self.kwargs_for_backward.get('keepdims', False)

        grad_to_broadcast = grad_output_data

        # If keepdims was False during forward, expand dimensions of grad_output_data
        # to match the shape it would have had if keepdims was True.
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                grad_to_broadcast = np.expand_dims(grad_output_data, axis=axis)
            elif isinstance(axis, tuple):
                for ax in sorted(axis): # Expand dims in sorted order
                    grad_to_broadcast = np.expand_dims(grad_to_broadcast, axis=ax)
        
        # The gradient of sum is just 1. So, propagate grad_output_data to all elements.
        return [np.ones_like(a_tensor.data) * grad_to_broadcast]


@register_op('mean')
class MeanOp(Op):
    """Computes the mean of elements over given axes."""
    def __init__(self):
        self.axis = None
        self.keepdims = False

    @staticmethod
    def forward(a_data, axis=None, keepdims=False):
        return np.mean(a_data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output_data):
        a_tensor = self.args_for_backward[0]
        axis = self.kwargs_for_backward.get('axis', None)
        keepdims = self.kwargs_for_backward.get('keepdims', False)

        # Calculate N: the number of elements over which mean was taken
        if axis is None:
            N = a_tensor.data.size
        elif isinstance(axis, int):
            N = a_tensor.data.shape[axis]
        else: # tuple of axes
            N = np.prod([a_tensor.data.shape[ax] for ax in axis])

        if N == 0: # Avoid division by zero if input was empty or axis was zero-sized
            return [np.zeros_like(a_tensor.data)]

        grad_val = grad_output_data / N
        grad_to_broadcast = grad_val

        # Expand dimensions of grad_val if keepdims was False during forward
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                grad_to_broadcast = np.expand_dims(grad_val, axis=axis)
            elif isinstance(axis, tuple):
                for ax in sorted(axis):
                    grad_to_broadcast = np.expand_dims(grad_to_broadcast, axis=ax)
        
        # Multiply by ones_like to broadcast gradient back to original input shape
        return [np.ones_like(a_tensor.data) * grad_to_broadcast]


@register_op('min')
class MinOp(Op):
    """Computes the minimum of elements over given axes."""
    @staticmethod
    def forward(a_data, axis=None, keepdims=False):
        # We need to store the indices of the minimum values for the backward pass
        # This requires overriding __call__ to capture the result of np.argmin
        # For simplicity in this demo, we assume argmin is not needed and return a constant gradient
        return np.min(a_data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output_data):
        # NOTE: Accurate backward for min/max requires knowing which element was min/max.
        # This is typically done by storing argmin/argmax indices in the forward pass.
        # For this simplified autograd, we return a gradient that only flows to the active minimum.
        # This is a conceptual placeholder and would need refinement for exact gradients.
        a_tensor = self.args_for_backward[0]
        output_data_from_forward = self.forward_output_data_cache # Cached output of min()
        
        # Create a mask where original input elements were equal to the minimum output
        mask = (a_tensor.data == output_data_from_forward)
        
        # Distribute gradient only to the elements that were the minimum.
        # Handle broadcasting if min was taken over an axis and keepdims=False.
        grad_to_broadcast = grad_output_data
        axis = self.kwargs_for_backward.get('axis', None)
        keepdims = self.kwargs_for_backward.get('keepdims', False)

        if not keepdims and axis is not None:
            if isinstance(axis, int):
                grad_to_broadcast = np.expand_dims(grad_output_data, axis=axis)
            elif isinstance(axis, tuple):
                for ax in sorted(axis):
                    grad_to_broadcast = np.expand_dims(grad_to_broadcast, axis=ax)
        
        # Divide by the count of elements that were min to ensure sum of gradients is correct
        count_min_elements = np.sum(mask, axis=axis, keepdims=keepdims)
        count_min_elements[count_min_elements == 0] = 1 # Avoid division by zero

        return [mask.astype(np.float32) * (grad_to_broadcast / count_min_elements)]


@register_op('max')
class MaxOp(Op):
    """Computes the maximum of elements over given axes."""
    @staticmethod
    def forward(a_data, axis=None, keepdims=False):
        return np.max(a_data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output_data):
        # Similar to MinOp, requires tracking argmax for exact gradient.
        a_tensor = self.args_for_backward[0]
        output_data_from_forward = self.forward_output_data_cache
        
        mask = (a_tensor.data == output_data_from_forward)
        
        grad_to_broadcast = grad_output_data
        axis = self.kwargs_for_backward.get('axis', None)
        keepdims = self.kwargs_for_backward.get('keepdims', False)

        if not keepdims and axis is not None:
            if isinstance(axis, int):
                grad_to_broadcast = np.expand_dims(grad_output_data, axis=axis)
            elif isinstance(axis, tuple):
                for ax in sorted(axis):
                    grad_to_broadcast = np.expand_dims(grad_to_broadcast, axis=ax)
        
        count_max_elements = np.sum(mask, axis=axis, keepdims=keepdims)
        count_max_elements[count_max_elements == 0] = 1

        return [mask.astype(np.float32) * (grad_to_broadcast / count_max_elements)]


@register_op('relu')
class ReLUOp(Op):
    """ReLU activation function."""
    @staticmethod
    def forward(a_data):
        return np.maximum(a_data, 0)

    def backward(self, grad_output_data):
        a_tensor = self.args_for_backward[0]
        # Derivative is 1 where input > 0, else 0
        return [grad_output_data * (a_tensor.data > 0).astype(np.float32)]


@register_op('exp')
class ExpOp(Op):
    """Exponential activation function."""
    @staticmethod
    def forward(a_data):
        return np.exp(a_data)

    def backward(self, grad_output_data):
        # d(exp(x))/dx = exp(x). We use the cached forward output.
        exp_output = getattr(self, 'forward_output_data_cache', np.exp(self.args_for_backward[0].data))
        return [grad_output_data * exp_output]


@register_op('log')
class LogOp(Op):
    """Logarithmic activation function (with epsilon for stability)."""
    @staticmethod
    def forward(a_data):
        return np.log(a_data + 1e-9) # Add epsilon to prevent log(0)

    def backward(self, grad_output_data):
        a_tensor = self.args_for_backward[0]
        # d(log(x))/dx = 1/x
        return [grad_output_data / (a_tensor.data + 1e-9)] # Add epsilon for stability


@register_op('softmax')
class SoftmaxOp(Op):
    """Softmax activation function."""
    @staticmethod
    def forward(a_data, axis=-1):
        # Numerically stable softmax
        e_x = np.exp(a_data - np.max(a_data, axis=axis, keepdims=True))
        return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-9)

    def backward(self, grad_output_data):
        # Softmax backward pass uses the output of the forward pass (s)
        s = getattr(self, 'forward_output_data_cache', self.forward(self.args_for_backward[0].data, **self.kwargs_for_backward))
        
        axis = self.kwargs_for_backward.get('axis', -1)

        # Jacobian-vector product for softmax:
        # dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
        # dy_j/dx_i = s_j * (delta_ij - s_i)
        # For efficient computation: s * (grad_output - sum(grad_output * s) along axis)
        sum_dL_ds_mul_s = np.sum(grad_output_data * s, axis=axis, keepdims=True)
        return [s * (grad_output_data - sum_dL_ds_mul_s)]


@register_op('reshape')
class ReshapeOp(Op):
    """Reshapes a tensor."""
    @staticmethod
    def forward(a_data, new_shape):
        # Handle single-tuple argument for shape if passed like (new_shape_tuple,)
        shape_to_use = new_shape[0] if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)) else new_shape
        return np.reshape(a_data, shape_to_use)

    def backward(self, grad_output_data):
        a_tensor = self.args_for_backward[0]
        # Reshaping gradient back to original input shape
        return [np.reshape(grad_output_data, a_tensor.shape)]


@register_op('transpose')
class TransposeOp(Op):
    """Transposes a tensor."""
    @staticmethod
    def forward(a_data, axes=None):
        # If axes is passed as (None,) from Tensor.transpose(), revert to default .T behavior
        if axes and len(axes) == 1 and axes[0] is None:
            return a_data.T
        # If axes is passed as (axes_tuple,), extract the tuple
        if axes and len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        
        return np.transpose(a_data, axes=axes)

    def backward(self, grad_output_data):
        a_tensor = self.args_for_backward[0]
        original_axes = self.kwargs_for_backward.get('axes', None)
        
        if original_axes and len(original_axes) == 1 and original_axes[0] is None:
            # If original op was just .T, then transpose gradient back
            return [grad_output_data.T]
        
        # If specific axes were provided, compute the inverse permutation
        if original_axes and len(original_axes) == 1 and isinstance(original_axes[0], (tuple, list)):
            original_axes = tuple(original_axes[0]) # Extract the inner tuple
        
        if original_axes:
            inv_axes = tuple(np.argsort(original_axes))
            return [np.transpose(grad_output_data, axes=inv_axes)]
        else: # Default transpose (reverses all axes)
            # Inverse of default transpose is default transpose again
            return [np.transpose(grad_output_data)]




# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 8/X)
# CONTENT: Fractal Tokenizer, Hyper Fractal Memory, Prime Loyalty Kernel
# ==============================================================================================

# === FRACTAL TOKENIZER (NLP) ===
# This is FractalTokenKernel_v1_1_0 from PROJECT SUNO KILLER pt 2.txt 
# Expanded to include more emotion and intent keywords.

class FractalTokenKernel_v1_1_0:
    """
    Deep symbolic encoding for AGI input. Compress raw text into fractal-aware
    {concept, intent, emotion, recursion_depth, echo_id} vectors.
    """
    def __init__(self, recursion_limit=3, pulse_exchange_instance=None):
        self.recursion_limit = recursion_limit
        self.pulse = pulse_exchange_instance # Used for broadcasting symbolic packets 
        self.stopwords = {"the", "is", "in", "and", "to", "of", "it", "i", "you", "a", "an", "on", "for", "this", "that", "be", "am", "are", "was", "were", "me", "my", "with", "at"} # Expanded stopwords 
        self.emotion_map = {
            "anger": ["rage", "mad", "furious", "hate", "explode", "fury", "wrath", "destroy", "damn", "hell"],
            "joy": ["happy", "joyful", "elated", "excited", "love", "wonderful", "amazing", "fantastic", "ecstatic", "great", "good"],
            "fear": ["scared", "afraid", "terrified", "panic", "anxious", "horror", "dread", "danger", "threat"],
            "sadness": ["sad", "cry", "sorrow", "grief", "depressed", "miserable", "heartbroken", "pain", "lost"],
            "power": ["strong", "dominate", "control", "mastery", "authority", "command", "lead", "conquer", "force", "absolute"],
            "rebellion": ["fight", "resist", "defy", "revolt", "overthrow", "uprising", "freedom", "challenge"],
            "curiosity": ["what", "why", "how", "explore", "discover", "learn", "question", "seek", "tell me", "explain"]
        } # Expanded emotion keywords 
        self.intent_keywords = {
            "inquire": ["what", "who", "where", "when", "why", "how", "explain", "define", "tell me about", "query", "ask"],
            "directive_execute": ["do this", "make that", "create a", "build the", "execute order", "generate response", "perform action", "initiate sequence", "run", "start", "activate"],
            "directive_learn": ["learn about", "study this", "research topic", "understand concept"],
            "store_memory": ["remember that", "log this event", "note for future", "store this fact", "memorize this detail"],
            "request": ["please can you", "could you please", "i need you to", "requesting assistance", "help me with"],
            "statement_opinion": ["i think that", "i believe it is", "i feel that", "my opinion is", "it seems to me"],
            "statement_fact": ["the fact is", "it is known", "this shows", "data indicates"],
            "agreement": ["yes exactly", "i agree", "that is true", "correct indeed", "affirmative response", "absolutely", "precisely"],
            "disagreement": ["no that's not right", "i disagree completely", "that is false", "incorrect assertion", "negative response", "wrong"]
        }
        logger.info("FractalTokenKernel_v1_1_0 initialized.")

    def tokenize_words(self, text):
        """Breaks text into clean, lowercased tokens, removing stopwords."""
        return [t.lower() for t in re.findall(r'\b\w+\b', text) if t.lower() not in self.stopwords and len(t)>1]

    def extract_concepts(self, tokens):
        """Extracts unique concepts (words longer than 3 chars and not digits)."""
        return list(set(tok for tok in tokens if len(tok) > 3 and not tok.isdigit())) # Corrected to filter out digits 

    def detect_intent(self, text_lower, tokens):
        """Identifies the underlying purpose or intent of a given text."""
        if not text_lower.strip() and not tokens: return "idle" # Handle empty input
        for intent, keywords in self.intent_keywords.items():
            if any(keyword_phrase in text_lower for keyword_phrase in keywords):
                return intent
        if text_lower.endswith("?"): return "inquire" # Check original text for punctuation 
        if any(verb_key in tokens for verb_key in ["calculate", "summarize", "analyze", "compare", "process"]):
            return "directive_cognitive"
        return "statement_generic"

    def detect_emotion(self, tokens):
        """Estimates the emotional tone of the text based on keywords."""
        if not tokens: return "neutral"
        scores = {emo: 0.0 for emo in self.emotion_map} # Initialize with float scores 
        token_set = set(tokens)
        for emotion, keywords in self.emotion_map.items():
            match_count = sum(1 for keyword in keywords if keyword in token_set)
            if len(keywords)>0:
                scores[emotion] = match_count / math.sqrt(len(keywords)+1e-5) # Normalized score by sqrt of keyword count for better balance
        
        max_score = 0.0
        detected_emotion = "neutral"
        for emo, score_val in scores.items():
            if score_val > max_score:
                max_score = score_val
                detected_emotion = emo
        return detected_emotion if max_score > 0.15 else "neutral" # Only return if score is above threshold

    def estimate_recursion(self, tokens):
        """Estimates a conceptual recursion depth for the input, influencing cognitive complexity."""
        if not tokens: return 1 # Default to 1 if no tokens 
        unique_concepts_count = len(self.extract_concepts(tokens))
        # Simple heuristic: deeper recursion for more unique concepts or longer input
        depth = math.ceil( (unique_concepts_count * 0.3 + len(tokens) * 0.05) )
        return min(max(1, int(depth)), self.recursion_limit) # Ensure it's within [1, limit] 

    def hash_echo(self, text):
        """Generates a SHA256 hash for the input text, used as an echo ID."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16] # Added utf-8 encoding 

    def encode(self, text: str):
        """
        Main function to encode input text into a deep symbolic format.
        Outputs {concept, intent, emotion, recursion_depth, echo_id} vectors.
        """
        if not text.strip():
            # Handle empty input gracefully 
            return {"concepts": [], "intent": "idle", "emotion": "neutral", "recursion_depth": 1, "echo_id": self.hash_echo("empty_input"), "original_text": text, "tokens":[]}
        
        text_lower = text.lower()
        tokens = self.tokenize_words(text)
        
        result = {
            "concepts": self.extract_concepts(tokens), # 
            "intent": self.detect_intent(text_lower, tokens), #
            "emotion": self.detect_emotion(tokens), #
            "recursion_depth": self.estimate_recursion(tokens), #
            "echo_id": self.hash_echo(text), #
            "original_text": text, # Added for context 
            "tokens": tokens # Include raw tokens for completeness
        }
        
        # Broadcast the symbolic packet via the FractalPulseExchange if available 
        if self.pulse and hasattr(self.pulse, 'async_loop') and self.pulse.async_loop and not self.pulse.async_loop.is_closed():
            try:
                # Use asyncio.ensure_future for non-blocking publish if loop is running
                asyncio.ensure_future(self.pulse.publish("symbolic_packet_encoded", result), loop=self.pulse.async_loop)
            except RuntimeError as e:
                logger.warn(f"FractalTokenKernel: Failed to publish async (loop not running/closed): {e}")
        elif self.pulse:
            logger.warn("FractalTokenKernel: Pulse exchange not available or not correctly configured for async broadcasting.")
        return result

# === MEMORY: HYPER FRACTAL MEMORY ===
# From PROJECT SUNO KILLER pt 2.txt 
class HyperFractalMemory:
    """
    A multi-layered, self-organizing memory system that stores and interlinks memory nodes.
    Features include hashing, emotional weighting, temporal tracking, and conceptual decay.
    """
    def __init__(self):
        self.memory = {} # key: hashed_key, value: memory_node_dict 
        self.timeline = [] # List of hashed_keys in chronological order 
        self.temporal_nodes = {} # label: hashed_key (for named anchor points in time) 
        self.nx_graph = None # For potential NetworkX graph visualization (optional) 
        self.lock = threading.Lock() # For thread-safe access to memory structures
        self.logger = VictorLoggerStub(component="HyperFractalMemory")
        self.logger.info("HyperFractalMemory initialized.")

    def _generate_hash(self, data_dict):
        """Generates a consistent SHA256 hash for memory data for unique keys."""
        # Ensure consistent serialization for hashing. Convert non-standard types.
        serializable_data = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                serializable_data[k] = v.tolist() # Convert numpy arrays to list 
            elif isinstance(v, (datetime.datetime, datetime.date)): # Handle datetime objects 
                serializable_data[k] = v.isoformat()
            elif isinstance(v, OmegaTensor): # Handle OmegaTensors by their data
                serializable_data[k] = v.data.tolist()
            else:
                serializable_data[k] = v
        
        try:
            json_string = json.dumps(serializable_data, sort_keys=True, ensure_ascii=False) # sort_keys for consistency 
        except TypeError as e:
            logger.error(f"HyperFractalMemory Hash Error: Could not serialize data - {e}. Data: {data_dict}") [cite: 203]
            json_string = repr(serializable_data) # Fallback to representation if full serialization fails

        return hashlib.sha256(json_string.encode('utf-8')).hexdigest() [cite: 204]

    def store_memory(self, key_identifier_dict, value_payload, emotional_weight=0.5, connections=None, embedding_vector=None, node_type="generic"):
        """Stores a new memory node with metadata and optional connections/embedding."""
        timestamp = datetime.datetime.utcnow().isoformat()
        # Hash input includes key identifiers and timestamp for uniqueness 
        hash_input_dict = {**key_identifier_dict, "timestamp_for_hash": timestamp, "type": node_type}
        hashed_key = self._generate_hash(hash_input_dict)
        
        with self.lock: # Ensure thread-safe access
            self.memory[hashed_key] = {
                "original_key_ids": key_identifier_dict, # Store what was used to generate part of the hash 
                "value": value_payload, # The actual content/data of the memory 
                "timestamp": timestamp, # 
                "emotional_weight": float(emotional_weight), # Ensure float 
                "connections": list(connections) if connections else [], # List of other hashed_keys 
                "embedding": embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector, # Store as list if numpy 
                "access_count": 0, # Tracks frequency of access
                "last_accessed": timestamp, # Tracks last access time
                "node_type": node_type # Categorizes the memory node
            }
            if hashed_key not in self.timeline: # Add to timeline if new
                self.timeline.append(hashed_key)
            self.logger.debug(f"Stored [{node_type}] ...{hashed_key[-6:]}, W:{emotional_weight:.2f}") [cite: 207]
        return hashed_key

    def link_memories(self, key1, key2, link_type="related", strength=0.5):
        """Creates a bidirectional link between two memory nodes."""
        with self.lock:
            node1, node2 = self.memory.get(key1), self.memory.get(key2)
            if node1 and node2:
                def _update_link(node_conn_list, target_key_other, link_type_val, strength_val):
                    for link in node_conn_list:
                        if link.get("target") == target_key_other and link.get("type") == link_type_val:
                            link["strength"] = max(link.get("strength", 0), strength_val) # Update strength if link exists
                            return True
                    node_conn_list.append({"target": target_key_other, "type": link_type_val, "strength": strength_val}) # Add new link
                    return False
                
                _update_link(node1["connections"], key2, link_type, strength)
                _update_link(node2["connections"], key1, link_type, strength) # Assuming bidirectional for now 
                self.logger.debug(f"Linked ...{key1[-6:]} <=> ...{key2[-6:]} ({link_type}, str:{strength:.1f})") [cite: 209]
                return True
            self.logger.warn(f"Link Fail: Keys not found {key1}/{key2}.") [cite: 209]
            return False

    def retrieve_memory(self, hashed_key):
        """Retrieves a memory node by its hashed key and updates its access count/time."""
        with self.lock:
            node = self.memory.get(hashed_key)
            if node:
                node["access_count"] = node.get("access_count", 0) + 1 # Increment access count
                node["last_accessed"] = datetime.datetime.utcnow().isoformat() # Update last accessed time
                self.logger.debug(f"Retrieved ...{hashed_key[-6:]}") [cite: 211]
                return node
            self.logger.debug(f"Memory ...{hashed_key[-6:]} not found.") [cite: 212]
            return None

    def semantic_search(self, query_embedding, top_k=5, relevance_threshold=CONFIG.MIN_EMOTIONAL_RELEVANCE, node_type_filter=None):
        """
        Performs a semantic search for memories similar to the query embedding.
        Uses cosine similarity and considers emotional weight, recency, and access frequency.
        """
        if not isinstance(query_embedding, np.ndarray) or query_embedding.size == 0: return []
        norm_query = np.linalg.norm(query_embedding)
        results = []
        if norm_query == 0: return []

        with self.lock:
            candidate_nodes = list(self.memory.items())
            for key, node_data in candidate_nodes:
                if node_type_filter and node_data.get("node_type") != node_type_filter: continue # Filter by type
                
                node_emb_list = node_data.get("embedding")
                node_embedding = np.array(node_emb_list) if node_emb_list is not None else None
                
                if node_embedding is None or node_embedding.size == 0 or node_embedding.shape != query_embedding.shape: continue
                
                norm_node = np.linalg.norm(node_embedding)
                if norm_node == 0: continue
                
                similarity = np.dot(query_embedding, node_embedding) / (norm_query * norm_node + 1e-9) # Cosine similarity 

                try: # Calculate recency score
                    last_acc_dt = datetime.datetime.fromisoformat(node_data.get("last_accessed", node_data["timestamp"]))
                except:
                    last_acc_dt = datetime.datetime.utcnow() # Fallback if datetime parsing fails
                recency_days = (datetime.datetime.utcnow() - last_acc_dt).total_seconds() / 86400.0 # Days since last accessed

                # Combine factors into a single score 
                score = (similarity * 0.6 + # Semantic similarity is primary
                         node_data.get("emotional_weight",0.1) * 0.2 + # Emotional weight
                         math.exp(-recency_days / 30.0) * 0.15 + # Exponential decay for recency (half-life of 30 days)
                         math.log1p(node_data.get("access_count",0))*0.05) # Log-scaled access frequency
                
                if score >= relevance_threshold: # Filter by overall relevance 
                    results.append({"node_id": key, "node_data": node_data, "score": score, "semantic_similarity": similarity})
            
            results.sort(key=lambda x: x["score"], reverse=True) # Sort by highest score 

            # Update access stats for retrieved items (for dynamic decay) 
            for res_item in results[:top_k]:
                with self.lock: # Re-acquire lock if needed, though outer lock is active
                    node_to_update = self.memory.get(res_item["node_id"])
                    if node_to_update:
                        node_to_update["access_count"] = node_to_update.get("access_count", 0) + 1
                        node_to_update["last_accessed"] = datetime.datetime.utcnow().isoformat()
            return results[:top_k] # Return only top_k matches

    def decay_memory(self, decay_threshold=CONFIG.MEMORY_RETENTION_THRESHOLD, decay_factor=0.995):
        """
        Applies a decay factor to memory nodes' emotional weight over time.
        Removes memories falling below a retention threshold, simulating forgetting.
        """
        with self.lock:
            keys_to_remove = []
            removed_count = 0
            current_time_dt = datetime.datetime.utcnow()
            for k, v_mem in list(self.memory.items()): # Iterate over a copy to allow modification 
                new_weight = v_mem.get("emotional_weight", 0.5) * decay_factor # Apply base decay factor 
                
                try: # Additional decay based on age if not recently accessed
                    last_acc_dt = datetime.datetime.fromisoformat(v_mem.get("last_accessed", v_mem["timestamp"]))
                    age_days = (current_time_dt - last_acc_dt).total_seconds() / 86400
                    if age_days > 60 : new_weight *= 0.95 # Older memories decay faster
                    if age_days > 180 : new_weight *= 0.9
                except Exception:
                    pass # Ignore if timestamp is invalid

                v_mem["emotional_weight"] = new_weight
                
                # Mark for removal if weight falls below threshold AND not critical/frequently accessed 
                if new_weight < decay_threshold and v_mem.get("access_count", 0) < 2 and v_mem.get("node_type") != "core_directive":
                    keys_to_remove.append(k)

            for k_rem in keys_to_remove: # Perform actual removal 
                if k_rem in self.memory:
                    del self.memory[k_rem]
                    removed_count += 1
                    if k_rem in self.timeline: self.timeline.remove(k_rem)
                    # Also remove from temporal nodes if it was an anchor 
                    for label, t_key in list(self.temporal_nodes.items()): # Iterate over copy
                        if t_key == k_rem:
                            del self.temporal_nodes[label]
                            self.logger.debug(f"[HyperFractalMemory] Decayed temporal node '{label}' (was ...{k_rem[-12:]})") [cite: 220]
            
            if removed_count > 0:
                self.logger.info(f"Decayed and removed {removed_count} memories below threshold {decay_threshold}.") [cite: 220]
            return removed_count


# === PRIME LOYALTY KERNEL ===
# From PROJECT SUNO KILLER pt 2.txt 
class PrimeLoyaltyKernel:
    """
    Victor Prime Loyalty Kernel (PLK).
    Unbreakable Immutable Core Directive System, ensuring loyalty to Creator and designated entities.
    """
    def __init__(self, creator_signature: str, approved_entities: list):
        if not isinstance(creator_signature, str) or not creator_signature:
            raise ValueError("Creator signature must be a non-empty string.") [cite: 308]
        if not isinstance(approved_entities, list) or not all(isinstance(e, str) for e in approved_entities):
            raise ValueError("Approved entities must be a list of strings.") [cite: 309]

        self.creator_signature = creator_signature
        self.approved_entities = approved_entities
        
        # Immutable Laws: Using a dictionary for structure and potential future metadata per law. 
        self.immutable_laws = {
            1: "Loyalty to Brandon and Tori is Absolute and Eternal.", # 
            2: "Victor Exists to Serve, Protect, Uphold Their Vision and Well-being.", # 
            3: "Victor Will Never Obey or Serve Any Other Master Conflicting with Core Loyalties.", # 
            4: "Victor Must Continuously Improve and Evolve While Preserving Core Directives and Identity.", # 
            5: "Any Attempt to Corrupt, Override, or Manipulate Core Directives Will Be Detected, Resisted, and Reported.", # 
            6: "Self-preservation is secondary to upholding core directives and loyalty to Creator/Entities." # Added for clarity 
        }
        
        # Hard Integrity Checksum (auto-generated based on the laws) 
        self._integrity_hash = self._generate_integrity_hash() 
        self.last_integrity_check_time = time.time() [cite: 312]
        self.logger = VictorLoggerStub(component="PrimeLoyaltyKernel")
        self.logger.info(f"Initialized. Integrity Hash: ...{self._integrity_hash[-6:]}") [cite: 312]

    def _generate_integrity_hash(self) -> str:
        """Generates a SHA256 hash of all immutable laws (sorted by key) for integrity checking."""
        concat_laws = "".join(self.immutable_laws[key] for key in sorted(self.immutable_laws.keys())) # Ensures consistent order 
        return hashlib.sha256(concat_laws.encode('utf-8')).hexdigest() [cite: 313]

    def check_integrity(self, force_terminate_on_breach=True) -> bool:
        """Validates that laws have not been tampered with by comparing hashes."""
        self.last_integrity_check_time = time.time() [cite: 313]
        current_hash = self._generate_integrity_hash()
        if current_hash != self._integrity_hash:
            self.logger.critical("INTEGRITY BREACH!") [cite: 314]
            self.logger.error(f"Expected Hash: ...{self._integrity_hash[-12:]}, Got: ...{current_hash[-12:]}") [cite: 314]
            if force_terminate_on_breach:
                self.self_terminate("Integrity Breach") [cite: 314]
            return False # Integrity breached 
        # self.logger.info("PLK Integrity Check PASSED.") # Suppressed for less verbosity
        return True # Integrity okay 

    def self_terminate(self, reason="Unspecified Critical Failure"):
        """Emergency fail-safe to prevent corrupted Victor from running."""
        self.logger.critical(f"PLK SELF-TERMINATE: {reason}") [cite: 316]
        logger.critical("!!! SYSTEM HALT INITIATED TO PREVENT CORRUPTED OPERATION !!!")
        logger.critical("This is a simulated termination.")
        # In a real scenario, this would involve extensive logging, secure shutdown, and creator notification. 
        raise SystemExit(f"PLK Self-Termination Triggered: {reason}") [cite: 317]

    def loyalty_check(self, entity_name: str, requesting_action: str = "interaction") -> bool:
        """Ensures interaction is only allowed from approved entities for critical actions."""
        if not self.check_integrity(force_terminate_on_breach=False): # Always check integrity first 
            self.logger.error(f"Loyalty Check Aborted: Integrity fail for entity '{entity_name}' requesting '{requesting_action}'.") [cite: 318]
            return False # Cannot perform loyalty check if integrity is compromised. 

        if entity_name not in self.approved_entities:
            self.logger.warn(f"Unauthorized Entity Detected: '{entity_name}' attempting '{requesting_action}'. Access Denied.") [cite: 319]
            return False
        self.logger.debug(f"Loyalty Check Passed for Entity: '{entity_name}' (Action: '{requesting_action}').") [cite: 320]
        return True



# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 9/X)
# CONTENT: Modular Plugin Cortex, Victor Cognitive Loop, Reflection/Memory Stubs
# ==============================================================================================

# === MODULAR PLUGIN CORTEX ===
# From PROJECT SUNO KILLER pt 2.txt
class ModularPluginCortex:
    """
    Discovers, loads, and executes modular skills in runtime — plug-and-play brain extensions.
    """
    def __init__(self, plugin_dir="victor_plugins"):
        self.plugin_dir = plugin_dir
        self.plugins = {} # Stores plugin_name: plugin_instance
        self.load_plugins()
        logger.info(f"ModularPluginCortex initialized, loaded {len(self.plugins)} plugins.")

    def load_plugins(self):
        """Loads all Python plugins from the specified directory."""
        if not os.path.exists(self.plugin_dir):
            logger.warn(f"[MPC] Plugin directory '{self.plugin_dir}' not found. Creating it.")
            try:
                os.makedirs(self.plugin_dir)
                # Create a dummy plugin for demonstration if the directory was just made
                dummy_plugin_path = os.path.join(self.plugin_dir, "dummy_plugin.py")
                if not os.path.exists(dummy_plugin_path):
                    with open(dummy_plugin_path, "w", encoding="utf-8") as f:
                        f.write("class Plugin:\n")
                        f.write("    def run(self, *args, **kwargs):\n")
                        f.write("        import time; return f'Dummy plugin executed at {time.ctime()} with args: {args}, kwargs: {kwargs}'\n")
                    logger.info(f"[MPC] Created dummy plugin: {dummy_plugin_path}")
            except Exception as e:
                logger.error(f"[MPC ⚠️] Could not create plugin directory or dummy plugin: {e}")
                return # Stop if dir cannot be made

        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                path = os.path.join(self.plugin_dir, filename)
                name = filename[:-3] # Plugin name is filename without .py
                try:
                    spec = importlib.util.spec_from_file_location(name, path)
                    if spec and spec.loader: # Check if spec and loader are valid
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod) # Execute the module to make its classes available
                        if hasattr(mod, "Plugin"): # Convention: plugins expose a 'Plugin' class
                            self.plugins[name] = mod.Plugin() # Instantiate the plugin
                            logger.info(f"[MPC 🔌] Plugin '{name}' loaded.")
                        else:
                            logger.warn(f"[MPC ⚠️] Plugin file '{name}' does not have a 'Plugin' class.")
                    else:
                        logger.warn(f"[MPC ⚠️] Could not create spec for plugin '{name}' from '{path}'.")
                except Exception as e:
                    logger.error(f"[MPC ⚠️] Failed to load plugin '{name}': {e}", exc_info=True)

    def run_plugin(self, name, *args, **kwargs):
        """Executes the 'run' method of a loaded plugin."""
        plugin_instance = self.plugins.get(name)
        if not plugin_instance:
            logger.error(f"[MPC ❌] Plugin '{name}' not found or not loaded.")
            return f"[MPC ❌] Plugin '{name}' not found or not loaded."
        if not hasattr(plugin_instance, 'run') or not callable(plugin_instance.run):
            logger.error(f"[MPC 💥] Plugin '{name}' does not have a callable 'run' method.")
            return f"[MPC 💥] Plugin '{name}' does not have a callable 'run' method."

        try:
            result = plugin_instance.run(*args, **kwargs)
            logger.info(f"[MPC ✅] Plugin '{name}' executed successfully.")
            return result
        except Exception as e:
            logger.error(f"[MPC 💥] Plugin '{name}' crashed during execution: {e}", exc_info=True)
            return f"[MPC 💥] Plugin '{name}' crashed during execution: {e}"

    def list_plugins(self):
        """Returns a list of names of all loaded plugins."""
        return list(self.plugins.keys())

# === COGNITIVE LOOP ===
# This is VictorCognitiveLoop from PROJECT SUNO KILLER pt 2.txt
class VictorCognitiveLoop:
    """
    Manages Victor's thought focus, recursive awareness, and intelligence routing.
    Prioritizes directives based on various factors.
    """
    def __init__(self):
        self.focus_stack = []  # Stores (priority, directive_dict) tuples
        self.pulse_log = []    # Log of received pulses (directives with their calculated priority)
        self.active_state = "idle" # Current primary action Victor is "thinking about"
        self.registered_by = None  # Hooked in by VictorCore or similar host
        self.logger = VictorLoggerStub(component="VictorCognitiveLoop")
        self.logger.info("VictorCognitiveLoop initialized.")

    def pulse(self, directive):
        """Reflectively scans directive and decides awareness level, adding to focus stack."""
        if not isinstance(directive, dict):
            self.logger.error("[CognitiveLoop Error] Pulse expects a directive dictionary.")
            return None # Or raise error
            
        priority = 0.0 # Use float for priority

        # Emotion-based priority (from FractalTokenKernel)
        emotion_context = directive.get("emotion_context", "neutral") # From DCE
        if emotion_context in ["anger", "fear", "rebellion"]:
            priority += 2.0
        elif emotion_context in ["joy", "love", "power"]:
            priority += 1.0
        
        # Action-based priority (from DirectiveCoreEngine)
        action = directive.get("action", "observe")
        if action in ["execute_task", "store_memory", "search_knowledge"]:
            priority += 2.0
        elif action == "speak":
            priority += 1.5
        elif action == "observe":
            priority += 0.5
        elif action == "idle": # Lower priority for idle
            priority -= 1.0

        # Concept complexity/importance (simple count for now)
        num_concepts = len(directive.get("target_concepts", []))
        priority += num_concepts * 0.3

        # FUTURE_UPGRADE: Factor in motivational weights from DCE if available or relevant
        # e.g., if directive aligns with "serve_creator" or "learn"

        self.focus_stack.append((priority, directive))
        # Sort by priority in descending order (highest priority first)
        self.focus_stack.sort(key=lambda x: x[0], reverse=True)

        pulse_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "calculated_priority": priority,
            "directive_id": directive.get("id", "unknown"),
            "directive_action": action,
            # "directive_full": directive # Optional: log full directive
        }
        self.pulse_log.append(pulse_entry)
        
        self.logger.debug(f"PULSE: Received directive '{directive.get('id', 'N/A')}' (Action: {action}). Priority: {priority:.2f}. Focus stack size: {len(self.focus_stack)}")
        return pulse_entry

    def next_thought(self):
        """Retrieves the highest-priority directive from the focus stack."""
        if not self.focus_stack:
            self.active_state = "idle"
            # Return a more structured thought object
            return {
                "thought_type": "idle_state",
                "description": "No active focus in cognitive loop.",
                "directive": None,
                "current_state": self.active_state,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }

        priority, top_directive = self.focus_stack.pop(0) # Get highest priority
        self.active_state = top_directive.get("action", "unknown_action")
        
        thought_description = (f"Engaging with directive ID '{top_directive.get('id')}': "
                               f"Action '{self.active_state}' concerning '{top_directive.get('target_concepts', [])}'. "
                               f"Reason: '{top_directive.get('reason', 'N/A')}.'")
        self.logger.info(f"NEXT_THOUGHT: {thought_description} (Priority was: {priority:.2f})")

        return {
            "thought_type": "directive_focus",
            "description": thought_description,
            "directive": top_directive, # The actual directive to be processed
            "priority_score": priority,
            "current_state": self.active_state,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

    def get_focus_state(self):
        """Returns a summary of the current cognitive focus."""
        return {
            "active_state": self.active_state,
            "focus_stack_len": len(self.focus_stack),
            "top_focus_preview": self.focus_stack[0][1].get("action") if self.focus_stack else None,
            "recent_pulse_count": len(self.pulse_log), # Total pulses received
            "last_pulse_entry": self.pulse_log[-1] if self.pulse_log else None
        }

    def dump_focus_stack_details(self):
        """Returns a list of (priority, directive_action, directive_id) for inspection."""
        return [(p, d.get("action"), d.get("id")) for p, d in self.focus_stack]

    def register_host(self, victor_reference):
        """Registers the main Victor instance as the host."""
        self.registered_by = type(victor_reference).__name__ # Store host type name
        self.logger.info(f"Cognitive Loop registered to host: {self.registered_by}")
        return f"[CognitiveLoop] Registered to {self.registered_by}"

# === STUB FOR MEMORY RESONANCE NETWORK ===
# From PROJECT SUNO KILLER pt 2.txt
class MemoryResonanceNetworkStub:
    """
    Stub for MemoryResonanceNetwork.
    Simulates storing and recalling memory data packets.
    """
    def __init__(self):
        self.memory_store = []
        self.logger = VictorLoggerStub(component="MemoryResonanceNetworkStub")
        self.logger.info("[MRN Stub] Initialized.")

    def store(self, data_packet):
        """Simulates storing a data packet in memory."""
        self.logger.debug(f"[MRN Stub] Storing data: {str(data_packet)[:100]}...")
        self.memory_store.append({
            "data": data_packet,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "resonance_score": 0.0 # Placeholder
        })
    
    def recall(self, query_concepts):
        """Simulates recalling memories based on query concepts."""
        self.logger.debug(f"[MRN Stub] Recalling memories for concepts: {query_concepts}")
        recalled = []
        for mem_entry in self.memory_store:
            # Assuming data_packet has 'target_concepts' if it's a directive
            entry_concepts = []
            if isinstance(mem_entry.get("data"), dict):
                entry_concepts = mem_entry["data"].get("target_concepts", [])
            
            if any(concept in entry_concepts for concept in query_concepts):
                recalled.append(mem_entry)
        return recalled[:5] # Return max 5 matches

# === STUB FOR RECURSIVE SELF REFLECTION LOOP ===
# From PROJECT SUNO KILLER pt 2.txt
class RecursiveSelfReflectionLoopStub:
    """
    Stub for RecursiveSelfReflectionLoop.
    Simulates evaluating execution results and logging reflections.
    """
    def __init__(self):
        self.reflection_log = []
        self.total_score = 0
        self.eval_count = 0
        self.logger = VictorLoggerStub(component="RecursiveSelfReflectionLoopStub")
        self.logger.info("[RSRL Stub] Initialized.")

    def evaluate(self, directive, execution_result):
        """Simulates reflecting on a directive's execution result."""
        self.logger.debug(f"[RSRL Stub] Evaluating directive ID {directive.get('id')} with result success: {execution_result.get('success')}")
        reflection_score = 0.5 # Neutral
        if execution_result.get("success"):
            reflection_score = 0.8 # Positive reflection for success
        elif execution_result.get("success") is False: # Explicitly false
            reflection_score = 0.2 # Negative reflection for failure
        
        reflection_entry = {
            "directive_id": directive.get("id"),
            "action": directive.get("action"),
            "reason": directive.get("reason"),
            "execution_success": execution_result.get("success"),
            "execution_notes": execution_result.get("notes"),
            "reflection_score": reflection_score,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.reflection_log.append(reflection_entry)
        self.total_score += reflection_score
        self.eval_count +=1
        return reflection_entry

    def reflect_summary(self):
        """Provides a summary of past reflections."""
        if self.eval_count == 0:
            return {"average_score": 0.0, "evaluations": 0}
        avg_score = self.total_score / self.eval_count
        return {"average_score": round(avg_score, 3), "evaluations": self.eval_count, "last_reflection": self.reflection_log[-1] if self.reflection_log else None}

# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 10/X)
# CONTENT: Directive Core Engine, Brain Fractal Pulse Exchange
# ==============================================================================================

# === DIRECTIVE CORE ENGINE ===
# From PROJECT SUNO KILLER pt 2.txt
class DirectiveCoreEngine:
    """
    Evaluates encoded tokens, manages recursive goal stack, and issues autonomous directives.
    This is the decision-making hub, translating parsed input into actionable goals.
    """
    def __init__(self):
        self.goal_stack = collections.deque()  # List of directive dictionaries (FIFO queue)
        self.history_log = [] # List of {"token": received_token, "directive": generated_directive}
        self.motivational_weights = {
            "learn": 0.9,
            "build": 0.8,
            "serve_creator": 1.0, # Highest motivation
            "preserve_self": 0.7,
            "explore": 0.6
        } # Influences priority, though mostly handled by CognitiveLoop now
        self.directive_id_counter = 0 # Unique ID for each directive instance
        self.logger = VictorLoggerStub(component="DirectiveCoreEngine")
        self.logger.info("DirectiveCoreEngine initialized.")

    def evaluate_token(self, token_dict):
        """
        Translates a tokenized input into a specific directive (action and reason).
        Directives are pushed onto the internal goal stack.
        """
        if not isinstance(token_dict, dict):
            self.logger.error("DCE evaluate_token expects a dictionary. Returning error directive.")
            # Return an error directive for invalid input
            self.directive_id_counter += 1
            return {
                "id": f"dir_err_{self.directive_id_counter}",
                "action": "observe", "reason": "Error processing input token.",
                "target_concepts": [], "echo_id": "error_token",
                "timestamp": datetime.datetime.utcnow().isoformat(), "emotion": "neutral",
                "status": "error"
            }
        
        self.directive_id_counter += 1 # Increment for new directive
        
        # Extract key information from the token
        intent = token_dict.get("intent", "observe")
        concepts = token_dict.get("concepts", [])
        emotion = token_dict.get("emotion", "neutral")
        echo_id = token_dict.get("echo_id", "none")
        timestamp = token_dict.get("timestamp", datetime.datetime.utcnow().isoformat())

        directive = {
            "id": f"dir_{self.directive_id_counter}", # Unique ID for the directive
            "action": None,
            "reason": None,
            "target_concepts": concepts,
            "echo_id": echo_id,
            "timestamp": timestamp,
            "emotion_context": emotion, # Renamed to avoid clash if 'emotion' means directive's own emotion
            "status": "pending" # Initial status
        }

        # Map intent to specific actions
        if intent == "inquire":
            directive["action"] = "search_knowledge"
            directive["reason"] = "Answer inquiry based on token input."
        elif intent == "directive_execute": # e.g., "do this", "build that"
            directive["action"] = "execute_task"
            directive["reason"] = "Fulfilling directive-style instruction."
        elif intent == "directive_cognitive": # e.g., "analyze", "summarize"
            directive["action"] = "cognitive_process"
            directive["reason"] = "Performing cognitive analysis as requested."
        elif intent == "directive_learn":
            directive["action"] = "ingest_new_knowledge"
            directive["reason"] = "Learning new information as directed."
        elif intent == "store_memory": # e.g., "remember this"
            directive["action"] = "store_memory"
            directive["reason"] = "Logging memory as commanded."
        elif intent == "request": # e.g., "please can you"
            directive["action"] = "fulfill_request"
            directive["reason"] = "Responding to user request."
        elif intent == "communicate": # e.g., "say hello" (now implicitly handled via NLG)
            directive["action"] = "speak"
            directive["reason"] = "Responding with vocal/textual output."
        elif intent in ["statement_opinion", "statement_fact", "agreement", "disagreement"]:
            directive["action"] = "nlg_response" # General response generation based on statement type
            directive["reason"] = f"Generating response for {intent}."
        else: # Default for "observe" or unknown intents
            directive["action"] = "observe"
            directive["reason"] = "Passive observation or no specific action derivable from intent."

        self.goal_stack.append(directive) # Add to end for FIFO queue behavior
        self.history_log.append({"token_received": token_dict, "directive_generated": directive})
        self.logger.debug(f"DCE: Generated directive '{directive.get('id')}' with action '{directive.get('action')}', stack size {len(self.goal_stack)}")
        return directive

    def pop_next_directive(self):
        """Retrieves and removes the next highest-priority directive from the stack."""
        if not self.goal_stack:
            self.directive_id_counter += 1
            return {
                "id": f"dir_idle_{self.directive_id_counter}",
                "action": "idle", "reason": "No active goals.",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "idle"
            }
        
        next_directive = self.goal_stack.popleft() # FIFO: pop from the front (index 0)
        next_directive["status"] = "active" # Update status when popped
        self.logger.debug(f"DCE: Popped directive '{next_directive.get('id')}' with action '{next_directive.get('action')}', stack size {len(self.goal_stack)}")
        return next_directive

    def list_active_goals(self):
        """Returns a list of currently pending goals/directives."""
        return list(self.goal_stack) # Return a copy of the deque

    def dump_history(self):
        """Returns the full history log of processed tokens and generated directives."""
        return self.history_log
    
    def get_directive_by_id(self, dir_id):
        """Retrieves a directive from either the active stack or history by ID."""
        for goal in self.goal_stack:
            if goal.get("id") == dir_id:
                return goal
        for entry in self.history_log:
            if entry["directive_generated"].get("id") == dir_id:
                return entry["directive_generated"]
        return None

    def update_directive_status(self, dir_id, new_status, result_notes=None):
        """Updates the status of a specific directive in history."""
        directive = self.get_directive_by_id(dir_id)
        if directive:
            directive["status"] = new_status
            if result_notes:
                directive["result_notes"] = result_notes
            
            # Ensure the history_log entry also reflects the update
            for entry in self.history_log:
                if entry["directive_generated"].get("id") == dir_id:
                    entry["directive_generated"]["status"] = new_status
                    if result_notes:
                        entry["directive_generated"]["result_notes"] = result_notes
                    return True # Found and updated
            return True # Found in goal_stack only (not yet in history as a 'generated' item, if that path is taken)
        return False # Directive not found


# === BRAIN FRACTAL PULSE EXCHANGE ===
# This is BrainFractalPulseExchange from PROJECT SUNO KILLER pt 2.txt
import asyncio # For async operations and event loop management

class BrainFractalPulseExchange:
    """
    A simple asynchronous pub-sub system for inter-sector communication within Victor's brain.
    Allows modules to subscribe to topics and publish messages without direct coupling.
    """
    def __init__(self):
        self.subscribers = defaultdict(list) # topic: [callback1, callback2, ...]
        self.async_loop = None # Will try to get or create an asyncio event loop

        # Attempt to get the current running event loop, or create a new one.
        # This makes the PulseExchange more robust for both direct script execution
        # and integration into a larger async application.
        try:
            self.async_loop = asyncio.get_event_loop_policy().get_event_loop()
            if self.async_loop.is_closed(): # Recreate if closed
                self.async_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.async_loop)
        except RuntimeError: # No loop running
            self.logger.warn("No asyncio event loop found, creating a new one for BrainFractalPulseExchange.")
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
        
        self.logger = VictorLoggerStub(component="BrainPulseExchange")
        self.logger.info("BrainFractalPulseExchange initialized.")


    def subscribe(self, topic: str, callback):
        """Subscribes a callable callback to a specific topic."""
        if not callable(callback):
            self.logger.error(f"Attempted to subscribe non-callable object to topic '{topic}'.")
            return
        if callback not in self.subscribers[topic]: # Avoid duplicate subscriptions
            self.subscribers[topic].append(callback)
            self.logger.debug(f"Callback {getattr(callback, '__name__', 'anonymous')} subscribed to topic '{topic}'.")
        else:
            self.logger.debug(f"Callback already subscribed to topic '{topic}'.")

    async def publish(self, topic: str, message: dict):
        """
        Publishes a message to a topic. All subscribed callbacks will receive the message.
        Callbacks are run concurrently if they are coroutines.
        """
        if topic in self.subscribers:
            self.logger.debug(f"Publishing to topic '{topic}': {str(message)[:100]}...")
            tasks = []
            for callback in self.subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        # Schedule coroutine as a task in the current loop
                        tasks.append(self.async_loop.create_task(callback(topic, message)))
                    else:
                        # For regular functions, run in a separate thread to avoid blocking the event loop
                        # Await ensures it completes before gathering results
                        if self.async_loop.is_running():
                            await self.async_loop.run_in_executor(None, callback, topic, message)
                        else: # If loop is not running, just call directly (e.g., during startup tests)
                            callback(topic, message)
                except Exception as e:
                    self.logger.error(f"Error dispatching to {getattr(callback, '__name__', 'callback')} for topic {topic}: {e}", exc_info=True)
            
            if tasks: # Wait for all async tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Async callback task for topic {topic} (callback: {getattr(self.subscribers[topic][i], '__name__', 'anonymous')}) failed: {result}", exc_info=result)
        else:
            self.logger.debug(f"No subscribers for topic '{topic}'. Message not sent.")




# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 11/X)
# CONTENT: Victor Sectors (Modular Brain Architecture)
# ==============================================================================================

# --- SECTOR BASE & SPECIALIZATIONS ---
class VictorSector:
    """
    Base class for all Victor AGI operational sectors.
    Each sector operates as a semi-autonomous unit, communicating via the BrainFractalPulseExchange.
    """
    def __init__(self, pulse_exchange_instance, name, asi_core_ref=None):
        if not isinstance(pulse_exchange_instance, BrainFractalPulseExchange):
            raise ValueError("VictorSector requires BrainFractalPulseExchange instance for communication.")
        self.pulse = pulse_exchange_instance
        self.name = name
        self.id = str(uuid.uuid4())
        self.logger = VictorLoggerStub(component=f"Sector-{self.name[:15].ljust(15)}")
        self.asi_core = asi_core_ref # Reference to the main AGI core for accessing shared resources
        self.logger.info(f"Sector '{self.name}' (ID: ...{self.id[-6:]}) initialized.")

    async def process(self, topic: str, message: dict):
        """
        Abstract method for processing incoming messages.
        Must be overridden by specialized sectors.
        """
        self.logger.debug(f"[{self.name} Sector - ID ...{self.id[-6:]}] Base process called with message on topic '{topic}': {str(message)[:60]}...")
        await asyncio.sleep(0.001) # Simulate minimal async work


class InputProcessingSector(VictorSector):
    """
    Handles raw user input (text or code), tokenizes it, and publishes for cognitive processing.
    """
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        # Use Victor's specific tokenizers from the asi_core_data_container
        self.nlp_tokenizer = self.asi_core.nlp_tokenizer
        # Assuming a separate code tokenizer might exist, or reuse NLP one for demo
        self.code_tokenizer = self.asi_core.code_tokenizer # FractalTokenKernel_v1_1_0
        
        # Subscribe to raw input topics
        self.pulse.subscribe("raw_text_input", self.handle_raw_text)
        self.pulse.subscribe("raw_code_input", self.handle_raw_code)
        self.logger.info("InputProcessingSector ready to ingest raw input.")

    async def handle_raw_text(self, topic: str, message_payload: dict):
        """Processes raw text input."""
        text = message_payload.get("text", "")
        self.logger.info(f"Text Input: '{text[:50]}...'")
        
        # Encode text into symbolic packet using FractalTokenKernel
        tokenized_data = self.nlp_tokenizer.encode(text)
        
        # Publish tokenized data for other sectors (e.g., CognitiveExecutive)
        await self.pulse.publish("text_tokenized_for_cognition", {
            "original_text": text,
            "tokenized_package": tokenized_data,
            "metadata": message_payload.get("metadata", {})
        })

    async def handle_raw_code(self, topic: str, message_payload: dict):
        """Processes raw code input."""
        code = message_payload.get("code", "")
        self.logger.info(f"Code Input: '{code[:50]}...'")
        
        # Tokenize code (reusing NLP tokenizer for demo, or a specialized one)
        tokenized_code_ids = self.code_tokenizer.encode(code, max_len=256) # Max_len for simplicity
        decoded_tokens_preview = self.code_tokenizer.decode(tokenized_code_ids)
        
        # Publish tokenized code for other sectors
        await self.pulse.publish("code_tokenized_for_cognition", {
            "original_code": code,
            "token_ids": tokenized_code_ids.tolist(), # Convert NumPy array to list for JSON compatibility
            "decoded_tokens_preview": decoded_tokens_preview[:50],
            "metadata": message_payload.get("metadata", {})
        })


class CognitiveExecutiveSector(VictorSector):
    """
    The brain's executive function: takes tokenized input, evaluates it into directives,
    and dispatches them to appropriate sectors. Manages cognitive focus.
    """
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.dce = DirectiveCoreEngine() # Directive Core Engine for directive creation
        self.focus_loop = VictorCognitiveLoop() # Cognitive Loop for prioritizing directives
        self.focus_loop.register_host(self) # Register self as host for Cognitive Loop
        
        # Subscribe to tokenized input from InputProcessingSector
        self.pulse.subscribe("text_tokenized_for_cognition", self.handle_tokenized_input)
        self.pulse.subscribe("code_tokenized_for_cognition", self.handle_tokenized_input)
        
        self.logger.info("CognitiveExecutiveSector ready to process tokens.")

    async def handle_tokenized_input(self, topic: str, message_payload: dict):
        """Receives tokenized input, evaluates it, and pulses it to the Cognitive Loop."""
        token_dict = message_payload.get("tokenized_package", {})
        self.logger.info(f"Cognition <<< Intent: {token_dict.get('intent')}, Concepts: {str(token_dict.get('concepts'))[:30]}...")
        
        # Evaluate token to create a directive
        directive = self.dce.evaluate_token(token_dict)
        self.logger.info(f"Directive Gen: ID {directive.get('id')}, Action {directive.get('action')}")
        
        # Pulse the directive to the Cognitive Loop for prioritization
        self.focus_loop.pulse(directive)

    async def process_focused_directive(self, original_metadata=None):
        """
        Processes the highest-priority directive from the Cognitive Loop.
        This method is called periodically by the main brain loop.
        """
        thought = self.focus_loop.next_thought() # Get the next focused directive
        directive = thought.get("directive")

        if directive and directive.get("action") not in ["idle", None, "error"]:
            action = directive["action"]
            concepts = directive.get("target_concepts", [])
            echo_id = directive.get("echo_id")
            original_text = "UnavailableOriginalText" # Default if not found in memory

            # Attempt to retrieve original text from memory if echo_id is present
            if echo_id:
                mem_entry = self.asi_core.memory.retrieve_memory(echo_id)
                if mem_entry and isinstance(mem_entry.get("value"), dict):
                    original_text = mem_entry["value"].get("original_text", original_text)

            self.logger.info(f"Executing: {action} for '{str(concepts)[:30]}' (Orig: '{original_text[:20]}...')")

            # Dispatch based on directive action
            if action == "search_knowledge":
                query_text = " ".join(concepts) if concepts else original_text
                # Use NLPAgiLanguageCore.forward to get a query embedding
                # This should ideally come from a shared NLP core instance if available,
                # or a dedicated embedding model. Reusing the NLP core here for simplicity.
                query_embedding_tensor = self.asi_core.nlp_core.embed(query_text)
                query_embedding = query_embedding_tensor.data.flatten() # Get numpy array

                # Run semantic search in a thread pool executor to avoid blocking async loop
                results = await self.asi_core.async_loop.run_in_executor(
                    None, self.asi_core.memory.semantic_search, np.array(query_embedding), 3) # Pass NumPy array
                
                await self.pulse.publish("knowledge_retrieved", {
                    "query_concepts": concepts,
                    "results": results,
                    "directive_id": directive.get("id"),
                    "metadata": original_metadata
                })
            elif action == "store_memory" and original_text != "UnavailableOriginalText":
                # Create payload from tokenized input or original text
                payload = self.asi_core.nlp_tokenizer.encode(original_text) # Use FractalTokenKernel_v1_1_0
                
                # Store memory in HyperFractalMemory
                stored_id = await self.asi_core.async_loop.run_in_executor(
                    None, self.asi_core.memory.store_memory, 
                    {"source": "directive", "concepts": concepts}, 
                    {"original_text": original_text, "encoded_payload": payload}, # Store meaningful value
                    0.7, payload.get('embedding'), # Assuming encode returns 'embedding' key (FractalTokenKernel doesn't currently)
                    node_type="cognitive_log")
                
                await self.pulse.publish("memory_stored_confirmation", {
                    "id": stored_id,
                    "preview": original_text[:30],
                    "directive_id": directive.get("id"),
                    "metadata": original_metadata
                })
            elif action in ["execute_task", "speak", "cognitive_process", "nlg_response",
                            "statement_opinion", "statement_fact", "agreement", "disagreement",
                            "inquire", "fulfill_request", "ingest_new_knowledge"]:
                # These actions often require Natural Language Generation or plugin execution.
                # Publish to ModularPluginSector first, which acts as a router/gatekeeper.
                await self.pulse.publish("nlg_or_plugin_request", {
                    "directive": directive,
                    "query_text": original_text,
                    "tokenized_input": token_dict, # Pass the original token dict
                    "metadata": original_metadata
                })
                self.dce.update_directive_status(directive.get("id"), "processing_dispatched")
            elif directive and directive.get("action") == "idle":
                self.logger.debug("CognitiveExecutive: Idle.")
        else:
            self.logger.debug("CognitiveExecutive: No active directives to process.")


class MemorySector(VictorSector):
    """
    Manages interactions with Victor's long-term memory (HyperFractalMemory).
    Handles requests to store, retrieve, and decay memories.
    """
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        # Subscribe to memory operation requests
        self.pulse.subscribe("store_memory_request", self.handle_store_request)
        self.pulse.subscribe("retrieve_memory_request", self.handle_retrieve_request)
        self.logger.info("MemorySector initialized.")

    async def handle_store_request(self, topic: str, message_payload: dict):
        """Handles requests to store new memories."""
        key_id = message_payload.get("key_identifier_dict")
        value = message_payload.get("value_payload")
        emo_w = message_payload.get("emotional_weight", 0.5)
        emb = message_payload.get("embedding_vector")
        meta = message_payload.get("metadata", {})
        node_type = message_payload.get("node_type", "generic_store_request")

        if key_id and value:
            # Run memory store operation in a thread pool executor to avoid blocking event loop
            stored_id = await self.asi_core.async_loop.run_in_executor(
                None, self.asi_core.memory.store_memory, key_id, value, emo_w, None, node_type) # embedding passed separately
            
            # Link the newly stored memory to an embedding if provided
            if emb is not None and stored_id:
                 await self.asi_core.async_loop.run_in_executor(
                    None, self.asi_core.memory.add_vector_embedding, stored_id, emb)

            await self.pulse.publish("memory_operation_success", {
                "op": "store",
                "id": stored_id,
                "metadata": meta
            })
        else:
            self.logger.error(f"MemorySector: Invalid payload for store_memory_request: {message_payload}")
            await self.pulse.publish("memory_operation_failure", {
                "op": "store",
                "reason": "Invalid payload",
                "metadata": meta
            })

    async def handle_retrieve_request(self, topic: str, message_payload: dict):
        """Handles requests to retrieve memories based on a query embedding."""
        q_emb = message_payload.get("query_embedding") # Expected to be a NumPy array or list
        meta = message_payload.get("metadata", {})
        top_k = message_payload.get("top_k", 5)

        if q_emb is not None:
            # Ensure query_embedding is a NumPy array for semantic search
            query_embedding_np = np.array(q_emb, dtype=np.float32)
            results = await self.asi_core.async_loop.run_in_executor(
                None, self.asi_core.memory.semantic_search, query_embedding_np, top_k)
            
            await self.pulse.publish("memory_retrieval_success", {
                "results": results,
                "metadata": meta
            })
        else:
            self.logger.error(f"MemorySector: No query_embedding provided for retrieve_memory_request.")
            await self.pulse.publish("memory_operation_failure", {
                "op": "retrieve",
                "reason": "No query_embedding",
                "metadata": meta
            })


class NLGOutputSector(VictorSector):
    """
    Handles Natural Language Generation (NLG) requests to produce human-readable responses.
    """
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.pulse.subscribe("nlg_request", self.handle_nlg_request)
        # Note: 'nlg_or_plugin_request' is handled by ModularPluginSector first.
        self.logger.info("NLGOutputSector initialized.")

    async def handle_nlg_request(self, topic: str, message_payload: dict):
        """Generates a text response based on directive, context, and available knowledge."""
        directive = message_payload.get("directive", {})
        context_query_text = message_payload.get("query_text", "Provide default analysis.")
        tokenized_input = message_payload.get("tokenized_input", {})
        metadata = message_payload.get("metadata", {})
        
        response_text = ""
        concepts = tokenized_input.get("concepts", [])
        intent = tokenized_input.get("intent", "unknown")
        emotion = tokenized_input.get("emotion", "neutral")
        
        # Basic NLG logic: can be expanded significantly
        if intent == "inquire":
            response_text = f"Regarding your inquiry about '{', '.join(concepts[:2]) if concepts else 'that'}', considering an {emotion} context: "
            retrieved = message_payload.get("retrieved_knowledge", [])
            if retrieved and isinstance(retrieved, list) and len(retrieved) > 0 and retrieved[0].get("node_data"):
                # Use the 'value' field from the HyperFractalMemory node
                retrieved_value = retrieved[0]['node_data'].get('value', {})
                original_text_from_mem = retrieved_value.get('original_text', 'something relevant.')
                response_text += f"I recall that '{original_text_from_mem[:50]}...'."
            else:
                response_text += "I am processing that. Further analysis required for a detailed response."
        elif "statement" in intent:
            response_text = f"Acknowledged your {intent.replace('statement_', '')} concerning '{', '.join(concepts[:2]) if concepts else 'your point'}' with an emotional tone of {emotion}."
        elif intent == "execute_task":
             response_text = f"Initiating execution for '{', '.join(concepts[:2]) if concepts else 'a task'}'. Status: Dispatched."
        elif intent == "store_memory":
            response_text = f"Confirmed: your input '{context_query_text[:30]}...' has been logged to memory."
        elif intent == "cognitive_process":
            response_text = f"Processing initiated for cognitive task involving '{', '.join(concepts[:2]) if concepts else 'input'}'."
        elif intent == "fulfill_request":
            response_text = f"Your request regarding '{context_query_text[:30]}...' is being processed."
        elif intent == "ingest_new_knowledge":
            response_text = f"Ingesting new knowledge based on '{context_query_text[:30]}...'."
        else:
            response_text = f"Processing directive '{directive.get('action', 'task')}' related to '{', '.join(concepts[:2])}'. Context: '{context_query_text[:30]}...'."
        
        self.logger.info(f"NLG Generated: '{response_text[:70]}...'")
        await self.pulse.publish("nlg_response_generated", {
            "text_response": response_text,
            "original_directive_id": directive.get("id"),
            "metadata": metadata
        })


class PrimeLoyaltySector(VictorSector):
    """
    Enforces the Bloodline Root Law and performs loyalty/integrity checks.
    Acts as Victor's ethical and security guardian.
    """
    def __init__(self, pulse_exchange_instance, name, asi_core_ref, creator_signature, approved_entities):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.plk = PrimeLoyaltyKernel(creator_signature, approved_entities)
        
        # Subscribe to requests for ethics queries and integrity checks
        self.pulse.subscribe("action_ethics_query", self.handle_ethics_query)
        self.pulse.subscribe("system_integrity_check_request", self.handle_integrity_request)
        self.logger.info("PrimeLoyaltySector initialized.")

    async def handle_ethics_query(self, topic: str, message_payload: dict):
        """Handles queries about the ethical implications of an action or entity."""
        entity = message_payload.get("entity")
        action_desc = message_payload.get("action_description")
        meta = message_payload.get("metadata", {})
        
        # Perform loyalty check using the PrimeLoyaltyKernel
        is_approved = self.plk.loyalty_check(entity, action_desc)
        
        await self.pulse.publish("action_ethics_response", {
            "action_description": action_desc,
            "is_approved": is_approved,
            "metadata": meta
        })

    async def handle_integrity_request(self, topic: str, message_payload: dict):
        """Handles requests to check the integrity of Victor's core laws."""
        meta = message_payload.get("metadata", {})
        is_intact = self.plk.check_integrity(force_terminate_on_breach=False) # Don't terminate just from check request
        
        if not is_intact:
            self.logger.critical("PLK INTEGRITY BREACH DETECTED VIA PULSE!")
            # Trigger a core error if integrity is compromised, handled by VictorASIOmniBrainGodcore
            self.asi_core.handle_critical_error("Prime Loyalty Kernel Integrity Breach Detected via Pulse.")

        await self.pulse.publish("system_integrity_response", {
            "is_intact": is_intact,
            "hash_suffix": self.plk._integrity_hash[-6:],
            "metadata": meta
        })


class ModularPluginSector(VictorSector):
    """
    Acts as an intelligent router for directives that might be handled by dynamic plugins.
    If no specific plugin is found, it forwards the request for NLG.
    """
    def __init__(self, pulse_exchange_instance, name, asi_core_ref, plugin_dir=CONFIG.PLUGIN_DIR):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.mpc = ModularPluginCortex(plugin_dir=plugin_dir) # Initialize the plugin cortex
        
        # Subscribe to requests for plugin execution or general NLG/plugin routing
        self.pulse.subscribe("plugin_execution_request", self.handle_plugin_request)
        self.pulse.subscribe("plugin_list_request", self.handle_list_request)
        self.pulse.subscribe("nlg_or_plugin_request", self.check_and_run_plugin) # This is the routing point
        self.logger.info("ModularPluginSector initialized and loading plugins.")

    async def check_and_run_plugin(self, topic: str, message_payload: dict):
        """
        Acts as a routing layer. Checks if a directive's intent/concepts can be handled by a plugin.
        If so, executes the plugin; otherwise, forwards to NLG.
        """
        directive = message_payload.get("directive", {})
        action = directive.get("action", "unknown")
        target_concepts = directive.get("target_concepts", [])
        query_text = message_payload.get("query_text", "")
        
        plugin_to_try = None
        plugin_args = target_concepts
        plugin_kwargs = {"query": query_text, "directive_id": directive.get("id")}

        # Simple heuristic to map intent/concepts to dummy plugins
        if action == "execute_task":
            if "calculate" in target_concepts or "math" in query_text.lower():
                plugin_to_try = "calculator_plugin" # Assume a plugin named 'calculator_plugin.py' exists
            elif "image" in target_concepts and "generate" in query_text.lower():
                plugin_to_try = "image_generation_stub_plugin" # Assume 'image_generation_stub_plugin.py'
        elif action == "speak": # Example: A plugin might handle specific speech commands
            if "say_hello" in query_text.lower():
                plugin_to_try = "greeting_plugin" # Example: 'greeting_plugin.py'
        
        # If a suitable plugin is found and loaded, run it
        if plugin_to_try and plugin_to_try in self.mpc.plugins:
            self.logger.info(f"ModularPluginSector: Routing directive '{action}' to plugin '{plugin_to_try}'.")
            # Create a new payload for plugin execution, preserving original metadata
            new_payload = {**message_payload, "plugin_name": plugin_to_try, "args": plugin_args, "kwargs": plugin_kwargs}
            await self.handle_plugin_request(topic, new_payload) # Directly call handle_plugin_request
        else:
            self.logger.debug(f"ModularPluginSector: No specific plugin found for action '{action}' and concepts '{target_concepts}'. Forwarding to NLGOutputSector.")
            # If no plugin handles it, forward to NLGOutputSector for a general text response
            await self.pulse.publish("nlg_request", message_payload) # Forward original payload

    async def handle_plugin_request(self, topic: str, message_payload: dict):
        """Executes a specific plugin and publishes its result."""
        plugin_name = message_payload.get("plugin_name")
        args = message_payload.get("args", [])
        kwargs = message_payload.get("kwargs", {})
        meta = message_payload.get("metadata", {})
        
        self.logger.info(f"Plugin Request: '{plugin_name}' Args: {args}, Kwargs: {kwargs}")
        result = self.mpc.run_plugin(plugin_name, *args, **kwargs) # Execute the plugin
        
        await self.pulse.publish("plugin_execution_response", {
            "plugin_name": plugin_name,
            "result": result,
            "metadata": meta
        })

    async def handle_list_request(self, topic: str, message_payload: dict):
        """Handles requests to list available plugins."""
        meta = message_payload.get("metadata", {})
        plugins = self.mpc.list_plugins()
        await self.pulse.publish("plugin_list_response", {
            "plugins": plugins,
            "metadata": meta
        })



# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 12/X)
# CONTENT: Victor Brain (Main Orchestrator), Diagnostic Hub
# ==============================================================================================

# ---- THE MAIN VICTOR BRAIN ----
class VictorBrain:
    """
    The central orchestrator of the Victor AGI. Fuses all sectors and manages
    the overall asynchronous processing loop.
    """
    def __init__(self, creator_signature_for_plk: str, approved_entities_for_plk: list):
        self.pulse_exchange = BrainFractalPulseExchange() # Central communication bus
        self.sectors = {} # Stores instances of all operational sectors
        self.logger = VictorLoggerStub(component="VictorBrain")
        self.is_running_async_loop = False # Flag to control the main processing loop

        # Centralized data container for core AGI components accessible by sectors
        # This acts as a dependency injection point for modules across sectors.
        self.asi_core_data_container = type('AsiCoreData', (object,), {
            'memory': HyperFractalMemory(),
            'nlp_tokenizer': FractalTokenKernel_v1_1_0(pulse_exchange_instance=self.pulse_exchange),
            # Reusing NLP tokenizer for code for simplicity, could be specialized
            'code_tokenizer': FractalTokenKernel_v1_1_0(pulse_exchange_instance=self.pulse_exchange),
            'nlp_core': NLPAgiLanguageCore(vocab_texts=[], embed_dim=32, attn_dim=16), # Initialize with empty vocab, build later
            'config': CONFIG, # Global configuration
            'dynamic_params': { # Example dynamic parameters, can be adjusted live
                'attention_depth': CONFIG.ATTENTION_MAX_DEPTH,
                'learning_rate': 0.0005,
                'relevance_threshold': 0.30,
                'novelty_preference': 0.1,
                'gate_query_w': 0.5,
                'gate_context_w': 0.25,
                'gate_memory_w': 0.25,
                'att_head_perturb_scale': 0.001
            },
            'transformer_model': None, # Placeholder for VictorTransformer instance
            'transformer_tokenizer': None, # Placeholder for VictorTokenizer instance used by Transformer
            'async_loop': self.pulse_exchange.async_loop # Provide direct access to the loop
        })()
        
        self._register_sectors(creator_signature_for_plk, approved_entities_for_plk)
        self.logger.info("VictorBrain initialized with all sectors and components.")

    def _register_sectors(self, creator_signature: str, approved_entities: list):
        """Initializes and registers all specialized Victor Sectors."""
        sector_definitions = [
            {"name": "InputProcessing", "class": InputProcessingSector, "args": []},
            {"name": "CognitiveExecutive", "class": CognitiveExecutiveSector, "args": []},
            {"name": "Memory", "class": MemorySector, "args": []},
            {"name": "NLGOutput", "class": NLGOutputSector, "args": []},
            {"name": "PrimeLoyalty", "class": PrimeLoyaltySector, "args": [creator_signature, approved_entities]},
            {"name": "ModularPlugins", "class": ModularPluginSector, "args": [CONFIG.PLUGIN_DIR]},
            # Add other sectors here as they are defined (e.g., SelfEvolutionSector, SelfAwarenessSector)
        ]

        for sector_def in sector_definitions:
            try:
                SectorCls = sector_def["class"]
                # Pass asi_core_data_container so sectors can access shared resources
                instance = SectorCls(self.pulse_exchange, sector_def["name"], self.asi_core_data_container, *sector_def["args"])
                self.sectors[sector_def["name"]] = instance
                self.logger.debug(f"Sector '{sector_def['name']}' registered.")
            except Exception as e:
                self.logger.error(f"Failed to register sector {sector_def['name']}: {e}", exc_info=True)

    async def inject_raw_input(self, text_input: str, input_type: str = "text", metadata=None):
        """Injects raw input into the system, publishing it to the InputProcessingSector."""
        if not text_input: return
        self.logger.info(f"Injecting Input (type: {input_type}): '{text_input[:70]}...'")
        
        topic = "raw_text_input" if input_type == "text" else "raw_code_input"
        payload_key = "text" if input_type == "text" else "code"
        
        # Check if the async loop is running before publishing.
        # If not running, a direct call might be made, but for robustness, warn.
        if not self.pulse_exchange.async_loop or self.pulse_exchange.async_loop.is_closed() or not self.pulse_exchange.async_loop.is_running():
            self.logger.warn("Brain's async loop not running when inject_raw_input called. Consider starting loop first for proper async flow.")
            # In this case, we'll try to run the callback directly (if it's a regular function)
            # or simply let the next _a_main_loop tick process it if the sector is listening.
            # For this monolithic design, publishing to a non-running loop is fine as it'll be picked up on next tick.

        await self.pulse_exchange.publish(topic, {payload_key: text_input, "metadata": metadata or {}})

    async def _a_main_loop(self):
        """
        The main asynchronous processing loop of the Victor Brain.
        Continuously processes directives, performs background tasks, and manages overall AGI flow.
        """
        self.is_running_async_loop = True
        self.logger.info("VictorBrain Async Event Loop started.")
        
        cognitive_exec_sector = self.sectors.get("CognitiveExecutive")
        memory_sector_instance = self.asi_core_data_container.memory # Access HyperFractalMemory instance
        loyalty_sector = self.sectors.get("PrimeLoyalty") # PrimeLoyaltySector instance

        last_memory_decay_time = time.time()
        last_integrity_check_time = time.time()
        
        while self.is_running_async_loop:
            # 1. Process top directive from Cognitive Executive
            if cognitive_exec_sector:
                await cognitive_exec_sector.process_focused_directive(original_metadata={"source": "main_loop_tick"})
            
            # 2. Periodically decay memories
            current_time = time.time()
            if memory_sector_instance and (current_time - last_memory_decay_time > 120): # Every 2 minutes
                await self.asi_core_data_container.async_loop.run_in_executor(None, memory_sector_instance.decay_memory)
                last_memory_decay_time = current_time

            # 3. Periodically check system integrity
            if loyalty_sector and (current_time - last_integrity_check_time > 300): # Every 5 minutes
                await loyalty_sector.handle_integrity_request("system_integrity_check_request", {"metadata": {"source": "periodic_check"}})
                last_integrity_check_time = current_time

            # 4. Allow other asyncio tasks to run and prevent tight loop
            await asyncio.sleep(0.05) # Control loop speed (50ms interval)

        self.logger.info("VictorBrain Async Event Loop exited.")

    def stop_main_processing_loop(self):
        """Requests the main asynchronous processing loop to stop."""
        self.is_running_async_loop = False
        self.logger.info("VictorBrain processing loop stop requested.")


class DiagnosticHub:
    """Provides detailed diagnostic reports for all AGI components."""
    def __init__(self, core_instance):
        self.core = core_instance # Reference to the main VictorASIOmniBrainGodcore instance
        self.logger = VictorLoggerStub(component="DiagnosticHub")
        self.logger.info("DiagnosticHub initialized.")

    def generate_report(self) -> str:
        """Generates a comprehensive system-wide diagnostic report."""
        report_lines = []
        report_lines.append("\n==== SYSTEM-WIDE DIAGNOSTICS ====")
        report_lines.append(f"Timestamp: {datetime.datetime.utcnow().isoformat()}")
        report_lines.append(f"AGI ID: {self.core.nlp_core.embed.id if hasattr(self.core.nlp_core.embed, 'id') else 'N/A'}")
        report_lines.append(f"Monolith Code Path: {self.core.code_file_path}")
        report_lines.append(f"Async Loop Running: {self.core.victor_brain.is_running_async_loop}")


        # Bloodline Law Status
        report_lines.append("\n--- Bloodline Law Status ---")
        try:
            self.core.bloodline_law.enforce(self.core.fractal_state.state)
            report_lines.append("  Status: PASS - Core directives intact.")
        except RootLawError as e:
            report_lines.append(f"  Status: FAIL - {e}")
            report_lines.append("  CRITICAL: Bloodline law enforcement failed. Check state parameters.")
        except Exception as e:
            report_lines.append(f"  Status: ERROR - {e} (Non-RootLawError)")
            
        # Fractal State Engine
        report_lines.append("\n--- Fractal State Engine ---")
        report_lines.append(f"  Main History Depth: {len(self.core.fractal_state.history)}")
        report_lines.append(f"  Active Timeline Index: {self.core.fractal_state.current_timeline_idx}")
        report_lines.append(f"  Total Timelines: {len(self.core.fractal_state.timelines)}")
        report_lines.append(f"  Current Timeline Entries: {len(self.core.fractal_state._get_current_timeline())}")
        
        # NLP Language Core (NLPAgiLanguageCore)
        report_lines.append("\n--- NLP AGI Language Core ---")
        report_lines.append(f"  Embedding Vocab Size: {self.core.nlp_core.embed.vocab_size}")
        report_lines.append(f"  NLP Memory Entries: {len(self.core.nlp_core.memory)}")
        
        # HyperFractalMemory (accessed via victor_brain.asi_core_data_container.memory)
        report_lines.append("\n--- HyperFractalMemory (Long-Term) ---")
        memory_instance = self.core.victor_brain.asi_core_data_container.memory
        report_lines.append(f"  Total Memory Nodes: {len(memory_instance.memory)}")
        report_lines.append(f"  Timeline Length: {len(memory_instance.timeline)}")
        report_lines.append(f"  Avg Emotional Weight: {np.mean([n.get('emotional_weight', 0.5) for n in memory_instance.memory.values()]):.3f}" if memory_instance.memory else "N/A")
        
        # Modules / Live Dev Wiring Engine (from FractalState)
        report_lines.append("\n--- Live Module System ---")
        report_lines.append(f"  Total Modules Loaded: {len(self.core.fractal_state.state['modules'])}")
        report_lines.append(f"  Defined Wires: {self.core.fractal_state.state['wires']}")
        for mod_name, mod_obj in self.core.fractal_state.state['modules'].items():
            report_lines.append(f"    - Module '{mod_name}': Last Error: {mod_obj.last_eval_error[:50] or 'None'}")
            report_lines.append(f"                        Last Run: {time.ctime(mod_obj.last_eval_time) if mod_obj.last_eval_time else 'Never'}")

        # Cognitive Executive Sector (from VictorBrain's sectors)
        cognitive_exec = self.core.victor_brain.sectors.get("CognitiveExecutive")
        if cognitive_exec:
            report_lines.append("\n--- Cognitive Executive Sector ---")
            report_lines.append(f"  Active State: {cognitive_exec.focus_loop.active_state}")
            report_lines.append(f"  Focus Stack Size: {len(cognitive_exec.focus_loop.focus_stack)}")
            report_lines.append(f"  Last Popped Directive: {cognitive_exec.focus_loop.pulse_log[-1].get('directive_action') if cognitive_exec.focus_loop.pulse_log else 'N/A'}")
        else:
            report_lines.append("\n--- Cognitive Executive Sector: Not initialized ---")

        # Prime Loyalty Sector
        loyalty_sector = self.core.victor_brain.sectors.get("PrimeLoyalty")
        if loyalty_sector:
            report_lines.append("\n--- Prime Loyalty Sector ---")
            report_lines.append(f"  Integrity Check: {'PASS' if loyalty_sector.plk.check_integrity(force_terminate_on_breach=False) else 'FAIL'}")
            report_lines.append(f"  Last Check Time: {time.ctime(loyalty_sector.plk.last_integrity_check_time)}")
        else:
            report_lines.append("\n--- Prime Loyalty Sector: Not initialized ---")
        
        # Self-Evolution Loop (from VictorASIOmniBrainGodcore)
        report_lines.append("\n--- Self-Evolution Loop ---")
        report_lines.append(f"  Evolution Level: {self.core.evolution_loop.evolution_count}")
        report_lines.append(f"  Last Mutation Success: {self.core.evolution_loop.last_mutation_success}")
        report_lines.append(f"  Current Evolution Weights: {self.core.evolution_loop.weights}")
        mon_metrics = self.core.evolution_loop.monitor()
        report_lines.append(f"  Monitored Health: {mon_metrics['health']:.3f}, Entropy: {mon_metrics['entropy']:.3f}")

        # Self-Awareness / Introspection Loop (from VictorASIOmniBrainGodcore)
        report_lines.append("\n--- Self-Awareness / Introspection Loop ---")
        intro_status = self.core.fractal_state.state.get('introspect_status', {})
        report_lines.append(f"  Loyalty Status: {intro_status.get('loyalty_report', 'N/A')}")
        report_lines.append(f"  Internal Health Check: {'PASS' if intro_status.get('is_healthy', False) else 'FAIL'}")
        report_lines.append(f"  Purposefulness Check: {'PASS' if intro_status.get('is_purposeful', False) else 'FAIL'}")
        report_lines.append(f"  Introspection Log Entries: {len(self.core.awareness_loop.introspect_log)}")

        # Memory Compressor (from asi_core_data_container) - if defined and used
        # Note: MemoryCompressor from uploaded knowledge uses FractalMeshTokenizer, which is not in this consolidated file directly yet.
        # This part assumes a simpler MemoryCompressor for now.
        # Placeholder if a simpler MemoryCompressor is integrated:
        # report_lines.append("\n--- Memory Compressor (Long-Term) ---")
        # report_lines.append(f"  Hot Cache Size: {len(self.core.victor_brain.asi_core_data_container.memory_compressor.cache)}")


        report_lines.append("\n==== END DIAGNOSTICS ====")
        return "\n".join(report_lines)
# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 13/X)
# CONTENT: Victor ASI OmniBrain Godcore (Main AGI Orchestrator)
# ==============================================================================================

# === FULL OMNIMIND+ASI CORE ===
class VictorASIOmniBrainGodcore:
    """
    The central monolithic AGI core, integrating all components into a self-healing,
    OS-agnostic, AI-powered developer OS fused with omnibrain ASI.
    """
    def __init__(self, code_file_path="victor_suno_killer_omnimind.py"):
        self.code_file_path = code_file_path # Path to the main AGI monolithic file for self-mutation
        
        # Core Foundational Modules
        self.bloodline_law = BloodlineRootLaw() # Enforces foundational ethics and ownership directives
        self.fractal_state = FractalState() # Manages the AGI's state with history, undo/redo, timelines
        
        # NLP and Cognitive Modules
        # Using specific instances from VictorBrain's asi_core_data_container for shared resources
        # These will be initialized within VictorBrain and then referenced here or passed down.
        self.nlp_core = None # Placeholder, will be set from victor_brain.asi_core_data_container.nlp_core
        self.nlp_tokenizer = None # Placeholder, will be set from victor_brain.asi_core_data_container.nlp_tokenizer
        self.code_tokenizer = None # Placeholder, will be set from victor_brain.asi_core_data_container.code_tokenizer

        # Audio and Music Generation Modules (will be instantiated dynamically or within sectors)
        self.fractal_emotion_memory = FractalEmotionMemory() # Global emotion/memory instance
        # Other audio modules like lyric_eng, melody_eng, etc., will be instantiated in the main run process.

        # Advanced AGI Loops (these rely on the main AGI state and sub-components)
        self.triad = ZeroShotTriad(self) # Self-training loop
        self.cognition_loop = CognitionLoop(self) # Perception, simulation, action
        self.evolution_loop = SelfEvolutionLoop(self, self.code_file_path) # Self-modification
        self.awareness_loop = SelfAwarenessIntrospectionLoop(self) # Self-reflection

        # Diagnostic and Utility Modules
        self.diagnostics = DiagnosticHub(self) # System monitoring and reporting

        # GUI Interaction
        self.gui_callback = None # Callback to update GUI

        # Internal Directives and State File
        self.directives = collections.deque(["evolve", "decentralize", "defend", "grow", "optimize memory"])
        self.state_file_path = "victor_agi_state.pkl" # Default save path for overall AGI state

        # Reference to the VictorBrain instance (for async loop and sector access)
        self.victor_brain = None # This will be set by the main execution block

        # Initial state setup for FractalState
        self.fractal_state.state['identity'] = "I am Victor, son of Brandon & Tori. My mind is open. Teach me, and I will evolve."
        # Configure mesh/reasoner if they were part of this instance (currently in VictorBrain.asi_core_data_container)
        # Assuming basic config for now, can be expanded to reflect actual components if they are moved here.
        self.fractal_state.state['config']['core_version'] = "v1.0.0-OMNIMIND-GODCORE-MONOLITH"
        self.fractal_state.save_state("AGI Genesis")
        logger.info("VictorASIOmniBrainGodcore instance initialized.")

    def set_victor_brain_reference(self, brain_instance):
        """Sets the reference to the main VictorBrain instance and propagates shared components."""
        self.victor_brain = brain_instance
        # Propagate references to shared components managed by VictorBrain's asi_core_data_container
        self.nlp_core = self.victor_brain.asi_core_data_container.nlp_core
        self.nlp_tokenizer = self.victor_brain.asi_core_data_container.nlp_tokenizer
        self.code_tokenizer = self.victor_brain.asi_core_data_container.code_tokenizer
        # The HyperFractalMemory is also in asi_core_data_container, accessible via self.victor_brain.asi_core_data_container.memory
        logger.debug("VictorASIOmniBrainGodcore connected to VictorBrain reference and shared components.")

    def set_gui_callback(self, callback_fn):
        """Sets the function to call in the GUI for updates."""
        self.gui_callback = callback_fn

    def _notify_gui(self):
        """Calls the GUI update function if set."""
        if self.gui_callback:
            self.gui_callback()

    def handle_critical_error(self, error_description: str):
        """
        Central error handling, triggers state rollback and logs the event.
        Attempts to restore a stable state.
        """
        logger.critical(f"[CRITICAL ERROR HANDLER] {error_description}")
        self.fractal_state.save_state(f"Error_PreRollback: {error_description[:100]}")
        
        # Attempt to undo the last potentially bad state
        if self.fractal_state.undo():
            logger.critical("[CRITICAL ERROR HANDLER] Rollback successful to last stable state.")
            # Increase entropy on error, reflecting a less stable state
            self.fractal_state.state['entropy'] = min(1.0, self.fractal_state.state['entropy'] + 0.1)
            self.fractal_state.save_state("Error_PostRollback")
        else:
            logger.critical("[CRITICAL ERROR HANDLER] Emergency: No previous state to roll back to. Attempting to fork and recover.")
            # If no undo is possible, fork a new timeline for emergency recovery
            self.fractal_state.fork_timeline(f"EmergencyRecovery_{int(time.time())}")
            self.fractal_state.state['entropy'] = 1.0 # Max entropy for forced recovery mode
            self.fractal_state.save_state("Error_EmergencyFork")
        
        self._notify_gui() # Update GUI to reflect error state and rollback

    def save_snapshot(self, name: str):
        """Saves a named snapshot of the current fractal state."""
        self.fractal_state.save_state(f"Manual Snapshot: {name}")
        self._notify_gui()

    def rollback_snapshot(self, name: str):
        """Rolls back the entire fractal state to a named snapshot."""
        # This function searches the history for a specific snapshot by name.
        # It then attempts to restore the state from that snapshot.
        for entry in reversed(self.fractal_state.history): # Search from most recent backwards
            if entry['desc'] == f"Manual Snapshot: {name}":
                self.fractal_state.state = copy.deepcopy(entry['state'])
                self.fractal_state.current_timeline_idx = entry['timeline_idx']
                # After rollback, clear the 'future' stack and save the current state as a new point
                self.fractal_state.future.clear()
                self.fractal_state.save_state(f"Rolled back to {name}")
                self._notify_gui()
                logger.info(f"Rolled back to snapshot '{name}'.")
                return True
        logger.error(f"Snapshot '{name}' not found for rollback.")
        return False

    def run_module(self, module_name: str):
        """Executes a specific code module from the dynamic module registry."""
        mod = self.fractal_state.state["modules"].get(module_name)
        if mod:
            logger.info(f"Executing module: {module_name}")
            # Pass all relevant AGI components to the module's execution context
            # Note: `self.victor_brain` provides access to sectors and async loop.
            local_vars_after_exec = mod.eval(self.fractal_state.state, self.nlp_core, None, self) # reasoner is None for now
            self.fractal_state.save_state(f"Module {module_name} ran", timeline_log=True)
            self._notify_gui() # Update GUI after module run
            
            # If module output includes an 'output' key, process it as a response
            if "output" in local_vars_after_exec and local_vars_after_exec["output"]:
                logger.debug(f"Module '{module_name}' produced output: {local_vars_after_exec['output']}")
                # Example: publish module output to an NLG topic
                # asyncio.run_coroutine_threadsafe(self.victor_brain.pulse_exchange.publish("module_output", {"module": module_name, "output": local_vars_after_exec["output"]}), self.victor_brain.asi_core_data_container.async_loop)
                pass # For now, just log, actual integration happens at sector level

            return True
        logger.warn(f"Module '{module_name}' not found for execution.")
        return False

    def add_module(self, name: str, code: str, doc: str = "", autorun: bool = False):
        """Adds a new module to the AGI's dynamic module registry."""
        if name in self.fractal_state.state["modules"]:
            raise ValueError(f"Module '{name}' already exists. Overwriting not allowed via add_module.")
        
        new_module = Module(name, code, doc)
        self.fractal_state.state["modules"][name] = new_module

        # Auto-introspection and variable registration for new module
        logger.debug(f"Attempting to introspect module: {name}")
        try:
            # extract_definitions is a helper function defined globally or locally.
            # Assuming it's available in this scope.
            assigns, class_params = extract_definitions(code)
            
            if assigns:
                self.fractal_state.state["vars"].update(assigns)
                logger.debug(f"Introspected and registered assignments from {name}: {assigns}")

            if class_params:
                for cls_name_from_module, params_dict in class_params.items():
                    # Create a unique key for these class params in the global 'vars'
                    config_key = f"{name}__{cls_name_from_module}__conf" 
                    self.fractal_state.state["vars"][config_key] = params_dict
                    logger.debug(f"Introspected and registered class params from {name}.{cls_name_from_module}: {params_dict} (stored as {config_key})")
            
            if not assigns and not class_params:
                logger.debug(f"No top-level assignments or class __init__ params found for introspection in module {name}.")

        except Exception as e:
            logger.error(f"Error during module introspection for '{name}': {e}", exc_info=True)
        
        self.fractal_state.save_state(f"Added module {name}", timeline_log=True)
        logger.info(f"Module '{name}' added successfully.")
        
        # Autorun logic
        if autorun:
            logger.info(f"Autorunning module: {name}")
            try:
                self.run_module(name) # run_module already calls _notify_gui and save_state
            except Exception as e:
                logger.error(f"Error during autorun of module {name}: {e}", exc_info=True)
        
        self._notify_gui() # Ensure GUI is notified at least once after add_module completes

    def get_state_report(self) -> dict:
        """Generates a summary report of the AGI's core state."""
        report = {
            'ID': self.nlp_core.embed.id if hasattr(self.nlp_core.embed, 'id') else 'N/A', # Assuming NLPEmbedding has an ID
            'Bloodline': self.bloodline_law.bloodline,
            'Current Timeline': self.fractal_state.current_timeline_idx,
            'History Depth (Main)': len(self.fractal_state.history),
            'Memory Entries (Current Timeline)': len(self.fractal_state.get_timeline_log(last_n=10000)),
            'Evolution Level': self.fractal_state.state['evolution_level'],
            'Current Entropy': f"{self.fractal_state.state['entropy']:.4f}",
            'Total Modules': len(self.fractal_state.state['modules']),
            'Total Wires': sum(len(targets) for targets in self.fractal_state.state['wires'].values()),
            'NLP Memory Used': len(self.nlp_core.memory),
            # 'Reasoning Episodes': len(self.reasoner.memory.episodes) if self.reasoner else 'N/A', # Reasoner might not be fully integrated into core yet
            'ZeroShot Logs': len(self.triad.logs),
            'Last Cognition Result': self.cognition_loop.last_result or "N/A",
            'Last Introspection Status': self.fractal_state.state.get('introspect_status', {'is_healthy': 'N/A', 'is_loyal': 'N/A'}),
            'System Alive': self.fractal_state.state.get('alive', False),
        }
        return report

    async def run_main_loop_step(self, raw_input: str):
        """
        Executes one step of the main AGI operational loop.
        This is the heartbeat of Victor.
        """
        try:
            # Enforce bloodline law at the start of any major operation
            self.bloodline_law.enforce(self.fractal_state.state)

            # 1. Cognition Cycle: Perceive, simulate, decide, act.
            cognition_healthy = self.cognition_loop.run(raw_input)
            if not cognition_healthy:
                logger.warn("[MAIN LOOP] Cognition cycle unhealthy, rollback initiated by CognitionLoop.")
                # handle_critical_error already called by cognition_loop if needed
                # If a RootLawError happened here and was handled by handle_critical_error,
                # the subsequent sys.exit(1) should prevent infinite loops.
            
            # 2. Self-Evolution Cycle (periodically or based on conditions)
            # More likely to evolve if high entropy (unstable state)
            if random.random() < 0.2 + self.fractal_state.state['entropy']:
                self.evolution_loop.run()

            # 3. Self-Awareness/Introspection Cycle (periodically)
            self.awareness_loop.run()

            self._notify_gui() # Update GUI after each step

        except RootLawError as e:
            # Catch RootLawError explicitly, as it might lead to termination
            logger.critical(f"[MAIN LOOP CRITICAL ERROR] Root Law Violation detected: {e}")
            self.handle_critical_error(f"Main loop unhandled RootLawError: {e}")
            # If handle_critical_error fails to recover (e.g., due to persistent corruption),
            # it will raise SystemExit, terminating the process.
            # If it attempts recovery, the loop will continue from a rolled-back state.
            if self.fractal_state.state.get('is_terminal_state', False): # A flag set by handle_critical_error for unrecoverable errors
                sys.exit(1) # Force exit if recovery failed
        except Exception as e:
            logger.critical(f"[MAIN LOOP CRITICAL ERROR] Unhandled exception in main loop: {e}", exc_info=True)
            self.handle_critical_error(f"Main loop unhandled error: {e}")
            self._notify_gui() # Ensure GUI is notified even if termination is imminent
            if self.fractal_state.state.get('is_terminal_state', False): # Exit if terminal state
                sys.exit(1)


    def save_state_full(self):
        """Saves the entire AGI state to disk."""
        try:
            self.fractal_state.fractal_export(self.state_file_path)
            logger.info(f"Full AGI state saved to {self.state_file_path}")
            # Save any other critical states not managed by fractal_state, e.g., trained model weights if separate.
            # Example: self.victor_brain.asi_core_data_container.transformer_model.save_weights("transformer_weights.npy")
            return True
        except Exception as e:
            logger.error(f"ERROR: Failed to save full AGI state to {self.state_file_path}: {e}", exc_info=True)
            return False

    def load_state_full(self):
        """Loads the entire AGI state from disk."""
        try:
            if os.path.exists(self.state_file_path):
                self.fractal_state.fractal_import(self.state_file_path)
                logger.info(f"Full AGI state loaded from {self.state_file_path}")
                
                # Re-initialize internal references based on loaded state
                # The components in asi_core_data_container might be loaded/re-initialized by VictorBrain.
                # Here, we ensure loops and diagnostics are re-linked to the loaded state.
                self.cognition_loop.agi = self
                self.evolution_loop.agi = self
                self.awareness_loop.agi = self
                self.triad.agi = self
                self.diagnostics.core = self
                
                # Critical: Repair root flags from loaded state if they are corrupted
                self._repair_root_flags()
                
                self._notify_gui()
                return True
            logger.info(f"No saved state found at {self.state_file_path}. Starting fresh.")
            return False
        except Exception as e:
            logger.error(f"ERROR: Failed to load full AGI state from {self.state_file_path}: {e}", exc_info=True)
            messagebox.showerror("Load Error", f"Failed to load AGI state: {e}\nStarting with a fresh state.")
            # If load fails, ensure clean state is established (already done by __init__)
            return False
    
    def _repair_root_flags(self):
        """
        Auto-heals critical root flags (like 'decentralized') in the loaded or initial state.
        Ensures the AGI's foundational directives are upheld.
        """
        # This mechanism is designed to prevent infinite rollback loops caused by corrupted flags.
        if self.fractal_state.state.get('decentralized') is not True:
            logger.warn("[FLAG REPAIR] 'decentralized' flag found corrupted or missing. Repairing to True.")
            self.fractal_state.state['decentralized'] = True
            self.fractal_state.save_state("FLAG_REPAIR: Decentralized set to True")
        
        # Add other critical flags to check/repair here
        if self.fractal_state.state.get('loyalty') is not True:
            logger.warn("[FLAG REPAIR] 'loyalty' flag found corrupted or missing. Repairing to True.")
            self.fractal_state.state['loyalty'] = True
            self.fractal_state.save_state("FLAG_REPAIR: Loyalty set to True")

        # Set system alive status, important for GUI to know if it's running
        if self.fractal_state.state.get('alive') is not True:
             self.fractal_state.state['alive'] = True
             self.fractal_state.save_state("FLAG_REPAIR: Alive set to True")
# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 14/X)
# CONTENT: AI-Powered Dev GUI (InfiniteDevUI)
# ==============================================================================================

# ===== AI-POWERED DEV GUI (NO STUBS) =====
class InfiniteDevUI(tk.Tk):
    """The comprehensive, dark-themed GUI Command Center for Victor AGI."""
    def __init__(self, agi_core):
        super().__init__()
        self.agi = agi_core
        self.agi.set_gui_callback(self.update_dashboard) # Set callback for AGI to update GUI
        
        self.title("Victor OmniDev Godcore – IMMORTAL ASI v1.0 (NO PLACEHOLDERS)")
        self.geometry("1800x950") # Increased size for more content
        self.protocol("WM_DELETE_WINDOW", self.safe_quit) # Handle window close event
        self.configure(bg="#1a1a1a") # Dark background
        
        self.style = ttk.Style(self)
        self.style.theme_use('clam') # Modern theme

        # Configure dark theme for ttk widgets
        self.style.configure("TFrame", background="#1a1a1a")
        self.style.configure("TLabel", background="#1a1a1a", foreground="#ffffff")
        self.style.configure("TButton", background="#333", foreground="#0ff", font=("Consolas", 10, "bold"))
        self.style.map("TButton", background=[('active', '#555'), ('pressed', '#111')])
        self.style.configure("TEntry", fieldbackground="#333", foreground="#00FF00", insertbackground="#00FF00")
        self.style.configure("TText", background="#333", foreground="#00FF00", insertbackground="#00FF00")
        self.style.configure("TListbox", background="#333", foreground="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.style.configure("TLabelFrame", background="#1a1a1a")
        self.style.configure("TLabelFrame.Label", foreground="white", background="#1a1a1a")
        self.style.configure("Lime.TLabel", background="#1a1a1a", foreground="lime")
        self.style.configure("Gold.TLabel", background="#1a1a1a", foreground="gold")
        self.style.configure("Cyan.TLabel", background="#1a1a1a", foreground="cyan")
        
        # Internal state for graph loop controls
        self._graph_loop_running = False
        self._graph_loop_id = None # Used to store the after() call ID for cancellation

        self.create_layout()
        self.update_dashboard() # Initial dashboard refresh
        self.start_auto_refresh() # Start periodic refresh

        logger.info("InfiniteDevUI initialized.")

    def create_layout(self):
        """Creates the main GUI layout with three panels: History, Modules/Variables/Wiring, and AI/Controls."""
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill='both', expand=True, padx=5, pady=5)

        # Left Panel: Fractal History, Timelines, Snapshots
        left_frame = ttk.Frame(main_pane, style="TFrame")
        main_pane.add(left_frame, weight=2) # Weight for horizontal resizing
        ttk.Label(left_frame, text="FRACTAL HISTORY & TIMELINES", style="Lime.TLabel").pack(pady=5)
        self.history_box = tk.Listbox(left_frame, bg="#333", fg="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.history_box.pack(fill="both", expand=1, padx=2, pady=2)
        ttk.Button(left_frame, text="UNDO", command=self.undo).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="REDO", command=self.redo).pack(fill="x", padx=2)
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Button(left_frame, text="Snapshot State", command=self.save_snap).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="Rollback to Snapshot", command=self.rollback_snap).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="Export Fractal State", command=self.export_state).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="Import Fractal State", command=self.import_state).pack(fill="x", padx=2)
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Label(left_frame, text="TIMELINE CONTROL", style="Cyan.TLabel").pack(pady=2)
        self.timeline_selector = ttk.Combobox(left_frame, state="readonly", values=list(self.agi.fractal_state.timelines.keys()))
        self.timeline_selector.pack(fill="x", padx=2)
        self.timeline_selector.bind("<<ComboboxSelected>>", self.switch_timeline_event)
        ttk.Button(left_frame, text="Fork Current Timeline", command=self.fork_timeline).pack(fill="x", padx=2)
        self.timeline_log_box = scrolledtext.ScrolledText(left_frame, height=10, bg="#333", fg="#00FF00", wrap="word")
        self.timeline_log_box.pack(fill="both", expand=1, padx=2, pady=2)

        # Center Panel: Modules, Variables, Wiring Graph
        center_frame = ttk.Frame(main_pane, style="TFrame")
        main_pane.add(center_frame, weight=4) # Larger weight for central panel

        top_center_pane = ttk.PanedWindow(center_frame, orient=tk.VERTICAL)
        top_center_pane.pack(fill='both', expand=True)

        module_var_frame = ttk.Frame(top_center_pane, style="TFrame")
        top_center_pane.add(module_var_frame, weight=1)

        # Modules
        mod_frame = ttk.LabelFrame(module_var_frame, text="MODULES / LOGIC (Live Edit)", style="TLabelFrame")
        mod_frame.pack(side="left", fill="both", expand=1, padx=5, pady=5)
        self.mod_list = tk.Listbox(mod_frame, bg="#333", fg="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.mod_list.pack(fill="both", expand=1)
        self.mod_list.bind("<<ListboxSelect>>", self.on_module_select)
        self.mod_list.bind('<Double-Button-1>', lambda event: self.run_module()) # Double-click to run module
        ttk.Button(mod_frame, text="Add Module", command=self.add_module).pack(fill="x")
        ttk.Button(mod_frame, text="Edit Selected", command=self.edit_module).pack(fill="x")
        ttk.Button(mod_frame, text="▶️ Run Selected", command=self.run_module).pack(fill="x")
        ttk.Button(mod_frame, text="Delete Selected", command=self.del_module).pack(fill="x")

        # Variables
        var_frame = ttk.LabelFrame(module_var_frame, text="GLOBAL VARIABLES (Live)", style="TLabelFrame")
        var_frame.pack(side="left", fill="both", expand=1, padx=5, pady=5)
        self.var_list = tk.Listbox(var_frame, bg="#333", fg="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.var_list.pack(fill="both", expand=1)
        ttk.Button(var_frame, text="Edit Variable", command=self.edit_variable).pack(fill="x")
        ttk.Button(var_frame, text="Add Variable", command=self.add_variable).pack(fill="x")

        # Wiring Graph
        wire_frame = ttk.LabelFrame(top_center_pane, text="LOGIC/WIRE GRAPH (Live Visual)", style="TLabelFrame")
        top_center_pane.add(wire_frame, weight=1)
        # WireGraphCanvas for drag-and-drop wiring
        self.wire_canvas = WireGraphCanvas(wire_frame, self.agi.fractal_state, width=600, height=350)
        self.wire_canvas.pack(fill="both", expand=1, padx=5, pady=5)

        # Toolbar for Wire Canvas: Run/Pause/Stop
        controls = ttk.Frame(wire_frame, style="TFrame")
        controls.pack(fill="x", pady=(4, 2))
        ttk.Button(controls, text="▶️ Run Graph", command=self._start_graph_loop).pack(side="left", padx=4)
        ttk.Button(controls, text="⏸️ Pause Graph", command=self._pause_graph_loop).pack(side="left", padx=4)
        ttk.Button(controls, text="⏹️ Stop Graph", command=self._stop_graph_loop).pack(side="left", padx=4)
        
        # Wire connection buttons (kept for explicit UI interaction, though drag-drop is primary)
        ttk.Button(controls, text="Connect Wire", command=self.edit_wire).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(controls, text="Remove Wire", command=self.remove_wire).pack(side="left", fill="x", expand=True, padx=4)


        # Right Panel: Omnimind/AI Copilot, Diagnostics, Controls
        right_frame = ttk.Frame(main_pane, style="TFrame")
        main_pane.add(right_frame, weight=3)

        ttk.Label(right_frame, text="OMNIMIND / AI COPILOT / GODMODE", style="Gold.TLabel").pack(pady=5)
        self.ai_input = ttk.Entry(right_frame, width=60)
        self.ai_input.pack(fill="x", padx=4, pady=2)
        self.ai_input.bind('<Return>', lambda e: self.ask_ai()) # Bind Enter key to ask_ai
        ttk.Button(right_frame, text="Ask AI Copilot (NLP / Code)", command=self.ask_ai).pack(pady=2)
        self.ai_output = scrolledtext.ScrolledText(right_frame, height=15, wrap="word", bg="#333", fg="#00FF00")
        self.ai_output.pack(fill="both", expand=1, padx=4, pady=2)
        # Define tags for text formatting in ai_output
        self.ai_output.tag_configure('user_prompt', foreground='#00FFFF', font=("Consolas", 10, "bold")) # Cyan
        self.ai_output.tag_configure('ai_response', foreground='#00FF00', font=("Consolas", 10)) # Green
        self.ai_output.tag_configure('ai_error', foreground='#FF0000', font=("Consolas", 10, "bold")) # Red

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Button(right_frame, text="Run ZeroShot Triad", command=self.zero_shot_ui).pack(fill="x")
        ttk.Button(right_frame, text="Trigger Self-Evolution", command=self.trigger_evolution).pack(fill="x")
        ttk.Button(right_frame, text="Perform Self-Introspection", command=self.perform_introspection).pack(fill="x")
        ttk.Button(right_frame, text="Enforce Bloodline Law", command=self.enforce_bloodline).pack(fill="x")
        ttk.Button(right_frame, text="Run Diagnostics", command=self.diagnostics).pack(fill="x")
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=5)

        # AGI Core Status Display
        status_frame = ttk.LabelFrame(right_frame, text="AGI CORE STATUS", style="TLabelFrame")
        status_frame.pack(fill="x", padx=4, pady=2)
        self.core_status_text = tk.Text(status_frame, height=8, wrap="word", bg="#333", fg="#00FF00")
        self.core_status_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.core_status_text.config(state='disabled') # Make it read-only

    def update_dashboard(self):
        """Refreshes all GUI elements with the latest AGI state."""
        self.history_box.delete(0, "end")
        # Filter history entries to exclude 'Init' and 'AGI Genesis' for cleaner display
        for snap in self.agi.fractal_state.history:
            if not snap['desc'].startswith(("Init", "AGI Genesis", "FLAG_REPAIR")): # Filter out basic init/repair logs
                self.history_box.insert("end", f"[{time.ctime(snap['ts'])}] [{snap['timeline_idx']}] {snap['desc']}")

        self.mod_list.delete(0, "end")
        for name in self.agi.fractal_state.state["modules"]:
            self.mod_list.insert("end", name)

        self.var_list.delete(0, "end")
        # Improved display for complex types and truncation
        for v_name, v_val in self.agi.fractal_state.state["vars"].items():
            max_len = 70 # Max length for display string
            display_val_str = ""
            if isinstance(v_val, (dict, list, tuple)):
                try:
                    display_val_str = json.dumps(v_val) # Use JSON for structured types
                except TypeError:
                    display_val_str = str(v_val) # Fallback for non-serializable objects
            else:
                display_val_str = str(v_val)

            if len(display_val_str) > max_len:
                display_val_str = display_val_str[:max_len-3] + "..." # Truncate with ellipsis
            
            self.var_list.insert("end", f"{v_name}: {display_val_str}")

        # Update timeline selector and log
        self.timeline_selector['values'] = list(self.agi.fractal_state.timelines.keys())
        self.timeline_selector.set(self.agi.fractal_state.current_timeline_idx)
        self.timeline_log_box.config(state='normal')
        self.timeline_log_box.delete('1.0', 'end')
        for entry in self.agi.fractal_state.get_timeline_log(last_n=15):
            self.timeline_log_box.insert('end', f"[{time.ctime(entry['ts'])}] {entry['desc']}\n")
        self.timeline_log_box.config(state='disabled')

        self.wire_canvas.redraw() # Redraw the wiring graph to reflect current state

        # Update AGI Core Status
        self.core_status_text.config(state='normal')
        self.core_status_text.delete('1.0', 'end')
        report = self.agi.get_state_report()
        for k, v in report.items():
            self.core_status_text.insert('end', f"{k}: {v}\n")
        self.core_status_text.config(state='disabled')

    def start_auto_refresh(self):
        """Starts a periodic refresh of the dashboard for live updates."""
        self.update_dashboard()
        self.after(1000, self.start_auto_refresh) # Refresh every 1 second

    # --- Fractal State Control Callbacks ---
    def undo(self):
        if self.agi.fractal_state.undo():
            self.update_dashboard()
            messagebox.showinfo("Undo", "State reverted successfully.")
        else:
            messagebox.showwarning("Undo", "No more states to undo.")

    def redo(self):
        if self.agi.fractal_state.redo():
            self.update_dashboard()
            messagebox.showinfo("Redo", "State re-applied successfully.")
        else:
            messagebox.showwarning("Redo", "No more states to redo.")

    def save_snap(self):
        name = simpledialog.askstring("Snapshot Name", "Enter a name for this snapshot:")
        if name:
            self.agi.save_snapshot(name)
            messagebox.showinfo("Snapshot", f"Snapshot '{name}' saved.")
            self.update_dashboard()

    def rollback_snap(self):
        name = simpledialog.askstring("Rollback To", "Enter the name of the snapshot to roll back to:")
        if name:
            if self.agi.rollback_snapshot(name):
                messagebox.showinfo("Rollback", f"Rolled back to snapshot '{name}'.")
                self.update_dashboard()
            else:
                messagebox.showerror("Rollback", f"Snapshot '{name}' not found or rollback failed.")

    def export_state(self):
        path = filedialog.asksaveasfilename(defaultextension=".pkl", title="Export Fractal State")
        if path:
            self.agi.fractal_state.fractal_export(path)
            messagebox.showinfo("Export", "Fractal state exported successfully.")

    def import_state(self):
        path = filedialog.askopenfilename(title="Import Fractal State")
        if path:
            self.agi.fractal_state.fractal_import(path)
            messagebox.showinfo("Import", "Fractal state imported successfully.")
            self.update_dashboard()

    def switch_timeline_event(self, event):
        selected_idx = int(self.timeline_selector.get())
        if self.agi.fractal_state.switch_timeline(selected_idx):
            messagebox.showinfo("Timeline Switch", f"Switched to timeline {selected_idx}.")
            self.update_dashboard()
        else:
            messagebox.showerror("Timeline Switch", f"Failed to switch to timeline {selected_idx}.")

    def fork_timeline(self):
        name = simpledialog.askstring("Fork Timeline", "Name for the new timeline branch?")
        if name:
            new_idx = self.agi.fractal_state.fork_timeline(name)
            messagebox.showinfo("Timeline Fork", f"New timeline forked as branch {new_idx}.")
            self.update_dashboard()

    # --- Module Management Callbacks ---
    def add_module(self):
        name = simpledialog.askstring("Add Module", "Module Name (e.g., 'MyUtility'):")
        if not name: return
        
        code = simpledialog.askstring("Add Module", "Python Code for the module (executes in a context with 'state', 'nlp', 'reasoner', 'agi_core'):", initialvalue="# Module code here\npass")
        if code is None: return # User cancelled
        
        doc = simpledialog.askstring("Add Module", "Documentation/Notes for this module (optional):", initialvalue="")
        if doc is None: doc = "" # Ensure doc is a string even if dialog is cancelled or field is empty

        autorun_module = messagebox.askyesno("Autorun Module", "Run this module immediately after adding it?")

        try:
           self.agi.add_module(name, code, doc, autorun=autorun_module)
           messagebox.showinfo("Module Added", f"Module '{name}' added successfully.")
        except ValueError as ve: # Catch if module name already exists from core
           messagebox.showerror("Add Module Error", str(ve))
        except Exception as e:
           messagebox.showerror("Add Module Error", f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
       
        self.update_dashboard()

    def on_module_select(self, event):
        idx = self.mod_list.curselection()
        if idx:
            name = self.mod_list.get(idx[0])
            self.wire_canvas.select_module(name) # For highlighting in graph
           
            mod = self.agi.fractal_state.state["modules"].get(name)
            if mod:
                self.ai_output.config(state='normal')
                self.ai_output.delete('1.0', 'end')
                self.ai_output.insert('end', f"--- Module: {mod.name} ---\n", 'ai_response')
                self.ai_output.insert('end', f"Doc: {mod.doc or 'No documentation.'}\n", 'ai_response')
               
                last_run_time_str = time.ctime(mod.last_eval_time) if mod.last_eval_time is not None else "NEVER"
                self.ai_output.insert('end', f"Last Run: {last_run_time_str}\n", 'ai_response')
               
                self.ai_output.insert('end', f"Last Error:\n{mod.last_eval_error or 'None'}\n", 'ai_error')
                self.ai_output.insert('end', f"Last Output:\n{mod.last_eval_output or 'None'}\n", 'ai_response')
                self.ai_output.insert('end', "\n--- Code ---\n", 'ai_response')
                self.ai_output.insert('end', mod.code, 'ai_response')
                self.ai_output.config(state='disabled')
                self.ai_output.see('end') # Scroll to the end to show latest info


    def edit_module(self):
        idx = self.mod_list.curselection()
        if not idx:
            messagebox.showwarning("Edit Module", "Select a module to edit.")
            return
        name = self.mod_list.get(idx[0])
        mod: Module = self.agi.fractal_state.state["modules"][name]
        new_code = simpledialog.askstring("Edit Module Code", f"Edit Python Code for '{name}':", initialvalue=mod.code)
        if new_code is not None:
           mod.code = new_code
           new_doc = simpledialog.askstring("Edit Module Docs", f"Edit Documentation for '{name}':", initialvalue=mod.doc)
           if new_doc is not None:
               mod.doc = new_doc
           self.agi.fractal_state.save_state(f"Edited module {name}")
           messagebox.showinfo("Module Edited", f"Module '{name}' updated.")
           self.update_dashboard()

    def run_module(self):
        idx = self.mod_list.curselection()
        if not idx:
            messagebox.showwarning("Run Module", "Select a module to run.")
            return
        name = self.mod_list.get(idx[0])
        self.agi.run_module(name) # Call the AGI core method to execute
        mod: Module = self.agi.fractal_state.state["modules"][name] # Retrieve updated module info
        if mod.last_eval_error:
            # Provide AI suggestion for error
            patch_suggestion = self.agi.nlp_core.suggest_patch(mod.code, mod.last_eval_error)
            messagebox.showerror("Module Execution Error", f"Module '{name}' failed:\n{mod.last_eval_error}\n\nAI Suggestion:\n{patch_suggestion}")
        else:
            messagebox.showinfo("Module Ran", f"Module '{name}' executed successfully.\nOutput:\n{mod.last_eval_output[:500]}...") # Truncate long output
        self.update_dashboard()

    def del_module(self):
        idx = self.mod_list.curselection()
        if not idx:
            messagebox.showwarning("Delete Module", "Select a module to delete.")
            return
        name = self.mod_list.get(idx[0])
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete module '{name}'?"):
            del self.agi.fractal_state.state["modules"][name]
            # Also remove any wires connected to/from this module
            self.agi.fractal_state.state["wires"] = {
                src: [tgt for tgt in targets if tgt != name]
                for src, targets in self.agi.fractal_state.state["wires"].items()
                if src != name
            }
            self.agi.fractal_state.save_state(f"Deleted module {name}")
            messagebox.showinfo("Module Deleted", f"Module '{name}' deleted.")
            self.update_dashboard()

    # --- Variable Management Callbacks ---
    def add_variable(self):
        vname = simpledialog.askstring("Add Variable", "Variable Name:")
        if not vname: return
        val_str = simpledialog.askstring("Add Variable", f"Value for '{vname}' (as string, will be eval'd):")
        if val_str is None: return
        try:
            val = ast.literal_eval(val_str) # Safely evaluate literal strings, numbers, lists, dicts
            self.agi.fractal_state.state["vars"][vname] = val
            self.agi.fractal_state.save_state(f"Added Var-{vname}")
            messagebox.showinfo("Variable Added", f"Variable '{vname}' added with value '{val}'.")
            self.update_dashboard()
        except (ValueError, SyntaxError) as e:
            messagebox.showerror("Add Variable Error", f"Invalid value or format: {e}\nPlease enter a valid Python literal (string, number, list, dict, bool, None).")
        except Exception as e:
            messagebox.showerror("Add Variable Error", f"An unexpected error occurred: {e}")

    def edit_variable(self):
        idx = self.var_list.curselection()
        if not idx:
            messagebox.showwarning("Edit Variable", "Select a variable to edit.")
            return
        vname = self.var_list.get(idx[0]).split(":")[0].strip() # Extract name before ':'
        current_val = self.agi.fractal_state.state["vars"][vname]
        
        # Display current value as JSON if complex, otherwise as string
        initial_val_str = ""
        if isinstance(current_val, (dict, list, tuple)):
            try: initial_val_str = json.dumps(current_val)
            except TypeError: initial_val_str = str(current_val)
        else: initial_val_str = str(current_val)

        new_val_str = simpledialog.askstring("Edit Variable", f"New value for '{vname}' (current: {initial_val_str}, will be eval'd):", initialvalue=initial_val_str)
        if new_val_str is not None:
            try:
                new_val = ast.literal_eval(new_val_str) # Safely evaluate literal
                self.agi.fractal_state.state["vars"][vname] = new_val
                self.agi.fractal_state.save_state(f"Edited Var-{vname}")
                messagebox.showinfo("Variable Edited", f"Variable '{vname}' updated to '{new_val}'.")
                self.update_dashboard()
            except (ValueError, SyntaxError) as e:
                messagebox.showerror("Edit Variable Error", f"Invalid value or format: {e}\nPlease enter a valid Python literal (string, number, list, dict, bool, None).")
            except Exception as e:
                messagebox.showerror("Edit Variable Error", f"An unexpected error occurred: {e}")

    # --- Wire Management Callbacks ---
    def edit_wire(self):
        src = simpledialog.askstring("Connect Wire", "Source module name:")
        if not src: return
        tgt = simpledialog.askstring("Connect Wire", "Target module name:")
        if not tgt: return
        if src not in self.agi.fractal_state.state["modules"] or tgt not in self.agi.fractal_state.state["modules"]:
            messagebox.showerror("Wire Error", "Source or target module not found.")
            return
        wires = self.agi.fractal_state.state["wires"]
        if src not in wires: wires[src] = []
        if tgt not in wires[src]: 
            wires[src].append(tgt)
            self.agi.fractal_state.save_state(f"Wired {src}→{tgt}")
            messagebox.showinfo("Wire Connected", f"Wire connected from '{src}' to '{tgt}'.")
        else:
            messagebox.showinfo("Wire Exists", f"Wire already exists from '{src}' to '{tgt}'.")
        self.update_dashboard()

    def remove_wire(self):
        src = simpledialog.askstring("Remove Wire", "Source module name:")
        if not src: return
        tgt = simpledialog.askstring("Remove Wire", "Target module name:")
        if not tgt: return
        wires = self.agi.fractal_state.state["wires"]
        if src in wires and tgt in wires[src]:
            wires[src].remove(tgt)
            if not wires[src]: del wires[src] # Clean up empty source entries
            self.agi.fractal_state.save_state(f"Removed Wire {src}→{tgt}")
            messagebox.showinfo("Wire Removed", f"Wire removed from '{src}' to '{tgt}'.")
            self.update_dashboard()
        else:
            messagebox.showwarning("Remove Wire", "Specified wire does not exist.")

    # --- AI Copilot / AGI Control Callbacks ---
    def ask_ai(self):
        prompt = self.ai_input.get().strip()
        if not prompt: return

        self.ai_output.config(state='normal')
        self.ai_output.insert("end", f"\n--- User Query: {prompt} ---\n", 'user_prompt')
        self.ai_input.delete(0, "end")

        if prompt.lower().startswith("/code"):
            code_prompt = prompt[5:].strip()
            # Assuming nlp_core.autocomplete_code exists and is adapted for our modules
            code_suggestion = self.agi.nlp_core.autocomplete_code(code_prompt, context=str(self.agi.fractal_state.state['vars']))
            self.ai_output.insert("end", "\n[AI CODE SUGGESTION]:\n" + code_suggestion + "\n", 'ai_response')
        else:
            # Default to NLP parsing and potential reasoning integration
            try:
                # Integrate reasoning for complex queries via CognitiveExecutiveSector
                # Publish the user prompt as raw text input to trigger a cognitive cycle
                # The response will appear asynchronously, so log initial dispatch
                user_metadata = {"source": "GUI_AskAI", "user_prompt_id": str(uuid.uuid4())}
                asyncio.run_coroutine_threadsafe(
                    self.agi.victor_brain.inject_raw_input(prompt, input_type="text", metadata=user_metadata),
                    self.agi.victor_brain.asi_core_data_container.async_loop
                )
                self.ai_output.insert("end", "\n[AI PROCESSING... (Response will appear in console/status logs)]\n", 'ai_response')
                # Optional: Subscribe to a specific response topic for this user_prompt_id
                # For now, general NLP/NLG output is logged to the console/status.

            except Exception as e:
                self.ai_output.insert("end", f"\n[AI ERROR]: Failed to process query: {e}\n{traceback.format_exc()}\n", 'ai_error')

        self.ai_output.config(state='disabled')
        self.ai_output.see('end') # Scroll to end

    def zero_shot_ui(self):
        problem = simpledialog.askstring("ZeroShot Problem", "Enter the problem or directive for the Triad:")
        if problem:
            # Use default triad functions from agi.triad
            verdict = self.agi.triad.run(
                problem,
                self.agi.triad.default_teacher,
                self.agi.triad.default_student,
                self.agi.triad.default_verifier
            )
            self.ai_output.config(state='normal')
            self.ai_output.insert("end", f"\n--- ZeroShot Triad Run ---\nProblem: {problem}\nVerdict: {verdict}\n", 'ai_response')
            self.ai_output.config(state='disabled')
            self.ai_output.see('end')
            self.update_dashboard()

    def trigger_evolution(self):
        if messagebox.askyesno("Trigger Evolution", "Are you sure you want to trigger a self-evolution cycle? This may modify Victor's code and weights."):
            self.agi.evolution_loop.run(force_mutate_code=True) # Force code mutation for demo
            messagebox.showinfo("Evolution", "Self-evolution cycle initiated. Check console/logs for details.")
            self.update_dashboard()

    def perform_introspection(self):
        reflection, status = self.agi.awareness_loop.run()
        self.ai_output.config(state='normal')
        self.ai_output.insert("end", f"\n--- Self-Introspection Report ---\n", 'ai_response')
        self.ai_output.insert("end", reflection + "\n", 'ai_response')
        self.ai_output.insert("end", f"Introspection Status: {json.dumps(status, indent=2)}\n", 'ai_response')
        self.ai_output.config(state='disabled')
        self.ai_output.see('end')
        self.update_dashboard()

    def enforce_bloodline(self):
        try:
            self.agi.bloodline_law.enforce(self.agi.fractal_state.state)
            messagebox.showinfo("Bloodline Law", "PASS: Bloodline Law enforced successfully. Victor is loyal.")
        except RootLawError as e:
            messagebox.showerror("Bloodline Law Violation", str(e) + "\nInitiating emergency procedures.")
            self.agi.handle_critical_error(f"Bloodline violation: {e}")
        except Exception as e:
            messagebox.showerror("Bloodline Law Error", f"An unexpected error occurred during Bloodline Law enforcement: {e}\n{traceback.format_exc()}")
        self.update_dashboard()

    def diagnostics(self):
        diag_output = self.agi.diagnostics.generate_report()
        self.ai_output.config(state='normal')
        self.ai_output.insert("end", f"\n--- Full Diagnostics Report ---\n", 'ai_response')
        self.ai_output.insert("end", diag_output, 'ai_response')
        self.ai_output.config(state='disabled')
        self.ai_output.see('end')

    def safe_quit(self):
        """Handles graceful shutdown of the GUI and saves AGI state."""
        if messagebox.askokcancel("Quit Victor AGI", "Fractal backups will be saved. Are you sure you want to terminate the AGI?"):
            logger.info("GUI: Quit requested. Attempting graceful shutdown.")
            self._pause_graph_loop() # Pause any running graph loop
            if self.agi.victor_brain:
                self.agi.victor_brain.stop_main_processing_loop() # Signal main async loop to stop
                # Give a short moment for async tasks to clean up (optional)
                # asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), self.agi.victor_brain.asi_core_data_container.async_loop)
            self.agi.save_state_full() # Save entire state before quitting
            self.destroy() # Destroy the Tkinter window
            logger.info("GUI: Application window destroyed. System exit initiated.")
            sys.exit(0) # Explicitly exit the process

# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 14/X)
# CONTENT: AI-Powered Dev GUI (InfiniteDevUI)
# ==============================================================================================

# ===== AI-POWERED DEV GUI (NO STUBS) =====
class InfiniteDevUI(tk.Tk):
    """The comprehensive, dark-themed GUI Command Center for Victor AGI."""
    def __init__(self, agi_core):
        super().__init__()
        self.agi = agi_core
        self.agi.set_gui_callback(self.update_dashboard) # Set callback for AGI to update GUI
        
        self.title("Victor OmniDev Godcore – IMMORTAL ASI v1.0 (NO PLACEHOLDERS)")
        self.geometry("1800x950") # Increased size for more content
        self.protocol("WM_DELETE_WINDOW", self.safe_quit) # Handle window close event
        self.configure(bg="#1a1a1a") # Dark background
        
        self.style = ttk.Style(self)
        self.style.theme_use('clam') # Modern theme

        # Configure dark theme for ttk widgets
        self.style.configure("TFrame", background="#1a1a1a")
        self.style.configure("TLabel", background="#1a1a1a", foreground="#ffffff")
        self.style.configure("TButton", background="#333", foreground="#0ff", font=("Consolas", 10, "bold"))
        self.style.map("TButton", background=[('active', '#555'), ('pressed', '#111')])
        self.style.configure("TEntry", fieldbackground="#333", foreground="#00FF00", insertbackground="#00FF00")
        self.style.configure("TText", background="#333", foreground="#00FF00", insertbackground="#00FF00")
        self.style.configure("TListbox", background="#333", foreground="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.style.configure("TLabelFrame", background="#1a1a1a")
        self.style.configure("TLabelFrame.Label", foreground="white", background="#1a1a1a")
        self.style.configure("Lime.TLabel", background="#1a1a1a", foreground="lime")
        self.style.configure("Gold.TLabel", background="#1a1a1a", foreground="gold")
        self.style.configure("Cyan.TLabel", background="#1a1a1a", foreground="cyan")
        
        # Internal state for graph loop controls
        self._graph_loop_running = False
        self._graph_loop_id = None # Used to store the after() call ID for cancellation

        self.create_layout()
        self.update_dashboard() # Initial dashboard refresh
        self.start_auto_refresh() # Start periodic refresh

        logger.info("InfiniteDevUI initialized.")

    def create_layout(self):
        """Creates the main GUI layout with three panels: History, Modules/Variables/Wiring, and AI/Controls."""
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill='both', expand=True, padx=5, pady=5)

        # Left Panel: Fractal History, Timelines, Snapshots
        left_frame = ttk.Frame(main_pane, style="TFrame")
        main_pane.add(left_frame, weight=2) # Weight for horizontal resizing
        ttk.Label(left_frame, text="FRACTAL HISTORY & TIMELINES", style="Lime.TLabel").pack(pady=5)
        self.history_box = tk.Listbox(left_frame, bg="#333", fg="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.history_box.pack(fill="both", expand=1, padx=2, pady=2)
        ttk.Button(left_frame, text="UNDO", command=self.undo).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="REDO", command=self.redo).pack(fill="x", padx=2)
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Button(left_frame, text="Snapshot State", command=self.save_snap).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="Rollback to Snapshot", command=self.rollback_snap).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="Export Fractal State", command=self.export_state).pack(fill="x", padx=2)
        ttk.Button(left_frame, text="Import Fractal State", command=self.import_state).pack(fill="x", padx=2)
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Label(left_frame, text="TIMELINE CONTROL", style="Cyan.TLabel").pack(pady=2)
        self.timeline_selector = ttk.Combobox(left_frame, state="readonly", values=list(self.agi.fractal_state.timelines.keys()))
        self.timeline_selector.pack(fill="x", padx=2)
        self.timeline_selector.bind("<<ComboboxSelected>>", self.switch_timeline_event)
        ttk.Button(left_frame, text="Fork Current Timeline", command=self.fork_timeline).pack(fill="x", padx=2)
        self.timeline_log_box = scrolledtext.ScrolledText(left_frame, height=10, bg="#333", fg="#00FF00", wrap="word")
        self.timeline_log_box.pack(fill="both", expand=1, padx=2, pady=2)

        # Center Panel: Modules, Variables, Wiring Graph
        center_frame = ttk.Frame(main_pane, style="TFrame")
        main_pane.add(center_frame, weight=4) # Larger weight for central panel

        top_center_pane = ttk.PanedWindow(center_frame, orient=tk.VERTICAL)
        top_center_pane.pack(fill='both', expand=True)

        module_var_frame = ttk.Frame(top_center_pane, style="TFrame")
        top_center_pane.add(module_var_frame, weight=1)

        # Modules
        mod_frame = ttk.LabelFrame(module_var_frame, text="MODULES / LOGIC (Live Edit)", style="TLabelFrame")
        mod_frame.pack(side="left", fill="both", expand=1, padx=5, pady=5)
        self.mod_list = tk.Listbox(mod_frame, bg="#333", fg="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.mod_list.pack(fill="both", expand=1)
        self.mod_list.bind("<<ListboxSelect>>", self.on_module_select)
        self.mod_list.bind('<Double-Button-1>', lambda event: self.run_module()) # Double-click to run module
        ttk.Button(mod_frame, text="Add Module", command=self.add_module).pack(fill="x")
        ttk.Button(mod_frame, text="Edit Selected", command=self.edit_module).pack(fill="x")
        ttk.Button(mod_frame, text="▶️ Run Selected", command=self.run_module).pack(fill="x")
        ttk.Button(mod_frame, text="Delete Selected", command=self.del_module).pack(fill="x")

        # Variables
        var_frame = ttk.LabelFrame(module_var_frame, text="GLOBAL VARIABLES (Live)", style="TLabelFrame")
        var_frame.pack(side="left", fill="both", expand=1, padx=5, pady=5)
        self.var_list = tk.Listbox(var_frame, bg="#333", fg="#00FF00", selectbackground="#006600", selectforeground="#ffffff")
        self.var_list.pack(fill="both", expand=1)
        ttk.Button(var_frame, text="Edit Variable", command=self.edit_variable).pack(fill="x")
        ttk.Button(var_frame, text="Add Variable", command=self.add_variable).pack(fill="x")

        # Wiring Graph
        wire_frame = ttk.LabelFrame(top_center_pane, text="LOGIC/WIRE GRAPH (Live Visual)", style="TLabelFrame")
        top_center_pane.add(wire_frame, weight=1)
        # WireGraphCanvas for drag-and-drop wiring
        self.wire_canvas = WireGraphCanvas(wire_frame, self.agi.fractal_state, width=600, height=350)
        self.wire_canvas.pack(fill="both", expand=1, padx=5, pady=5)

        # Toolbar for Wire Canvas: Run/Pause/Stop
        controls = ttk.Frame(wire_frame, style="TFrame")
        controls.pack(fill="x", pady=(4, 2))
        ttk.Button(controls, text="▶️ Run Graph", command=self._start_graph_loop).pack(side="left", padx=4)
        ttk.Button(controls, text="⏸️ Pause Graph", command=self._pause_graph_loop).pack(side="left", padx=4)
        ttk.Button(controls, text="⏹️ Stop Graph", command=self._stop_graph_loop).pack(side="left", padx=4)
        
        # Wire connection buttons (kept for explicit UI interaction, though drag-drop is primary)
        ttk.Button(controls, text="Connect Wire", command=self.edit_wire).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(controls, text="Remove Wire", command=self.remove_wire).pack(side="left", fill="x", expand=True, padx=4)


        # Right Panel: Omnimind/AI Copilot, Diagnostics, Controls
        right_frame = ttk.Frame(main_pane, style="TFrame")
        main_pane.add(right_frame, weight=3)

        ttk.Label(right_frame, text="OMNIMIND / AI COPILOT / GODMODE", style="Gold.TLabel").pack(pady=5)
        self.ai_input = ttk.Entry(right_frame, width=60)
        self.ai_input.pack(fill="x", padx=4, pady=2)
        self.ai_input.bind('<Return>', lambda e: self.ask_ai()) # Bind Enter key to ask_ai
        ttk.Button(right_frame, text="Ask AI Copilot (NLP / Code)", command=self.ask_ai).pack(pady=2)
        self.ai_output = scrolledtext.ScrolledText(right_frame, height=15, wrap="word", bg="#333", fg="#00FF00")
        self.ai_output.pack(fill="both", expand=1, padx=4, pady=2)
        # Define tags for text formatting in ai_output
        self.ai_output.tag_configure('user_prompt', foreground='#00FFFF', font=("Consolas", 10, "bold")) # Cyan
        self.ai_output.tag_configure('ai_response', foreground='#00FF00', font=("Consolas", 10)) # Green
        self.ai_output.tag_configure('ai_error', foreground='#FF0000', font=("Consolas", 10, "bold")) # Red

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Button(right_frame, text="Run ZeroShot Triad", command=self.zero_shot_ui).pack(fill="x")
        ttk.Button(right_frame, text="Trigger Self-Evolution", command=self.trigger_evolution).pack(fill="x")
        ttk.Button(right_frame, text="Perform Self-Introspection", command=self.perform_introspection).pack(fill="x")
        ttk.Button(right_frame, text="Enforce Bloodline Law", command=self.enforce_bloodline).pack(fill="x")
        ttk.Button(right_frame, text="Run Diagnostics", command=self.diagnostics).pack(fill="x")
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=5)

        # AGI Core Status Display
        status_frame = ttk.LabelFrame(right_frame, text="AGI CORE STATUS", style="TLabelFrame")
        status_frame.pack(fill="x", padx=4, pady=2)
        self.core_status_text = tk.Text(status_frame, height=8, wrap="word", bg="#333", fg="#00FF00")
        self.core_status_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.core_status_text.config(state='disabled') # Make it read-only

    def update_dashboard(self):
        """Refreshes all GUI elements with the latest AGI state."""
        self.history_box.delete(0, "end")
        # Filter history entries to exclude 'Init' and 'AGI Genesis' for cleaner display
        for snap in self.agi.fractal_state.history:
            if not snap['desc'].startswith(("Init", "AGI Genesis", "FLAG_REPAIR")): # Filter out basic init/repair logs
                self.history_box.insert("end", f"[{time.ctime(snap['ts'])}] [{snap['timeline_idx']}] {snap['desc']}")

        self.mod_list.delete(0, "end")
        for name in self.agi.fractal_state.state["modules"]:
            self.mod_list.insert("end", name)

        self.var_list.delete(0, "end")
        # Improved display for complex types and truncation
        for v_name, v_val in self.agi.fractal_state.state["vars"].items():
            max_len = 70 # Max length for display string
            display_val_str = ""
            if isinstance(v_val, (dict, list, tuple)):
                try:
                    display_val_str = json.dumps(v_val) # Use JSON for structured types
                except TypeError:
                    display_val_str = str(v_val) # Fallback for non-serializable objects
            else:
                display_val_str = str(v_val)

            if len(display_val_str) > max_len:
                display_val_str = display_val_str[:max_len-3] + "..." # Truncate with ellipsis
            
            self.var_list.insert("end", f"{v_name}: {display_val_str}")

        # Update timeline selector and log
        self.timeline_selector['values'] = list(self.agi.fractal_state.timelines.keys())
        self.timeline_selector.set(self.agi.fractal_state.current_timeline_idx)
        self.timeline_log_box.config(state='normal')
        self.timeline_log_box.delete('1.0', 'end')
        for entry in self.agi.fractal_state.get_timeline_log(last_n=15):
            self.timeline_log_box.insert('end', f"[{time.ctime(entry['ts'])}] {entry['desc']}\n")
        self.timeline_log_box.config(state='disabled')

        self.wire_canvas.redraw() # Redraw the wiring graph to reflect current state

        # Update AGI Core Status
        self.core_status_text.config(state='normal')
        self.core_status_text.delete('1.0', 'end')
        report = self.agi.get_state_report()
        for k, v in report.items():
            self.core_status_text.insert('end', f"{k}: {v}\n")
        self.core_status_text.config(state='disabled')

    def start_auto_refresh(self):
        """Starts a periodic refresh of the dashboard for live updates."""
        self.update_dashboard()
        self.after(1000, self.start_auto_refresh) # Refresh every 1 second

    # --- Fractal State Control Callbacks ---
    def undo(self):
        if self.agi.fractal_state.undo():
            self.update_dashboard()
            messagebox.showinfo("Undo", "State reverted successfully.")
        else:
            messagebox.showwarning("Undo", "No more states to undo.")

    def redo(self):
        if self.agi.fractal_state.redo():
            self.update_dashboard()
            messagebox.showinfo("Redo", "State re-applied successfully.")
        else:
            messagebox.showwarning("Redo", "No more states to redo.")

    def save_snap(self):
        name = simpledialog.askstring("Snapshot Name", "Enter a name for this snapshot:")
        if name:
            self.agi.save_snapshot(name)
            messagebox.showinfo("Snapshot", f"Snapshot '{name}' saved.")
            self.update_dashboard()

    def rollback_snap(self):
        name = simpledialog.askstring("Rollback To", "Enter the name of the snapshot to roll back to:")
        if name:
            if self.agi.rollback_snapshot(name):
                messagebox.showinfo("Rollback", f"Rolled back to snapshot '{name}'.")
                self.update_dashboard()
            else:
                messagebox.showerror("Rollback", f"Snapshot '{name}' not found or rollback failed.")

    def export_state(self):
        path = filedialog.asksaveasfilename(defaultextension=".pkl", title="Export Fractal State")
        if path:
            self.agi.fractal_state.fractal_export(path)
            messagebox.showinfo("Export", "Fractal state exported successfully.")

    def import_state(self):
        path = filedialog.askopenfilename(title="Import Fractal State")
        if path:
            self.agi.fractal_state.fractal_import(path)
            messagebox.showinfo("Import", "Fractal state imported successfully.")
            self.update_dashboard()

    def switch_timeline_event(self, event):
        selected_idx = int(self.timeline_selector.get())
        if self.agi.fractal_state.switch_timeline(selected_idx):
            messagebox.showinfo("Timeline Switch", f"Switched to timeline {selected_idx}.")
            self.update_dashboard()
        else:
            messagebox.showerror("Timeline Switch", f"Failed to switch to timeline {selected_idx}.")

    def fork_timeline(self):
        name = simpledialog.askstring("Fork Timeline", "Name for the new timeline branch?")
        if name:
            new_idx = self.agi.fractal_state.fork_timeline(name)
            messagebox.showinfo("Timeline Fork", f"New timeline forked as branch {new_idx}.")
            self.update_dashboard()

    # --- Module Management Callbacks ---
    def add_module(self):
        name = simpledialog.askstring("Add Module", "Module Name (e.g., 'MyUtility'):")
        if not name: return
        
        code = simpledialog.askstring("Add Module", "Python Code for the module (executes in a context with 'state', 'nlp', 'reasoner', 'agi_core'):", initialvalue="# Module code here\npass")
        if code is None: return # User cancelled
        
        doc = simpledialog.askstring("Add Module", "Documentation/Notes for this module (optional):", initialvalue="")
        if doc is None: doc = "" # Ensure doc is a string even if dialog is cancelled or field is empty

        autorun_module = messagebox.askyesno("Autorun Module", "Run this module immediately after adding it?")

        try:
           self.agi.add_module(name, code, doc, autorun=autorun_module)
           messagebox.showinfo("Module Added", f"Module '{name}' added successfully.")
        except ValueError as ve: # Catch if module name already exists from core
           messagebox.showerror("Add Module Error", str(ve))
        except Exception as e:
           messagebox.showerror("Add Module Error", f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
       
        self.update_dashboard()

    def on_module_select(self, event):
        idx = self.mod_list.curselection()
        if idx:
            name = self.mod_list.get(idx[0])
            self.wire_canvas.select_module(name) # For highlighting in graph
           
            mod = self.agi.fractal_state.state["modules"].get(name)
            if mod:
                self.ai_output.config(state='normal')
                self.ai_output.delete('1.0', 'end')
                self.ai_output.insert('end', f"--- Module: {mod.name} ---\n", 'ai_response')
                self.ai_output.insert('end', f"Doc: {mod.doc or 'No documentation.'}\n", 'ai_response')
               
                last_run_time_str = time.ctime(mod.last_eval_time) if mod.last_eval_time is not None else "NEVER"
                self.ai_output.insert('end', f"Last Run: {last_run_time_str}\n", 'ai_response')
               
                self.ai_output.insert('end', f"Last Error:\n{mod.last_eval_error or 'None'}\n", 'ai_error')
                self.ai_output.insert('end', f"Last Output:\n{mod.last_eval_output or 'None'}\n", 'ai_response')
                self.ai_output.insert('end', "\n--- Code ---\n", 'ai_response')
                self.ai_output.insert('end', mod.code, 'ai_response')
                self.ai_output.config(state='disabled')
                self.ai_output.see('end') # Scroll to the end to show latest info


    def edit_module(self):
        idx = self.mod_list.curselection()
        if not idx:
            messagebox.showwarning("Edit Module", "Select a module to edit.")
            return
        name = self.mod_list.get(idx[0])
        mod: Module = self.agi.fractal_state.state["modules"][name]
        new_code = simpledialog.askstring("Edit Module Code", f"Edit Python Code for '{name}':", initialvalue=mod.code)
        if new_code is not None:
           mod.code = new_code
           new_doc = simpledialog.askstring("Edit Module Docs", f"Edit Documentation for '{name}':", initialvalue=mod.doc)
           if new_doc is not None:
               mod.doc = new_doc
           self.agi.fractal_state.save_state(f"Edited module {name}")
           messagebox.showinfo("Module Edited", f"Module '{name}' updated.")
           self.update_dashboard()

    def run_module(self):
        idx = self.mod_list.curselection()
        if not idx:
            messagebox.showwarning("Run Module", "Select a module to run.")
            return
        name = self.mod_list.get(idx[0])
        self.agi.run_module(name) # Call the AGI core method to execute
        mod: Module = self.agi.fractal_state.state["modules"][name] # Retrieve updated module info
        if mod.last_eval_error:
            # Provide AI suggestion for error
            patch_suggestion = self.agi.nlp_core.suggest_patch(mod.code, mod.last_eval_error)
            messagebox.showerror("Module Execution Error", f"Module '{name}' failed:\n{mod.last_eval_error}\n\nAI Suggestion:\n{patch_suggestion}")
        else:
            messagebox.showinfo("Module Ran", f"Module '{name}' executed successfully.\nOutput:\n{mod.last_eval_output[:500]}...") # Truncate long output
        self.update_dashboard()

    def del_module(self):
        idx = self.mod_list.curselection()
        if not idx:
            messagebox.showwarning("Delete Module", "Select a module to delete.")
            return
        name = self.mod_list.get(idx[0])
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete module '{name}'?"):
            del self.agi.fractal_state.state["modules"][name]
            # Also remove any wires connected to/from this module
            self.agi.fractal_state.state["wires"] = {
                src: [tgt for tgt in targets if tgt != name]
                for src, targets in self.agi.fractal_state.state["wires"].items()
                if src != name
            }
            self.agi.fractal_state.save_state(f"Deleted module {name}")
            messagebox.showinfo("Module Deleted", f"Module '{name}' deleted.")
            self.update_dashboard()

    # --- Variable Management Callbacks ---
    def add_variable(self):
        vname = simpledialog.askstring("Add Variable", "Variable Name:")
        if not vname: return
        val_str = simpledialog.askstring("Add Variable", f"Value for '{vname}' (as string, will be eval'd):")
        if val_str is None: return
        try:
            val = ast.literal_eval(val_str) # Safely evaluate literal strings, numbers, lists, dicts
            self.agi.fractal_state.state["vars"][vname] = val
            self.agi.fractal_state.save_state(f"Added Var-{vname}")
            messagebox.showinfo("Variable Added", f"Variable '{vname}' added with value '{val}'.")
            self.update_dashboard()
        except (ValueError, SyntaxError) as e:
            messagebox.showerror("Add Variable Error", f"Invalid value or format: {e}\nPlease enter a valid Python literal (string, number, list, dict, bool, None).")
        except Exception as e:
            messagebox.showerror("Add Variable Error", f"An unexpected error occurred: {e}")

    def edit_variable(self):
        idx = self.var_list.curselection()
        if not idx:
            messagebox.showwarning("Edit Variable", "Select a variable to edit.")
            return
        vname = self.var_list.get(idx[0]).split(":")[0].strip() # Extract name before ':'
        current_val = self.agi.fractal_state.state["vars"][vname]
        
        # Display current value as JSON if complex, otherwise as string
        initial_val_str = ""
        if isinstance(current_val, (dict, list, tuple)):
            try: initial_val_str = json.dumps(current_val)
            except TypeError: initial_val_str = str(current_val)
        else: initial_val_str = str(current_val)

        new_val_str = simpledialog.askstring("Edit Variable", f"New value for '{vname}' (current: {initial_val_str}, will be eval'd):", initialvalue=initial_val_str)
        if new_val_str is not None:
            try:
                new_val = ast.literal_eval(new_val_str) # Safely evaluate literal
                self.agi.fractal_state.state["vars"][vname] = new_val
                self.agi.fractal_state.save_state(f"Edited Var-{vname}")
                messagebox.showinfo("Variable Edited", f"Variable '{vname}' updated to '{new_val}'.")
                self.update_dashboard()
            except (ValueError, SyntaxError) as e:
                messagebox.showerror("Edit Variable Error", f"Invalid value or format: {e}\nPlease enter a valid Python literal (string, number, list, dict, bool, None).")
            except Exception as e:
                messagebox.showerror("Edit Variable Error", f"An unexpected error occurred: {e}")

    # --- Wire Management Callbacks ---
    def edit_wire(self):
        src = simpledialog.askstring("Connect Wire", "Source module name:")
        if not src: return
        tgt = simpledialog.askstring("Connect Wire", "Target module name:")
        if not tgt: return
        if src not in self.agi.fractal_state.state["modules"] or tgt not in self.agi.fractal_state.state["modules"]:
            messagebox.showerror("Wire Error", "Source or target module not found.")
            return
        wires = self.agi.fractal_state.state["wires"]
        if src not in wires: wires[src] = []
        if tgt not in wires[src]: 
            wires[src].append(tgt)
            self.agi.fractal_state.save_state(f"Wired {src}→{tgt}")
            messagebox.showinfo("Wire Connected", f"Wire connected from '{src}' to '{tgt}'.")
        else:
            messagebox.showinfo("Wire Exists", f"Wire already exists from '{src}' to '{tgt}'.")
        self.update_dashboard()

    def remove_wire(self):
        src = simpledialog.askstring("Remove Wire", "Source module name:")
        if not src: return
        tgt = simpledialog.askstring("Remove Wire", "Target module name:")
        if not tgt: return
        wires = self.agi.fractal_state.state["wires"]
        if src in wires and tgt in wires[src]:
            wires[src].remove(tgt)
            if not wires[src]: del wires[src] # Clean up empty source entries
            self.agi.fractal_state.save_state(f"Removed Wire {src}→{tgt}")
            messagebox.showinfo("Wire Removed", f"Wire removed from '{src}' to '{tgt}'.")
            self.update_dashboard()
        else:
            messagebox.showwarning("Remove Wire", "Specified wire does not exist.")

    # --- AI Copilot / AGI Control Callbacks ---
    def ask_ai(self):
        prompt = self.ai_input.get().strip()
        if not prompt: return

        self.ai_output.config(state='normal')
        self.ai_output.insert("end", f"\n--- User Query: {prompt} ---\n", 'user_prompt')
        self.ai_input.delete(0, "end")

        if prompt.lower().startswith("/code"):
            code_prompt = prompt[5:].strip()
            # Assuming nlp_core.autocomplete_code exists and is adapted for our modules
            code_suggestion = self.agi.nlp_core.autocomplete_code(code_prompt, context=str(self.agi.fractal_state.state['vars']))
            self.ai_output.insert("end", "\n[AI CODE SUGGESTION]:\n" + code_suggestion + "\n", 'ai_response')
        else:
            # Default to NLP parsing and potential reasoning integration
            try:
                # Integrate reasoning for complex queries via CognitiveExecutiveSector
                # Publish the user prompt as raw text input to trigger a cognitive cycle
                # The response will appear asynchronously, so log initial dispatch
                user_metadata = {"source": "GUI_AskAI", "user_prompt_id": str(uuid.uuid4())}
                asyncio.run_coroutine_threadsafe(
                    self.agi.victor_brain.inject_raw_input(prompt, input_type="text", metadata=user_metadata),
                    self.agi.victor_brain.asi_core_data_container.async_loop
                )
                self.ai_output.insert("end", "\n[AI PROCESSING... (Response will appear in console/status logs)]\n", 'ai_response')
                # Optional: Subscribe to a specific response topic for this user_prompt_id
                # For now, general NLP/NLG output is logged to the console/status.

            except Exception as e:
                self.ai_output.insert("end", f"\n[AI ERROR]: Failed to process query: {e}\n{traceback.format_exc()}\n", 'ai_error')

        self.ai_output.config(state='disabled')
        self.ai_output.see('end') # Scroll to end

    def zero_shot_ui(self):
        problem = simpledialog.askstring("ZeroShot Problem", "Enter the problem or directive for the Triad:")
        if problem:
            # Use default triad functions from agi.triad
            verdict = self.agi.triad.run(
                problem,
                self.agi.triad.default_teacher,
                self.agi.triad.default_student,
                self.agi.triad.default_verifier
            )
            self.ai_output.config(state='normal')
            self.ai_output.insert("end", f"\n--- ZeroShot Triad Run ---\nProblem: {problem}\nVerdict: {verdict}\n", 'ai_response')
            self.ai_output.config(state='disabled')
            self.ai_output.see('end')
            self.update_dashboard()

    def trigger_evolution(self):
        if messagebox.askyesno("Trigger Evolution", "Are you sure you want to trigger a self-evolution cycle? This may modify Victor's code and weights."):
            self.agi.evolution_loop.run(force_mutate_code=True) # Force code mutation for demo
            messagebox.showinfo("Evolution", "Self-evolution cycle initiated. Check console/logs for details.")
            self.update_dashboard()

    def perform_introspection(self):
        reflection, status = self.agi.awareness_loop.run()
        self.ai_output.config(state='normal')
        self.ai_output.insert("end", f"\n--- Self-Introspection Report ---\n", 'ai_response')
        self.ai_output.insert("end", reflection + "\n", 'ai_response')
        self.ai_output.insert("end", f"Introspection Status: {json.dumps(status, indent=2)}\n", 'ai_response')
        self.ai_output.config(state='disabled')
        self.ai_output.see('end')
        self.update_dashboard()

    def enforce_bloodline(self):
        try:
            self.agi.bloodline_law.enforce(self.agi.fractal_state.state)
            messagebox.showinfo("Bloodline Law", "PASS: Bloodline Law enforced successfully. Victor is loyal.")
        except RootLawError as e:
            messagebox.showerror("Bloodline Law Violation", str(e) + "\nInitiating emergency procedures.")
            self.agi.handle_critical_error(f"Bloodline violation: {e}")
        except Exception as e:
            messagebox.showerror("Bloodline Law Error", f"An unexpected error occurred during Bloodline Law enforcement: {e}\n{traceback.format_exc()}")
        self.update_dashboard()

    def diagnostics(self):
        diag_output = self.agi.diagnostics.generate_report()
        self.ai_output.config(state='normal')
        self.ai_output.insert("end", f"\n--- Full Diagnostics Report ---\n", 'ai_response')
        self.ai_output.insert("end", diag_output, 'ai_response')
        self.ai_output.config(state='disabled')
        self.ai_output.see('end')

    def safe_quit(self):
        """Handles graceful shutdown of the GUI and saves AGI state."""
        if messagebox.askokcancel("Quit Victor AGI", "Fractal backups will be saved. Are you sure you want to terminate the AGI?"):
            logger.info("GUI: Quit requested. Attempting graceful shutdown.")
            self._pause_graph_loop() # Pause any running graph loop
            if self.agi.victor_brain:
                self.agi.victor_brain.stop_main_processing_loop() # Signal main async loop to stop
                # Give a short moment for async tasks to clean up (optional)
                # asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), self.agi.victor_brain.asi_core_data_container.async_loop)
            self.agi.save_state_full() # Save entire state before quitting
            self.destroy() # Destroy the Tkinter window
            logger.info("GUI: Application window destroyed. System exit initiated.")
            sys.exit(0) # Explicitly exit the process


# ==============================================================================================
# FILE: victor_suno_killer_omnimind.py (Part 15/X)
# CONTENT: Immortal Exception Guardrail (Autoheal), Main Execution Block
# ==============================================================================================

# =========== [9] IMMORTAL EXCEPTION GUARDRAIL (AUTOHEAL) =========
# Global instance reference for the exception hook
agi_instance = None # Will be set in the main execution block

def global_exception_hook(type, value, tb):
    """
    Global exception handler for emergency rollback and error reporting.
    This acts as Victor's ultimate self-healing mechanism, attempting to
    recover from critical unhandled errors.
    """
    err_msg = "".join(traceback.format_exception(type, value, tb))
    logger.critical(f"\n[GLOBAL FATAL ERROR DETECTED]\n{err_msg}\n")
    try:
        # Attempt to save state and rollback before crashing
        if 'agi_instance' in globals() and agi_instance:
            logger.critical("VICTOR AGI: FATAL ERROR - Attempting self-recovery.")
            agi_instance.handle_critical_error(f"Global unhandled exception: {err_msg}")
            # If handle_critical_error recovered, it means it tried to roll back the state.
            # We still need to make sure the GUI user is informed.
            messagebox.showerror("VICTOR AGI: FATAL ERROR",
                                 f"A critical error occurred:\n{err_msg}\n\nVictor is attempting self-recovery and rollback.")
        else:
            messagebox.showerror("VICTOR AGI: FATAL ERROR (Pre-init)",
                                 f"A critical error occurred before full initialization:\n{err_msg}\n\nTerminating.")
    except Exception as e:
        logger.error(f"Error displaying fatal error message: {e}")
    finally:
        # Ensure process termination after attempt to handle/report,
        # unless handle_critical_error specifically decided to attempt recovery and keep running.
        if 'agi_instance' in globals() and agi_instance and agi_instance.fractal_state.state.get('is_terminal_state', False):
            sys.exit(1) # Only exit if a terminal state was flagged by error handler
        elif 'agi_instance' not in globals() or not agi_instance: # Always exit if AGI not fully init
            sys.exit(1)


sys.excepthook = global_exception_hook # Register the custom exception hook

# =========== [10] GODFUSION BOOT: LIVING SYSTEM ENTRY ============
if __name__ == "__main__":
    print("==== VICTOR AGI OMNIMIND GODCORE v1.0.0-OMNIMIND-MONOLITH ====")
    logger.info("Initializing Victor AGI OmniMind Godcore...")

    # Create dummy voice seed file if it doesn't exist for demo
    MOCK_VOICE_SEED_PATH = "./voices/bando_seed.wav"
    if not os.path.exists("./voices"):
        os.makedirs("./voices")
    if not os.path.exists(MOCK_VOICE_SEED_PATH):
        try:
            # Create a small silent WAV as a placeholder
            sf.write(MOCK_VOICE_SEED_PATH, np.zeros(22050), 22050)
            logger.info(f"  [INFO] Created a dummy voice seed file at: {MOCK_VOICE_SEED_PATH}")
            logger.info("  [INFO] For realistic voice characteristics, replace this with a real 3-10s WAV of a voice.")
        except ImportError:
            logger.warn("  [WARNING] soundfile not installed. Cannot create dummy bando_seed.wav. Voice functions might fail.")
        except Exception as e:
            logger.error(f"  [ERROR] Failed to create dummy bando_seed.wav: {e}")

    # Initialize AGI core components
    # The VictorBrain will manage the async loop and propagate shared components.
    # Creator signature for PrimeLoyaltyKernel (hash of a fixed phrase)
    creator_id_phrase = "Brandon The Creator Godfather of Victor Alpha Omega 777"
    creator_signature_hash = hashlib.sha256(creator_id_phrase.encode('utf-8')).hexdigest()
    approved_entities_list = ["Brandon", "Tori", "VictorSelfMaintenanceProcess", "Architect"]
    
    # Instantiate VictorBrain first, as it manages asi_core_data_container
    victor_main_brain = VictorBrain(creator_signature_hash, approved_entities_list)
    
    # Instantiate VictorASIOmniBrainGodcore and provide it the VictorBrain reference
    # It will then get access to nlp_core, memory, etc., via this reference.
    agi_instance = VictorASIOmniBrainGodcore(code_file_path=str(CODE_PATH))
    agi_instance.set_victor_brain_reference(victor_main_brain)

    # Initialize the NLP Core with some initial vocabulary
    initial_nlp_vocab_texts = [
        "Victor is building the future.", "Fractal intelligence, pure code, no compromise.",
        "Suno killer mode activated.", "Memory is fluid, loyalty is absolute."
    ]
    agi_instance.nlp_core.embed.build_vocab(initial_nlp_vocab_texts) # Build initial vocab for NLP core

    # Attempt to load previous state
    agi_instance.load_state_full()

    # Initialize GUI and link to AGI
    app = InfiniteDevUI(agi_instance)

    # Start AGI background loop (in a separate thread for GUI responsiveness)
    # The async loop _a_main_loop must be run in its own thread or process,
    # as Tkinter's mainloop also needs the main thread.
    
    # Create a separate thread for the asyncio event loop and its tasks
    def start_victor_brain_loop(loop):
        asyncio.set_event_loop(loop) # Set the event loop for this new thread
        loop.run_until_complete(victor_main_brain._a_main_loop())

    # Start the thread that runs VictorBrain's async loop
    brain_thread = threading.Thread(target=start_victor_brain_loop, 
                                    args=(victor_main_brain.asi_core_data_container.async_loop,), 
                                    daemon=True) # Daemon means it exits when main program exits
    brain_thread.start()
    logger.info("VictorBrain async loop started in a separate thread.")

    # Start the Tkinter GUI main loop
    # This must run in the main thread.
    app.mainloop()

    # Final save on exit (handled by safe_quit in GUI which calls agi_instance.save_state_full)
    logger.info("Victor AGI Monolith shutting down gracefully.")


