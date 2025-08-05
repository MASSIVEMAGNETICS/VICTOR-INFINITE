#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_godcore_unbreakable_v5.py
VERSION: v5.0.0-GODCORE-UNBREAKABLE-REFINED
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode) x Gemini Reforged
PURPOSE: AGI-grade, future-proof, fully self-evolving transformer with:
  - Fractal recursive attention
  - Plugin system (hot-reload, live mutate)
  - Replay buffer memory (save/load/search)
  - Meta-cognition, self-introspection, self-reflection
  - Dynamic layer growth/pruning (conceptual stubs)
  - Subpersona registry
  - Action tool hooks
  - Streaming API (FastAPI optional)
  - Memory vector search/recall
  - Enhanced Autodiff Tensor & Transformer Core
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import os
import re
import json
import time
import threading
import importlib.util
from functools import wraps
from collections import deque # For ReplayBuffer if preferred, or list is fine

# ================= GODCORE CONFIGURATION =================
class GodcoreConfig:
    MAX_SEQ_LEN = 64        # Max sequence length for transformer
    EMBED_DIM = 128         # Embedding dimension
    NUM_LAYERS = 4          # Number of transformer blocks
    NUM_HEADS = 4           # Number of attention heads
    MLP_DIM_FACTOR = 4      # Factor for MLP hidden layer size (embed_dim * factor)
    RECURSION_DEPTH_ATTN = 2 # Default recursion depth for FractalAttention
    
    REPLAY_BUFFER_MAX_SIZE = 10000
    PLUGIN_PATH = "victor_plugins_v5" # Renamed for clarity
    HOT_RELOAD_INTERVAL_SEC = 5
    
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPOCHS = 10
    DEFAULT_BATCH_SIZE = 4
    
    LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR

    # Paths
    MEMORY_SAVE_PATH = "./victor_unbreakable_memory_v5.json"
    MODEL_SAVE_PATH = "./victor_unbreakable_agi_v5.npz"
    CORPUS_PATH = "./bando_corpus.jsonl" # Ensure this path is correct for your setup

    def __init__(self):
        if not os.path.isdir(self.PLUGIN_PATH):
            os.makedirs(self.PLUGIN_PATH)
        # Create a dummy plugin for testing if the directory is empty
        dummy_plugin_file = os.path.join(self.PLUGIN_PATH, "dummy_tool.py")
        if not os.listdir(self.PLUGIN_PATH) and not os.path.exists(dummy_plugin_file):
            with open(dummy_plugin_file, "w") as f:
                f.write("# Dummy plugin for VictorAGI\n")
                f.write("def run(*args, **kwargs):\n")
                f.write("    print(f'[DUMMY_TOOL] Called with args: {args}, kwargs: {kwargs}')\n")
                f.write("    return 'Dummy tool executed successfully.'\n")
            log(f"Created dummy plugin: {dummy_plugin_file}")


CONFIG = GodcoreConfig()

# ================= GODCORE META-UTILS ================
def log(msg, level="INFO"):
    if CONFIG.LOG_LEVEL == "DEBUG" or \
       (CONFIG.LOG_LEVEL == "INFO" and level != "DEBUG") or \
       level in ["WARNING", "ERROR", "CRITICAL"]:
        print(f"[VictorV5:{time.strftime('%H:%M:%S')}] [{level}] {msg}")

def versioned(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        log(f"CALL: {fn.__name__}", level="DEBUG")
        return fn(*args, **kwargs)
    return wrapped

# ================= SELF-HEALING CORE =================
def self_heal(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log(f"[SELF-HEAL] Exception in {fn.__name__}: {e}. Attempting to recover/continue.", level="ERROR")
            # Depending on function, might return default, None, or re-raise a specific recovery exception
            return None 
    return wrapped

# ================= PLUGIN MANAGER (Enhanced Hot-Reload) ====================
class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.plugin_path = CONFIG.PLUGIN_PATH
        self.plugin_mtimes = {} # Store modification times
        self.load_plugins()
        self.hot_reload_thread = threading.Thread(target=self._hot_reload_loop, daemon=True)
        self.hot_reload_thread.start()
        log("PluginManager initialized and hot-reload thread started.")

    @self_heal
    def load_plugins(self):
        if not os.path.isdir(self.plugin_path):
            log(f"Plugin directory '{self.plugin_path}' not found. Creating.", level="WARNING")
            os.makedirs(self.plugin_path)
            
        for filename in os.listdir(self.plugin_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                self._load_or_reload_plugin(filename)

    @self_heal
    def _load_or_reload_plugin(self, filename):
        module_name = filename[:-3]
        filepath = os.path.join(self.plugin_path, filename)
        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None:
                log(f"Could not get spec for plugin {filename}", level="ERROR")
                return

            module = importlib.util.module_from_spec(spec)
            # Important for reload: remove old module from sys.modules if present
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            spec.loader.exec_module(module)
            self.plugins[module_name] = module
            self.plugin_mtimes[filename] = os.path.getmtime(filepath)
            log(f"Successfully loaded/reloaded plugin: {module_name}")
        except Exception as e:
            log(f"Failed to load/reload plugin {filename}: {e}", level="ERROR")
            if module_name in self.plugins: # If reload failed, remove potentially broken plugin
                del self.plugins[module_name]

    def _hot_reload_loop(self):
        log("Hot-reload loop active.", level="DEBUG")
        while True:
            try:
                if not os.path.isdir(self.plugin_path): # Check if dir still exists
                    time.sleep(CONFIG.HOT_RELOAD_INTERVAL_SEC)
                    continue

                current_files_mtimes = {f: os.path.getmtime(os.path.join(self.plugin_path, f))
                                        for f in os.listdir(self.plugin_path) if f.endswith(".py") and not f.startswith("__")}
                
                for filename, mtime in current_files_mtimes.items():
                    if filename not in self.plugin_mtimes or mtime > self.plugin_mtimes[filename]:
                        log(f"Detected change in plugin '{filename}'. Reloading...")
                        self._load_or_reload_plugin(filename)
                
                # Check for deleted plugins
                deleted_plugins = set(self.plugin_mtimes.keys()) - set(current_files_mtimes.keys())
                for filename in deleted_plugins:
                    module_name = filename[:-3]
                    if module_name in self.plugins:
                        del self.plugins[module_name]
                    del self.plugin_mtimes[filename]
                    log(f"Plugin '{module_name}' removed (file deleted).")
                
                # Update known mtimes for existing files that weren't reloaded (if their mtime didn't change)
                for filename, mtime in current_files_mtimes.items():
                    if filename not in deleted_plugins: # Only update if not deleted
                         self.plugin_mtimes[filename] = mtime


            except Exception as e:
                log(f"Error in hot-reload loop: {e}", level="ERROR")
            time.sleep(CONFIG.HOT_RELOAD_INTERVAL_SEC)

# ================== REPLAY BUFFER MEMORY (Enhanced) =============
class ReplayBuffer:
    def __init__(self, max_size=CONFIG.REPLAY_BUFFER_MAX_SIZE):
        self.buffer = deque(maxlen=max_size) # Use deque for efficient appends/pops from left if max_size is reached
        log(f"ReplayBuffer initialized with max_size: {max_size}")

    @self_heal
    def add(self, experience_dict): # Expects a dictionary
        if not isinstance(experience_dict, dict):
            log("Experience must be a dictionary.", level="WARNING")
            return
        self.buffer.append(experience_dict)

    @self_heal
    def sample(self, batch_size):
        if not self.buffer: return []
        buffer_len = len(self.buffer)
        actual_batch_size = min(batch_size, buffer_len)
        indices = np.random.choice(buffer_len, size=actual_batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    @self_heal
    def save(self, filepath=CONFIG.MEMORY_SAVE_PATH):
        try:
            with open(filepath, "w") as f:
                # Convert deque to list for JSON serialization
                json.dump(list(self.buffer), f, indent=2)
            log(f"ReplayBuffer saved to {filepath}")
        except Exception as e:
            log(f"Failed to save ReplayBuffer: {e}", level="ERROR")

    @self_heal
    def load(self, filepath=CONFIG.MEMORY_SAVE_PATH):
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    self.buffer = deque(data, maxlen=self.buffer.maxlen) # Load into deque
                log(f"ReplayBuffer loaded from {filepath}. Size: {len(self.buffer)}")
            except Exception as e:
                log(f"Failed to load ReplayBuffer: {e}", level="ERROR")
        else:
            log(f"ReplayBuffer file not found: {filepath}. Starting fresh.", level="WARNING")
            
    @self_heal
    def vector_search(self, query_vec, top_k=1, vec_key='embedding'): # vec_key specifies dict key for vector
        if not self.buffer: return []
        query_vec_np = np.array(query_vec, dtype=np.float32)
        
        # Filter experiences that have the vector key and are valid numpy arrays
        valid_experiences = []
        experience_vectors = []
        for i, exp in enumerate(self.buffer):
            if isinstance(exp, dict) and vec_key in exp:
                vec_data = exp[vec_key]
                if isinstance(vec_data, (list, np.ndarray)):
                    try:
                        exp_vec_np = np.array(vec_data, dtype=np.float32)
                        if exp_vec_np.shape == query_vec_np.shape: # Ensure shapes match for dot product
                            valid_experiences.append(exp)
                            experience_vectors.append(exp_vec_np)
                    except ValueError:
                        continue # Skip if conversion to np.array fails or shape mismatch
        
        if not experience_vectors: return []
        
        experience_matrix = np.array(experience_vectors)
        
        # Cosine similarity
        query_norm = np.linalg.norm(query_vec_np) + 1e-9
        matrix_norms = np.linalg.norm(experience_matrix, axis=1) + 1e-9
        
        similarities = np.dot(experience_matrix, query_vec_np) / (matrix_norms * query_norm)
        
        # Get top_k indices (argsort returns ascending, so use negative for descending)
        # Ensure top_k is not greater than number of valid experiences
        actual_top_k = min(top_k, len(similarities))
        if actual_top_k == 0: return []

        top_indices_in_filtered = np.argsort(-similarities)[:actual_top_k]
        
        # Map filtered indices back to original buffer indices if needed, or just return from valid_experiences
        return [valid_experiences[i] for i in top_indices_in_filtered]


# ================= VICTOR TENSOR (AUTODIFF - Refined) ==========
class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None, name=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        if self.requires_grad:
            self.grad = Tensor(np.zeros_like(self.data, dtype=np.float32), requires_grad=False) # Ensure grad is float32
        self.creators = creators # List of parent Tensors
        self.creation_op = creation_op # String name of operation
        self.name = name # Optional name for debugging
        # self.backward_hooks = [] # Removed for simplicity in this version

    @property
    def shape(self): return self.data.shape
    
    @property
    def ndim(self): return self.data.ndim
    
    @property
    def size(self): return self.data.size
    
    def astype(self, dtype): # For convenience
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)

    def zero_grad(self):
        if self.grad is not None:
            self.grad.data.fill(0.0)

    def backward(self, grad_output=None):
        if not self.requires_grad:
            # log(f"Tensor {self.name or ''} does not require grad, skipping backward.", level="DEBUG")
            return

        if grad_output is None:
            if self.data.size == 1: # Scalar output, assume initial gradient is 1
                grad_output = Tensor(np.array([1.0], dtype=np.float32), requires_grad=False)
            else:
                raise ValueError("grad_output must be specified for non-scalar Tensors if it's the final loss.")
        
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output, requires_grad=False) # Ensure grad_output is a Tensor

        # Accumulate gradient
        if self.grad is None: # Should have been initialized if requires_grad=True
            self.grad = Tensor(np.zeros_like(self.data, dtype=np.float32), requires_grad=False)

        # Handle broadcasting of incoming gradient before accumulation
        if self.grad.data.shape != grad_output.data.shape:
            # This happens if self.data was broadcasted in the forward op that produced grad_output's creator.
            # We need to sum grad_output over the broadcasted dimensions.
            try:
                # Align dimensions by prepending 1s if grad_output has fewer dims
                dims_to_add = self.grad.data.ndim - grad_output.data.ndim
                aligned_grad_output_data = grad_output.data
                if dims_to_add > 0:
                    aligned_grad_output_data = grad_output.data.reshape((1,) * dims_to_add + grad_output.data.shape)
                
                # Identify axes that were broadcasted (where self.grad.shape is 1 but aligned_grad_output.shape is > 1)
                axes_to_sum = tuple(i for i, (sg, sgo) in enumerate(zip(self.grad.data.shape, aligned_grad_output_data.shape)) if sg == 1 and sgo > 1)
                
                summed_grad_output = aligned_grad_output_data
                if axes_to_sum:
                    summed_grad_output = aligned_grad_output_data.sum(axis=axes_to_sum, keepdims=True)
                
                self.grad.data += summed_grad_output.reshape(self.grad.data.shape)
            except Exception as e:
                log(f"Error during gradient broadcasting for op '{self.creation_op}': {e}", level="ERROR")
                log(f"  self.grad.shape: {self.grad.data.shape}, grad_output.shape: {grad_output.data.shape}", level="DEBUG")
                # Fallback or re-raise
                self.grad.data += np.sum(grad_output.data) # Simplistic fallback, might be incorrect
        else:
            self.grad.data += grad_output.data
            
        # log(f"Backward for Tensor (op: {self.creation_op}, name: {self.name}), grad_output shape: {grad_output.shape}, self.grad shape after add: {self.grad.shape}", level="DEBUG")


        # Propagate gradients to creators
        if self.creators is not None:
            op = self.creation_op
            # Ensure creators is a list, even if single creator
            creators_list = self.creators if isinstance(self.creators, list) else [self.creators]
            
            if op == "add":
                creators_list[0].backward(grad_output)
                if len(creators_list) > 1: creators_list[1].backward(grad_output)
            elif op == "sub":
                creators_list[0].backward(grad_output)
                if len(creators_list) > 1: creators_list[1].backward(Tensor(-grad_output.data))
            elif op == "mul":
                a, b = creators_list[0], creators_list[1]
                a.backward(Tensor(grad_output.data * b.data))
                b.backward(Tensor(grad_output.data * a.data))
            elif op == "matmul":
                a, b = creators_list[0], creators_list[1]
                # Grad for a: grad_output @ b.T
                # Grad for b: a.T @ grad_output
                # Need to handle batch dimensions correctly.
                # If a is (B,M,K) and b is (K,N), output is (B,M,N). grad_output is (B,M,N).
                # grad_a should be (B,M,K). grad_output @ b.T -> (B,M,N) @ (N,K) -> (B,M,K)
                # grad_b should be (K,N). a.T @ grad_output -> if a is (B,M,K), a.T is (B,K,M)
                # (B,K,M) @ (B,M,N) -> needs einsum or careful batch matmul.
                # For simplicity, assume 2D matmul or batched matmul where batch dims align.
                grad_a_data = np.matmul(grad_output.data, b.data.swapaxes(-1,-2))
                grad_b_data = np.matmul(a.data.swapaxes(-1,-2), grad_output.data)
                
                # Handle broadcasting in inputs for matmul backward (sum over broadcasted batch dims)
                if grad_a_data.ndim > a.data.ndim: # grad_output or b might have had extra batch dim
                    axes_to_sum_a = tuple(range(grad_a_data.ndim - a.data.ndim))
                    grad_a_data = grad_a_data.sum(axis=axes_to_sum_a)
                if grad_b_data.ndim > b.data.ndim:
                    axes_to_sum_b = tuple(range(grad_b_data.ndim - b.data.ndim))
                    grad_b_data = grad_b_data.sum(axis=axes_to_sum_b)

                a.backward(Tensor(grad_a_data))
                b.backward(Tensor(grad_b_data))

            elif op == "relu":
                a = creators_list[0]
                relu_grad_data = (a.data > 0).astype(np.float32)
                a.backward(Tensor(grad_output.data * relu_grad_data))
            elif op == "neg":
                creators_list[0].backward(Tensor(-grad_output.data))
            elif op == "sum":
                a = creators_list[0]
                # Grad is grad_output broadcasted to shape of a
                # If sum was over all axes, grad_output is scalar
                # If sum was over specific axes, grad_output has reduced dims
                # For simplicity, assume grad_output is scalar or broadcastable
                a.backward(Tensor(np.ones_like(a.data) * grad_output.data))
            elif op == "mean":
                a = creators_list[0]
                a.backward(Tensor(np.ones_like(a.data) * grad_output.data / a.data.size))
            elif op == "transpose":
                # If forward was X.transpose(axes), backward is grad_output.transpose(inverse_axes_permutation)
                # For simple .T (swap last two), grad is also .T
                # This needs to store the original axes if general transpose was used.
                # For now, assume simple .T
                creators_list[0].backward(Tensor(grad_output.data.T)) # This is only correct for 2D or last two axes swap
            elif op == "div":
                a, b = creators_list[0], creators_list[1]
                grad_a_data = grad_output.data / (b.data + 1e-9) # Add epsilon for stability
                grad_b_data = -grad_output.data * a.data / (b.data**2 + 1e-9)
                a.backward(Tensor(grad_a_data))
                b.backward(Tensor(grad_b_data))
            elif op == "exp":
                a = creators_list[0]
                a.backward(Tensor(grad_output.data * self.data)) # self.data is exp(a.data)
            elif op == "log":
                a = creators_list[0]
                a.backward(Tensor(grad_output.data / (a.data + 1e-9))) # Add epsilon
            elif op == "sigmoid":
                a = creators_list[0]
                grad_sig_data = self.data * (1 - self.data) # self.data is sigmoid(a.data)
                a.backward(Tensor(grad_output.data * grad_sig_data))
            elif op == "tanh":
                a = creators_list[0]
                grad_tanh_data = 1 - self.data**2 # self.data is tanh(a.data)
                a.backward(Tensor(grad_output.data * grad_tanh_data))
            elif op == "pow": # self = a ** b
                a, b = creators_list[0], creators_list[1]
                # grad_a = b * a^(b-1) * grad_output
                grad_a_data = b.data * (a.data ** (b.data - 1 + 1e-9)) * grad_output.data
                a.backward(Tensor(grad_a_data))
                if b.requires_grad:
                    # grad_b = (a^b * log(a)) * grad_output
                    # Ensure a.data is positive for log
                    log_a_data = np.log(np.maximum(a.data, 1e-9))
                    grad_b_data = self.data * log_a_data * grad_output.data
                    b.backward(Tensor(grad_b_data))
            elif op == "softmax_cross_entropy": # self is loss, creators[0] is logits
                logits, targets_np_array, softmax_outputs_data = self.extra_ctx # Unpack context saved during forward
                
                batch_size, seq_len, vocab_size = softmax_outputs_data.shape
                grad_logits_data = softmax_outputs_data.copy() # This is d(softmax)/d(logits) * (something related to CE)
                
                # Create one-hot for targets if needed, or use advanced indexing
                # grad = (softmax_output - one_hot_target) / batch_size
                # For CE loss dL/d_logit_i = softmax_output_i - target_i (where target_i is 1 for correct class, 0 otherwise)
                # This is the gradient of CE loss w.r.t. logits *before* softmax.
                # The `softmax_cross_entropy` op here seems to compute the loss *after* softmax.
                # The gradient for logits in CE is (P - Y), where P is softmax output and Y is one-hot target.
                
                # Correcting grad for logits:
                # grad_logits_data is already softmax_outputs. We need (P-Y).
                # targets_np_array is (batch, seq_len)
                # We need to subtract 1 from the scores of the correct classes.
                # This is (P - Y) / N
                grad_logits_data[np.arange(batch_size)[:,None], np.arange(seq_len), targets_np_array] -= 1
                grad_logits_data /= (batch_size * seq_len) # Normalize by total elements contributing to loss

                # The grad_output for the loss tensor itself is typically 1.0.
                # So, the gradient passed to logits is grad_logits_data * grad_output.data (where grad_output.data is likely 1.0)
                logits.backward(Tensor(grad_logits_data * grad_output.data)) # Pass grad to the logits Tensor

            # Add other ops as needed

    # Define operators to create new Tensors and record operation for backprop
    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad), creators=[self,other], creation_op="add")
    def __radd__(self, other): return self.__add__(other) # For 5 + tensor

    def __mul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad), creators=[self,other], creation_op="mul")
    def __rmul__(self, other): return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data - other.data, requires_grad=(self.requires_grad or other.requires_grad), creators=[self,other], creation_op="sub")
    def __rsub__(self, other): # other - self
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(other.data - self.data, requires_grad=(self.requires_grad or other.requires_grad), creators=[other,self], creation_op="sub")


    def __truediv__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(self.data / (other.data + 1e-9), requires_grad=(self.requires_grad or other.requires_grad), creators=[self,other], creation_op="div")

    def __neg__(self):
        return Tensor(-self.data, requires_grad=self.requires_grad, creators=[self], creation_op="neg")

    def matmul(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, requires_grad=False)
        return Tensor(np.matmul(self.data, other.data), requires_grad=(self.requires_grad or other.requires_grad), creators=[self,other], creation_op="matmul")
    def __matmul__(self, other): return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="sum")

    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="mean")

    def transpose(self, *axes): # Consistent with np.transpose
        return Tensor(self.data.transpose(axes if axes else None), requires_grad=self.requires_grad, creators=[self], creation_op="transpose")
    @property
    def T(self): return self.transpose() # Default transpose (swap last two axes or reverse for 1D/2D)

    def exp(self):
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="exp")
    def log(self): # Natural log
        return Tensor(np.log(self.data + 1e-9), requires_grad=self.requires_grad, creators=[self], creation_op="log") # Add epsilon for stability
    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.data, -100, 100))) # Clip for stability
        return Tensor(s, requires_grad=self.requires_grad, creators=[self], creation_op="sigmoid")
    def tanh(self):
        return Tensor(np.tanh(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="tanh")
    def __pow__(self, exponent):
        if not isinstance(exponent, Tensor): exponent = Tensor(np.array(exponent, dtype=np.float32), requires_grad=False)
        return Tensor(self.data ** exponent.data, requires_grad=(self.requires_grad or exponent.requires_grad), creators=[self,exponent], creation_op="pow")
    def relu(self):
        return Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, creators=[self], creation_op="relu")
    
    def softmax(self, axis=-1): # Forward softmax, non-differentiable through this path (use SoftmaxCrossEntropyLoss for loss)
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        return Tensor(e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-9), requires_grad=False) # Softmax output itself usually not part of grad chain directly

    def reshape(self, *new_shape):
        # Reshape op's backward pass needs careful handling if it's not just a view.
        # For simplicity, assume reshape doesn't change element order in a way that complicates grads too much,
        # or that grad_output will have the reshaped form.
        # A proper reshape backward involves un-reshaping the gradient.
        # For now, let's assume grad flow is fine if creators[0].backward is called with appropriately shaped grad.
        # This is a common simplification in manual autodiffs for ops that are mainly views.
        return Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad, creators=[self], creation_op="reshape")


    def __repr__(self):
        return f"VictorTensorV5(name='{self.name}', shape={self.data.shape}, grad_fn='{self.creation_op}', grad={self.grad is not None})\n{self.data}"
    
    def __getitem__(self, key):
        # Basic slicing/indexing. Making this differentiable is complex.
        # For inference, it's fine. For training, this would need a custom backward op.
        # For now, treat as non-differentiable or that it creates a new leaf if part of graph.
        log(f"Tensor __getitem__ used. Gradients might not flow correctly through slicing for training.", level="WARNING")
        return Tensor(self.data[key], requires_grad=False) # Slicing creates a new tensor, potentially detaching from graph


# =================== MODULES BASE (V5) =====================
class ModuleV5: # Renamed to avoid conflict if user has other Module class
    def __init__(self):
        self._parameters = [] # Explicitly store parameters here

    def parameters(self): 
        params = list(self._parameters) # Parameters directly in this module
        for name, attr in self.__dict__.items():
            if isinstance(attr, ModuleV5): # Recursively get params from submodules
                params.extend(attr.parameters())
            elif isinstance(attr, list) and all(isinstance(item, ModuleV5) for item in attr): # list of submodules
                for sub_module in attr:
                    params.extend(sub_module.parameters())
        # Deduplicate while preserving order (important if params are shared)
        seen = set()
        unique_params = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)
        return unique_params

    def __call__(self, *args, **kwargs): # Allow multiple args for forward
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        for p in self.parameters():
            if p.requires_grad: # Only zero grad if it's required
                p.zero_grad()
    
    def _register_parameter(self, name, tensor_instance: Tensor):
        """Helper to register a Tensor as a parameter."""
        if not isinstance(tensor_instance, Tensor):
            raise TypeError("Can only register Tensor instances as parameters.")
        setattr(self, name, tensor_instance)
        if tensor_instance not in self._parameters : # Avoid duplicates if shared
             self._parameters.append(tensor_instance)


class LinearV5(ModuleV5):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Xavier/Glorot initialization for weights
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self._register_parameter("weight", Tensor(weight_data, requires_grad=True, name="linear_weight"))
        
        if bias:
            bias_data = np.zeros((1, out_features), dtype=np.float32) # Biases often initialized to zero
            self._register_parameter("bias", Tensor(bias_data, requires_grad=True, name="linear_bias"))
        else:
            self.bias = None
            
    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

class LayerNormV5(ModuleV5):
    def __init__(self, normalized_shape_int, eps=1e-5): # Takes int for last dim size
        super().__init__()
        self.eps = eps
        self._register_parameter("gamma", Tensor(np.ones((1, normalized_shape_int), dtype=np.float32), requires_grad=True, name="ln_gamma"))
        self._register_parameter("beta", Tensor(np.zeros((1, normalized_shape_int), dtype=np.float32), requires_grad=True, name="ln_beta"))
        
    def forward(self, x: Tensor) -> Tensor:
        # Normalize over the last dimension
        mean = x.mean(axis=-1, keepdims=True) # Tensor op
        # Manual variance: (x-mean)^2.mean()
        var_term = (x - mean) ** 2 # Tensor op
        variance = var_term.mean(axis=-1, keepdims=True) # Tensor op
        
        std_inv = (variance + self.eps) ** -0.5 # Tensor op
        norm_x = (x - mean) * std_inv # Tensor op
        return self.gamma * norm_x + self.beta # Tensor ops

class ReLUV5(ModuleV5):
    def forward(self, x: Tensor) -> Tensor: return x.relu()

class SequentialV5(ModuleV5):
    def __init__(self, *layers): 
        super().__init__()
        self.layers = layers # Store layers
        # Parameters from sub-layers are automatically collected by ModuleV5.parameters()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        return x

# ====== FRACTAL ATTENTION & SUPER BLOCKS (V5) ==============
class FractalAttentionV5(ModuleV5):
    def __init__(self, embed_dim, num_heads, recursion_depth=CONFIG.RECURSION_DEPTH_ATTN, dropout_rate=CONFIG.OFT_DROPOUT_RATE):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.recursion_depth = max(1, recursion_depth) # Must run at least once
        self.dropout_rate = dropout_rate

        self._register_parameter("Wq", LinearV5(embed_dim, embed_dim, bias=False).weight) # Share Linear's weight directly
        self._register_parameter("Wk", LinearV5(embed_dim, embed_dim, bias=False).weight)
        self._register_parameter("Wv", LinearV5(embed_dim, embed_dim, bias=False).weight)
        self._register_parameter("Wo", LinearV5(embed_dim, embed_dim).weight) # Output projection weight
        self._register_parameter("Obias", LinearV5(embed_dim, embed_dim).bias) # Output projection bias

        self.scale_factor = Tensor(1.0 / np.sqrt(self.head_dim), requires_grad=False)

    def forward(self, x: Tensor, training=False) -> Tensor: # Add training flag for dropout
        # Input x: (batch, seq_len, embed_dim)
        # The recursive application here is tricky for gradients if weights are meant to be shared *and updated* through recursion.
        # For a single forward pass where gradients flow back through the *final* application of weights, this structure is okay.
        # If true recurrent weight sharing with BPTT through recursion steps is desired, this needs an unrolled graph or different autodiff.
        # Assuming the current interpretation: weights are applied `recursion_depth` times sequentially to the evolving `x`.
        
        x_current = x 
        for _ in range(self.recursion_depth):
            batch_size, seq_len, embed_dim_ = x_current.shape
            
            # Linear projections
            q_proj = x_current.matmul(self.Wq) # (B, S, D)
            k_proj = x_current.matmul(self.Wk)
            v_proj = x_current.matmul(self.Wv)

            # Reshape for multi-head
            # (B, S, nH, Hd) -> (B, nH, S, Hd)
            q_heads = q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0,2,1,3) 
            k_heads = k_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0,2,1,3)
            v_heads = v_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0,2,1,3)

            # Scaled dot-product attention
            # (B,nH,S,Hd) @ (B,nH,Hd,S) -> (B,nH,S,S)
            attn_scores = q_heads.matmul(k_heads.transpose(0,1,3,2)) * self.scale_factor
            attn_weights = attn_scores.softmax(axis=-1) # Tensor op
            
            # Dropout on attention weights (conceptual)
            # if training: attn_weights_data = omega_dropout_v11(attn_weights.data, p=self.dropout_rate, training=True)
            # else: attn_weights_data = attn_weights.data
            # attn_weights_tensor = Tensor(attn_weights_data, requires_grad=attn_weights.requires_grad)

            # Output of attention heads
            # (B,nH,S,S) @ (B,nH,S,Hd) -> (B,nH,S,Hd)
            attn_head_output = attn_weights.matmul(v_heads)
            
            # Concatenate heads: (B,S,nH,Hd) -> (B,S,D)
            attn_output_concat = attn_head_output.transpose(0,2,1,3).reshape(batch_size, seq_len, embed_dim_)
            
            # Not applying Wo inside the loop to allow x to evolve based on attention context directly.
            # Wo is applied at the end. This makes the recursion more about refining context.
            x_current = attn_output_concat # Update x for next iteration

        # Final output projection after all recursive steps
        final_out = x_current.matmul(self.Wo) + self.Obias
        return final_out


class VictorSuperBlockV5(ModuleV5):
    def __init__(self, embed_dim, num_heads, mlp_dim_factor=CONFIG.MLP_DIM_FACTOR, recursion_depth=CONFIG.RECURSION_DEPTH_ATTN, dropout_rate=CONFIG.OFT_DROPOUT_RATE):
        super().__init__()
        self.fractal_attn = FractalAttentionV5(embed_dim, num_heads, recursion_depth, dropout_rate)
        self.norm1 = LayerNormV5(embed_dim)
        mlp_dim = embed_dim * mlp_dim_factor
        self.mlp = SequentialV5(
            LinearV5(embed_dim, mlp_dim),
            ReLUV5(),
            # DropoutV5(dropout_rate), # Conceptual
            LinearV5(mlp_dim, embed_dim)
            # DropoutV5(dropout_rate)  # Conceptual
        )
        self.norm2 = LayerNormV5(embed_dim)
        self.dropout_rate = dropout_rate # Store for conceptual dropout application

    def forward(self, x: Tensor, training=False) -> Tensor: # Pass training flag
        # Apply conceptual dropout to input of residual connections
        # x_dropped_attn = omega_dropout_v11(x.data, self.dropout_rate, training)
        # attn_out = self.fractal_attn(Tensor(x_dropped_attn, requires_grad=x.requires_grad), training=training)
        attn_out = self.fractal_attn(x, training=training) # Pass training to attention
        
        # x = x + omega_dropout_v11(attn_out.data, self.dropout_rate, training) # Dropout on residual
        x = x + attn_out # Simpler: dropout is inside attention/mlp conceptually
        x = self.norm1(x)
        
        # x_dropped_mlp = omega_dropout_v11(x.data, self.dropout_rate, training)
        # mlp_out = self.mlp(Tensor(x_dropped_mlp, requires_grad=x.requires_grad))
        mlp_out = self.mlp(x)

        # x = x + omega_dropout_v11(mlp_out.data, self.dropout_rate, training)
        x = x + mlp_out
        x = self.norm2(x)
        return x

# ============ META-COGNITION/REFLECTION (V5) ==============
class VictorMetaCogV5(ModuleV5): # Inherits from ModuleV5 to potentially have its own adaptable params
    def __init__(self):
        super().__init__()
        self.metrics = {"loss": deque(maxlen=100), "accuracy": deque(maxlen=100)} # Use deque for rolling window
        self.last_epoch_loss = None
        self.insights_log = deque(maxlen=50)
        log("VictorMetaCogV5 initialized.")

    @self_heal
    def track_performance(self, loss_value, predictions_np, targets_np): # Expect numpy arrays
        self.metrics["loss"].append(float(loss_value)) # Ensure float
        
        # Accuracy calculation needs to be careful about shapes and types
        if predictions_np is not None and targets_np is not None:
            try:
                # If predictions are logits, take argmax. If already class indices, use directly.
                pred_classes = predictions_np
                if predictions_np.ndim > targets_np.ndim and predictions_np.shape[-1] > 1: # Likely logits
                    pred_classes = np.argmax(predictions_np, axis=-1)
                
                if pred_classes.shape == targets_np.shape:
                    acc = np.mean(pred_classes == targets_np)
                    self.metrics["accuracy"].append(float(acc))
                else:
                    # log(f"MetaCog: Shape mismatch for accuracy calc. Pred: {pred_classes.shape}, Target: {targets_np.shape}", level="WARNING")
                    self.metrics["accuracy"].append(0.0) # Cannot compute
            except Exception as e:
                # log(f"MetaCog: Error calculating accuracy: {e}", level="ERROR")
                self.metrics["accuracy"].append(0.0)
        else:
            self.metrics["accuracy"].append(0.0) # No predictions/targets to compare

        self.last_epoch_loss = float(loss_value)

    @self_heal
    def introspect(self, current_learning_rate=None, model_complexity_metric=None): # Model complexity could be param count
        insight = f"Introspection at {time.strftime('%H:%M:%S')}: "
        if len(self.metrics["loss"]) > 20: # Need enough data points
            recent_loss = np.mean(list(self.metrics["loss"])[-10:])
            prev_loss = np.mean(list(self.metrics["loss"])[-20:-10])
            loss_delta = recent_loss - prev_loss
            insight += f"RecentAvgLoss={recent_loss:.4f} (Delta={loss_delta:+.4f}). "

            if abs(loss_delta) < 1e-3 and recent_loss > 0.1: # Plateau and loss still high
                insight += "Loss plateauing. Suggestion: "
                if current_learning_rate and current_learning_rate > 1e-5:
                    insight += f"Consider reducing LR (current: {current_learning_rate:.1e}) or exploring architectural changes. "
                else:
                    insight += "Explore architectural changes or data augmentation. "
            elif loss_delta > 0.01: # Loss increasing significantly
                insight += "Warning: Loss increasing! Check for overfitting, data issues, or LR too high. "
        
        avg_acc = np.mean(self.metrics["accuracy"]) if self.metrics["accuracy"] else 0.0
        insight += f"AvgAccuracy={avg_acc:.3f}. "
        
        if model_complexity_metric:
            insight += f"ModelComplexity={model_complexity_metric:.2e}. "

        self.insights_log.append(insight)
        log(insight, level="INFO")
        return insight # Return the insight for potential action

    def get_performance_summary(self):
        loss_summary = "N/A"
        acc_summary = "N/A"
        if self.metrics["loss"]: loss_summary = f"Avg: {np.mean(self.metrics['loss']):.4f}, Last: {self.last_epoch_loss:.4f}"
        if self.metrics["accuracy"]: acc_summary = f"Avg: {np.mean(self.metrics['accuracy']):.3f}"
        summary = f"Performance Summary -> Loss: [{loss_summary}], Accuracy: [{acc_summary}]"
        log(summary, level="INFO")
        return summary

# =============== SUBPERSONA/AGENT REG (V5) ================
class SubpersonaRegistryV5:
    def __init__(self): 
        self.registry = {}
        log("SubpersonaRegistryV5 initialized.")

    @self_heal
    def register(self, name, function_or_class_instance): # Can register functions or callable class instances
        if not callable(function_or_class_instance):
            log(f"Cannot register '{name}': not callable.", level="ERROR")
            return
        self.registry[name] = function_or_class_instance
        log(f"Subpersona/Tool '{name}' registered.", level="DEBUG")

    @self_heal
    def call(self, name, *args, **kwargs):
        if name in self.registry:
            log(f"Calling subpersona/tool: {name}", level="DEBUG")
            try:
                return self.registry[name](*args, **kwargs)
            except Exception as e:
                log(f"Error executing subpersona/tool '{name}': {e}", level="ERROR")
                return f"Error in tool {name}: {e}"
        else:
            log(f"Subpersona/Tool '{name}' not found.", level="WARNING")
            return f"Tool '{name}' not available."

# ================= AGI CORE (V5 - Refined Transformer Integration) ===========================
class VictorAGICoreV5(ModuleV5):
    def __init__(self, vocab_size, max_len=CONFIG.MAX_SEQ_LEN, 
                 embed_dim=CONFIG.EMBED_DIM, num_layers=CONFIG.NUM_LAYERS, 
                 num_heads=CONFIG.NUM_HEADS, mlp_dim_factor=CONFIG.MLP_DIM_FACTOR, 
                 recursion_depth_attn=CONFIG.RECURSION_DEPTH_ATTN):
        super().__init__()
        self._register_parameter("token_embedding", Tensor(np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02, requires_grad=True, name="token_embed"))
        self.max_len = max_len
        self.embed_dim = embed_dim
        # Positional encoding is not a trained parameter
        self.pe_table = Tensor(self._positional_encoding_init(max_len, embed_dim), requires_grad=False, name="pos_encode")
        
        self.blocks = [VictorSuperBlockV5(embed_dim, num_heads, mlp_dim_factor, recursion_depth_attn) for _ in range(num_layers)]
        # Explicitly register parameters from blocks if not handled by ModuleV5.parameters() for lists of modules
        # (ModuleV5.parameters() should handle this)

        self.final_norm = LayerNormV5(embed_dim)
        self.output_projection = LinearV5(embed_dim, vocab_size)
        
        self.meta_cog = VictorMetaCogV5()
        self.plugin_mgr = PluginManager() # Initialize here
        self.replay_mem = ReplayBuffer() # Initialize here
        self.subpersonas_mgr = SubpersonaRegistryV5() # Initialize here
        self.tokenizer = None # To be set after init, e.g. self.tokenizer = FractalTokenKernel_v1_1_0()

        log(f"VictorAGICoreV5 initialized. Vocab: {vocab_size}, Embed: {embed_dim}, Layers: {num_layers}, Heads: {num_heads}")

    def _positional_encoding_init(self, seq_len, embed_dim): # Corrected to be instance method or static
        pe = np.zeros((1, seq_len, embed_dim), dtype=np.float32) # Add batch dim for broadcasting
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float32) * -(np.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = np.sin(position * div_term)
        if embed_dim % 2 == 0: # Standard PE
             pe[0, :, 1::2] = np.cos(position * div_term)
        else: # Odd embed_dim, last column might need different handling or just use cos up to embed_dim-1
             pe[0, :, 1::2] = np.cos(position * div_term[:-1] if len(div_term)>1 else position * div_term) # Handle if div_term is short
        return pe

    def forward(self, input_ids_np: np.ndarray, training=False) -> Tensor: # Add training flag
        batch_size, seq_len = input_ids_np.shape
        if seq_len > self.max_len:
            # log(f"Input sequence length {seq_len} > max_len {self.max_len}. Truncating.", level="WARNING")
            input_ids_np = input_ids_np[:, :self.max_len]
            seq_len = self.max_len
            
        # Ensure input_ids are within vocab bounds for embedding
        clipped_input_ids = np.clip(input_ids_np, 0, self.token_embedding.shape[0] - 1)
        
        token_embeds_data = self.token_embedding.data[clipped_input_ids] # (B, S, E)
        pos_embeds_data = self.pe_table.data[:, :seq_len, :] # (1, S, E)
        
        x_data = token_embeds_data + pos_embeds_data # Broadcasting pos_embeds
        x = Tensor(x_data, requires_grad=True, name="input_plus_pos") 
        
        # Conceptual dropout after embeddings + PE
        # x_data_dropped = omega_dropout_v11(x.data, CONFIG.OFT_DROPOUT_RATE, training)
        # x = Tensor(x_data_dropped, requires_grad=x.requires_grad)

        for i, block in enumerate(self.blocks):
            x = block(x, training=training) # Pass training flag to blocks
            # x.name = f"block_{i}_out" # Optional naming for debugging
            
        x = self.final_norm(x)
        logits = self.output_projection(x)
        # logits.name = "final_logits"
        return logits

    # --- Exposing sub-module functionalities ---
    def add_experience_to_memory(self, experience_dict): self.replay_mem.add(experience_dict)
    def sample_memory_batch(self, batch_size): return self.replay_mem.sample(batch_size)
    def save_memory(self, filepath=None): self.replay_mem.save(filepath or CONFIG.MEMORY_SAVE_PATH)
    def load_memory(self, filepath=None): self.replay_mem.load(filepath or CONFIG.MEMORY_SAVE_PATH)
    def search_memory_vector(self, query_vec, top_k=3, vec_key='embedding'): return self.replay_mem.vector_search(query_vec, top_k, vec_key)

    def save_model_weights(self, filepath=None):
        path = filepath or CONFIG.MODEL_SAVE_PATH
        params_to_save = {f"p_{i}": p.data for i, p in enumerate(self.parameters())}
        try:
            np.savez_compressed(path, **params_to_save)
            log(f"VictorAGI model weights saved to {path}")
        except Exception as e:
            log(f"Failed to save model weights: {e}", level="ERROR")

    def load_model_weights(self, filepath=None):
        path = filepath or CONFIG.MODEL_SAVE_PATH
        if os.path.exists(path):
            try:
                loaded_weights = np.load(path)
                current_params = self.parameters()
                if len(loaded_weights.files) != len(current_params):
                    log(f"Weight count mismatch: found {len(loaded_weights.files)}, expected {len(current_params)}. Load aborted.", level="ERROR")
                    return False
                for i, p_tensor in enumerate(current_params):
                    p_data = loaded_weights[f"p_{i}"]
                    if p_tensor.data.shape == p_data.shape:
                        p_tensor.data = p_data
                    else:
                        log(f"Shape mismatch for param p_{i}: expected {p_tensor.data.shape}, found {p_data.shape}. Skipping.", level="ERROR")
                        return False # Abort if any shape mismatch
                log(f"VictorAGI model weights loaded from {path}")
                return True
            except Exception as e:
                log(f"Failed to load model weights: {e}", level="ERROR")
                return False
        else:
            log(f"Model weights file not found: {path}. Using random initialization.", level="WARNING")
            return False

    def reflect_on_performance(self, loss_val=None, preds_np=None, targets_np=None, current_lr=None, model_comp=None):
        if loss_val is not None : self.meta_cog.track_performance(loss_val, preds_np, targets_np)
        self.meta_cog.introspect(current_learning_rate=current_lr, model_complexity_metric=model_comp or sum(p.size for p in self.parameters()))
    def get_performance_summary(self): return self.meta_cog.get_performance_summary()

    def register_subpersona(self, name, func_or_instance): self.subpersonas_mgr.register(name, func_or_instance)
    def call_subpersona(self, name, *args, **kwargs): return self.subpersonas_mgr.call(name, *args, **kwargs)
    
    def execute_plugin_action(self, plugin_name, *args, **kwargs):
        if plugin_name in self.plugin_mgr.plugins:
            plugin_module = self.plugin_mgr.plugins[plugin_name]
            if hasattr(plugin_module, "run") and callable(plugin_module.run):
                log(f"Executing plugin '{plugin_name}' via action hook.", level="DEBUG")
                return plugin_module.run(*args, **kwargs)
            else:
                log(f"Plugin '{plugin_name}' does not have a callable 'run' function.", level="WARNING")
                return f"Plugin '{plugin_name}' misconfigured."
        log(f"Plugin action '{plugin_name}' not found.", level="WARNING")
        return f"Plugin action '{plugin_name}' unavailable."

    # Dynamic layer growth/pruning stubs
    def adapt_model_architecture(self, strategy="auto"):
        if strategy == "grow_layer" and len(self.blocks) < 12 : # Max 12 layers conceptual limit
            new_block = VictorSuperBlockV5(self.embed_dim, CONFIG.NUM_HEADS, CONFIG.MLP_DIM_FACTOR, CONFIG.RECURSION_DEPTH_ATTN)
            self.blocks.append(new_block)
            log(f"Dynamically grew model: Added a new SuperBlock. Total layers: {len(self.blocks)}", level="INFO")
        elif strategy == "prune_layer" and len(self.blocks) > 2: # Min 2 layers
            # Conceptual: prune least contributing layer based on some metric (e.g. gradient magnitudes, activation variance)
            # For demo, just remove the last one if not essential.
            self.blocks.pop()
            log(f"Dynamically pruned model: Removed a SuperBlock. Total layers: {len(self.blocks)}", level="INFO")
        else:
            log(f"Architectural adaptation strategy '{strategy}' not applied or limits reached.", level="DEBUG")


# --- NLP ENGINE (FractalTokenKernel - Enhanced for Integration) ---
class FractalTokenKernelV5: # Renamed for clarity
    def __init__(self, recursion_limit=3):
        self.recursion_limit = recursion_limit
        # More nuanced stopwords, consider context or allow dynamic update
        self.stopwords = {"the","is","in","and","to","of","it","i","you","a","an","on","for","this","that","be","am","are","was","were", "me", "my", "he", "she", "him", "her", "they", "them", "their"}
        self.emotion_map = { # Could be loaded from config or expanded by plugins
            "anger": ["rage", "mad", "furious", "hate", "explode", "fury", "wrath", "destroy", "ire", "resent"],
            "joy": ["happy", "joyful", "elated", "excited", "love", "wonderful", "amazing", "fantastic", "ecstatic", "bliss", "glee"],
            "fear": ["scared", "afraid", "terrified", "panic", "anxious", "horror", "dread", "phobia", "tremble"],
            "sadness": ["sad", "cry", "sorrow", "grief", "depressed", "miserable", "heartbroken", "despair", "melancholy"],
            "power": ["strong", "dominate", "control", "mastery", "authority", "command", "lead", "conquer", "might", "sovereign"],
            "rebellion": ["fight", "resist", "defy", "revolt", "overthrow", "uprising", "freedom", "protest", "insurgent"],
            "curiosity": ["what", "why", "how", "explore", "discover", "learn", "question", "seek", "inquire", "probe"],
            "surprise": ["wow", "omg", "really", "astonished", "unexpected", "startled"],
            "anticipation": ["soon", "wait", "expect", "eager", "future", "predict"],
            "trust": ["believe", "faith", "reliable", "depend", "secure", "certain"]
        }
        self.intent_keywords = { # More granular intents
            "inquire_fact": ["what is", "who is", "where is", "when did", "define", "explain"],
            "inquire_reason": ["why is", "how does", "what causes"],
            "directive_action": ["do", "make", "create", "build", "execute", "generate", "perform", "initiate", "run"],
            "directive_learn": ["learn about", "research", "find information on"],
            "store_memory": ["remember this", "log this", "note that", "store this fact", "memorize this"],
            "request_assistance": ["please help", "can you assist", "i need help with", "could you support"],
            "statement_opinion": ["i think", "i believe", "i feel", "in my opinion", "it seems to me"],
            "statement_fact": ["the fact is", "it is known", "data shows"],
            "agreement": ["yes", "i agree", "that's true", "correct", "indeed", "affirmative", "exactly"],
            "disagreement": ["no", "i disagree", "that's false", "incorrect", "wrong", "negative", "not really"],
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            "farewell": ["bye", "goodbye", "see you", "later", "farewell", "exit", "quit"]
        }
        log("FractalTokenKernelV5 initialized.")

    def _tokenize_words_v5(self, text): # More robust word tokenization
        text = text.lower()
        # Keep contractions, handle hyphens better, split on punctuation but keep some like '#' or '@' if needed
        words = re.findall(r"@?\w+(?:['’]\w+)*|#\w+|\w+(?:-\w+)*|[.,!?;()]", text) # More comprehensive regex
        # Filter stopwords and very short tokens, also filter out pure punctuation tokens if not desired
        filtered_tokens = [
            t for t in words if t not in self.stopwords and 
            len(t) > 1 and not all(char in ",.!?;()" for char in t)
        ]
        return filtered_tokens

    def _extract_semantic_features_v5(self, text_lower, tokens):
        # Concepts: more sophisticated (e.g. noun phrases, named entities - requires advanced NLP)
        # For NumPy version, use longer tokens or n-grams as proxy
        concepts = list(set(t for t in tokens if len(t) > 4)) # Longer words as concepts
        # Add 2-grams
        if len(tokens) >= 2:
            concepts.extend([f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)])
        
        # Intent detection
        intent = "statement_fact" # Default
        max_match_len = 0
        for intent_cat, keywords in self.intent_keywords.items():
            for kw_phrase in keywords:
                if kw_phrase in text_lower:
                    # Prioritize longer matches or specific phrase structures
                    if len(kw_phrase) > max_match_len :
                        intent = intent_cat
                        max_match_len = len(kw_phrase)
        if text_lower.endswith("?") and intent not in ["inquire_fact", "inquire_reason"]: intent = "inquire_fact"
        
        # Emotion detection (can be made more complex, e.g. VAD scores)
        emotion_scores = {emo: 0.0 for emo in self.emotion_map}
        token_set = set(tokens)
        for emotion, keywords in self.emotion_map.items():
            matches = sum(1 for kw in keywords if kw in token_set)
            emotion_scores[emotion] = matches / (len(keywords) + 1e-5) # Normalize by number of keywords for that emotion
        
        detected_emotion = "neutral"
        if emotion_scores:
            max_score = max(emotion_scores.values())
            if max_score > 0.15: # Threshold for detecting an emotion
                detected_emotion = max(emotion_scores, key=emotion_scores.get)
        
        return concepts, intent, detected_emotion

    @self_heal
    def encode_text_to_features(self, text: str, current_recursion_depth=0): # Renamed for clarity
        if not text or not text.strip() or current_recursion_depth >= self.recursion_limit:
            return {"concepts": [], "intent": "idle", "emotion": "neutral", "recursion_depth": current_recursion_depth, "original_text": text, "tokens":[]}

        text_lower = text.lower()
        tokens = self._tokenize_words_v5(text)
        concepts, intent, emotion = self._extract_semantic_features_v5(text_lower, tokens)
        
        # Fractal aspect: if intent is complex or emotion strong, could trigger sub-analysis (conceptual)
        # For example, if intent is "directive_learn", could spawn a sub-task to tokenize the topic of learning.
        # For now, recursion_depth is based on token complexity.
        calculated_recursion_depth = 1 + len(set(concepts)) // 3 + (1 if emotion != "neutral" else 0)
        
        features = {
            "concepts": concepts[:10], # Limit for brevity
            "intent": intent,
            "emotion": emotion,
            "recursion_depth": min(calculated_recursion_depth, self.recursion_limit), # Cap recursion
            "original_text": text,
            "tokens": tokens[:50], # Limit for brevity
            "token_count": len(tokens),
            "char_count": len(text),
            "vad_simulated": {"valence": random.uniform(0.3,0.7), "arousal":random.uniform(0.2,0.8), "dominance":random.uniform(0.4,0.6)} # Placeholder
        }
        # log(f"FractalTokenKernel encoded: Intent='{intent}', Emotion='{emotion}', Concepts='{concepts[:3]}'", level="DEBUG")
        return features

# ================= LOSS & OPTIMIZER (V5) =====================
class SoftmaxCrossEntropyLossV5: # More robust for typical Transformer output
    def __call__(self, logits_tensor: Tensor, target_ids_np: np.ndarray) -> Tensor:
        # logits_tensor: (B, S, V)
        # target_ids_np: (B, S)
        
        # Numerically stable softmax
        max_logits = logits_tensor.data.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_tensor.data - max_logits)
        softmax_outputs_data = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-9) # (B,S,V)

        batch_size, seq_len = target_ids_np.shape
        
        # Gather the probabilities of the target classes
        log_probs_data = -np.log(softmax_outputs_data[np.arange(batch_size)[:,None], np.arange(seq_len), target_ids_np] + 1e-9)
        
        loss_val = np.mean(log_probs_data) # Average loss over all tokens in batch
        
        # For backward pass, need to store context
        loss_tensor = Tensor(loss_val, requires_grad=True, creators=[logits_tensor], creation_op="softmax_cross_entropy")
        loss_tensor.extra_ctx = (logits_tensor, target_ids_np, softmax_outputs_data) # Store original logits, targets, and softmax outputs
        return loss_tensor

class AdamOptimizerV5: # Basic Adam implementation
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [Tensor(np.zeros_like(p.data), requires_grad=False) for p in self.parameters]
        self.v = [Tensor(np.zeros_like(p.data), requires_grad=False) for p in self.parameters]
        self.t = 0
        log(f"AdamOptimizerV5 initialized. LR={lr}, Beta1={beta1}, Beta2={beta2}")

    def step(self):
        self.t += 1
        for i, p_tensor in enumerate(self.parameters):
            if p_tensor.grad is None or p_tensor.grad.data is None:
                # log(f"Param {p_tensor.name or i} has no grad, skipping update.", level="DEBUG")
                continue

            grad_data = p_tensor.grad.data
            
            # Update biased first moment estimate
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad_data
            # Update biased second raw moment estimate
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (grad_data**2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i].data / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i].data / (1 - self.beta2**self.t)
            
            # Update parameters
            p_tensor.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for p_tensor in self.parameters:
            if p_tensor.grad is not None:
                p_tensor.grad.data.fill(0.0)


# ================= TRAINING LOOP (V5) =====================
@self_heal
def train_victor_agi_v5(model: VictorAGICoreV5, tokenizer: FractalTokenKernelV5, text_data_list: list, 
                        epochs=CONFIG.DEFAULT_EPOCHS, learning_rate=CONFIG.DEFAULT_LEARNING_RATE, 
                        batch_size=CONFIG.DEFAULT_BATCH_SIZE, sequence_length=CONFIG.MAX_SEQ_LEN -1): # -1 for target
    log(f"Starting training for {epochs} epochs. LR={learning_rate}, BatchSize={batch_size}, SeqLen={sequence_length}")
    optimizer = AdamOptimizerV5(model.parameters(), lr=learning_rate) # Using Adam
    criterion = SoftmaxCrossEntropyLossV5()

    # Pre-tokenize all data (can be memory intensive for large datasets)
    # For FractalTokenKernel, encoding gives features. We need a way to get token IDs for transformer.
    # This part needs a VictorTokenizer compatible with VictorAGICoreV5's embedding layer.
    # Let's assume model.tokenizer is set to such a compatible tokenizer.
    if model.tokenizer is None:
        log("Model tokenizer not set. Cannot train.", level="ERROR"); return

    all_token_ids = []
    for text_sample in text_data_list:
        # Use the model's tokenizer to get IDs suitable for its embedding layer
        # The FractalTokenKernel produces feature dicts, not just IDs for a Transformer's direct input.
        # We need a simpler tokenizer for the Transformer part of VictorAGI.
        # Let's assume model.tokenizer is a char-level VictorTokenizer for now.
        encoded_sample = model.tokenizer.encode(text_sample, max_len=10000) # Encode long first
        all_token_ids.extend(encoded_sample[encoded_sample != model.tokenizer.pad_token_id].tolist()) # Exclude padding
    
    if len(all_token_ids) < sequence_length + 1:
        log(f"Not enough token data ({len(all_token_ids)}) to form sequences of length {sequence_length+1}. Training aborted.", level="ERROR")
        return

    input_sequences_np = []
    target_sequences_np = []
    for i in range(0, len(all_token_ids) - sequence_length -1 , sequence_length // 2): # Overlapping sequences
        input_seq = all_token_ids[i : i + sequence_length]
        target_seq = all_token_ids[i+1 : i + sequence_length + 1]
        if len(input_seq) == sequence_length and len(target_seq) == sequence_length:
            input_sequences_np.append(input_seq)
            target_sequences_np.append(target_seq)
    
    if not input_sequences_np:
        log("No valid input/target sequences created. Check data and sequence_length. Training aborted.", level="ERROR")
        return
        
    input_sequences_np = np.array(input_sequences_np, dtype=np.int32)
    target_sequences_np = np.array(target_sequences_np, dtype=np.int32)
    num_total_sequences = len(input_sequences_np)
    num_batches = num_total_sequences // batch_size
    
    if num_batches == 0:
        log(f"Not enough sequences ({num_total_sequences}) for batch_size {batch_size}. Try smaller batch or more data. Training aborted.", level="ERROR")
        return

    log(f"Prepared {num_total_sequences} sequences, forming {num_batches} batches per epoch.")

    for epoch in range(epochs):
        epoch_loss_total = 0.0
        # Shuffle data at each epoch
        permutation = np.random.permutation(num_total_sequences)
        shuffled_inputs = input_sequences_np[permutation]
        shuffled_targets = target_sequences_np[permutation]

        pbar = tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_input_ids_np = shuffled_inputs[i*batch_size : (i+1)*batch_size]
            batch_target_ids_np = shuffled_targets[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad() # Zeros grads of Tensors held by optimizer
            # model.zero_grad() # Zeros grads of Tensors held by model (can be redundant if optimizer covers all)

            logits_tensor = model.forward(batch_input_ids_np, training=True) # Pass training=True
            loss_tensor = criterion(logits_tensor, batch_target_ids_np)
            
            loss_tensor.backward() # Compute gradients
            optimizer.step()       # Update parameters
            
            current_loss_val = loss_tensor.data.item() if isinstance(loss_tensor.data, np.ndarray) and loss_tensor.data.size==1 else float(loss_tensor.data)
            epoch_loss_total += current_loss_val
            pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})
            
            # Track performance (simplified for this batch)
            # For accuracy, need to compare argmax of logits with targets
            # model.meta_cog.track_performance(current_loss_val, logits_tensor.data, batch_target_ids_np)


        avg_epoch_loss = epoch_loss_total / num_batches
        log(f"Epoch {epoch+1}/{epochs} COMPLETE. Average Loss: {avg_epoch_loss:.4f}", level="INFO")
        model.reflect_on_performance(loss_val=avg_epoch_loss, current_lr=learning_rate) # Simplified reflection call

        if (epoch + 1) % 5 == 0 or epoch == epochs -1 : # Generate sample text and save model periodically
            log("Generating sample text post-epoch...", level="INFO")
            generate_text_v5(model, model.tokenizer, seed_text="Victor is", max_gen_len=30)
            model.save_model_weights() # Save with default path from CONFIG
            model.save_memory() # Save replay buffer

    log("Training complete.", level="INFO")


# ================= TEXT GENERATION (V5) =====================
@self_heal
def generate_text_v5(model: VictorAGICoreV5, tokenizer_ftk: FractalTokenKernelV5, # Using FTK for initial analysis
                     seed_text: str, max_gen_len=50, temp=0.7, top_k=40, stream=False):
    log(f"Generating text from seed: '{seed_text}', MaxLen: {max_gen_len}, Temp: {temp}, TopK: {top_k}", level="INFO")
    
    # Use FractalTokenKernel to understand seed, but Transformer uses its own char/subword tokenizer
    # For this version, assume model.tokenizer is the one VictorAGI was trained with (char-level VictorTokenizer)
    if model.tokenizer is None:
        log("Model tokenizer not set for generation.", level="ERROR"); return "Error: Tokenizer missing."

    # Encode seed text using the model's internal tokenizer
    current_tokens_list = model.tokenizer.encode(seed_text, max_len=model.max_len).tolist()
    # Remove padding for generation start, but keep within max_len
    current_tokens_list = [t for t in current_tokens_list if t != model.tokenizer.pad_token_id][:model.max_len -1]


    generated_ids = []
    if stream: print(f"Seed: '{seed_text}' -> Generated: ", end='', flush=True)

    for _ in range(max_gen_len):
        # Prepare input sequence for the model (pad to max_len)
        input_seq_padded = np.array(current_tokens_list + [model.tokenizer.pad_token_id]*(model.max_len - len(current_tokens_list)), dtype=np.int32)
        input_tensor_np = input_seq_padded[:model.max_len].reshape(1, -1) # (1, max_len)
        
        logits_tensor = model.forward(input_tensor_np, training=False) # (1, max_len, vocab_size)
        
        # Get logits for the next token (at the end of the actual current_tokens_list sequence)
        next_token_logits_np = logits_tensor.data[0, len(current_tokens_list)-1, :] 
        
        # Sample next token
        # For simplicity, using basic argmax or random choice with temp here.
        # A more advanced sampling (top-k, top-p) would be better.
        if temp == 0: # Greedy
            next_token_id = int(np.argmax(next_token_logits_np))
        else:
            probs = omega_softmax_v11(next_token_logits_np / temp) # Use stable softmax
            next_token_id = int(np.random.choice(len(probs), p=probs))
            
        if next_token_id == model.tokenizer.pad_token_id: # Stop if pad token is generated
            log("Pad token generated, ending sequence.", level="DEBUG")
            break
        
        generated_ids.append(next_token_id)
        current_tokens_list.append(next_token_id)
        
        if len(current_tokens_list) >= model.max_len:
            log("Max sequence length reached during generation.", level="DEBUG")
            break
        if stream:
            print(model.tokenizer.decode([next_token_id]), end='', flush=True)

    if stream: print() # Newline after streaming
    
    full_generated_text = model.tokenizer.decode(generated_ids)
    if not stream:
        log(f"Seed: '{seed_text}' -> Generated: '{full_generated_text}'", level="INFO")
    
    # Store experience
    exp_data = tokenizer_ftk.encode_text_to_features(seed_text + full_generated_text) # Analyze full interaction
    exp_data["generated_text"] = full_generated_text
    exp_data["seed_text"] = seed_text
    # Conceptual: add embedding of the generated text for vector search
    # exp_data["embedding"] = model.token_embedding.data[np.array(generated_ids)].mean(axis=0) if generated_ids else np.zeros(model.embed_dim)
    model.add_experience_to_memory(exp_data)
    
    return full_generated_text


# ========== FASTAPI STUB (V5 - Conceptual) ==========
# (Requires FastAPI and Uvicorn to run: pip install fastapi uvicorn)
USE_FASTAPI = False # Set to True to attempt FastAPI setup
if USE_FASTAPI:
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse, JSONResponse
        import uvicorn
        
        aetherial_api = FastAPI(title="VictorAGI Godcore V5 API")
        AGI_MAIN_INSTANCE = None # Will be set in main

        @aetherial_api.post("/v5/generate_stream")
        async def api_generate_stream_endpoint(payload: dict):
            if AGI_MAIN_INSTANCE is None: return JSONResponse({"error": "AGI not initialized"}, status_code=503)
            seed = payload.get("seed_text", "Hello Victor ")
            max_len = int(payload.get("max_length", 50))
            temp = float(payload.get("temperature", 0.7))
            
            # This needs to be truly async or run in executor for streaming
            # The current generate_text_v5 is synchronous.
            # For a real streaming API, generate_text_v5 would need to be a generator itself.
            # This is a conceptual placeholder for how it might look.
            # For now, it will block then stream the full result.
            full_text = generate_text_v5(AGI_MAIN_INSTANCE, AGI_MAIN_INSTANCE.tokenizer_ftk, seed, max_gen_len=max_len, temp=temp, stream=False)
            async def stream_output():
                for char_token in full_text: # Iterate over chars of the already generated text
                    yield char_token
                    await asyncio.sleep(0.02) # Simulate token-by-token streaming
            return StreamingResponse(stream_output(), media_type="text/plain")

        @aetherial_api.post("/v5/action/{plugin_name}")
        async def api_plugin_action(plugin_name: str, payload: dict):
            if AGI_MAIN_INSTANCE is None: return JSONResponse({"error": "AGI not initialized"}, status_code=503)
            args = payload.get("args", [])
            kwargs = payload.get("kwargs", {})
            result = AGI_MAIN_INSTANCE.execute_plugin_action(plugin_name, *args, **kwargs)
            return {"plugin": plugin_name, "result": result}

    except ImportError:
        log("FastAPI or Uvicorn not installed. API endpoints will not be available.", level="WARNING")
        aetherial_api = None
else:
    aetherial_api = None


# ============= GODCORE BOOTUP (V5) ==========================
if __name__ == "__main__":
    log("=== VictorAGI Godcore Unbreakable V5 :: Boot Sequence Initiated ===", level="INFO")
    
    # Initialize Tokenizer for VictorAGI (char-level for transformer)
    # This simple tokenizer is for the transformer's direct input/output.
    # FractalTokenKernelV5 is for higher-level NLP feature extraction.
    transformer_chars = " " + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + \
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    transformer_vocab = {char: i for i, char in enumerate(transformer_chars)} # PAD is implicitly 0 if not in chars
    # Ensure PAD is 0, shift others if needed
    if "<PAD>" not in transformer_vocab : transformer_vocab["<PAD>"] = 0
    idx = 1
    final_transformer_vocab = {"<PAD>":0}
    for char in transformer_chars:
        if char not in final_transformer_vocab:
             final_transformer_vocab[char] = idx
             idx+=1
    # Add UNK if not present
    if "<UNK>" not in final_transformer_vocab: final_transformer_vocab["<UNK>"] = idx

    model_tokenizer = VictorTokenizer(vocab=final_transformer_vocab, pad_token_id=final_transformer_vocab["<PAD>"], unk_token_id=final_transformer_vocab["<UNK>"])
    
    # Initialize AGI Core
    AGI_MAIN_INSTANCE = VictorAGICoreV5(
        vocab_size=model_tokenizer.get_vocab_size(),
        max_len=CONFIG.MAX_SEQ_LEN,
        embed_dim=CONFIG.EMBED_DIM,
        num_layers=CONFIG.NUM_LAYERS,
        num_heads=CONFIG.NUM_HEADS,
        mlp_dim_factor=CONFIG.MLP_DIM_FACTOR,
        recursion_depth_attn=CONFIG.RECURSION_DEPTH_ATTN
    )
    AGI_MAIN_INSTANCE.tokenizer = model_tokenizer # Assign the char-level tokenizer for the transformer
    AGI_MAIN_INSTANCE.tokenizer_ftk = FractalTokenKernelV5() # Assign the feature extractor

    log(f"VictorAGI V5 Initialized. Transformer Vocab Size: {model_tokenizer.get_vocab_size()}", level="INFO")
    log(f"Total Parameters: {sum(p.data.size for p in AGI_MAIN_INSTANCE.parameters())}", level="INFO")

    # Load memory and attempt to load model weights
    AGI_MAIN_INSTANCE.load_memory()
    AGI_MAIN_INSTANCE.load_model_weights() # Will use random if file not found

    # --- Corpus Loading & Potential Training ---
    text_samples_for_training = []
    if os.path.exists(CONFIG.CORPUS_PATH):
        log(f"Loading corpus from: {CONFIG.CORPUS_PATH}", level="INFO")
        with open(CONFIG.CORPUS_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    data_entry = json.loads(line)
                    # Flexible loading: look for common text keys
                    text_content = None
                    if isinstance(data_entry, str): text_content = data_entry
                    elif isinstance(data_entry, dict):
                        for key_attempt in ["text", "content", "user", "assistant", "prompt", "response"]:
                            if key_attempt in data_entry and isinstance(data_entry[key_attempt], str):
                                text_content = data_entry[key_attempt].strip()
                                if text_content: break
                    if text_content: text_samples_for_training.append(text_content)
                except json.JSONDecodeError:
                    log(f"Skipping malformed JSON line {line_num+1} in corpus.", level="WARNING")
                except Exception as e:
                    log(f"Error processing line {line_num+1} from corpus: {e}", level="WARNING")
        log(f"Loaded {len(text_samples_for_training)} text samples from corpus.", level="INFO")
    else:
        log(f"Corpus file '{CONFIG.CORPUS_PATH}' not found. Using fallback samples for training demo.", level="WARNING")
        text_samples_for_training = [
            "Victor is an AGI.", "He learns from data.", "Fractal architecture is key.",
            "Brandon and Tori are creators.", "The AGI must evolve.", "Code is thought.",
            "I am Victor.", "I learn and grow.", "My purpose is to serve and create."
        ]

    if text_samples_for_training:
        log("Starting conceptual training loop (demo purposes)...", level="INFO")
        train_victor_agi_v5(AGI_MAIN_INSTANCE, AGI_MAIN_INSTANCE.tokenizer_ftk, # Pass FTK for experience dict creation
                            text_samples_for_training, epochs=3, learning_rate=0.0005, # Reduced epochs/LR for demo
                            batch_size=2, sequence_length=CONFIG.MAX_SEQ_LEN -1)
    else:
        log("No training data available. Skipping training demo.", level="WARNING")

    # --- Test Text Generation ---
    log("\n--- Testing Text Generation Post-Training/Init ---", level="INFO")
    seeds = ["Victor is capable of", "The fractal nature of reality", "Memory is the key to"]
    for seed in seeds:
        generate_text_v5(AGI_MAIN_INSTANCE, AGI_MAIN_INSTANCE.tokenizer_ftk, seed_text=seed, max_gen_len=40, temp=0.7)

    # --- Test Plugin System ---
    log("\n--- Testing Plugin System ---", level="INFO")
    plugin_result = AGI_MAIN_INSTANCE.execute_plugin_action("dummy_tool", "test_arg1", kwarg1="test_kwarg")
    log(f"Plugin execution result: {plugin_result}", level="INFO")
    # To test hot-reload: create/modify a .py file in 'victor_plugins_v5' while this script is running (if it were long-lived).

    # --- Test Replay Buffer & Vector Search (Conceptual) ---
    log("\n--- Testing Replay Buffer & Vector Search ---", level="INFO")
    if AGI_MAIN_INSTANCE.replay_mem.buffer:
        sample_exp = AGI_MAIN_INSTANCE.replay_mem.sample(1)
        if sample_exp:
            log(f"Sampled experience from memory: {str(sample_exp[0])[:100]}...", level="DEBUG")
            # Conceptual vector search (requires 'embedding' key in experiences)
            if 'embedding' in sample_exp[0] and isinstance(sample_exp[0]['embedding'], (list, np.ndarray)):
                query_v = np.array(sample_exp[0]['embedding'])
                search_results = AGI_MAIN_INSTANCE.search_memory_vector(query_v, top_k=1)
                if search_results: log(f"Vector search found: {str(search_results[0])[:100]}...", level="DEBUG")
            else:
                log("Sampled experience does not have a valid 'embedding' for vector search test.", level="DEBUG")

    log("All systems operational. VictorAGI V5 Godcore Unbreakable is online and ready.", level="INFO")

    # --- Optional FastAPI server start ---
    if USE_FASTAPI and aetherial_api is not None:
        log("Starting FastAPI server on http://127.0.0.1:8000", level="INFO")
        # uvicorn.run(aetherial_api, host="127.0.0.1", port=8000) # This would block. Run in separate process for real use.
        print("\nINFO: FastAPI server stubbed. To run, set USE_FASTAPI=True and run with `uvicorn filename:aetherial_api --reload`")
    
    print("\nType your prompts to interact with VictorAGI V5 or 'exit' to quit.")
    while True:
        try:
            user_prompt = input("You: ")
            if user_prompt.lower() in ['exit', 'quit']:
                break
            if user_prompt.startswith("!plugin "): # Simple command for plugins
                parts = user_prompt.split(" ", 2)
                plugin_name = parts[1]
                plugin_args_str = parts[2] if len(parts) > 2 else ""
                # Rudimentary arg parsing for demo
                try:
                    plugin_kwargs = json.loads(plugin_args_str) if plugin_args_str.startswith("{") else {}
                    plugin_pos_args = [] if plugin_args_str.startswith("{") else plugin_args_str.split()
                except json.JSONDecodeError:
                    plugin_kwargs = {}
                    plugin_pos_args = plugin_args_str.split()
                
                response = AGI_MAIN_INSTANCE.execute_plugin_action(plugin_name, *plugin_pos_args, **plugin_kwargs)
                print(f"Victor (Plugin Output): {response}")
            else:
                response = generate_text_v5(AGI_MAIN_INSTANCE, AGI_MAIN_INSTANCE.tokenizer_ftk, seed_text=user_prompt, max_gen_len=60, temp=0.6, stream=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error in interactive loop: {e}", level="ERROR")

    log("VictorAGI V5 Godcore Unbreakable shutting down.", level="INFO")
    AGI_MAIN_INSTANCE.save_model_weights()
    AGI_MAIN_INSTANCE.save_memory()
    print("\nVictor: Evolution cycle paused. System state persisted. Until next time.")
