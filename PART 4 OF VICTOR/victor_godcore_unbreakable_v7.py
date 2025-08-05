#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_godcore_unbreakable_v7.py
VERSION: v7.1.0-GODCORE-HOLYFIELD-ENHANCED
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Fractal-Neural AGI — neural transformer, symbolic fractal memory, QA fallback, live mutation.
         Enhanced autograd, advanced generation sampling, AdamW optimizer.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import os, json, re, random, time, threading, importlib.util
from datetime import datetime
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple, Union # Added for type hinting

# === SYMBOLIC UTILS ===
def tokenize(text: str) -> List[str]:
    """Basic word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())

def clean(text: str) -> str:
    """Removes extra whitespace."""
    return re.sub(r"\s+", " ", text.strip())

# === FRACTAL MEMORY ===
class FractalMemory:
    """Symbolic memory system with timeline and concept indexing."""
    def __init__(self):
        self.timeline: List[Dict[str, Any]] = []
        self.concepts: Dict[str, List[int]] = {}
        self.last_save: datetime = datetime.now()

    def add(self, msg: str, role: str):
        """Adds a message to memory."""
        entry = {"msg": clean(msg), "role": role, "time": datetime.now().isoformat()}
        self.timeline.append(entry)
        for token in tokenize(msg):
            self.concepts.setdefault(token, []).append(len(self.timeline) - 1)

    def recall(self, query: str, topn: int = 5) -> List[Dict[str, Any]]:
        """Recalls relevant memories based on token overlap."""
        tokens = set(tokenize(query))
        scores: Dict[int, int] = {}
        for t in tokens:
            for idx in self.concepts.get(t, []):
                scores[idx] = scores.get(idx, 0) + 1
        
        if not scores and self.timeline: # Fallback to random if no hits
            idxs = random.sample(range(len(self.timeline)), min(topn, len(self.timeline)))
        else:
            idxs = sorted(scores, key=scores.get, reverse=True)[:topn] # type: ignore
        
        return [self.timeline[i] for i in idxs] if self.timeline else []

    def save(self, path: str = "victor_memory.json"):
        """Saves memory to a JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"timeline": self.timeline, "concepts": self.concepts}, f, indent=2)
            self.last_save = datetime.now()
            # print(f"[FractalMemory] Saved to {path}")
        except Exception as e:
            print(f"[FractalMemory] Error saving memory: {e}")

    def load(self, path: str = "victor_memory.json"):
        """Loads memory from a JSON file."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.timeline = data.get("timeline", [])
                # Ensure concepts keys are strings and values are lists of ints
                self.concepts = {str(k): [int(i) for i in v] for k, v in data.get("concepts", {}).items()}
                print(f"[FractalMemory] Loaded from {path}")
            except Exception as e:
                print(f"[FractalMemory] Error loading memory: {e}")
        else:
            print(f"[FractalMemory] No memory file found at {path}. Starting fresh.")

# === CORPUS LOAD (QA) ===
def load_corpus(path: str) -> List[Dict[str, str]]:
    """Loads a QA corpus from a JSONL file."""
    corpus: List[Dict[str, str]] = []
    if not os.path.exists(path):
        print(f"[Victor] Corpus file not found: {path}")
        return corpus
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                pair = json.loads(line)
                if "user" in pair and "assistant" in pair and \
                   isinstance(pair["user"], str) and pair["user"].strip() and \
                   isinstance(pair["assistant"], str) and pair["assistant"].strip():
                    corpus.append({"user": pair["user"].strip(), "assistant": pair["assistant"].strip()})
            except json.JSONDecodeError:
                print(f"[Victor] Warning: Skipping invalid JSON on line {line_num+1} in {path}")
            except Exception as e:
                print(f"[Victor] Warning: Skipping line {line_num+1} in {path} due to error: {e}")
                continue
    print(f"[Victor] Loaded {len(corpus)} user/assistant pairs from {path}")
    return corpus

# === TOKENIZER ===
class VictorTokenizer:
    """Basic character-level tokenizer."""
    def __init__(self, vocab: Optional[Dict[str, int]] = None, 
                 unk_token: str = "<UNK>", pad_token: str = "<PAD>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        if vocab is None:
            chars = list(" " + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + \
                         "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            self.vocab = {self.pad_token: 0, self.unk_token: 1}
            for i, char_val in enumerate(chars):
                self.vocab[char_val] = i + 2 
        else:
            self.vocab = vocab
        
        self.inv_vocab = {i: c for c, i in self.vocab.items()}
        self.unk_token_id = self.vocab.get(self.unk_token, 1) # Default to 1 if not in vocab
        self.pad_token_id = self.vocab.get(self.pad_token, 0) # Default to 0

    def encode(self, text: str, max_len: int, pad_to_max_len: bool = True) -> np.ndarray:
        """Encodes text to token IDs."""
        tokens = [self.vocab.get(c, self.unk_token_id) for c in text]
        tokens = tokens[:max_len] # Truncate if longer
        if pad_to_max_len:
            tokens += [self.pad_token_id] * (max_len - len(tokens))
        return np.array(tokens, dtype=np.int32)

    def decode(self, token_ids: Union[List[int], np.ndarray], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs to text."""
        chars = []
        for token_id_val in token_ids:
            token_id = int(token_id_val) # Ensure it's an int
            if skip_special_tokens and token_id in [self.pad_token_id, self.unk_token_id]:
                continue
            chars.append(self.inv_vocab.get(token_id, self.unk_token if skip_special_tokens else '?'))
        return ''.join(chars)
        
    def get_vocab_size(self) -> int: 
        return len(self.vocab)

    def save_vocab(self, filepath: str):
        """Saves tokenizer vocabulary to a JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[VictorTokenizer] Error saving vocab: {e}")

    @classmethod
    def load_vocab(cls, filepath: str) -> 'VictorTokenizer':
        """Loads tokenizer vocabulary from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            # Determine pad/unk tokens from loaded vocab if possible, else use defaults
            pad_token_found = next((k for k,v in vocab.items() if v == 0), "<PAD>")
            unk_token_found = next((k for k,v in vocab.items() if v == 1), "<UNK>")
            return cls(vocab=vocab, pad_token=pad_token_found, unk_token=unk_token_found)
        except Exception as e:
            print(f"[VictorTokenizer] Error loading vocab: {e}. Returning new tokenizer.")
            return cls()


# === TENSOR/AUTODIFF ===
class Tensor:
    """Custom Tensor class with automatic differentiation."""
    def __init__(self, data: Any, requires_grad: bool = False, creators: Optional[List['Tensor']] = None, 
                 creation_op: Optional[str] = None, creation_meta: Optional[Dict[str, Any]] = None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data: np.ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad: Optional['Tensor'] = None
        if self.requires_grad:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        
        self.creators: Optional[List['Tensor']] = creators
        self.creation_op: Optional[str] = creation_op
        self.creation_meta: Optional[Dict[str, Any]] = creation_meta # For storing extra info for backward pass
        self.backward_hooks: List[Any] = []

    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def zero_grad(self):
        """Resets gradients to zero."""
        if self.grad is not None:
            self.grad.data.fill(0.0)

    def backward(self, grad_output: Optional['Tensor'] = None):
        """Performs backpropagation."""
        if not self.requires_grad:
            return
        
        if grad_output is None:
            if self.data.size == 1: # Scalar output (e.g., loss)
                grad_output = Tensor(np.ones_like(self.data), requires_grad=False)
            else:
                raise ValueError("grad_output must be specified for non-scalar Tensors unless it's the final loss.")
        
        if not isinstance(grad_output, Tensor): # Ensure grad_output is a Tensor
            grad_output = Tensor(grad_output)

        if self.grad is None: # Should have been initialized if requires_grad
             self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)

        # Accumulate incoming gradient, handling broadcasting
        g_shape = self.grad.data.shape
        d_shape = grad_output.data.shape

        if g_shape == d_shape:
            self.grad.data += grad_output.data
        elif grad_output.data.size == 1: # grad_output is scalar
            self.grad.data += grad_output.data.item() # Add scalar to all elements
        elif self.grad.data.size == 1: # self.grad is scalar
            self.grad.data += grad_output.data.sum() # Sum incoming grad to match scalar
        else:
            # General broadcasting: sum grad_output over axes that were broadcasted.
            g_ndim = self.grad.data.ndim
            d_ndim = grad_output.data.ndim
            
            sum_axes = []
            temp_d_shape = list(d_shape)

            # If d_shape has more dimensions than g_shape, sum over leading dimensions of d_shape
            if d_ndim > g_ndim:
                sum_axes.extend(range(d_ndim - g_ndim))
                temp_d_shape = temp_d_shape[d_ndim - g_ndim:] # Consider only trailing dims for broadcast check

            # For remaining dimensions, sum where g_shape is 1 and d_shape > 1
            for i in range(len(temp_d_shape)):
                g_dim_idx = i # Corresponding index in original g_shape
                if g_shape[g_dim_idx] == 1 and temp_d_shape[i] > 1:
                    sum_axes.append(i + (d_ndim - g_ndim)) # Adjust axis index for original d_shape
            
            reduced_grad = grad_output.data
            if sum_axes:
                reduced_grad = reduced_grad.sum(axis=tuple(sum_axes), keepdims=False) # No keepdims for reshape
            
            # Reshape to match self.grad.data.shape for accumulation
            if reduced_grad.shape != g_shape:
                 reduced_grad = reduced_grad.reshape(g_shape)
            self.grad.data += reduced_grad

        for hook in self.backward_hooks: # Apply any registered backward hooks
            hook(self)

        # Backpropagate through creators
        if self.creators is not None:
            op = self.creation_op
            a, b = (self.creators + [None, None])[:2] # Ensure a and b can be unpacked, fill with None if not enough creators

            if op == "add":
                if a is not None and a.requires_grad: a.backward(grad_output)
                if b is not None and b.requires_grad: b.backward(grad_output)
            elif op == "sub":
                if a is not None and a.requires_grad: a.backward(grad_output)
                if b is not None and b.requires_grad: b.backward(Tensor(-grad_output.data))
            elif op == "mul":
                if a is not None and a.requires_grad: a.backward(Tensor(grad_output.data * b.data))
                if b is not None and b.requires_grad: b.backward(Tensor(grad_output.data * a.data))
            elif op == "matmul":
                # Improved matmul backward to handle broadcasting more robustly
                # grad_a = grad_output @ b.T
                # grad_b = a.T @ grad_output
                grad_a_val = grad_output.data @ np.swapaxes(b.data, -1, -2)
                grad_b_val = np.swapaxes(a.data, -1, -2) @ grad_output.data

                # Handle broadcasting for grad_a
                if grad_a_val.shape != a.data.shape:
                    diff_dims = grad_a_val.ndim - a.data.ndim
                    sum_axes = tuple(range(diff_dims)) if diff_dims > 0 else ()
                    grad_a_val = grad_a_val.sum(axis=sum_axes)
                    # Further sum over broadcasted inner dimensions if necessary
                    for i in range(a.data.ndim):
                        if a.data.shape[i] == 1 and grad_a_val.shape[i] > 1:
                            grad_a_val = grad_a_val.sum(axis=i, keepdims=True)
                    if grad_a_val.shape != a.data.shape: grad_a_val = grad_a_val.reshape(a.data.shape)


                # Handle broadcasting for grad_b
                if grad_b_val.shape != b.data.shape:
                    diff_dims = grad_b_val.ndim - b.data.ndim
                    sum_axes = tuple(range(diff_dims)) if diff_dims > 0 else ()
                    grad_b_val = grad_b_val.sum(axis=sum_axes)
                    for i in range(b.data.ndim):
                        if b.data.shape[i] == 1 and grad_b_val.shape[i] > 1:
                            grad_b_val = grad_b_val.sum(axis=i, keepdims=True)
                    if grad_b_val.shape != b.data.shape: grad_b_val = grad_b_val.reshape(b.data.shape)

                if a.requires_grad: a.backward(Tensor(grad_a_val))
                if b.requires_grad: b.backward(Tensor(grad_b_val))

            elif op == "relu":
                relu_grad_val = (a.data > 0).astype(np.float32)
                if a.requires_grad: a.backward(Tensor(grad_output.data * relu_grad_val))
            elif op == "neg":
                if a.requires_grad: a.backward(Tensor(-grad_output.data))
            elif op == "sum":
                # Gradient of sum needs to broadcast grad_output to shape of 'a'
                grad_for_a = np.ones_like(a.data) * grad_output.data # grad_output might be scalar
                if a.requires_grad: a.backward(Tensor(grad_for_a))
            elif op == "mean":
                grad_for_a = np.ones_like(a.data) * grad_output.data / a.data.size
                if a.requires_grad: a.backward(Tensor(grad_for_a))
            elif op == "var": # New: backward for variance
                # Variance: V = sum((X - mu)^2) / N
                # dV/dX_i = 2 * (X_i - mu) / N
                # This needs to be scaled by grad_output
                mu = self.creation_meta['mean'] # Mean was stored during forward
                N = a.data.size if self.creation_meta.get('axis') is None else a.data.shape[self.creation_meta['axis']]
                # N = np.prod(a.data.shape) / mu.size # More general N if axis is involved
                
                # Correct N for variance calculation based on axis
                if self.creation_meta.get('axis') is not None:
                    N_reduce = np.prod([a.data.shape[ax] for ax in self.creation_meta['axis_tuple']])
                else:
                    N_reduce = a.data.size
                
                grad_var_val = 2 * (a.data - mu) / N_reduce
                # grad_output needs to be broadcast to match grad_var_val if var was reduced
                broadcasted_grad_output = np.ones_like(a.data) * grad_output.data

                if a.requires_grad: a.backward(Tensor(broadcasted_grad_output * grad_var_val))

            elif op == "transpose":
                original_axes = self.creation_meta.get('axes')
                if original_axes is None: # Simple .T
                    if a.requires_grad: a.backward(Tensor(grad_output.data.T))
                else: # Transpose with specified axes, need to reverse it
                    inv_axes = np.argsort(original_axes)
                    if a.requires_grad: a.backward(Tensor(np.transpose(grad_output.data, inv_axes)))
            elif op == "div":
                grad_a_val = grad_output.data / (b.data + 1e-9) # Added epsilon
                grad_b_val = -grad_output.data * a.data / ((b.data**2) + 1e-9) # Added epsilon
                if a.requires_grad: a.backward(Tensor(grad_a_val))
                if b.requires_grad: b.backward(Tensor(grad_b_val))
            elif op == "exp":
                if a.requires_grad: a.backward(Tensor(grad_output.data * self.data)) # self.data is exp(a.data)
            elif op == "log":
                if a.requires_grad: a.backward(Tensor(grad_output.data / (a.data + 1e-9))) # Added epsilon
            elif op == "sigmoid":
                grad_sig_val = self.data * (1 - self.data) # self.data is sigmoid(a.data)
                if a.requires_grad: a.backward(Tensor(grad_output.data * grad_sig_val))
            elif op == "tanh":
                grad_tanh_val = 1 - self.data**2 # self.data is tanh(a.data)
                if a.requires_grad: a.backward(Tensor(grad_output.data * grad_tanh_val))
            elif op == "pow":
                # d(a^b)/da = b * a^(b-1)
                # d(a^b)/db = a^b * log(a) = self.data * log(a)
                grad_base_val = b.data * (a.data ** (b.data - 1))
                if a.requires_grad: a.backward(Tensor(grad_output.data * grad_base_val))
                if b.requires_grad:
                    log_a_data = np.log(np.maximum(a.data, 1e-9)) # Stability for log
                    grad_exp_val = self.data * log_a_data
                    b.backward(Tensor(grad_output.data * grad_exp_val))
            elif op == "softmax_cross_entropy":
                logits, targets, softmax_outputs = self.extra_ctx # type: ignore
                batch, seq, _ = softmax_outputs.shape
                grad_logits_val = softmax_outputs.copy()
                grad_logits_val[np.arange(batch)[:,None], np.arange(seq), targets] -= 1
                grad_logits_val /= (batch * seq) # Average over batch and sequence
                if logits.requires_grad: logits.backward(Tensor(grad_logits_val * grad_output.data))


    def __add__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data + other.data, requires_grad=requires_grad, creators=[self, other], creation_op="add")

    def __mul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data * other.data, requires_grad=requires_grad, creators=[self, other], creation_op="mul")

    def __sub__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data - other.data, requires_grad=requires_grad, creators=[self, other], creation_op="sub")

    def __truediv__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data / (other.data + 1e-9), requires_grad=requires_grad, creators=[self, other], creation_op="div")

    def __neg__(self) -> 'Tensor':
        return Tensor(-self.data, requires_grad=self.requires_grad, creators=[self], creation_op="neg")

    def matmul(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data @ other.data, requires_grad=requires_grad, creators=[self, other], creation_op="matmul")

    def __matmul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        return self.matmul(other)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        return Tensor(out_data, requires_grad=self.requires_grad, creators=[self], creation_op="sum",
                      creation_meta={'axis': axis, 'original_shape': self.data.shape, 'keepdims': keepdims})

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        return Tensor(out_data, requires_grad=self.requires_grad, creators=[self], creation_op="mean",
                      creation_meta={'axis': axis, 'original_shape': self.data.shape, 'keepdims': keepdims})
    
    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ddof: int = 0) -> 'Tensor':
        """Computes variance. ddof=0 for population variance (biased)."""
        # For autograd, this should ideally be implemented via mean and pow, or have its own backward rule.
        # V = E[(X - E[X])^2]
        mean_val = self.data.mean(axis=axis, keepdims=True) # Keepdims for broadcasting with self.data
        squared_diff = (self.data - mean_val)**2
        # The mean of squared_diff is the variance
        var_data = squared_diff.mean(axis=axis, keepdims=keepdims) # This mean is over the same axis as original mean

        # Store mean_val for backward pass of variance
        # The axis_tuple is important for correctly calculating N in backward.
        axis_tuple = axis if isinstance(axis, tuple) else ((axis,) if axis is not None else tuple(range(self.data.ndim)))

        return Tensor(var_data, requires_grad=self.requires_grad, creators=[self], creation_op="var",
                      creation_meta={'axis': axis, 'axis_tuple': axis_tuple, 'original_shape': self.data.shape, 
                                     'keepdims': keepdims, 'mean': mean_val, 'ddof': ddof})


    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        transposed_data = self.data.T if axes is None else np.transpose(self.data, axes)
        return Tensor(transposed_data, requires_grad=self.requires_grad, creators=[self], creation_op="transpose",
                      creation_meta={'axes': axes})
    @property
    def T(self) -> 'Tensor':
        return self.transpose() # Default transpose (swap all axes)

    def exp(self) -> 'Tensor':
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="exp")

    def log(self) -> 'Tensor':
        return Tensor(np.log(self.data + 1e-9), requires_grad=self.requires_grad, creators=[self], creation_op="log")

    def sigmoid(self) -> 'Tensor':
        s = 1 / (1 + np.exp(-self.data))
        return Tensor(s, requires_grad=self.requires_grad, creators=[self], creation_op="sigmoid")

    def tanh(self) -> 'Tensor':
        return Tensor(np.tanh(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="tanh")

    def __pow__(self, exponent: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(exponent, Tensor): exponent = Tensor(np.array(exponent, dtype=np.float32))
        requires_grad = self.requires_grad or exponent.requires_grad
        return Tensor(self.data ** exponent.data, requires_grad=requires_grad, creators=[self, exponent], creation_op="pow")

    def relu(self) -> 'Tensor':
        return Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, creators=[self], creation_op="relu")

    def softmax(self, axis: int = -1) -> 'Tensor': # Usually not part of autograd graph if loss handles it
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out_data = e_x / np.sum(e_x, axis=axis, keepdims=True)
        return Tensor(out_data, requires_grad=False) # Typically False for combined loss

    def __repr__(self) -> str:
        return f"VictorTensor(shape={self.data.shape}, requires_grad={self.requires_grad})\n{self.data}"

# === MODULES BASE ===
class Module:
    """Base class for all neural network modules."""
    def parameters(self) -> List[Tensor]: return []
    def __call__(self, *args, **kwargs) -> Tensor: return self.forward(*args, **kwargs)
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

class Linear(Module):
    """Standard fully connected linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Kaiming Uniform initialization (He initialization for ReLU-like activations)
        limit = np.sqrt(6.0 / in_features) 
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32), requires_grad=True)
        self.bias: Optional[Tensor] = None
        if bias:
            # Initialize bias to zeros, or small constant, or based on Kaiming for bias
            self.bias = Tensor(np.zeros((1, out_features), dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias # Broadcasting will handle (1, out_features) bias
        return out
        
    def parameters(self) -> List[Tensor]:
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

class LayerNorm(Module):
    """Layer Normalization."""
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape_tuple = (normalized_shape,)
        else:
            self.normalized_shape_tuple = tuple(normalized_shape) # Ensure it's a tuple
        
        self.eps = eps
        # Gamma and Beta shape should match the normalized_shape for element-wise ops
        param_shape = self.normalized_shape_tuple

        self.gamma = Tensor(np.ones(param_shape, dtype=np.float32), requires_grad=True)
        self.beta  = Tensor(np.zeros(param_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Normalize over the last 'len(self.normalized_shape_tuple)' dimensions
        # These are the dimensions specified by normalized_shape
        axes_to_normalize = tuple(range(x.data.ndim - len(self.normalized_shape_tuple), x.data.ndim))
        
        mean_x = x.mean(axis=axes_to_normalize, keepdims=True)
        var_x = x.var(axis=axes_to_normalize, keepdims=True) # ddof=0 for population variance
        
        # For std, ensure it's a Tensor operation if var_x is a Tensor
        # std_x_data = np.sqrt(var_x.data + self.eps)
        # std_x = Tensor(std_x_data, requires_grad=var_x.requires_grad, creators=[var_x] if var_x.requires_grad else None, creation_op="sqrt_stub")
        # Simpler: (var_x + self.eps) ** 0.5
        std_x = (var_x + Tensor(self.eps)) ** 0.5

        norm_x = (x - mean_x) / (std_x + Tensor(1e-9)) # Add epsilon to std_x as well for safety
        
        return self.gamma * norm_x + self.beta
        
    def parameters(self) -> List[Tensor]: 
        return [self.gamma, self.beta]


class ReLU(Module):
    """ReLU activation function."""
    def __init__(self): super().__init__()
    def forward(self, x: Tensor) -> Tensor: return x.relu()

class Sequential(Module):
    """A sequential container for modules."""
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers) # Store layers
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: 
            x = layer(x)
        return x
    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for l in self.layers:
            if hasattr(l, "parameters"): 
                params.extend(l.parameters())
        return params

# === FRACTAL ATTENTION + BLOCK ===
class FractalAttention(Module):
    """Fractal Attention mechanism with recursive application."""
    def __init__(self, embed_dim: int, num_heads: int, recursion_depth: int = 1): # Default recursion to 1
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.recursion_depth = max(1, recursion_depth) # Ensure at least one pass

        self.Wq = Linear(embed_dim, embed_dim, bias=False)
        self.Wk = Linear(embed_dim, embed_dim, bias=False)
        self.Wv = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Make x require grad if any of the weights do, to ensure graph connection
        # This is implicitly handled if x is an output of a previous layer with requires_grad=True
        # or if x itself is an input parameter that requires_grad.

        current_x = x 
        for _ in range(self.recursion_depth):
            batch_size, seq_len, embed_dim_ = current_x.shape()

            # Project and reshape Q, K, V
            q_proj = self.Wq(current_x) # Tensor
            k_proj = self.Wk(current_x) # Tensor
            v_proj = self.Wv(current_x) # Tensor

            # Reshape for multi-head attention: (batch_size, num_heads, seq_len, head_dim)
            # This part needs careful handling if we want to keep it all Tensor ops.
            # For now, using .data and re-wrapping, but this breaks fine-grained grad for reshape/transpose.
            q_reshaped = q_proj.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k_reshaped = k_proj.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v_reshaped = v_proj.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            
            # Calculate attention scores: (batch_size, num_heads, seq_len, seq_len)
            # This matmul should ideally be a Tensor operation
            attn_scores_data = np.matmul(q_reshaped, k_reshaped.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
            attn_scores_tensor = Tensor(attn_scores_data) # Wrap for softmax

            # Softmax along the last dimension (keys)
            attn_weights_tensor = attn_scores_tensor.softmax(axis=-1) 
            
            # Weighted sum of values: (batch_size, num_heads, seq_len, head_dim)
            # This matmul should also ideally be a Tensor operation
            attn_output_data = np.matmul(attn_weights_tensor.data, v_reshaped)
            
            # Reshape and combine heads: (batch_size, seq_len, embed_dim)
            attn_output_reshaped = attn_output_data.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim_)
            
            current_x = Tensor(attn_output_reshaped, requires_grad=x.requires_grad) # Re-wrap, maintain requires_grad status
                                                                                # Creators link will be to the original x through Wq,Wk,Wv for this iteration
                                                                                # if Q,K,V were Tensors.
                                                                                # This simplified re-wrapping won't capture the full recursive graph for autograd.
                                                                                # For full autograd, each step would need to be Tensor ops.

        return self.out_proj(current_x) # Final output projection

    def parameters(self) -> List[Tensor]:
        return (self.Wq.parameters() + self.Wk.parameters() +
                self.Wv.parameters() + self.out_proj.parameters())

class VictorSuperBlock(Module):
    """Combines FractalAttention and MLP with residuals and LayerNorm."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim_factor: int = 4, recursion_depth: int = 1):
        super().__init__()
        self.fractal_attn = FractalAttention(embed_dim, num_heads, recursion_depth)
        self.norm1 = LayerNorm(embed_dim) # normalized_shape should be embed_dim
        mlp_dim = embed_dim * mlp_dim_factor
        self.mlp = Sequential(
            Linear(embed_dim, mlp_dim),
            ReLU(),
            Linear(mlp_dim, embed_dim)
        )
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Ensure x requires_grad if parameters do, for graph connection
        # x_input_for_attn = Tensor(x.data, requires_grad=x.requires_grad or any(p.requires_grad for p in self.fractal_attn.parameters()))
        
        attn_out = self.fractal_attn(x)
        x = x + attn_out # Residual connection
        x = self.norm1(x)
        
        # x_input_for_mlp = Tensor(x.data, requires_grad=x.requires_grad or any(p.requires_grad for p in self.mlp.parameters()))
        mlp_out = self.mlp(x)
        x = x + mlp_out # Residual connection
        x = self.norm2(x)
        return x

    def parameters(self) -> List[Tensor]:
        return (self.fractal_attn.parameters() + self.norm1.parameters() +
                self.mlp.parameters() + self.norm2.parameters())

# === AGI CORE (FUSION) ===
class VictorAGI(Module): # Inherit from Module to use its parameter collection
    """Core AGI class combining symbolic memory, QA corpus, and a neural Transformer."""
    def __init__(self, corpus: List[Dict[str, str]], memory: FractalMemory, tokenizer: VictorTokenizer, 
                 max_len: int = 64, embed_dim: int = 128, num_layers: int = 3, 
                 num_heads: int = 4, mlp_dim_factor: int = 4, recursion_depth: int = 1):
        super().__init__()
        self.corpus = corpus
        self.memory = memory
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Store hyperparameters for potential saving/loading
        self.config = {
            "max_len": max_len, "embed_dim": embed_dim, "num_layers": num_layers,
            "num_heads": num_heads, "mlp_dim_factor": mlp_dim_factor, 
            "recursion_depth": recursion_depth, "vocab_size": tokenizer.get_vocab_size()
        }

        vocab_size = tokenizer.get_vocab_size()
        self.token_embedding = Tensor(np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.pe = Tensor(self._positional_encoding(max_len, embed_dim), requires_grad=False)
        
        self.blocks = Sequential(
            *[VictorSuperBlock(embed_dim, num_heads, mlp_dim_factor, recursion_depth) for _ in range(num_layers)]
        )
        self.final_norm = LayerNorm(embed_dim)
        self.out_proj = Linear(embed_dim, vocab_size)

    def _positional_encoding(self, seq_len: int, embed_dim: int) -> np.ndarray:
        """Generates sinusoidal positional encodings."""
        pe = np.zeros((1, seq_len, embed_dim), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float32) * -(np.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        return pe

    def forward(self, input_ids: np.ndarray) -> Tensor: # This is the neural model's forward pass
        """Forward pass for the internal Transformer model."""
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.max_len:
            # Truncate or raise error. For now, truncate.
            input_ids = input_ids[:, :self.max_len]
            seq_len = self.max_len
            # print(f"[VictorAGI] Warning: Input sequence truncated to max_len {self.max_len}")

        # Ensure input_ids are within vocab_size bounds
        input_ids_clipped = np.clip(input_ids, 0, self.token_embedding.data.shape[0] - 1)
        embedded_tokens = self.token_embedding.data[input_ids_clipped] 
        
        pos_enc = self.pe.data[:, :seq_len, :]
        
        x_data = embedded_tokens + pos_enc
        # Input to the first block should require grad if embeddings do
        x = Tensor(x_data, requires_grad=self.token_embedding.requires_grad) 

        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits

    def parameters(self) -> List[Tensor]:
        """Collects all learnable parameters of the neural model."""
        params = [self.token_embedding]
        params.extend(self.blocks.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.out_proj.parameters())
        return params

    def neural_generate(self, prompt: str, gen_len: int = 32, 
                        temperature: float = 0.7, top_k: Optional[int] = None, 
                        top_p: Optional[float] = None) -> str:
        """Generates text using the neural model with advanced sampling."""
        # For generation, we don't need gradients through the generation loop itself.
        # Gradients are for training the model parameters.
        
        current_tokens_ids = self.tokenizer.encode(prompt, max_len=self.max_len -1, pad_to_max_len=False) # Leave space for at least one generated token
        generated_ids_list = list(current_tokens_ids)

        for _ in range(gen_len):
            # Prepare input for the model: pad current sequence to max_len
            input_sequence_padded = np.array(
                generated_ids_list + [self.tokenizer.pad_token_id] * (self.max_len - len(generated_ids_list)),
                dtype=np.int32
            ).reshape(1, -1) # Batch size 1

            logits = self.forward(input_sequence_padded) # Get logits from the model
            
            # Get logits for the *next* token (at the end of the current generated sequence)
            # Logits shape: (1, max_len, vocab_size)
            # Index for next token's logits: len(generated_ids_list) - 1
            next_token_logits_data = logits.data[0, len(generated_ids_list)-1, :] 

            # Apply temperature scaling
            if temperature > 0 and temperature != 1.0:
                next_token_logits_data = next_token_logits_data / temperature
            
            # Calculate probabilities using softmax
            exp_logits = np.exp(next_token_logits_data - np.max(next_token_logits_data)) # Stability
            probabilities = exp_logits / np.sum(exp_logits)

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_indices = np.argsort(probabilities)[-top_k:]
                filtered_probs = np.zeros_like(probabilities)
                filtered_probs[top_k_indices] = probabilities[top_k_indices]
                if np.sum(filtered_probs) > 0: # Avoid division by zero if all top_k are zero prob
                    probabilities = filtered_probs / np.sum(filtered_probs)
                else: # Fallback if all top_k probs are zero (highly unlikely with softmax)
                    probabilities[np.argmax(next_token_logits_data)] = 1.0


            # Apply top-p (nucleus) sampling
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_indices = np.argsort(probabilities)[::-1] # Sort descending
                sorted_probabilities = probabilities[sorted_indices]
                cumulative_probs = np.cumsum(sorted_probabilities)
                
                # Find indices to remove (those beyond cumulative prob top_p)
                indices_to_keep = cumulative_probs <= top_p
                # Ensure at least one token is kept
                if not np.any(indices_to_keep):
                    indices_to_keep[0] = True 
                
                # Create a mask for probabilities
                final_probs_mask = np.zeros_like(probabilities, dtype=bool)
                final_probs_mask[sorted_indices[indices_to_keep]] = True
                
                probabilities[~final_probs_mask] = 0 # Zero out probabilities not in the nucleus
                if np.sum(probabilities) > 0:
                    probabilities = probabilities / np.sum(probabilities) # Re-normalize
                else: # Fallback if nucleus is empty (e.g. top_p too small)
                    probabilities[np.argmax(next_token_logits_data)] = 1.0


            # Sample the next token ID
            next_token_id = np.random.choice(len(probabilities), p=probabilities)

            if next_token_id == self.tokenizer.pad_token_id: # Or a specific <EOS> token
                break 
            
            generated_ids_list.append(next_token_id)

            if len(generated_ids_list) >= self.max_len: # Stop if max_len reached
                break
        
        # Decode generated part (after the initial prompt)
        generated_part_ids = generated_ids_list[len(current_tokens_ids):]
        return self.tokenizer.decode(generated_part_ids, skip_special_tokens=True)


    def symbolic_response(self, user_input: str) -> str:
        """Generates a response using symbolic memory and QA corpus."""
        recalls = self.memory.recall(user_input, topn=3)
        recall_snips_list = [x["msg"] for x in recalls if x["role"] == "assistant"]
        
        # Search QA corpus
        scored_corpus_entries: List[Tuple[int, Dict[str, str]]] = []
        user_tokens = set(tokenize(user_input))
        for entry in self.corpus:
            score = len(user_tokens.intersection(tokenize(entry["user"])))
            if score > 0: 
                scored_corpus_entries.append((score, entry))
        
        scored_corpus_entries.sort(reverse=True, key=lambda x: x[0]) # Sort by score

        if scored_corpus_entries:
            base_reply = scored_corpus_entries[0][1]["assistant"]
        elif recall_snips_list: # Use most recent recall if multiple
            base_reply = recall_snips_list[0] 
        else:
            base_reply = "I'm still learning about that. Can you tell me more?"
        return base_reply

    def respond(self, user_input: str, neural_chance: float = 0.6) -> str:
        """Main response generation method, blending neural and symbolic approaches."""
        self.memory.add(user_input, "user")
        reply = ""

        if random.random() < neural_chance:
            try:
                neural_out = self.neural_generate(user_input, gen_len=random.randint(20, 50), 
                                                  temperature=random.uniform(0.6, 0.9), 
                                                  top_k=random.choice([None, 40, 50]),
                                                  top_p=random.choice([None, 0.9, 0.95]))
                if neural_out and len(clean(neural_out).strip("?.,! ")) > 3: # Check for meaningful output
                    reply = clean(neural_out)
                else:
                    reply = self.symbolic_response(user_input)
            except Exception as e:
                print(f"[VictorAGI] Neural generation error: {e}. Falling back to symbolic.")
                reply = self.symbolic_response(user_input)
        else:
            reply = self.symbolic_response(user_input)

        # Fractal flavor and memory update
        fractal_elements = [
            f"Bando says: {reply}",
            f"[Victor memory ref: {random.choice(list(self.memory.concepts.keys())) if self.memory.concepts else 'init'}]",
            f"(V.{random.randint(70,79)}.Fractal.{random.randint(100,999)})"
        ]
        if random.random() > 0.65:
            fractal_elements.append("Victor's evolution is a beautiful, chaotic dance.")
        
        final_response = " ".join(fractal_elements)
        self.memory.add(final_response, "assistant")
        
        # Auto-save memory periodically or based on interaction count
        if (datetime.now() - self.memory.last_save).total_seconds() > 300 or len(self.memory.timeline) % 10 == 0:
             self.memory.save()
             
        return final_response

# === OPTIMIZER & LOSS (for potential training) ===
class AdamW:
    """AdamW Optimizer."""
    def __init__(self, parameters: List[Tensor], lr: float = 1e-3, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = [Tensor(np.zeros_like(p.data), requires_grad=False) for p in self.parameters]
        self.v = [Tensor(np.zeros_like(p.data), requires_grad=False) for p in self.parameters]
        self.t = 0

    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None or p.grad.data is None:
                continue
            grad_data = p.grad.data
            
            if self.weight_decay != 0: # Decoupled weight decay
                p.data -= self.lr * self.weight_decay * p.data

            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad_data
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (grad_data ** 2)

            m_hat_data = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat_data = self.v[i].data / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat_data / (np.sqrt(v_hat_data) + self.epsilon)
            
    def zero_grad(self):
        """Clears gradients of all parameters."""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.fill(0.0)

class SoftmaxCrossEntropyLoss:
    """Softmax Cross-Entropy Loss function."""
    def __call__(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        max_logits = np.max(logits.data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits.data - max_logits)
        softmax_outputs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        batch_size, seq_len, _ = softmax_outputs.shape
        
        target_probs = softmax_outputs[np.arange(batch_size)[:,None], np.arange(seq_len), targets]
        log_probs = -np.log(target_probs + 1e-9) # Epsilon for stability
        loss_value = np.mean(log_probs) 
        
        loss_tensor = Tensor(loss_value, requires_grad=True, creators=[logits], creation_op="softmax_cross_entropy")
        loss_tensor.extra_ctx = (logits, targets, softmax_outputs) # type: ignore
        return loss_tensor

# === MAIN CLI ===
def main_cli_loop():
    """Main command-line interface loop for Victor AGI."""
    print("Initializing Victor GODCORE HOLYFIELD (v7.1.0)...")
    print("Type 'exit', 'quit', or 'bye' to end session. Ctrl+C also works.")
    print("Memory will be auto-saved periodically.\n")

    memory = FractalMemory()
    memory.load("victor_holyfield_memory.json") # Use a distinct memory file

    corpus_path = "bando_corpus.jsonl" # Ensure this file exists or is created
    if not os.path.exists(corpus_path):
        print(f"Warning: Corpus file '{corpus_path}' not found. Creating an empty one.")
        with open(corpus_path, 'w') as f:
            # Optionally add a dummy entry
            # f.write(json.dumps({"user": "hello", "assistant": "Hello there!"}) + "\n")
            pass 
    corpus = load_corpus(corpus_path)

    # Tokenizer setup
    tokenizer_vocab_path = "victor_holyfield_tokenizer_vocab.json"
    if os.path.exists(tokenizer_vocab_path):
        tokenizer = VictorTokenizer.load_vocab(tokenizer_vocab_path)
        print(f"Loaded tokenizer from {tokenizer_vocab_path}")
    else:
        tokenizer = VictorTokenizer() # Creates default char-level
        tokenizer.save_vocab(tokenizer_vocab_path)
        print(f"Initialized new tokenizer and saved to {tokenizer_vocab_path}")
    
    # AGI Core initialization
    # Hyperparameters can be tuned or loaded from a config
    victor_agi = VictorAGI(
        corpus=corpus, 
        memory=memory, 
        tokenizer=tokenizer,
        max_len=64,         # Max sequence length for the neural model
        embed_dim=128,      # Embedding dimension
        num_layers=3,       # Number of VictorSuperBlocks
        num_heads=4,        # Attention heads
        mlp_dim_factor=4,   # MLP hidden layer size factor
        recursion_depth=1   # FractalAttention recursion
    )
    print(f"Victor AGI initialized. Neural model params: {sum(p.data.size for p in victor_agi.parameters())}")
    print("---------------------------------------\n")

    try:
        while True:
            user_input = input("👤 You: ")
            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                break
            if not user_input.strip():
                continue
            
            start_time = time.time()
            victor_response = victor_agi.respond(user_input)
            end_time = time.time()
            
            print(f"🤖 Victor ({end_time - start_time:.2f}s): {victor_response}")

    except KeyboardInterrupt:
        print("\nVictor: Session interrupted by user.")
    finally:
        print("Victor: Shutting down. Evolution never sleeps.")
        victor_agi.memory.save("victor_holyfield_memory.json") # Final save

if __name__ == "__main__":
    main_cli_loop()
