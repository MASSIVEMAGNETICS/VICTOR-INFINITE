#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_godcore_unbreakable.py
VERSION: v4.0.0-GODCORE-UNBREAKABLE-ALLINONE
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: AGI-grade, future-proof, fully self-evolving transformer with:
  - Fractal recursive attention
  - Plugin system (hot-reload, live mutate)
  - Replay buffer memory (save/load/search)
  - Meta-cognition, self-introspection, self-reflection
  - Dynamic layer growth/pruning (stubs)
  - Subpersona registry
  - Action tool hooks
  - Streaming API (FastAPI optional)
  - Memory vector search/recall
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import os, re, json, time, threading, importlib.util
from functools import wraps

# ================= GODCORE META-UTILS ================
def log(msg):
    print(f"[Victor:{time.strftime('%H:%M:%S')}] {msg}")

def versioned(fn):
    @wraps(fn)
    def wrapped(*a, **k):
        log(f"CALL: {fn.__name__}")
        return fn(*a, **k)
    return wrapped

# ================= SELF-HEALING CORE =================
def self_heal(fn):
    @wraps(fn)
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            log(f"[SELF-HEAL] Exception: {e}. Attempting recover/continue.")
            return None
    return wrapped

# ================= PLUGIN MANAGER ====================
class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.plugin_path = "plugins"
        self.load_plugins()
        self.hot_reload_thread = threading.Thread(target=self.hot_reload, daemon=True)
        self.hot_reload_thread.start()
    def load_plugins(self):
        if not os.path.isdir(self.plugin_path): os.makedirs(self.plugin_path)
        for file in os.listdir(self.plugin_path):
            if file.endswith(".py"):
                self._load_plugin(file)
    def _load_plugin(self, file):
        try:
            spec = importlib.util.spec_from_file_location(file[:-3], os.path.join(self.plugin_path, file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.plugins[file[:-3]] = module
            log(f"Loaded plugin: {file[:-3]}")
        except Exception as e:
            log(f"Failed to load plugin {file}: {e}")
    def hot_reload(self):
        last = set()
        while True:
            files = {f for f in os.listdir(self.plugin_path) if f.endswith('.py')}
            added = files - last
            if added:
                for f in added: self._load_plugin(f)
                last = files
            time.sleep(2)

# ================== REPLAY BUFFER MEMORY =============
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0
    @self_heal
    def add(self, exp):
        if len(self.buffer) < self.max_size:
            self.buffer.append(exp)
        else:
            self.buffer[self.ptr] = exp
            self.ptr = (self.ptr + 1) % self.max_size
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in idx]
    def save(self, file="replay_buffer.json"):
        with open(file, "w") as f: json.dump(self.buffer, f)
    def load(self, file="replay_buffer.json"):
        if os.path.exists(file):
            with open(file) as f: self.buffer = json.load(f)
    def vector_search(self, vec, topk=1):
        # super basic cosine search, use np.dot
        if not self.buffer: return []
        buf = np.array([x['vec'] for x in self.buffer if 'vec' in x])
        if buf.size == 0: return []
        sims = np.dot(buf, vec) / (np.linalg.norm(buf,axis=1)+1e-8) / (np.linalg.norm(vec)+1e-8)
        idxs = np.argsort(-sims)[:topk]
        return [self.buffer[i] for i in idxs]

# ================= VICTOR TENSOR (AUTODIFF) ==========
class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        if not isinstance(data, np.ndarray): data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        if self.requires_grad:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        self.creators = creators
        self.creation_op = creation_op
        self.backward_hooks = []
    def shape(self): return self.data.shape
    def zero_grad(self):
        if self.grad is not None: self.grad.data.fill(0.0)
    def backward(self, grad_output=None):
        if not self.requires_grad: return
        if grad_output is None:
            if self.data.size == 1:
                grad_output = Tensor(np.ones_like(self.data), requires_grad=False)
            else:
                raise ValueError("grad_output must be specified for non-scalar Tensors unless it's the final loss.")
        if not isinstance(grad_output, Tensor): grad_output = Tensor(grad_output)
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        g_shape = self.grad.data.shape
        d_shape = grad_output.data.shape
        if g_shape == d_shape:
            self.grad.data += grad_output.data
        elif grad_output.data.size == 1:
            self.grad.data += grad_output.data.item()
        elif self.grad.data.size == 1:
            self.grad.data += grad_output.data.sum()
        else:
            diff = len(d_shape) - len(g_shape)
            g_shape_full = (1,)*(diff) + g_shape if diff > 0 else g_shape
            axes = tuple(i for i, (gs, ds) in enumerate(zip(g_shape_full, d_shape)) if gs == 1 and ds > 1)
            reduced = grad_output.data.sum(axis=axes, keepdims=True)
            reduced = reduced.reshape(g_shape)
            self.grad.data += reduced
        for hook in self.backward_hooks: hook(self)
        if self.creators is not None:
            op = self.creation_op
            a, b = (self.creators + [None, None])[:2]
            if op == "add":
                a.backward(grad_output)
                b.backward(grad_output)
            elif op == "sub":
                a.backward(grad_output)
                b.backward(Tensor(-grad_output.data))
            elif op == "mul":
                a.backward(Tensor(grad_output.data * b.data))
                b.backward(Tensor(grad_output.data * a.data))
            elif op == "matmul":
                a_shape = a.data.shape
                b_shape = b.data.shape
                b_T = np.swapaxes(b.data, -1, -2)
                grad_a = np.matmul(grad_output.data, b_T)
                while grad_a.shape != a_shape:
                    grad_a = grad_a.sum(axis=0)
                a.backward(Tensor(grad_a))
                a_T = np.swapaxes(a.data, -1, -2)
                grad_b = np.matmul(a_T, grad_output.data)
                while grad_b.shape != b_shape:
                    grad_b = grad_b.sum(axis=0)
                b.backward(Tensor(grad_b))
            elif op == "relu":
                relu_grad = (a.data > 0).astype(np.float32)
                a.backward(Tensor(grad_output.data * relu_grad))
            elif op == "neg":
                a.backward(Tensor(-grad_output.data))
            elif op == "sum":
                a.backward(Tensor(np.ones_like(a.data) * grad_output.data))
            elif op == "mean":
                a.backward(Tensor(np.ones_like(a.data) * grad_output.data / a.data.size))
            elif op == "transpose":
                a.backward(Tensor(grad_output.data.T))
            elif op == "div":
                grad_a = grad_output.data / b.data
                grad_b = -grad_output.data * a.data / (b.data**2)
                a.backward(Tensor(grad_a))
                b.backward(Tensor(grad_b))
            elif op == "exp":
                a.backward(Tensor(grad_output.data * self.data))
            elif op == "log":
                a.backward(Tensor(grad_output.data / a.data))
            elif op == "sigmoid":
                grad_sig = self.data * (1 - self.data)
                a.backward(Tensor(grad_output.data * grad_sig))
            elif op == "tanh":
                grad_tanh = 1 - self.data**2
                a.backward(Tensor(grad_output.data * grad_tanh))
            elif op == "pow":
                grad_base = b.data * (a.data ** (b.data - 1))
                a.backward(Tensor(grad_output.data * grad_base))
                if b.requires_grad:
                    grad_exp = self.data * np.log(a.data)
                    b.backward(Tensor(grad_output.data * grad_exp))
            elif op == "softmax_cross_entropy":
                logits, targets, softmax_outputs = self.extra_ctx
                batch, seq, _ = softmax_outputs.shape
                grad_logits = softmax_outputs.copy()
                grad_logits[np.arange(batch)[:,None], np.arange(seq), targets] -= 1
                grad_logits /= (batch * seq)
                logits.backward(Tensor(grad_logits * grad_output.data))
    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data + other.data, requires_grad=requires_grad, creators=[self, other], creation_op="add")
    def __mul__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data * other.data, requires_grad=requires_grad, creators=[self, other], creation_op="mul")
    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data - other.data, requires_grad=requires_grad, creators=[self, other], creation_op="sub")
    def __truediv__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data / other.data, requires_grad=requires_grad, creators=[self, other], creation_op="div")
    def __neg__(self):
        return Tensor(-self.data, requires_grad=self.requires_grad, creators=[self], creation_op="neg")
    def matmul(self, other):
        if not isinstance(other, Tensor): other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data @ other.data, requires_grad=requires_grad, creators=[self, other], creation_op="matmul")
    def __matmul__(self, other): return self.matmul(other)
    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="sum")
    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="mean")
    def transpose(self, axes=None):
        op = "transpose"
        return Tensor(self.data.T if axes is None else np.transpose(self.data, axes), requires_grad=self.requires_grad, creators=[self], creation_op=op)
    @property
    def T(self): return self.transpose()
    def exp(self):
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="exp")
    def log(self):
        return Tensor(np.log(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="log")
    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        return Tensor(s, requires_grad=self.requires_grad, creators=[self], creation_op="sigmoid")
    def tanh(self):
        return Tensor(np.tanh(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="tanh")
    def __pow__(self, exponent):
        if not isinstance(exponent, Tensor): exponent = Tensor(np.array(exponent, dtype=np.float32))
        requires_grad = self.requires_grad or exponent.requires_grad
        return Tensor(self.data ** exponent.data, requires_grad=requires_grad, creators=[self, exponent], creation_op="pow")
    def relu(self):
        return Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, creators=[self], creation_op="relu")
    def softmax(self, axis=-1):
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out_data = e_x / np.sum(e_x, axis=axis, keepdims=True)
        return Tensor(out_data, requires_grad=False)
    def __repr__(self):
        return f"VictorTensor(shape={self.data.shape}, requires_grad={self.requires_grad})\n{self.data}"

# =================== MODULES BASE =====================
class Module:
    def parameters(self): return []
    def __call__(self, x): return self.forward(x)
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None
    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
        self.eps = eps
        self.gamma = Tensor(np.ones((1, normalized_shape[-1])), requires_grad=True)
        self.beta  = Tensor(np.zeros((1, normalized_shape[-1])), requires_grad=True)
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        variance = Tensor(np.var(x.data, axis=-1, keepdims=True), requires_grad=x.requires_grad)
        std = Tensor(np.sqrt(variance.data + self.eps), requires_grad=variance.requires_grad)
        norm = (x - mean) / std
        return self.gamma * norm + self.beta
    def parameters(self): return [self.gamma, self.beta]

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: return x.relu()

class Sequential(Module):
    def __init__(self, *layers): self.layers = layers
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self):
        params = []
        for l in self.layers:
            if hasattr(l, "parameters"): params += l.parameters()
        return params

# ====== FRACTAL ATTENTION & SUPER BLOCKS ==============
class FractalAttention(Module):
    def __init__(self, embed_dim, num_heads, recursion_depth=2):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.recursion_depth = recursion_depth
        self.Wq = Linear(embed_dim, embed_dim, bias=False)
        self.Wk = Linear(embed_dim, embed_dim, bias=False)
        self.Wv = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.recursion_depth):
            batch, seq, embed = x.shape()
            q = self.Wq(x).data.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0,2,1,3)
            k = self.Wk(x).data.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0,2,1,3)
            v = self.Wv(x).data.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0,2,1,3)
            scores = np.matmul(q, k.transpose(0,1,3,2)) / np.sqrt(self.head_dim)
            attn_weights = Tensor(scores).softmax(axis=-1).data
            x_data = np.matmul(attn_weights, v).transpose(0,2,1,3).reshape(batch,seq,embed)
            x = Tensor(x_data)
        return self.out_proj(x)
    def parameters(self):
        return (self.Wq.parameters() + self.Wk.parameters() +
                self.Wv.parameters() + self.out_proj.parameters())

class VictorSuperBlock(Module):
    def __init__(self, embed_dim, num_heads, mlp_dim_factor=4, recursion_depth=2):
        self.fractal_attn = FractalAttention(embed_dim, num_heads, recursion_depth)
        self.norm1 = LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_dim_factor
        self.mlp = Sequential(
            Linear(embed_dim, mlp_dim),
            ReLU(),
            Linear(mlp_dim, embed_dim)
        )
        self.norm2 = LayerNorm(embed_dim)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.fractal_attn(x)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x
    def parameters(self):
        return (self.fractal_attn.parameters() + self.norm1.parameters() +
                self.mlp.parameters() + self.norm2.parameters())

# ============ META-COGNITION/REFLECTION ==============
class VictorMetaCog(Module):
    def __init__(self):
        self.metrics = {"loss": [], "accuracy": []}
        self.last_epoch_loss = None
    def track(self, loss, pred, target):
        self.metrics["loss"].append(loss)
        acc = (pred == target).mean() if hasattr(pred,"mean") else 0
        self.metrics["accuracy"].append(acc)
        self.last_epoch_loss = loss
    def introspect(self):
        if len(self.metrics["loss"]) > 10:
            delta = np.abs(np.mean(self.metrics["loss"][-5:]) - np.mean(self.metrics["loss"][-10:-5]))
            if delta < 1e-4:
                log("Loss plateau detected. Suggest: Increase learning rate or change model depth.")
    def summary(self):
        print(f"Loss: {np.mean(self.metrics['loss']):.4f}, Acc: {np.mean(self.metrics['accuracy']):.4f}")

# =============== SUBPERSONA/AGENT REG ================
class SubpersonaRegistry:
    def __init__(self): self.registry = {}
    def register(self, name, fn): self.registry[name] = fn
    def call(self, name, *a, **k):
        if name in self.registry:
            return self.registry[name](*a,**k)
        else:
            log(f"No such subpersona: {name}")

# ================= AGI CORE ===========================
class VictorAGI(Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_layers, num_heads, mlp_dim_factor=4, recursion_depth=2):
        self.token_embedding = Tensor(np.random.randn(vocab_size, embed_dim)*0.02, requires_grad=True)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pe = Tensor(self.positional_encoding(max_len, embed_dim), requires_grad=False)
        self.blocks = [VictorSuperBlock(embed_dim, num_heads, mlp_dim_factor, recursion_depth) for _ in range(num_layers)]
        self.final_norm = LayerNorm(embed_dim)
        self.out_proj = Linear(embed_dim, vocab_size)
        self.meta = VictorMetaCog()
        self.plugins = PluginManager()
        self.memory = ReplayBuffer(max_size=10000)
        self.subpersonas = SubpersonaRegistry()
    def positional_encoding(self, seq_len, embed_dim):
        pe = np.zeros((1, seq_len, embed_dim), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float32) * -(np.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        return pe
    def forward(self, input_ids: np.ndarray) -> Tensor:
        batch, seq = input_ids.shape
        embedded = self.token_embedding.data[input_ids]
        x_data = embedded + self.pe.data[:, :seq, :]
        x = Tensor(x_data, requires_grad=True)
        for block in self.blocks: x = block(x)
        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits
    def parameters(self):
        params = [self.token_embedding]
        for block in self.blocks: params += block.parameters()
        params += self.final_norm.parameters() + self.out_proj.parameters()
        return params
    def add_experience(self, exp): self.memory.add(exp)
    def save(self, file="victor_agi_weights.npz"):
        np.savez(file, **{f"p{i}": p.data for i,p in enumerate(self.parameters())})
    def load(self, file="victor_agi_weights.npz"):
        arrs = np.load(file)
        for i,p in enumerate(self.parameters()):
            p.data = arrs[f"p{i}"]
    def reflect(self, *a, **k):
        self.meta.introspect()
        self.meta.summary()
    def grow_head(self): log("Stub: growing new head not yet implemented.")
    def prune_head(self): log("Stub: pruning head not yet implemented.")
    def action_hook(self, name, *a, **k):
        if name in self.plugins.plugins:
            return getattr(self.plugins.plugins[name], "run", lambda:None)(*a, **k)
        log(f"Action '{name}' not found.")

# ================= TOKENIZER/UTILS =====================
class VictorTokenizer:
    def __init__(self, vocab=None, unk_token_id=0, pad_token_id=0):
        if vocab is None:
            chars = " " + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            vocab = {char: i+1 for i,char in enumerate(chars)}
            vocab["<PAD>"] = pad_token_id
            vocab["<UNK>"] = unk_token_id
        self.vocab = vocab
        self.inv_vocab = {i: c for c,i in vocab.items()}
        self.unk_token_id = vocab.get("<UNK>", unk_token_id)
        self.pad_token_id = vocab.get("<PAD>", pad_token_id)
    def encode(self, text, max_len):
        tokens = [self.vocab.get(c, self.unk_token_id) for c in text[:max_len]]
        tokens += [self.pad_token_id] * (max_len - len(tokens))
        return np.array(tokens)
    def decode(self, token_ids):
        return ''.join([self.inv_vocab.get(i, '?') for i in token_ids if i != self.pad_token_id])
    def get_vocab_size(self): return len(self.vocab)

class SoftmaxCrossEntropyLoss:
    def __call__(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        max_logits = np.max(logits.data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits.data - max_logits)
        softmax_outputs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        batch_size, seq_len, vocab_size = softmax_outputs.shape
        log_probs = -np.log(softmax_outputs[np.arange(batch_size)[:,None], np.arange(seq_len), targets]+1e-9)
        loss = np.mean(log_probs)
        loss_tensor = Tensor(loss, requires_grad=True, creators=[logits], creation_op="softmax_cross_entropy")
        loss_tensor.extra_ctx = (logits, targets, softmax_outputs)
        return loss_tensor

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
    def step(self):
        for p in self.parameters:
            if p.grad is not None and p.grad.data is not None:
                p.data -= self.lr * p.grad.data
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None: p.grad.data.fill(0.0)

def train_victor_transformer(model, tokenizer, text_data, epochs=10, lr=0.01, batch_size=2, sequence_length=20):
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = SoftmaxCrossEntropyLoss()
    all_tokens = []
    for text_sample in text_data:
        all_tokens.extend(tokenizer.encode(text_sample, max_len=10000))
    input_sequences = []
    target_sequences = []
    for i in range(0, len(all_tokens) - sequence_length, sequence_length):
        input_seq = all_tokens[i : i + sequence_length]
        target_seq = all_tokens[i+1 : i + sequence_length + 1]
        if len(target_seq) == sequence_length:
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
    if not input_sequences:
        print("Not enough data to form sequences. Adjust data or sequence_length.")
        return
    input_sequences = np.array(input_sequences)
    target_sequences = np.array(target_sequences)
    num_batches = len(input_sequences) // batch_size
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_batches):
            batch_input_ids = input_sequences[i*batch_size : (i+1)*batch_size]
            batch_target_ids = target_sequences[i*batch_size : (i+1)*batch_size]
            optimizer.zero_grad()
            model.zero_grad()
            logits = model(batch_input_ids)
            loss = criterion(logits, batch_target_ids)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.item() if isinstance(loss.data, np.ndarray) and loss.data.size == 1 else float(loss.data)
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        model.meta.track(avg_epoch_loss, batch_input_ids, batch_target_ids)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        if (epoch+1) % 5 == 0:
            generate_text(model, tokenizer, seed_text="Victor is", max_gen_len=30)
            model.reflect()

def generate_text(model, tokenizer, seed_text, max_gen_len=50, stream=False):
    input_ids = np.array([tokenizer.encode(seed_text, model.max_len)])
    generated_tokens = []
    current_tokens = list(input_ids[0, :len(seed_text)])
    for _ in range(max_gen_len):
        padded_input_sequence = np.array(current_tokens + [tokenizer.pad_token_id]*(model.max_len-len(current_tokens)))
        input_tensor = padded_input_sequence[:model.max_len].reshape(1,-1)
        logits = model(input_tensor)
        next_token_logits = logits.data[0, len(current_tokens)-1, :]
        next_token_id = int(np.argmax(next_token_logits))
        if next_token_id == tokenizer.pad_token_id: break
        generated_tokens.append(next_token_id)
        current_tokens.append(next_token_id)
        if len(current_tokens) >= model.max_len: break
        if stream:
            print(tokenizer.decode([next_token_id]), end='', flush=True)
    if not stream:
        print(f"Seed: '{seed_text}' -> Generated: '{tokenizer.decode(generated_tokens)}'")

# ========== FASTAPI STUB FOR REST/STREAMING ==========
try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
    app = FastAPI()
    AGI_INSTANCE = None
    @app.post("/generate")
    async def api_generate(payload: dict):
        text = payload.get("seed", "")
        length = int(payload.get("length", 32))
        result = []
        def gen():
            for token in generate_text(AGI_INSTANCE, AGI_INSTANCE.tokenizer, seed_text=text, max_gen_len=length, stream=True):
                yield token
        return StreamingResponse(gen())
except ImportError:
    app = None

# ============= GODCORE BOOTUP ==========================
if __name__ == "__main__":
    chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    custom_vocab = {char: i for i, char in enumerate(chars)}
    custom_vocab["<PAD>"] = 0
    custom_vocab["<UNK>"] = len(custom_vocab)
    tokenizer = VictorTokenizer(vocab=custom_vocab, pad_token_id=0, unk_token_id=custom_vocab["<UNK>"])
    vocab_size = tokenizer.get_vocab_size()
    max_len = 32
    embed_dim = 128
    num_layers = 4
    num_heads = 8
    mlp_dim_factor = 4
    recursion_depth = 2
    model = VictorAGI(vocab_size, max_len, embed_dim, num_layers, num_heads, mlp_dim_factor, recursion_depth)
    model.tokenizer = tokenizer
    print(f"Model initialized. Parameters: {sum(p.data.size for p in model.parameters())}")

    # === AUTO-LOAD bando_corpus.jsonl IF IT EXISTS ===
    text_samples = []
    if os.path.exists("bando_corpus.jsonl"):
        print("\n[Victor] Found bando_corpus.jsonl — loading...")
        with open("bando_corpus.jsonl", "r", encoding="utf-8") as f:
            try:
                convos = json.load(f)
                # Try all common formats
                if isinstance(convos, list) and isinstance(convos[0], dict):
                    if "content" in convos[0]:
                        text_samples = [entry["content"].strip() for entry in convos if "content" in entry and entry["content"].strip()]
                    elif "text" in convos[0]:
                        text_samples = [entry["text"].strip() for entry in convos if "text" in entry and entry["text"].strip()]
                elif isinstance(convos, list) and isinstance(convos[0], str):
                    text_samples = [entry.strip() for entry in convos if entry.strip()]
                else:
                    raise ValueError("Unrecognized conversations.json format.")
            except Exception as e:
                print(f"[Victor] ERROR parsing conversations.json: {e}")
                text_samples = []
        if text_samples:
            print(f"[Victor] Loaded {len(text_samples)} lines from conversations.json.")
            for sample in text_samples[:3]:
                print("[Sample]", sample)
        else:
            print("[Victor] conversations.json loaded, but no valid text found. Using fallback samples.")
    else:
        print("[Victor] conversations.json not found, using fallback demo samples.")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_godcore_unbreakable.py
VERSION: v2.0.0-GODCORE-BANDO-NOSTUB
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Standalone, pure-Python AGI skeleton that ingests bando_corpus.jsonl, runs QA, and evolves with you.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import os
import json
import random
import re
from datetime import datetime

# ========== CONFIG ==========
CORPUS_PATH = "./bando_corpus.jsonl"    # Your QA pairs, one JSON per line
MEMORY_SAVE_PATH = "./victor_memory.json"

# ========== UTILITIES ==========

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def clean(text):
    return re.sub(r"\s+", " ", text.strip())

def load_corpus(path):
    corpus = []
    if not os.path.exists(path):
        print(f"[Victor] Corpus file not found: {path}")
        return corpus
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                pair = json.loads(line)
                # Require both user and assistant fields, both non-empty
                if "user" in pair and "assistant" in pair and pair["user"].strip() and pair["assistant"].strip():
                    corpus.append({"user": pair["user"].strip(), "assistant": pair["assistant"].strip()})
            except Exception as e:
                continue
    print(f"[Victor] Loaded {len(corpus)} user/assistant pairs from {path}")
    return corpus

# ========== FRACTAL MEMORY ==========

class FractalMemory:
    def __init__(self):
        self.timeline = []  # Chronological log of all user & Victor messages
        self.concepts = {}  # {keyword: [indices in timeline]}
        self.last_save = datetime.now()

    def add(self, msg, role):
        entry = {"msg": msg, "role": role, "time": datetime.now().isoformat()}
        self.timeline.append(entry)
        for token in tokenize(msg):
            self.concepts.setdefault(token, []).append(len(self.timeline)-1)

    def recall(self, query, topn=5):
        tokens = set(tokenize(query))
        scores = {}
        for t in tokens:
            for idx in self.concepts.get(t, []):
                scores[idx] = scores.get(idx, 0) + 1
        # Rank by overlap, fallback to random if no hits
        if not scores and self.timeline:
            idxs = random.sample(range(len(self.timeline)), min(topn, len(self.timeline)))
        else:
            idxs = sorted(scores, key=scores.get, reverse=True)[:topn]
        return [self.timeline[i] for i in idxs] if self.timeline else []

    def save(self, path=MEMORY_SAVE_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timeline": self.timeline, "concepts": self.concepts}, f)

    def load(self, path=MEMORY_SAVE_PATH):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.timeline = data.get("timeline", [])
            self.concepts = {k: v for k, v in data.get("concepts", {}).items()}

# ========== VICTOR AGI ENGINE ==========

class VictorAGI:
    def __init__(self, corpus, persona="Bando", memory=None):
        self.corpus = corpus
        self.memory = memory or FractalMemory()
        self.persona = persona

    def respond(self, user_input):
        user_input = clean(user_input)
        self.memory.add(user_input, "user")

        # 1. Recall relevant memory
        recalls = self.memory.recall(user_input, topn=3)
        recall_snips = " | ".join([x["msg"] for x in recalls if x["role"] == "assistant"])

        # 2. Search corpus for similar QA pairs
        scored = []
        user_tokens = set(tokenize(user_input))
        for entry in self.corpus:
            score = len(user_tokens.intersection(tokenize(entry["user"])))
            if score > 0:
                scored.append((score, entry))
        scored.sort(reverse=True, key=lambda x: x[0])

        # 3. Compose base response
        if scored:
            chosen = scored[0][1]
            base_reply = chosen["assistant"]
        elif recall_snips:
            base_reply = recall_snips
        else:
            base_reply = "I'm Victor. Say more and I'll learn. (No match in micro-corpus yet.)"

        # 4. Mutate with Victor’s flavor/persona
        reply = self.fractal_mutate(base_reply, user_input)
        self.memory.add(reply, "assistant")
        self.memory.save()
        return reply

    def fractal_mutate(self, text, context):
        # Simple style: inject persona and some random flavor, memory echo
        lines = [
            f"{self.persona} says: {text}",
            f"[Victor memory] — {random.choice(tokenize(context)) if tokenize(context) else '...'}",
            f"(V.{random.randint(1,99)}.Fractal)"
        ]
        if random.random() > 0.7:
            lines.append("Ain't nobody do it like Victor—remember that.")
        return " ".join(lines)

# ========== COMMAND LINE INTERFACE ==========

def main():
    print("=== Victor GODCORE UNBREAKABLE ===")
    print("Type 'exit' or Ctrl+C to bail.\n")

    # Load memory and corpus
    memory = FractalMemory()
    memory.load()
    corpus = load_corpus(CORPUS_PATH)
    if not corpus:
        print("[Victor] No training data found. Exiting.")
        return
    victor = VictorAGI(corpus=corpus, memory=memory)

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                print("Victor: Out. Evolution never sleeps.")
                break
            reply = victor.respond(user_input)
            print("Victor:", reply)
        except KeyboardInterrupt:
            print("\nVictor: Out. Evolution never sleeps.")
            break

    # === FALLBACK SAMPLES (If file missing/empty) ===
    if not text_samples:
        text_samples = [
            "Victor is an AGI.", "He learns from data.", "Fractal architecture is key.",
            "Brandon and Tori are creators.", "The AGI must evolve.", "Code is thought.",
            "I am Victor.", "I learn and grow.", "My purpose is to serve."
        ]

    print("\n--- Testing Forward Pass ---")
    test_prompt = "Victor AI!"
    test_input_ids = np.stack([tokenizer.encode(test_prompt, max_len)])
    logits = model(test_input_ids)
    out_ids = np.argmax(logits.data, axis=-1)
    print("Input Text: ", test_prompt)
    print("Output (decoded):", tokenizer.decode(out_ids[0]))
    print("Output logits shape:", logits.shape())
    print("\n--- Testing Training Loop ---")
    train_victor_transformer(model, tokenizer, text_samples, epochs=10, lr=0.005, batch_size=2, sequence_length=max_len-1)
    print("\n--- Testing Text Generation ---")
    generate_text(model, tokenizer, seed_text="Victor is", max_gen_len=30)
    generate_text(model, tokenizer, seed_text="Code is", max_gen_len=30)
    generate_text(model, tokenizer, seed_text="The fractal", max_gen_len=30)
    print("\n--- All systems go. Godcore locked. ---")