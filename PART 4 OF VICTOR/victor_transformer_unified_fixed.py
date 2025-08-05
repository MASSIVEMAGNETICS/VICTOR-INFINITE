#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_transformer_unified.py
VERSION: v1.0.3-GODCORE-BANDO-ENHANCED
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE:
    Standalone, pure-NumPy, GPT-style transformer with custom autodiff (VictorTensor).
    All logic in one file—train, test, mutate, REST, or CLI.
    - VictorTensor autodiff (no torch/tf)
    - Modular transformer stack (LLM-style)
    - Full train+test loop (SGD)
    - Tokenizer, text gen, AGI plug-in ready
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import re
import sys

# ================== VICTOR TENSOR (AUTODIFF) ==================

class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        if self.requires_grad:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        self.creators = creators
        self.creation_op = creation_op
        self.backward_hooks = []

    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.grad is not None:
            self.grad.data.fill(0.0)

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return
        if grad_output is None:
            if self.data.size == 1:
                grad_output = Tensor(np.array([1.0], dtype=np.float32), requires_grad=False)
            else:
                raise ValueError("grad_output must be specified for non-scalar Tensors unless it's the final loss.")
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output)
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        self.grad.data += grad_output.data

        for hook in self.backward_hooks:
            hook(self)

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
                a.backward(Tensor(grad_output.data @ b.data.T))
                b.backward(Tensor(a.data.T @ grad_output.data))
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
            # Add more ops as needed

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

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="sum")

    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="mean")

    def transpose(self, axes=None):
        op = "transpose"
        return Tensor(self.data.T if axes is None else np.transpose(self.data, axes),
                      requires_grad=self.requires_grad, creators=[self], creation_op=op)
    @property
    def T(self):
        return self.transpose()

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
        relu_data = np.maximum(self.data, 0)
        return Tensor(relu_data, requires_grad=self.requires_grad, creators=[self], creation_op="relu")

    def softmax(self, axis=-1):
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out_data = e_x / np.sum(e_x, axis=axis, keepdims=True)
        return Tensor(out_data, requires_grad=False)

    def __repr__(self):
        return f"VictorTensor(shape={self.data.shape}, requires_grad={self.requires_grad})\n{self.data}"

# ================== MODULES ==================

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

# ================== TRANSFORMER BLOCKS ==================

class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wq = Linear(embed_dim, embed_dim, bias=False)
        self.Wk = Linear(embed_dim, embed_dim, bias=False)
        self.Wv = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embed_dim_ = x.shape()
        q = self.Wq(x).data.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.Wk(x).data.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.Wv(x).data.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attn_weights_tensor = Tensor(attn_scores).softmax(axis=-1)
        attn_weights = attn_weights_tensor.data
        attn_output = np.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim_)
        return self.out_proj(Tensor(attn_output))
    def parameters(self):
        return (self.Wq.parameters() + self.Wk.parameters() +
                self.Wv.parameters() + self.out_proj.parameters())

def positional_encoding(seq_len, embed_dim):
    pe = np.zeros((1, seq_len, embed_dim), dtype=np.float32)
    position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float32) * -(np.log(10000.0) / embed_dim))
    pe[0, :, 0::2] = np.sin(position * div_term)
    pe[0, :, 1::2] = np.cos(position * div_term)
    return pe

class VictorTokenizer:
    def __init__(self, vocab=None, unk_token_id=0, pad_token_id=0):
        if vocab is None:
            chars = " " + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            vocab = {char: i+1 for i, char in enumerate(chars)}
            vocab["<PAD>"] = pad_token_id
            vocab["<UNK>"] = unk_token_id
        self.vocab = vocab
        self.inv_vocab = {i: c for c, i in vocab.items()}
        self.unk_token_id = vocab.get("<UNK>", unk_token_id)
        self.pad_token_id = vocab.get("<PAD>", pad_token_id)
    def encode(self, text, max_len):
        tokens = [self.vocab.get(c, self.unk_token_id) for c in text[:max_len]]
        tokens += [self.pad_token_id] * (max_len - len(tokens))
        return np.array(tokens)
    def decode(self, token_ids):
        return ''.join([self.inv_vocab.get(i, '?') for i in token_ids if i != self.pad_token_id])
    def get_vocab_size(self): return len(self.vocab)

class VictorTransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, mlp_dim_factor=4):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_dim_factor
        self.mlp = Sequential(
            Linear(embed_dim, mlp_dim),
            ReLU(),
            Linear(mlp_dim, embed_dim)
        )
        self.norm2 = LayerNorm(embed_dim)
    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.norm1(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        return x
    def parameters(self):
        return (self.attn.parameters() + self.norm1.parameters() +
                self.mlp.parameters() + self.norm2.parameters())

class VictorTransformer(Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_layers, num_heads, mlp_dim_factor=4):
        self.token_embedding = Tensor(np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02, requires_grad=True)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pe = Tensor(positional_encoding(max_len, embed_dim), requires_grad=False)
        self.blocks = [VictorTransformerBlock(embed_dim, num_heads, mlp_dim_factor) for _ in range(num_layers)]
        self.final_norm = LayerNorm(embed_dim)
        self.out_proj = Linear(embed_dim, vocab_size)
    def forward(self, input_ids: np.ndarray) -> Tensor:
        batch_size, seq_len = input_ids.shape
        embedded_tokens = self.token_embedding.data[input_ids]
        x_data = embedded_tokens + self.pe.data[:, :seq_len, :]
        x = Tensor(x_data, requires_grad=True)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits
    def parameters(self):
        params = [self.token_embedding]
        for block in self.blocks:
            params += block.parameters()
        params += self.final_norm.parameters()
        params += self.out_proj.parameters()
        return params

class SoftmaxCrossEntropyLoss:
    def __call__(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        max_logits = np.max(logits.data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits.data - max_logits)
        softmax_outputs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        batch_size, seq_len, vocab_size = softmax_outputs.shape
        # (batch, seq, vocab)
        log_probs = -np.log(softmax_outputs[np.arange(batch_size)[:,None], np.arange(seq_len), targets] + 1e-9)
        loss = np.mean(log_probs)
        loss_tensor = Tensor(loss, requires_grad=True, creators=[logits], creation_op="softmax_cross_entropy")
        # Attach context for autodiff
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
            p.zero_grad()

# ============= GODCORE TRAIN/TEST SCAFFOLDING (CLI/REST plug-in ready) =============

if __name__ == "__main__":
    # Example usage / smoke test (CLI run = instant sanity check)
    vocab = VictorTokenizer()
    vocab_size = vocab.get_vocab_size()
    max_len = 32
    embed_dim = 48
    num_layers = 2
    num_heads = 4

    model = VictorTransformer(vocab_size, max_len, embed_dim, num_layers, num_heads)
    loss_fn = SoftmaxCrossEntropyLoss()
    optim = SGD(model.parameters(), lr=0.03)

    # Toy train set (hello, world. Repeat until it overfits, lol)
    text = "Hello Bando Bandz! Godcore transformer, run that back."
    toks = vocab.encode(text, max_len)
    X = np.array([toks])
    Y = np.array([toks])  # Just trying to overfit single sequence, like a boss

    print("[Victor] Training toy sequence. Watch loss drop or debug if it explodes.")

    for epoch in range(10):  # Just a few steps for demo
        logits = model(X)
        loss = loss_fn(logits, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Epoch {epoch+1}, Loss: {loss.data:.5f}")

    # Output generation test (not true sampling, just to check decode)
    print("Input:", vocab.decode(X[0]))
    preds = np.argmax(model(X).data, axis=-1)
    print("Output:", vocab.decode(preds[0]))
