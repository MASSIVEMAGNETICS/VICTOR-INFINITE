#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_transformer_unified.py
VERSION: v1.0.4-GODCORE-BANDO-FIXED-NDGRAD
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Standalone, pure-NumPy, GPT-style transformer with custom autodiff (VictorTensor),
         all logic in one file. Extend, train, and API as you wish.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import re

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
                grad_output = Tensor(np.ones_like(self.data), requires_grad=False)
            else:
                raise ValueError("grad_output must be specified for non-scalar Tensors unless it's the final loss.")
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output)
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)

        # === Bando Broadcast & Reduce Fix (Final Form) ===
        g_shape = self.grad.data.shape
        d_shape = grad_output.data.shape
        if g_shape == d_shape:
            self.grad.data += grad_output.data
        elif grad_output.data.size == 1:
            self.grad.data += grad_output.data.item()
        elif self.grad.data.size == 1:
            self.grad.data += grad_output.data.sum()
        else:
            # Reduce grad_output to self.grad shape if needed (sum over broadcast axes)
            diff = len(d_shape) - len(g_shape)
            g_shape_full = (1,)*(diff) + g_shape if diff > 0 else g_shape
            axes = tuple(i for i, (gs, ds) in enumerate(zip(g_shape_full, d_shape)) if gs == 1 and ds > 1)
            reduced = grad_output.data.sum(axis=axes, keepdims=True)
            reduced = reduced.reshape(g_shape)
            self.grad.data += reduced

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
                # ND Batch-aware backward (PyTorch-like)
                a_shape = a.data.shape
                b_shape = b.data.shape
                # Grad w.r.t. a: np.matmul(grad_output, b^T)
                b_T = np.swapaxes(b.data, -1, -2)
                grad_a = np.matmul(grad_output.data, b_T)
                # Reduce grad_a to match a's shape
                while grad_a.shape != a_shape:
                    grad_a = grad_a.sum(axis=0)
                a.backward(Tensor(grad_a))
                # Grad w.r.t. b: np.matmul(a^T, grad_output)
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
        return Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, creators=[self], creation_op="relu")

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
        log_probs = -np.log(softmax_outputs[np.arange(batch_size)[:,None], np.arange(seq_len), targets] + 1e-9)
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
            if p.grad is not None:
                p.grad.data.fill(0.0)

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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            generate_text(model, tokenizer, seed_text="Victor is", max_gen_len=30)

def generate_text(model, tokenizer, seed_text, max_gen_len=50):
    input_ids = np.array([tokenizer.encode(seed_text, model.max_len)])
    generated_tokens = []
    current_tokens = list(input_ids[0, :len(seed_text)])
    for _ in range(max_gen_len):
        padded_input_sequence = np.array(current_tokens + [tokenizer.pad_token_id] * (model.max_len - len(current_tokens)))
        input_tensor = padded_input_sequence[:model.max_len].reshape(1, -1)
        logits = model(input_tensor)
        next_token_logits = logits.data[0, len(current_tokens)-1, :]
        next_token_id = np.argmax(next_token_logits)
        if next_token_id == tokenizer.pad_token_id: break
        generated_tokens.append(next_token_id)
        current_tokens.append(next_token_id)
        if len(current_tokens) >= model.max_len: break
    print(f"Seed: '{seed_text}' -> Generated: '{tokenizer.decode(generated_tokens)}'")

if __name__ == "__main__":
    chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    custom_vocab = {char: i for i, char in enumerate(chars)}
    custom_vocab["<PAD>"] = 0
    custom_vocab["<UNK>"] = len(custom_vocab)
    tokenizer = VictorTokenizer(vocab=custom_vocab, pad_token_id=0, unk_token_id=custom_vocab["<UNK>"])
    vocab_size = tokenizer.get_vocab_size()
    max_len = 32
    embed_dim = 64
    num_layers = 3
    num_heads = 4
    mlp_dim_factor = 4
    model = VictorTransformer(vocab_size, max_len, embed_dim, num_layers, num_heads, mlp_dim_factor)
    print(f"Model initialized. Parameters: {sum(p.data.size for p in model.parameters())}")
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
