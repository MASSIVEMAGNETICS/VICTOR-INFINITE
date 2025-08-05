#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_aetherial_audiocore_godcore.py
VERSION: v2.0.0-GODCORE-BANDO-FULL
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Standalone, pure-NumPy, fractal audio TTS pipeline.
         Each stage (semantic, coarse, fine) is its own VictorTransformer, with VictorTensor autodiff backend.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import random
import os

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

# ================== FULL PIPELINE ==================

class VictorAetherialAudioCore:
    def __init__(self, vocab_size, max_len, embed_dim, n_layers, n_heads, mlp_factor):
        self.semantic_transformer = VictorTransformer(vocab_size, max_len, embed_dim, n_layers, n_heads, mlp_factor)
        self.coarse_transformer   = VictorTransformer(vocab_size, max_len, embed_dim, n_layers, n_heads, mlp_factor)
        self.fine_transformer     = VictorTransformer(vocab_size, max_len, embed_dim, n_layers, n_heads, mlp_factor)
        self.tokenizer = VictorTokenizer()

    def forward_semantic(self, prompt, max_gen=32, temp=1.0):
        tokens = self.tokenizer.encode(prompt, self.semantic_transformer.max_len)
        input_ids = np.stack([tokens])
        generated = list(tokens[:len(prompt)])
        for i in range(len(prompt), max_gen):
            logits = self.semantic_transformer(input_ids)
            next_logits = logits.data[0, i-1] if i > 0 else logits.data[0, 0]
            if temp == 0:
                next_token = np.argmax(next_logits)
            else:
                probs = np.exp(next_logits / temp)
                probs /= np.sum(probs)
                next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            input_ids[0, i] = next_token
            if next_token == self.tokenizer.pad_token_id:
                break
        return np.array(generated, dtype=np.int32)

    def forward_coarse(self, semantic_tokens, max_gen=32, temp=1.0):
        coarse_input = semantic_tokens[:max_gen]
        input_ids = np.stack([coarse_input])
        generated = list(coarse_input)
        for i in range(len(coarse_input), max_gen):
            logits = self.coarse_transformer(input_ids)
            next_logits = logits.data[0, i-1] if i > 0 else logits.data[0, 0]
            if temp == 0:
                next_token = np.argmax(next_logits)
            else:
                probs = np.exp(next_logits / temp)
                probs /= np.sum(probs)
                next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            input_ids[0, i] = next_token
        return np.tile(np.array(generated), (2,1))  # Dummy 2 codebooks

    def forward_fine(self, coarse_tokens, max_gen=32, temp=1.0):
        seq_len = coarse_tokens.shape[1]
        fine_codes = []
        for c in range(8):
            input_ids = coarse_tokens[0:1, :]  # Use only first codebook for context
            logits = self.fine_transformer(input_ids)
            tokens = []
            for i in range(seq_len):
                next_logits = logits.data[0, i]
                if temp == 0:
                    next_token = np.argmax(next_logits)
                else:
                    probs = np.exp(next_logits / temp)
                    probs /= np.sum(probs)
                    next_token = np.random.choice(len(probs), p=probs)
                tokens.append(next_token)
            fine_codes.append(tokens)
        return np.array(fine_codes, dtype=np.int32)

    def decode_audio(self, fine_tokens_matrix, sample_rate=24000):
        seq_len = fine_tokens_matrix.shape[1]
        t = np.linspace(0, seq_len/75, seq_len*320, endpoint=False)
        waveform = np.zeros_like(t)
        for i, codes in enumerate(fine_tokens_matrix):
            freq = 200 + 80*i
            waveform += 0.0125 * np.sin(2 * np.pi * freq * t + (codes[:len(t)//320].mean() % np.pi))
        waveform = waveform / np.max(np.abs(waveform))
        return waveform.astype(np.float32)

    def run(self, prompt, output_path="bando_victor_unified.wav"):
        print(f"SEMANTIC: {prompt}")
        sem_tokens = self.forward_semantic(prompt)
        print("SEMANTIC TOKENS:", sem_tokens)
        print("SEMANTIC DECODED:", self.tokenizer.decode(sem_tokens))
        coarse = self.forward_coarse(sem_tokens)
        fine = self.forward_fine(coarse)
        audio = self.decode_audio(fine)
        try:
            import soundfile as sf
            sf.write(output_path, audio, 24000)
            print(f"Wrote audio: {output_path}")
        except ImportError:
            print("soundfile not installed. Skipping .wav output.")
        return audio

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
    mlp_factor = 4

    godcore = VictorAetherialAudioCore(vocab_size, max_len, embed_dim, num_layers, num_heads, mlp_factor)
    godcore.tokenizer = tokenizer  # Use custom vocab/tokenizer

    prompt = "I am Bando Bandz and Victor is my voice!"
    godcore.run(prompt, output_path="bando_victor_unified.wav")
