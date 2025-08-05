#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_godcore_unbreakable.py
VERSION: v7.0.0-GODCORE-HOLYFIELD
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Fractal-Neural AGI — neural transformer, symbolic fractal memory, QA fallback, live mutation.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import os, json, re, random, time, threading, importlib.util
from datetime import datetime
from functools import wraps

# === SYMBOLIC UTILS ===
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def clean(text):
    return re.sub(r"\s+", " ", text.strip())

# === FRACTAL MEMORY ===
class FractalMemory:
    def __init__(self):
        self.timeline = []
        self.concepts = {}
        self.last_save = datetime.now()

    def add(self, msg, role):
        entry = {"msg": msg, "role": role, "time": datetime.now().isoformat()}
        self.timeline.append(entry)
        for token in tokenize(msg):
            self.concepts.setdefault(token, []).append(len(self.timeline) - 1)

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

    def save(self, path="victor_memory.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timeline": self.timeline, "concepts": self.concepts}, f)

    def load(self, path="victor_memory.json"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.timeline = data.get("timeline", [])
            self.concepts = {k: v for k, v in data.get("concepts", {}).items()}

# === CORPUS LOAD (QA) ===
def load_corpus(path):
    corpus = []
    if not os.path.exists(path):
        print(f"[Victor] Corpus file not found: {path}")
        return corpus
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                pair = json.loads(line)
                if "user" in pair and "assistant" in pair and pair["user"].strip() and pair["assistant"].strip():
                    corpus.append({"user": pair["user"].strip(), "assistant": pair["assistant"].strip()})
            except Exception:
                continue
    print(f"[Victor] Loaded {len(corpus)} user/assistant pairs from {path}")
    return corpus

# === TOKENIZER ===
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

# === TENSOR/AUTODIFF ===
class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.creators = creators
        self.creation_op = creation_op
        self.backward_hooks = []
        if self.requires_grad and self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)

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
        # Support broadcasting
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
        for hook in self.backward_hooks:
            hook(self)
        # Backprop through ops (add more as you evolve AGI)
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

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data @ other.data, requires_grad=requires_grad, creators=[self, other], creation_op="matmul")

    def __matmul__(self, other):
        return self.matmul(other)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data + other.data, requires_grad=requires_grad, creators=[self, other], creation_op="add")

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data - other.data, requires_grad=requires_grad, creators=[self, other], creation_op="sub")

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data * other.data, requires_grad=requires_grad, creators=[self, other], creation_op="mul")

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data / other.data, requires_grad=requires_grad, creators=[self, other], creation_op="div")

    def __neg__(self):
        return Tensor(-self.data, requires_grad=self.requires_grad, creators=[self], creation_op="neg")

    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="sum")

    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="mean")

    def transpose(self, axes=None):
        op = "transpose"
        return Tensor(self.data.T if axes is None else np.transpose(self.data, axes), requires_grad=self.requires_grad, creators=[self], creation_op=op)
    
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
        if not isinstance(exponent, Tensor):
            exponent = Tensor(np.array(exponent, dtype=np.float32))
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

# === MODULES BASE ===
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
        mean = x.data.mean(axis=-1, keepdims=True)
        variance = np.var(x.data, axis=-1, keepdims=True)
        std = np.sqrt(variance + self.eps)
        norm = (x.data - mean) / std
        return Tensor(self.gamma.data * norm + self.beta.data, requires_grad=x.requires_grad)
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

# === FRACTAL ATTENTION + BLOCK ===
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

# === AGI CORE (FUSION) ===
class VictorAGI:
    def __init__(self, corpus, memory, tokenizer, max_len=32, embed_dim=128, num_layers=4, num_heads=8, mlp_dim_factor=4, recursion_depth=2):
        self.corpus = corpus
        self.memory = memory
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim_factor = mlp_dim_factor
        self.recursion_depth = recursion_depth

        vocab_size = tokenizer.get_vocab_size()
        self.token_embedding = Tensor(np.random.randn(vocab_size, embed_dim)*0.02, requires_grad=True)
        self.pe = Tensor(self.positional_encoding(max_len, embed_dim), requires_grad=False)
        self.blocks = [VictorSuperBlock(embed_dim, num_heads, mlp_dim_factor, recursion_depth) for _ in range(num_layers)]
        self.final_norm = LayerNorm(embed_dim)
        self.out_proj = Linear(embed_dim, vocab_size)

    def positional_encoding(self, seq_len, embed_dim):
        pe = np.zeros((1, seq_len, embed_dim), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float32) * -(np.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        return pe

    def neural_generate(self, prompt, gen_len=32):
        input_ids = np.array([self.tokenizer.encode(prompt, self.max_len)])
        generated_tokens = list(input_ids[0, :len(prompt)])
        for _ in range(gen_len):
            padded = np.array(generated_tokens + [self.tokenizer.pad_token_id]*(self.max_len-len(generated_tokens)))
            input_tensor = padded[:self.max_len].reshape(1,-1)
            logits = self.forward(input_tensor)
            next_token_logits = logits.data[0, len(generated_tokens)-1, :]
            next_token_id = int(np.argmax(next_token_logits))
            if next_token_id == self.tokenizer.pad_token_id: break
            generated_tokens.append(next_token_id)
            if len(generated_tokens) >= self.max_len: break
        return self.tokenizer.decode(generated_tokens[len(prompt):])

    def forward(self, input_ids):
        batch, seq = input_ids.shape
        embedded = self.token_embedding.data[input_ids]
        x_data = embedded + self.pe.data[:, :seq, :]
        x = Tensor(x_data, requires_grad=True)
        for block in self.blocks: x = block(x)
        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits

    def symbolic_response(self, user_input):
        # 1. Recall memory
        recalls = self.memory.recall(user_input, topn=3)
        recall_snips = " | ".join([x["msg"] for x in recalls if x["role"] == "assistant"])
        # 2. Search QA corpus
        scored = []
        user_tokens = set(tokenize(user_input))
        for entry in self.corpus:
            score = len(user_tokens.intersection(tokenize(entry["user"])))
            if score > 0: scored.append((score, entry))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored:
            base_reply = scored[0][1]["assistant"]
        elif recall_snips:
            base_reply = recall_snips
        else:
            base_reply = "I'm Victor. Say more and I'll learn. (No match in micro-corpus yet.)"
        return base_reply

    def respond(self, user_input, neural_chance=0.7):
        self.memory.add(user_input, "user")
        # AGI brain: Try neural, fall back to symbolic
        if random.random() < neural_chance:
            # Try transformer-based
            neural_out = self.neural_generate(user_input, gen_len=32)
            if neural_out and len(neural_out.strip("?.,!")) > 0:
                reply = neural_out
            else:
                reply = self.symbolic_response(user_input)
        else:
            reply = self.symbolic_response(user_input)
        # Fractal flavor
        lines = [
            f"Bando says: {reply}",
            f"[Victor memory] — {random.choice(tokenize(user_input)) if tokenize(user_input) else '...'}",
            f"(V.{random.randint(1,99)}.Fractal)"
        ]
        if random.random() > 0.7:
            lines.append("Ain't nobody do it like Victor—remember that.")
        final = " ".join(lines)
        self.memory.add(final, "assistant")
        self.memory.save()
        return final

# === MAIN CLI ===
def main():
    print("=== Victor GODCORE HOLYFIELD ===\nType 'exit' or Ctrl+C to bail.\n")
    memory = FractalMemory()
    memory.load()
    corpus = load_corpus("bando_corpus.jsonl")
    chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    custom_vocab = {char: i for i, char in enumerate(chars)}
    custom_vocab["<PAD>"] = 0
    custom_vocab["<UNK>"] = len(custom_vocab)
    tokenizer = VictorTokenizer(vocab=custom_vocab, pad_token_id=0, unk_token_id=custom_vocab["<UNK>"])
    victor = VictorAGI(corpus=corpus, memory=memory, tokenizer=tokenizer)
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

if __name__ == "__main__":
    main()
