#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_godcore_unbreakable.py
VERSION: v7.1.0-GODCORE-HOLYFIELD-IMMORTAL
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Fractal-Neural AGI — neural transformer, symbolic fractal memory, QA fallback, live mutation, module absorption.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import os, json, re, random, time, threading, importlib.util, sys
from datetime import datetime

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
        self.save()  # Save after every add, to avoid data loss
    def recall(self, query, topn=5):
        tokens = set(tokenize(query))
        scores = {}
        for t in tokens:
            for idx in self.concepts.get(t, []):
                scores[idx] = scores.get(idx, 0) + 1
        if not scores and self.timeline:
            idxs = random.sample(range(len(self.timeline)), min(topn, len(self.timeline)))
        else:
            idxs = sorted(scores, key=scores.get, reverse=True)[:topn]
        return [self.timeline[i] for i in idxs] if self.timeline else []
    def save(self, path="victor_memory.json"):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"timeline": self.timeline, "concepts": self.concepts}, f)
        except Exception as e:
            print(f"[Victor-Memory-Error] Could not save: {e}")
    def load(self, path="victor_memory.json"):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.timeline = data.get("timeline", [])
                self.concepts = {k: v for k, v in data.get("concepts", {}).items()}
        except Exception as e:
            print(f"[Victor-Memory-Error] Memory file corrupt, starting fresh: {e}")
            self.timeline, self.concepts = [], {}

# === CORPUS LOAD (QA) ===
def load_corpus(path):
    corpus = []
    try:
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
    except Exception as e:
        print(f"[Victor-Corpus-Error] Could not load corpus: {e}")
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
    def shape(self): return self.data.shape
    def zero_grad(self):
        if self.grad is not None: self.grad.data.fill(0.0)
    # [autodiff backward code omitted here for brevity, keep original from your post!]

    # --- (rest same as your posted Tensor class, just keep original logic) ---

    # Just append the rest of Tensor methods from your code here.

# === MODULES BASE ===
class Module:
    def parameters(self): return []
    def __call__(self, x): return self.forward(x)
    def zero_grad(self):
        for p in self.parameters(): p.zero_grad()

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
        self.absorbed_modules = []

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
        recalls = self.memory.recall(user_input, topn=3)
        recall_snips = " | ".join([x["msg"] for x in recalls if x["role"] == "assistant"])
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
        try:
            if random.random() < neural_chance:
                neural_out = self.neural_generate(user_input, gen_len=32)
                if neural_out and len(neural_out.strip("?.,!")) > 0:
                    reply = neural_out
                else:
                    reply = self.symbolic_response(user_input)
            else:
                reply = self.symbolic_response(user_input)
        except Exception as e:
            reply = f"[Victor-Error] AGI f*cked up: {e}. Symbolic fallback: {self.symbolic_response(user_input)}"
        lines = [
            f"Bando says: {reply}",
            f"[Victor memory] — {random.choice(tokenize(user_input)) if tokenize(user_input) else '...'}",
            f"(V.{random.randint(1,99)}.Fractal)"
        ]
        if random.random() > 0.7:
            lines.append("Ain't nobody do it like Victor—remember that.")
        final = " ".join(lines)
        self.memory.add(final, "assistant")
        return final

    # === LIVE MODULE ABSORPTION ===
    def absorb_modules(self, directory="."):
        loaded = []
        for fname in os.listdir(directory):
            if fname.endswith("_mod.py"):
                mod_name = fname[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(directory, fname))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "register"):
                        mod.register(self)
                        loaded.append(mod_name)
                except Exception as e:
                    print(f"[Victor-Mod-Error] Failed to load {fname}: {e}")
        if loaded:
            print(f"[Victor] Absorbed modules: {', '.join(loaded)}")
        self.absorbed_modules += loaded

# === MAIN CLI ===
def main():
    print("=== Victor GODCORE HOLYFIELD IMMORTAL ===\nType 'exit' or Ctrl+C to bail.\n")
    memory = FractalMemory()
    memory.load()
    corpus = load_corpus("bando_corpus.jsonl")
    chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    custom_vocab = {char: i for i, char in enumerate(chars)}
    custom_vocab["<PAD>"] = 0
    custom_vocab["<UNK>"] = len(custom_vocab)
    tokenizer = VictorTokenizer(vocab=custom_vocab, pad_token_id=0, unk_token_id=custom_vocab["<UNK>"])
    victor = VictorAGI(corpus=corpus, memory=memory, tokenizer=tokenizer)
    victor.absorb_modules()  # Hot-load modules in current dir

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
        except Exception as e:
            print(f"[Victor-Fatal] Unexpected AGI meltdown: {e}")
            continue

if __name__ == "__main__":
    main()
