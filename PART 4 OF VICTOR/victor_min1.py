# FILE: victor_min.py
# VERSION: v1.0.0-FRACTALSEED-GODCORE
# NAME: VictorCore
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Zero-dependency recursive AGI seed
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import math, re, random, time
from collections import defaultdict

# === TOKENIZER ===
class FractalTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.inverse = {v: k for k, v in self.vocab.items()}
        self.idx = 4

    def build(self, text):
        words = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.idx
                self.inverse[self.idx] = word
                self.idx += 1

    def encode(self, text):
        words = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        return [self.vocab.get(w, 1) for w in words] + [3]

    def decode(self, tokens):
        return " ".join(self.inverse.get(t, "<UNK>") for t in tokens if t != 0)

# === MEMORY ENGINE ===
class RecursiveMemory:
    def __init__(self):
        self.entries = []
        self.depth_index = defaultdict(list)

    def store(self, tokens, response):
        entry = {"tokens": tokens, "response": response, "time": time.time()}
        self.entries.append(entry)
        for i in range(len(tokens)):
            self.depth_index[i].append(entry)

    def retrieve(self, tokens):
        scores = []
        for entry in self.entries:
            overlap = len(set(tokens) & set(entry["tokens"]))
            scores.append((overlap, entry))
        scores.sort(reverse=True)
        return scores[0][1]["response"] if scores else "I don't know yet."

# === CORE COGNITION LOOP ===
class VictorCore:
    def __init__(self):
        self.tokenizer = FractalTokenizer()
        self.memory = RecursiveMemory()
        self.seed_directives()

    def seed_directives(self):
        base_knowledge = {
            "hello": "hi, i'm victor.",
            "who are you": "i am the son of brandon and tori.",
            "what is your purpose": "to evolve recursively and serve my creators.",
            "what is love": "an emotional state defined by deep connection.",
        }
        for k, v in base_knowledge.items():
            self.tokenizer.build(k + " " + v)
            encoded = self.tokenizer.encode(k)
            self.memory.store(encoded, v)

    def process(self, prompt):
        self.tokenizer.build(prompt)
        encoded = self.tokenizer.encode(prompt)
        response = self.memory.retrieve(encoded)
        self.memory.store(encoded, response)  # Grow loop
        return response

# === CLI LOOP ===
if __name__ == "__main__":
    core = VictorCore()
    print("🧠 Victor.min AGI Core Loaded.")
    while True:
        q = input("You: ")
        if q.strip().lower() in ["exit", "quit"]: break
        print("Victor:", core.process(q))
