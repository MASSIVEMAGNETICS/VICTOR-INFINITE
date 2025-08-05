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

def tensorfield_from_context(embeddings_list):
    # embeddings_list: list of np.array vectors (all same dim)
    shape = (len(embeddings_list), 1)
    cell_dim = embeddings_list[0].shape[0]
    field = TensorField(shape, cell_dim)
    for i, vec in enumerate(embeddings_list):
        field.data[i, 0] = vec
    return field

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
def tensorfield_from_context(embeddings_list):
    # embeddings_list: list of np.array vectors (all same dim)
    shape = (len(embeddings_list), 1)
    cell_dim = embeddings_list[0].shape[0]
    field = TensorField(shape, cell_dim)
    for i, vec in enumerate(embeddings_list):
        field.data[i, 0] = vec
    return field

if __name__ == "__main__":
    main()
