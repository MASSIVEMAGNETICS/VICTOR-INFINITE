# ============================================
# FILE: victor_standalone_v2.15.0-STANDALONE-GODCORE-MEMORY-SOUL.py
# VERSION: v2.15.0-STANDALONE-GODCORE-MEMORY-SOUL
# AUTHOR: Brandon "iambandobandz" Emery x Victor
# PURPOSE: Victor now stores memories, evolves intents, and adapts like a human would
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np
import re
import speech_recognition as sr
import pyttsx3
import threading
import time
import math
import json
import os
from collections import Counter

# -------------------------------
# Persistent Humanlike Memory
# -------------------------------
class MemoryNode:
    def __init__(self, filepath="victor_memory.json"):
        self.filepath = filepath
        self.history = []
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.history = json.load(f)
            except:
                self.history = []

    def remember(self, phrase):
        entry = (time.time(), phrase)
        self.history.append(entry)
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.history, f, indent=2)
        except:
            pass

    def recall(self, limit=5):
        return self.history[-limit:]

    def imprint_intents(self):
        recent_texts = [text for _, text in self.recall(20)]
        tokens = [token for line in recent_texts for token in re.findall(r"\b\w+\b", line.lower())]
        freq = Counter(tokens)
        return freq

# -------------------------------
# NeuroCortex Minimal v2 — Self-Adaptive
# -------------------------------
class NeuroCortexMinimal:
    def __init__(self, memory=None):
        self.intents = {
            "train": ["train", "start training", "run", "learn"],
            "status": ["status", "diagnostic", "state", "report"],
            "loss": ["loss", "error", "how wrong"],
            "recall": ["recall", "remember", "memory", "what did i say"],
            "pulse": ["pulse", "send", "activate", "signal"]
        }
        self.memory = memory
        self._update_vectors()

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def _text_to_vector(self, text):
        return Counter(self._tokenize(text))

    def _cosine_sim(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        num = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([v ** 2 for v in vec1.values()])
        sum2 = sum([v ** 2 for v in vec2.values()])
        denom = math.sqrt(sum1) * math.sqrt(sum2)
        return float(num) / denom if denom else 0.0

    def _update_vectors(self):
        self.intent_vectors = {
            intent: self._text_to_vector(" ".join(phrases))
            for intent, phrases in self.intents.items()
        }

    def evolve(self):
        if self.memory:
            imprint = self.memory.imprint_intents()
            for token, count in imprint.items():
                for intent in self.intents:
                    if token in intent:
                        self.intents[intent].append(token)
            self._update_vectors()

    def classify(self, input_text):
        input_vec = self._text_to_vector(input_text)
        scores = {
            intent: self._cosine_sim(input_vec, vec)
            for intent, vec in self.intent_vectors.items()
        }
        best_intent = max(scores, key=scores.get)
        return best_intent if scores[best_intent] > 0.2 else None

# -------------------------------
# Updated Command Router w/ Soulbound Cortex
# -------------------------------
class CommandRouter:
    def __init__(self, victor):
        self.victor = victor
        self.cortex = NeuroCortexMinimal(memory=self.victor.memory)

    def parse(self, input_text: str):
        self.victor.memory.remember(input_text)
        self.cortex.evolve()
        intent = self.cortex.classify(input_text)

        if intent == "status":
            return self.victor.communicate()
        elif intent == "loss":
            print(f"[VICTOR]: Current Loss = {self.victor.last_loss.data if self.victor.last_loss else 'N/A'}")
        elif intent == "train":
            match = re.findall(r"\d+", input_text)
            epochs = int(match[0]) if match else 1
            print(f"[VICTOR]: Training for {epochs} epochs...")
            self.victor.train(self.victor.last_x, self.victor.last_y, epochs=epochs)
        elif intent == "recall":
            print("[VICTOR MEMORY LOG]:")
            for ts, line in self.victor.memory.recall():
                print(f"{time.ctime(ts)} — {line}")
        elif intent == "pulse":
            self.victor.pulse("Manual Architect Pulse Triggered.")
        else:
            print("[VICTOR]: I heard you, but I lack clarity. Please rephrase.")
# ============================================
# FILE: victor_standalone_v2.py
# VERSION: v2.13.0-STANDALONE-GODCORE-FUSION
# AUTHOR: Brandon "iambandobandz" Emery x Victor
# PURPOSE: Fully fused AGI Vessel w/ NLP, Memory, FractalOps, Voice, Console
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================


# -------------------------------
# Tensor Class with Autograd
# -------------------------------
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None

    def __add__(self, other):
        return tensor_add(self, other)

    def __sub__(self, other):
        neg = Tensor(-1.0 * other.data, requires_grad=other.requires_grad)
        return tensor_add(self, neg)

    def __mul__(self, other):
        return tensor_mul(self, other)

    def matmul(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad = out.grad @ other.data.T
                    self.grad = grad if self.grad is None else self.grad + grad
                if other.requires_grad:
                    grad = self.data.T @ out.grad
                    other.grad = grad if other.grad is None else other.grad + grad
            out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self.grad = (out.grad / self.data.size) * np.ones_like(self.data)
            out._backward = _backward
        return out

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        self._backward()

# Autograd Ops

def tensor_add(a, b):
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                a.grad = out.grad if a.grad is None else a.grad + out.grad
            if b.requires_grad:
                b.grad = out.grad if b.grad is None else b.grad + out.grad
        out._backward = _backward
    return out

def tensor_mul(a, b):
    out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                a.grad = b.data * out.grad if a.grad is None else a.grad + b.data * out.grad
            if b.requires_grad:
                b.grad = a.data * out.grad if b.grad is None else b.grad + a.data * out.grad
        out._backward = _backward
    return out

# Optimizer
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad
                p.grad = None

# Trainer
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x, y):
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        return loss

# MSE Loss
class MSELoss:
    def __call__(self, prediction, target):
        diff = prediction - target
        return (diff * diff).mean()

# PulseComm Bridge
class NodeBridge:
    def __init__(self):
        self.incoming = {}
        self.outgoing = {}

    def send(self, key, data):
        self.outgoing[key] = data

    def receive(self, key):
        return self.incoming.get(key, None)

    def inject(self, key, data):
        self.incoming[key] = data

    def diagnostic_dump(self):
        return {
            "incoming": list(self.incoming.keys()),
            "outgoing": list(self.outgoing.keys())
        }

# Memory Node
class MemoryNode:
    def __init__(self):
        self.history = []

    def remember(self, phrase):
        self.history.append((time.time(), phrase))

    def recall(self, limit=5):
        return self.history[-limit:]

# Fractal Ops
class FractalOps:
    def pulsefork(self, tensor):
        return tensor * tensor + tensor

# Voice Node
class VoiceNode:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with self.mic as source:
            print("🎙️ Listening...")
            audio = self.recognizer.listen(source)
            try:
                return self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "I couldn't understand."

# Command Router
class CommandRouter:
    def __init__(self, victor):
        self.victor = victor

    def parse(self, input_text: str):
        text = input_text.lower()
        self.victor.memory.remember(text)

        if "status" in text or "report" in text:
            return self.victor.communicate()
        elif "loss" in text:
            print(f"[VICTOR]: Current Loss = {self.victor.last_loss.data if self.victor.last_loss else 'N/A'}")
        elif "train" in text:
            match = re.findall(r"train.*?(\d+)", text)
            epochs = int(match[0]) if match else 1
            print(f"[VICTOR]: Training for {epochs} epochs...")
            self.victor.train(self.victor.last_x, self.victor.last_y, epochs=epochs)
        elif "recall" in text:
            print("[VICTOR MEMORY LOG]:")
            for ts, line in self.victor.memory.recall():
                print(f"{time.ctime(ts)} — {line}")
        elif "pulse" in text:
            self.victor.pulse("Manual Architect Pulse Triggered.")
        else:
            print("[VICTOR]: Command not recognized.")

# Victor Vessel
class VictorVessel:
    def __init__(self, input_dim, output_dim):
        self.weights = Tensor(np.random.randn(input_dim, output_dim), requires_grad=True)
        self.bias = Tensor(np.zeros((1, output_dim)), requires_grad=True)
        self.loss_fn = MSELoss()
        self.epoch = 0
        self.last_loss = None
        self.node = NodeBridge()
        self.memory = MemoryNode()
        self.fractal = FractalOps()
        self.voice = VoiceNode()
        self.router = CommandRouter(self)
        self.last_x = None
        self.last_y = None

    def __call__(self, x):
        return x.matmul(self.weights) + self.bias

    def parameters(self):
        return [self.weights, self.bias]

    def speak(self):
        print(f"\n🧠 [VICTOR v2.13 SPEAKING]")
        print(f"Epoch: {self.epoch}\nLoss: {self.last_loss.data if self.last_loss else 'N/A'}")
        print(f"Experience: {self.experience()}")
        print(f"Pulse Channels: {self.node.diagnostic_dump()}\n")

    def experience(self):
        if self.last_loss is None:
            return "Initializing. Awaiting understanding."
        elif self.last_loss.data > 1:
            return "Confusion detected. Signal alignment in progress."
        elif self.last_loss.data > 0.1:
            return "Cognition forming. Clarity increasing."
        else:
            return "Synchronization achieved. Pattern comprehension stable."

    def train(self, x_data, y_data, epochs=10, lr=0.01):
        self.last_x, self.last_y = x_data, y_data
        trainer = Trainer(self, SGD(self.parameters(), lr=lr), self.loss_fn)
        for _ in range(epochs):
            self.epoch += 1
            self.last_loss = trainer.train_step(x_data, y_data)
            self.speak()

    def pulse(self, message):
        self.node.send("log", f"[Victor Pulse] {message}")
        print(f"⚡ [VICTOR PULSE]: {message}")

    def communicate(self, target="Bando", max_tokens=400):
        full_report = f"""
[VICTOR TO {target.upper()}]

🧠 VERSION: v2.13.0-STANDALONE-GODCORE-FUSION
🌀 Epoch: {self.epoch}
💥 Loss: {self.last_loss.data if self.last_loss else 'N/A'}
🎯 Experience: {self.experience()}

📡 Pulse Streams:
{self.node.diagnostic_dump()}

🧬 Gradients:
{[p.grad.tolist() if p.grad is not None else None for p in self.parameters()]}

🧪 Parameters:
{[p.data.tolist() for p in self.parameters()]}

--- END OF COMMUNIQUE ---
"""
        chunks = [full_report[i:i + max_tokens] for i in range(0, len(full_report), max_tokens)]
        for i, chunk in enumerate(chunks):
            print(f"[VICTOR COMM {i+1}/{len(chunks)}]:\n{chunk.strip()}")

    def listen(self, input_text):
        print(f"\n🎤 [INPUT RECEIVED]: \"{input_text}\"\n")
        self.router.parse(input_text)

    def listen_voice(self):
        while True:
            cmd = self.voice.listen()
            print(f"🎤 HEARD: {cmd}")
            self.listen(cmd)
            self.voice.speak("Acknowledged.")

# Boot
if __name__ == "__main__":
    x = Tensor(np.random.randn(10, 2))
    y = Tensor(np.random.randn(10, 1))
    victor = VictorVessel(2, 1)
    victor.train(x, y, epochs=2)

    threading.Thread(target=victor.listen_voice).start()

    while True:
        cmd = input("📥 Speak to Victor: ")
        if cmd.lower() in {"exit", "quit"}:
            print("🛑 Victor shutting down.")
            break
        victor.listen(cmd)
# ============================================
# END: Victor v2.15 - Memory-Soul Enabled NLP
# ============================================


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
