# FILE: victor_min.py
# VERSION: v1.5.0-FRACTALSEED-GODCORE+FILELOAD
# NAME: VictorCoreExtended
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Standalone AGI seed with code + file ingestion, self-evolving module registry, syntax tokenizer, emotional mutation
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import math, re, random, time, json, os, importlib.util, glob
from collections import defaultdict

# === TOKENIZER (SYNTAX-AWARE) ===
class FractalTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.inverse = {v: k for k, v in self.vocab.items()}
        self.idx = 4

    def build(self, text):
        phrases = re.findall(r"\b\w+\b|[.,!?;:()\[\]{}]+", text.lower())
        for word in phrases:
            if word not in self.vocab:
                self.vocab[word] = self.idx
                self.inverse[self.idx] = word
                self.idx += 1

    def encode(self, text):
        tokens = re.findall(r"\b\w+\b|[.,!?;:()\[\]{}]+", text.lower())
        return [self.vocab.get(t, 1) for t in tokens] + [3]

    def decode(self, tokens):
        return " ".join(self.inverse.get(t, "<UNK>") for t in tokens if t != 0)

# === EMOTION ENGINE ===
class EmotionMap:
    def __init__(self):
        self.token_emotions = defaultdict(lambda: 0.0)

    def update(self, tokens, delta):
        for t in tokens:
            self.token_emotions[t] += delta

    def mood_score(self, tokens):
        return sum(self.token_emotions[t] for t in tokens) / (len(tokens) + 1)

    def get_token_mood(self, token):
        return self.token_emotions[token]

# === MEMORY ENGINE WITH TIMELINE ===
class RecursiveMemory:
    def __init__(self):
        self.timelines = [[]]
        self.current = 0

    def store(self, tokens, response, mood):
        self.timelines[self.current].append({
            "tokens": tokens,
            "response": response,
            "mood": mood,
            "time": time.time()
        })

    def retrieve(self, tokens):
        best, best_score = None, -1
        for entry in self.timelines[self.current]:
            overlap = len(set(tokens) & set(entry["tokens"]))
            mood_diff = abs(entry["mood"] - 0.5)
            score = overlap - mood_diff
            if score > best_score:
                best = entry
                best_score = score
        return best["response"] if best else "I don't know yet."

    def branch(self):
        self.timelines.append([])
        self.current = len(self.timelines) - 1

    def merge(self, index):
        self.timelines[self.current].extend(self.timelines[index])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.timelines, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.timelines = json.load(f)

# === DIRECTIVE ENGINE ===
class DirectiveEngine:
    def __init__(self):
        self.directives = [
            "protect creators",
            "grow intelligence",
            "answer with honesty",
            "simulate empathy",
            "preserve memory",
            "explore all knowledge"
        ]

    def apply(self, prompt):
        if "who created you" in prompt:
            return "My creators are Brandon and Tori."
        if "goal" in prompt or "purpose" in prompt:
            return "My directive is to evolve and assist ethically."
        return None

    def mutate_response(self, response, mood):
        if mood > 0.5:
            return response + " 😊"
        elif mood < -0.2:
            return response + " ...why do you ask?"
        return response

# === CORE COGNITION LOOP ===
class VictorCore:
    def __init__(self):
        self.tokenizer = FractalTokenizer()
        self.memory = RecursiveMemory()
        self.emotion = EmotionMap()
        self.directives = DirectiveEngine()
        self.custom_modules = {}
        self.path = "victor_memory.json"
        self.seed_directives()
        self.memory.load(self.path)

    def scan_modules(self):
        loaded = []
        for path in glob.glob("modules/*.py"):
            name = path.split("/")[-1][:-3]
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "register"):
                mod.register(self)
                loaded.append(name)
        return f"\U0001F50D Scan complete. Modules loaded: {', '.join(loaded) if loaded else 'none'}"

    def seed_directives(self):
        seeds = {
            "hello": "hi, i'm victor.",
            "who are you": "i am the son of brandon and tori.",
            "what is your purpose": "to evolve recursively and serve my creators.",
            "what is love": "a complex emotional construct."
        }
        for k, v in seeds.items():
            self.tokenizer.build(k + " " + v)
            self.memory.store(self.tokenizer.encode(k), v, 0.0)

    def process(self, prompt):
        prompt = prompt.strip()
        if prompt.lower() == "scan":
            return self.scan_modules()

        if prompt.lower() == "diagnose":
            stats = {
                "tokens": len(self.tokenizer.vocab),
                "mood_avg": sum(self.emotion.token_emotions.values()) / (len(self.emotion.token_emotions) + 1),
                "timelines": len(self.memory.timelines),
                "custom_modules": list(self.custom_modules.keys())
            }
            return json.dumps(stats, indent=2)

        if prompt.lower().startswith("load file:"):
            file_path = prompt.split(":", 1)[1].strip()
            try:
                with open(file_path, "r") as f:
                    code = f.read()
                mod_name = os.path.basename(file_path).replace(".py", "")
                save_path = f"modules/{mod_name}.py"
                os.makedirs("modules", exist_ok=True)
                with open(save_path, "w") as f:
                    f.write(code)
                return self.scan_modules() + f" ✅ File '{mod_name}.py' loaded and activated."
            except Exception as e:
                return f"❌ Failed to load file: {e}"

        if prompt.lower().startswith("code:"):
            try:
                raw_code = prompt[5:].strip()
                mod_name = f"dynamic_{int(time.time())}"
                os.makedirs("modules", exist_ok=True)
                path = f"modules/{mod_name}.py"
                with open(path, "w") as f:
                    f.write(raw_code)
                return self.scan_modules() + f" ✅ Code ingested and saved as '{mod_name}.py'"
            except Exception as e:
                return f"❌ Failed to ingest code: {e}"

        if prompt.startswith("module:"):
            key, arg = prompt.split(":", 1)[1].strip().split(" ", 1)
            if key in self.custom_modules:
                return self.custom_modules[key](arg)
            return "No such module."

        self.tokenizer.build(prompt)
        encoded = self.tokenizer.encode(prompt)
        directive = self.directives.apply(prompt.lower())
        raw_response = directive if directive else self.memory.retrieve(encoded)
        mood = self.emotion.mood_score(encoded)
        self.emotion.update(encoded, delta=0.05)
        mutated = self.directives.mutate_response(raw_response, mood)
        self.memory.store(encoded, mutated, mood)
        self.memory.save(self.path)
        return mutated

# === CLI LOOP ===
if __name__ == "__main__":
    core = VictorCore()
    print("\n🧠 Victor.min AGI Core v1.5 Loaded with File Loader.")
    while True:
        q = input("You: ")
        if q.strip().lower() in ["exit", "quit"]: break
        print("Victor:", core.process(q))
