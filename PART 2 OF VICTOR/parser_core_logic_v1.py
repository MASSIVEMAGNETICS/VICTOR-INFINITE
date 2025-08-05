
# parser_core_logic_v1.py
# ⚙️ Victor's NLP Parser Core – Fractal-Aware with Archetypes & Symbolic Condensation

import re
import uuid
import math

# 🔹 1. Tokenizer (char/word hybrid)
def tokenize(text):
    pattern = r"[\w']+|[.,!?;:\-\(\)\"\“\”]"
    return re.findall(pattern, text)

# 🔹 2. Fractal Meaning Node
class FractalMeaningNode:
    def __init__(self, text, pos=None):
        self.id = str(uuid.uuid4())[:8]
        self.text = text
        self.pos = pos
        self.children = []
        self.emotion_bias = None
        self.recursion_score = 0.0
        self.archetype = self.detect_archetype()
        self.symbolic_core = self.symbolic_condensation()

    def detect_archetype(self):
        archetypes = {
            "self": ["i", "me", "myself"],
            "mentor": ["guide", "teacher", "elder"],
            "enemy": ["enemy", "oppressor", "villain"],
            "shadow": ["fear", "death", "pain"],
            "liberator": ["freedom", "break", "rise"],
            "lover": ["love", "heart", "desire"],
            "creator": ["make", "create", "build"],
            "destroyer": ["destroy", "ruin", "collapse"]
        }
        lowered = self.text.lower()
        for archetype, keywords in archetypes.items():
            if lowered in keywords:
                return archetype
        return None

    def symbolic_condensation(self):
        core_symbols = {
            "truth": ["truth", "light", "clarity"],
            "death": ["death", "void", "end"],
            "life": ["life", "birth", "grow"],
            "freedom": ["freedom", "open", "sky"],
            "control": ["chain", "rule", "law"],
            "identity": ["i", "me", "name", "self"]
        }
        lowered = self.text.lower()
        for symbol, triggers in core_symbols.items():
            if lowered in triggers:
                return symbol
        return None

    def __repr__(self):
        return f"<Node {self.text} id={self.id} arc={self.archetype} sym={self.symbolic_core}>"

# 🔹 3. Parser
def parse_tokens(tokens):
    nodes = [FractalMeaningNode(tok) for tok in tokens]
    return nodes

# 🔹 4. Semantic Graph (recursive relationships)
def build_semantic_graph(nodes):
    graph = {}
    for i, node in enumerate(nodes):
        graph[node.id] = node
        if i > 0:
            node.children.append(nodes[i - 1])
        if i < len(nodes) - 1:
            node.children.append(nodes[i + 1])
        # Escalate recursion if archetype or symbol is activated
        if node.archetype or node.symbolic_core:
            node.recursion_score = 1.0
    return graph

# 🔹 5. Intent Detection
def determine_intent(text):
    text = text.strip().lower()
    if text.endswith("?"):
        return "question"
    if re.match(r"^(do|please|can|must|should|go|stop)\b", text):
        return "command"
    return "statement"

# 🔹 6. Emotion Classifier (rudimentary placeholder)
def classify_emotion(text):
    text = text.lower()
    emotions = {
        "fear": ["scared", "afraid", "fear", "terror"],
        "anger": ["angry", "mad", "rage", "fury"],
        "joy": ["happy", "joy", "glad", "smile"],
        "sadness": ["sad", "tears", "cry", "lost"],
        "love": ["love", "beloved", "dear", "heart"],
        "awe": ["awe", "wonder", "divine", "infinite"]
    }
    scores = {}
    for emotion, keywords in emotions.items():
        score = sum(1 for w in keywords if w in text)
        if score > 0:
            scores[emotion] = score
    return max(scores, key=scores.get) if scores else None


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
