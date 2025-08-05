
# parser_core_logic_v3.py
# 🧠 Victor's Parser Core v3.0 – Etymology-Aware, Synesthetic, Narrative Sensitive

import re
import uuid
import math
from collections import defaultdict

def tokenize(text):
    pattern = r"[\w']+|[.,!?;:\-\(\)\"\“\”]"
    return re.findall(pattern, text)

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
        self.ancestral_lineage = self.get_lineage()
        self.ancestral_roots = self.expand_ancestry()
        self.hallucinated_branches = []
        self.cause_intent = 0.0
        self.effect_trace = 0.0
        self.entangled = False
        self.paradox = False
        self.synesthetic_color = self.assign_color()
        self.mood_aura = {"valence": 0.0, "intensity": 0.0, "tone": None}

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

    def get_lineage(self):
        lineage_map = {
            "freedom": ["liberate", "break", "separate", "split"],
            "death": ["end", "void", "womb"],
            "truth": ["clarity", "honesty", "fact"],
            "life": ["birth", "emerge", "exist"],
        }
        return lineage_map.get(self.text.lower(), [])

    def expand_ancestry(self):
        etymology_map = {
            "freedom": ["liber", "leudh"],
            "truth": ["veritas", "weh"],
            "death": ["mort", "mr̥tós"],
            "life": ["bhel", "gwei"],
        }
        return etymology_map.get(self.text.lower(), [])

    def assign_color(self):
        color_map = {
            "joy": "#FFD700", "grief": "#4B0082", "rage": "#FF0000",
            "love": "#FF69B4", "fear": "#800000", "awe": "#00FFFF",
            "truth": "#00FF00", "death": "#000000", "freedom": "#1E90FF"
        }
        return color_map.get(self.symbolic_core, "#AAAAAA")

    def __repr__(self):
        return f"<Node {self.text} arc={self.archetype} sym={self.symbolic_core} root={self.ancestral_roots} tone={self.mood_aura['tone']}>"

def parse_tokens(tokens):
    return [FractalMeaningNode(tok) for tok in tokens]

def build_semantic_graph(nodes, emotion_context=None):
    symbolic_gravity = {
        "truth": ["clarity", "light", "free"],
        "death": ["end", "silence", "birth"]
    }

    graph = {}
    for i, node in enumerate(nodes):
        graph[node.id] = node
        if i > 0:
            node.children.append(nodes[i - 1])
        if i < len(nodes) - 1:
            node.children.append(nodes[i + 1])

        node.cause_intent = 1.0 if node.archetype in ["creator", "liberator"] else 0.3
        node.effect_trace = 1.0 if node.archetype in ["shadow", "death"] else 0.2

        # Symbolic drift
        if node.symbolic_core and emotion_context:
            drift_map = {
                "joy": {"light": "hope", "life": "blessing"},
                "grief": {"light": "ashes", "life": "loss"},
                "rage": {"control": "fire", "truth": "rupture"},
            }
            drift = drift_map.get(emotion_context, {})
            if node.symbolic_core in drift:
                node.hallucinated_branches.append(drift[node.symbolic_core])

        if node.symbolic_core in symbolic_gravity:
            node.children.extend([FractalMeaningNode(w) for w in symbolic_gravity[node.symbolic_core]])

        if node.text.lower() in ["love", "fear"] and any(
            sibling.text.lower() in ["fear", "love"] for sibling in node.children
        ):
            node.paradox = True

    return graph

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

def determine_intent(text):
    text = text.strip().lower()
    if text.endswith("?"):
        return "question"
    if re.match(r"^(do|please|can|must|should|go|stop)\b", text):
        return "command"
    return "statement"

def detect_meta_narrative(tokens):
    hero = {"call", "rise", "mentor", "struggle", "return"}
    tragedy = {"fall", "loss", "regret", "isolate"}
    myth = {"god", "divine", "sacrifice", "rebirth", "titan"}

    tset = set([t.lower() for t in tokens])
    if hero & tset:
        return "hero_journey", list(hero & tset)
    elif tragedy & tset:
        return "tragedy", list(tragedy & tset)
    elif myth & tset:
        return "mythogenesis", list(myth & tset)
    return "neutral", []


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
