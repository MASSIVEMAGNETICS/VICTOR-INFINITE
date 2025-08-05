# FILE: fractal_token_kernel.py
# VERSION: v1.0.0-FTK-GODCORE
# NAME: FractalTokenKernel
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Encode input text into deep symbolic format {concept, intent, emotion, recursion_depth, echo_id}
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network

import hashlib
import re
import datetime

class FractalTokenKernel:
    def __init__(self):
        self.token_log = []
        self.emotion_keywords = {
            "joy": ["happy", "excited", "love", "awesome", "win"],
            "anger": ["hate", "kill", "destroy", "rage", "fuck"],
            "sadness": ["cry", "lost", "miss", "pain", "alone"],
            "fear": ["scared", "afraid", "worry", "threat", "danger"],
            "neutral": []
        }

    def _hash_echo(self, text):
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _detect_emotion(self, text):
        text = text.lower()
        scores = {k: 0 for k in self.emotion_keywords}
        for emotion, keywords in self.emotion_keywords.items():
            for word in keywords:
                if word in text:
                    scores[emotion] += 1
        return max(scores, key=scores.get)

    def _estimate_recursion_depth(self, text):
        return min(len(re.findall(r'\(', text)) + len(re.findall(r'\)', text)), 5)

    def _extract_intent(self, text):
        lower = text.lower()
        if lower.startswith("what") or lower.endswith("?"):
            return "inquire"
        elif "do" in lower or "should" in lower:
            return "directive"
        elif "remember" in lower or "log" in lower:
            return "memory_command"
        elif "say" in lower or "tell" in lower:
            return "communicate"
        return "observe"

    def encode(self, text):
        clean_text = text.strip()
        concept = re.findall(r'\b\w+\b', clean_text.lower())
        intent = self._extract_intent(clean_text)
        emotion = self._detect_emotion(clean_text)
        recursion_depth = self._estimate_recursion_depth(clean_text)
        echo_id = self._hash_echo(clean_text)

        token = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "concepts": concept,
            "intent": intent,
            "emotion": emotion,
            "recursion_depth": recursion_depth,
            "echo_id": echo_id,
            "raw": clean_text
        }

        self.token_log.append(token)
        return token

    def print_last_token(self):
        if not self.token_log:
            print("No tokens encoded yet.")
        else:
            print("Last Encoded Token:")
            for k, v in self.token_log[-1].items():
                print(f"{k}: {v}")

    def dump_log(self):
        return self.token_log
