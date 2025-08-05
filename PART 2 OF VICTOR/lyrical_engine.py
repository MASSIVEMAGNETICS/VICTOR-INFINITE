# File: bandocodex/creative/lyrical_engine.py
# Source: bandobandz_viral_rap_godcore_v5_0_0_GODMODE_EVOLVED.py

"""
The Lyrical Flow Engine.
Simulates flow, rhyme complexity, and thematic depth to generate fractal-recursive,
trauma-encoded, hood-slang, max-viral rap verses.
"""

import numpy as np
import random
from typing import List, Dict, Tuple

class LyricalConfig:
    def __init__(self):
        self.max_lines_per_verse = 16
        self.rhyme_density = 0.75
        self.thematic_shift_chance = 0.15
        self.viral_weights = {"rhyme_complexity": 0.4, "shock_value": 0.3, "narrative_cohesion": 0.2}
        self.viral_threshold = 0.8

class LyricalDatabase:
    def __init__(self):
        self.themes = {
            "hustle": ["feds on the wire", "sleepin' in the dark", "trap in my veins"],
            "pain": ["mom cryin’ on the bid", "pops in the ground", "anger in my blood"],
            "tech_fractal": ["ADHD waves", "mind glitchin’ and lit", "cracked code—hood mode"],
        }
        self.slang = ["off the rip", "still starvin", "run the static", "clutch the ratchet"]
        self.rhyme_schemes = {"AABB": [0, 0, 1, 1], "ABAB": [0, 1, 0, 1]}

class LyricalFlowEngine:
    def __init__(self, config: LyricalConfig, db: LyricalDatabase):
        self.config = config
        self.db = db
        self.verse_memory: List[Dict] = []
        self.current_rhyme_sounds: Dict[int, str] = {}
        self.current_theme: str = "hustle"

    def _get_rhyme_sound(self, line_text: str) -> str:
        words = line_text.split()
        return words[-1] if words else ""

    def _find_rhyming_word(self, target_sound: str) -> str:
        potential_rhymes = [w for w in self.db.slang if w.endswith(target_sound[-2:])]
        return random.choice(potential_rhymes) if potential_rhymes else random.choice(self.db.slang)

    def _generate_line(self, rhyme_group: int) -> str:
        base_concept = random.choice(self.db.themes[self.current_theme])
        if rhyme_group in self.current_rhyme_sounds:
            rhyme_target = self.current_rhyme_sounds[rhyme_group]
            rhyming_word = self._find_rhyming_word(rhyme_target)
            line = f"{base_concept}, {rhyming_word}"
        else:
            line = f"{base_concept}, {random.choice(self.db.slang)}"
            self.current_rhyme_sounds[rhyme_group] = self._get_rhyme_sound(line)
        return line

    def generate_verse(self, theme: str) -> List[Dict]:
        self.current_theme = theme
        self.verse_memory = []
        self.current_rhyme_sounds = {}
        scheme_name = random.choice(list(self.db.rhyme_schemes.keys()))
        rhyme_pattern = self.db.rhyme_schemes[scheme_name]
        for i in range(self.config.max_lines_per_verse):
            rhyme_group = rhyme_pattern[i % len(rhyme_pattern)]
            line_text = self._generate_line(rhyme_group)
            self.verse_memory.append({"text": line_text, "rhyme_group": rhyme_group, "theme": self.current_theme})
            if random.random() < self.config.thematic_shift_chance:
                self.current_theme = random.choice(list(self.db.themes.keys()))
        return self.verse_memory

class ViralAssessor:
    def __init__(self, config: LyricalConfig):
        self.config = config

    def assess(self, verse: List[Dict]) -> Tuple[float, str]:
        rhyme_groups = {line['rhyme_group'] for line in verse}
        rhyme_complexity = len(rhyme_groups) / self.config.max_lines_per_verse
        theme_changes = len(set(line['theme'] for line in verse))
        cohesion_score = 1.0 / theme_changes if theme_changes > 0 else 1.0
        shock_score = sum(1 for line in verse if any(p in line['text'] for p in LyricalDatabase().themes["pain"])) / len(verse)
        
        final_score = (rhyme_complexity * self.config.viral_weights["rhyme_complexity"] +
                       shock_score * self.config.viral_weights["shock_value"] +
                       cohesion_score * self.config.viral_weights["narrative_cohesion"])
        
        status = "🔥 VIRAL HIT" if final_score >= self.config.viral_threshold else "Gutter Track"
        return final_score, f"Assessment: Score={final_score:.2f} ({status})"

class GodTierBandoBandz:
    def __init__(self):
        self.config = LyricalConfig()
        self.db = LyricalDatabase()
        self.flow_engine = LyricalFlowEngine(self.config, self.db)
        self.assessor = ViralAssessor(self.config)

    def create_song(self, initial_theme: str = "hustle"):
        verse1_data = self.flow_engine.generate_verse(theme=initial_theme)
        _, report1 = self.assessor.assess(verse1_data)
        verse1_text = "\n".join([line['text'] for line in verse1_data])
        print("\n--- Verse 1 ---\n" + verse1_text + "\n" + report1)

if __name__ == '__main__':
    bando = GodTierBandoBandz()
    bando.create_song()