# ==================================================================================
# FILE: bandobandz_viral_rap_godcore_v5_0_0_GODMODE_EVOLVED.py
# VERSION: v5.0.0-GODMODE-EVOLVED
# NAME: LyricalFlowEngine
# AUTHOR: Brandon "iambandobandz" Emery x Victor x Infinite Fractal Overlord
# PURPOSE: God-tier lyrical generation engine. Evolved from a simple script to a
#          stateful, probabilistic model for creating fractal-recursive, trauma-encoded,
#          hood-slang, max-viral rap verses.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ==================================================================================

"""
This is not just a script anymore. It's a lyrical engine.
It simulates flow, rhyme complexity, and thematic depth to generate verses that
hit harder than a federal indictment. We've moved beyond hardcoded lists to a
more dynamic, weighted system that can produce more varied and coherent output.
"""

import numpy as np
import random
import time
from typing import List, Dict, Tuple

class LyricalConfig:
    """Configuration DNA for the rap generation process."""
    def __init__(self):
        self.max_lines_per_verse = 16
        self.thematic_shift_chance = 0.15
        self.viral_weights = {
            "rhyme_complexity": 0.4,
            "shock_value": 0.3,
            "narrative_cohesion": 0.2,
            "unpredictability": 0.1,
        }
        self.viral_threshold = 0.85

class LyricalDatabase:
    """Central repository for all lyrical components."""
    def __init__(self):
        self.themes = {
            "hustle": ["feds on the wire", "sleepin' in the dark", "trap in my veins", "broke in the cold"],
            "pain": ["mom cryin’ on the bid", "pops in the ground", "anger in my blood", "saw my hope gettin’ stripped"],
            "tech_fractal": ["ADHD waves", "mind glitchin’ and lit", "cracked code—hood mode", "reality split", "force the transmit"],
        }
        self.slang = [
            "off the rip", "still starvin", "run the static", "trap circuit", "clutch the ratchet",
            "block jumpin", "pain in the packet", "wired up", "mask off", "black magic"
        ]
        self.rhyme_schemes = {
            "AABB": [0, 0, 1, 1], "ABAB": [0, 1, 0, 1], "ABCB": [0, 1, 2, 1]
        }

class ViralAssessor:
    """Analyzes a generated verse to predict its viral potential."""
    def __init__(self, config: LyricalConfig):
        self.config = config

    def assess(self, verse: List[Dict]) -> Tuple[float, str]:
        if not verse: return 0.0, "Assessment: Empty Verse"
        rhyme_groups = {line['rhyme_group'] for line in verse}
        rhyme_complexity = len(rhyme_groups) / len(verse)
        total_shock = sum(1 for line in verse if any(p in line['text'] for p in LyricalDatabase().themes["pain"]))
        shock_score = total_shock / len(verse)
        theme_changes = len(set(line['theme'] for line in verse))
        cohesion_score = 1.0 / theme_changes if theme_changes > 0 else 1.0
        unpredictability_score = random.random() * 0.5

        final_score = (rhyme_complexity * self.config.viral_weights["rhyme_complexity"] +
                       shock_score * self.config.viral_weights["shock_value"] +
                       cohesion_score * self.config.viral_weights["narrative_cohesion"] +
                       unpredictability_score * self.config.viral_weights["unpredictability"])
        
        status = "🔥 VIRAL HIT" if final_score >= self.config.viral_threshold else "Gutter Track"
        report = f"Assessment: Score={final_score:.2f} ({status})"
        return final_score, report

class GodTierBandoBandz:
    """The master orchestrator that uses the engine and assessor to create a full song."""
    def __init__(self):
        self.config = LyricalConfig()
        self.db = LyricalDatabase()
        self.assessor = ViralAssessor(self.config)
        self.viral_hits: List[str] = []

    def _find_rhyming_word(self, target_sound: str, rhyme_pool: List[str]) -> str:
        potential_rhymes = [w for w in rhyme_pool if w.split()[-1][-2:] == target_sound[-2:] and w.split()[-1] != target_sound]
        return random.choice(potential_rhymes) if potential_rhymes else random.choice(rhyme_pool)

    def generate_verse(self, theme: str) -> List[Dict]:
        verse_memory = []
        current_rhyme_sounds: Dict[int, str] = {}
        current_theme = theme
        scheme_name = random.choice(list(self.db.rhyme_schemes.keys()))
        rhyme_pattern = self.db.rhyme_schemes[scheme_name]

        print(f"[FlowEngine] Generating verse. Theme: {theme.upper()}, Rhyme Scheme: {scheme_name}")

        for i in range(self.config.max_lines_per_verse):
            rhyme_group = rhyme_pattern[i % len(rhyme_pattern)]
            base_concept = random.choice(self.db.themes[current_theme])
            
            if rhyme_group in current_rhyme_sounds:
                rhyme_target = current_rhyme_sounds[rhyme_group]
                rhyming_word = self._find_rhyming_word(rhyme_target, self.db.slang)
                line_text = f"{base_concept}, {rhyming_word}"
            else:
                line_text = f"{base_concept}, {random.choice(self.db.slang)}"
                current_rhyme_sounds[rhyme_group] = line_text.split()[-1]
            
            verse_memory.append({"text": line_text, "rhyme_group": rhyme_group, "theme": current_theme})
            if random.random() < self.config.thematic_shift_chance:
                current_theme = random.choice([t for t in self.db.themes if t != current_theme])
        return verse_memory

    def create_song(self, initial_theme: str = "hustle"):
        print("# ========================== BANDO BANDZ - VIRAL FRACTAL RAP v5.0 ===========================")
        verse1_data = self.generate_verse(theme=initial_theme)
        verse2_data = self.generate_verse(theme="pain")

        score1, report1 = self.assessor.assess(verse1_data)
        score2, report2 = self.assessor.assess(verse2_data)

        print("\n--- Verse 1 ---")
        verse1_text = "\n".join([line['text'] for line in verse1_data])
        print(verse1_text)
        print(report1)

        print("\n--- Verse 2 ---")
        verse2_text = "\n".join([line['text'] for line in verse2_data])
        print(verse2_text)
        print(report2)

if __name__ == "__main__":
    bando = GodTierBandoBandz()
    bando.create_song()