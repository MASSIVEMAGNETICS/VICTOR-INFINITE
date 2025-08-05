#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: VictorAudioGenesis_V5_QuantumHarmony.py
VERSION: v5.0.0-QUANTUMHARMONY-GODCORE_2029
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode) x Gemini Quantum Weaver
PURPOSE: 2029-level, proprietary, self-evolving, emotion/cognition-driven, explainable,
         modular AI music engine with advanced music theory, harmony, and structural generation.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network / Quantum Harmony Labs
"""

import random
import math
import numpy as np
import time
import threading
import json # For structured logging and state
from collections import deque

# --- CONFIGURATION ---
SR = 44100  # Sample Rate
STEREO = True
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING

# --- UTILITY & LOGGING ---
def scale_value(value, old_min, old_max, new_min, new_max):
    """Scales a value from one range to another."""
    if old_max == old_min: return new_min
    return (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

def log_message(module_name, level, message, data=None):
    if LOG_LEVEL == "DEBUG" or (LOG_LEVEL == "INFO" and level != "DEBUG"):
        log_entry = f"[{time.strftime('%H:%M:%S')}] [{level}] [{module_name}] {message}"
        if data:
            log_entry += f" | Data: {json.dumps(data, sort_keys=True, default=lambda o: '<not serializable>')[:100]}" # Truncate long data
        print(log_entry)

# ========== [QUANTUM EMOTION-COGNITION CORE (QECC)] ==========
class QuantumEmotionCognitionCore:
    """
    Manages the AI's emotional (VAD) and cognitive state (themes, motifs, harmony).
    Drives musical decisions based on this evolving internal landscape.
    """
    def __init__(self):
        self.state = {
            "vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}, # Valence, Arousal, Dominance (0-1)
            "current_key": "C",
            "current_mode": "major", # major, minor, dorian, etc.
            "current_scale_notes": self.get_scale_notes("C", "major"), # MIDI note numbers
            "current_chord_progression": [], # List of chord names or MIDI note sets
            "active_melodic_motifs": [], # List of short melodic sequences (tuples of (pitch, duration))
            "active_rhythmic_patterns": [], # List of rhythmic patterns (tuples of durations)
            "lyrical_themes": ["introspection", "cosmos"],
            "complexity_level": 0.5, # 0-1, influences density, ornamentation
            "recursion_depth_factor": 1.0 # Multiplier for recursive processes
        }
        self.state_timeline = deque(maxlen=20) # Store history of major state changes
        self.memory_bank = {} # Stores 'MusicalIdeaNodes' - key: idea_id, value: node_data
        self.log_message("QECC", "INFO", "Initialized.")

    def get_scale_notes(self, key_root_note="C", mode="major"):
        # Simplified mapping for demonstration
        notes = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11}
        root_midi = notes.get(key_root_note.upper(), 0) + 60 # Start at C4 octave

        intervals_map = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "blues": [0, 3, 5, 6, 7, 10] # Hexatonic blues
        }
        intervals = intervals_map.get(mode, intervals_map["major"])
        return sorted(list(set([(root_midi + i) % 12 + (octave * 12) for i in intervals for octave in range(3, 7)]))) # Multi-octave

    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            if key == "vad_update": # expects dict like {'valence_delta': 0.1, ...}
                for vad_key, delta in value.items():
                    self.state["vad"][vad_key] = np.clip(self.state["vad"].get(vad_key,0.5) + delta, 0, 1)
            elif key in self.state:
                self.state[key] = value
        
        # If key/mode changed, update scale notes
        if "current_key" in kwargs or "current_mode" in kwargs:
            self.state["current_scale_notes"] = self.get_scale_notes(self.state["current_key"], self.state["current_mode"])

        self.state_timeline.append(json.loads(json.dumps(self.state, default=lambda o: '<state_obj>'))) # Store a deep copy
        self.log_message("QECC", "DEBUG", "State updated", kwargs)

    def store_musical_idea(self, idea_type, content, related_tags=None):
        idea_id = f"{idea_type}_{int(time.time()*1000)}_{random.randint(0,999)}"
        self.memory_bank[idea_id] = {
            "type": idea_type, # "lyric_fragment", "melodic_motif", "rhythmic_pattern", "chord_sequence"
            "content": content, # The actual data (text, list of notes, etc.)
            "vad_at_creation": dict(self.state["vad"]),
            "tags": related_tags or [],
            "timestamp": time.time(),
            "usage_count": 0
        }
        self.log_message("QECC", "DEBUG", f"Stored musical idea: {idea_id}")

    def retrieve_musical_idea(self, idea_type, current_vad, num_ideas=1, similarity_threshold=0.3):
        # Simplified retrieval: find ideas of the type, somewhat matching VAD
        # A real system would use embeddings and semantic search.
        candidates = []
        for idea_id, idea_data in self.memory_bank.items():
            if idea_data["type"] == idea_type:
                vad_c = idea_data["vad_at_creation"]
                # Simple VAD distance (Euclidean)
                dist = math.sqrt( (vad_c['valence'] - current_vad['valence'])**2 + \
                                  (vad_c['arousal'] - current_vad['arousal'])**2 + \
                                  (vad_c['dominance'] - current_vad['dominance'])**2 )
                # Lower distance is better, convert to similarity (0-1, 1 is best)
                similarity = np.clip(1.0 - (dist / math.sqrt(3)), 0, 1) # Max distance is sqrt(1^2+1^2+1^2)
                if similarity >= similarity_threshold:
                    candidates.append((idea_data, similarity))
        
        candidates.sort(key=lambda x: x[1], reverse=True) # Sort by similarity
        if candidates:
             # Increment usage count for retrieved ideas
            for idea_data, _ in candidates[:num_ideas]:
                idea_data["usage_count"] +=1 # This modifies the dict in memory_bank directly
            return [c[0]['content'] for c in candidates[:num_ideas]]
        return []

    def explain_state(self):
        return f"Current Emotion (V/A/D): {self.state['vad']['valence']:.2f}/{self.state['vad']['arousal']:.2f}/{self.state['vad']['dominance']:.2f}. Key: {self.state['current_key']} {self.state['current_mode']}. Complexity: {self.state['complexity_level']:.2f}."


# ========== [SEMANTIC LYRIC WEAVER (SLW)] ==========
class SemanticLyricWeaver:
    def __init__(self, qecc_ref, topic="transformation", persona="QuantumBando"):
        self.qecc = qecc_ref
        self.topic = topic
        self.persona = persona
        self.logs = [] # For explainability
        self.log_message("SLW", "INFO", f"Initialized with topic: {topic}, persona: {persona}.")

    def _get_themed_vocab(self):
        # Dynamically build vocab based on QECC state (themes, VAD)
        base_vocab = {"time":1, "space":1, "mind":1, "light":1, "dark":1, "dream":1, "real":1, "echo":1, "fractal":1, "code":1, "flow":1}
        for theme in self.qecc.state["lyrical_themes"]:
            if theme == "cosmos": base_vocab.update({"star":1, "nebula":1, "galaxy":1, "void":1, "pulse":1})
            if theme == "introspection": base_vocab.update({"soul":1, "self":1, "thought":1, "within":1, "reflect":1})
            if theme == "transformation": base_vocab.update({"change":1, "shift":1, "evolve":1, "become":1, "new":1})

        # VAD influence on vocab choice (conceptual: weight words by emotional association)
        vad = self.qecc.state["vad"]
        if vad['valence'] > 0.7: base_vocab.update({"joy":2, "shine":2, "soar":2}) # Higher weight
        if vad['valence'] < 0.3: base_vocab.update({"pain":2, "break":2, "fade":2})
        if vad['arousal'] > 0.7: base_vocab.update({"storm":2, "surge":2, "wild":2})
        if vad['arousal'] < 0.3: base_vocab.update({"still":2, "calm":2, "hush":2})
        
        # Retrieve related lyric fragments from memory
        retrieved_fragments = self.qecc.retrieve_musical_idea("lyric_fragment", vad, num_ideas=2)
        for fragment in retrieved_fragments:
            for word in fragment.split(): base_vocab[word.lower()] = base_vocab.get(word.lower(), 0) + 1 # Add/boost words from memory
        
        return list(base_vocab.keys()) # For now, just use keys; weights could be used for probabilistic choice

    def _generate_fractal_line(self, vocab, line_length_avg=7, rhyme_target=None):
        # More sophisticated line generation using Markov-like approach or weighted random choice
        line_words = []
        current_length = 0
        max_len = line_length_avg + random.randint(-2,2)
        
        # Start line with a thematically relevant word or one matching VAD profile
        # This is still simplified. A real system would have semantic word embeddings.
        if not vocab: return "Silent thoughts echo."
        
        # Attempt to find a starting word matching current VAD. This is highly conceptual.
        # For now, just pick a random word.
        current_word = random.choice(vocab)
        line_words.append(current_word)
        current_length += 1
        
        while current_length < max_len:
            # Simplified: pick next word that "fits" (randomly from vocab for now)
            # Future: use n-grams or semantic similarity
            next_word_candidates = [w for w in vocab if w != current_word] # Avoid immediate repetition
            if not next_word_candidates: break
            
            # If rhyme_target, try to find a rhyming word (very basic end-sound match)
            if rhyme_target and current_length >= max_len -1 : # Try to rhyme last word
                rhyming_words = [w for w in next_word_candidates if w.endswith(rhyme_target[-2:])] # Simple 2-char end rhyme
                if rhyming_words:
                    current_word = random.choice(rhyming_words)
                else: # No rhyme found, pick randomly
                    current_word = random.choice(next_word_candidates)
            else:
                current_word = random.choice(next_word_candidates)

            line_words.append(current_word)
            current_length += 1
        
        line = " ".join(line_words).capitalize()
        # Store generated fragment
        self.qecc.store_musical_idea("lyric_fragment", line, self.qecc.state["lyrical_themes"])
        return line
        
    def generate_lyrics(self, num_lines=8, structure="AABB"): # structure e.g. AABB, ABAB
        self.log_message("SLW", "INFO", f"Generating {num_lines} lines of lyrics. Structure: {structure}.")
        vocab = self._get_themed_vocab()
        verse = []
        rhyme_map = {} # Stores rhyme for 'A', 'B', etc.
        
        for i in range(num_lines):
            line_rhyme_char = structure[i % len(structure)] # A, B, etc.
            rhyme_target = None
            # If this line needs to rhyme with a previous line of the same char
            if line_rhyme_char in rhyme_map and (i // len(structure)) > (list(rhyme_map.keys()).index(line_rhyme_char) // len(structure)) : # Crude check if it's a subsequent rhyme
                 rhyme_target_line = rhyme_map[line_rhyme_char]
                 if rhyme_target_line and len(rhyme_target_line.split()) > 0:
                    rhyme_target = rhyme_target_line.split()[-1] # Last word of the target line

            line = self._generate_fractal_line(vocab, rhyme_target=rhyme_target)
            
            # If this line sets a new rhyme for its character
            if line_rhyme_char not in rhyme_map or (i % len(structure) < len(structure)/2) : # Store first instance of A, B as rhyme masters
                if len(line.split()) > 0:
                    rhyme_map[line_rhyme_char] = line # Store the whole line, rhyme target will be its last word

            verse.append(line)
            self.logs.append({"line_num": i+1, "text": line, "rhyme_char": line_rhyme_char, "rhyme_target": rhyme_target, "emotion_at_gen": dict(self.qecc.state["vad"])})
        
        self.qecc.update_state(memory_lyrics=verse) # Update QECC with the generated lyrics
        self.log_message("SLW", "INFO", "Lyrics generation complete.")
        return verse

    def explain(self):
        explanation = ["SLW Explainability Log:"]
        for log_entry in self.logs[-10:]: # Last 10 log entries for brevity
            explanation.append(f"  Line {log_entry['line_num']} ('{log_entry['text'][:30]}...'): Rhyme '{log_entry['rhyme_char']}', Target='{log_entry['rhyme_target']}'. VAD(V/A/D): {log_entry['emotion_at_gen']['valence']:.2f}/{log_entry['emotion_at_gen']['arousal']:.2f}/{log_entry['emotion_at_gen']['dominance']:.2f}")
        return "\n".join(explanation)
    
    def log_message(self, level, message, data=None): # Helper for internal logging if needed
        log_message("SLW", level, message, data)


# ========== [HARMONIC MELODY MORPHING ENGINE (HMME)] ==========
class HarmonicMelodyMorphingEngine:
    def __init__(self, qecc_ref, measures=4, beats_per_measure=4, subdivisions_per_beat=4):
        self.qecc = qecc_ref
        self.measures = measures
        self.beats_per_measure = beats_per_measure
        self.subdivisions_per_beat = subdivisions_per_beat # e.g., 4 = 16th notes
        self.total_steps = measures * beats_per_measure * subdivisions_per_beat
        self.logs = []
        self.log_message("HMME", "INFO", f"Initialized for {measures} measures.")

    def _select_note_from_scale(self, scale_notes, last_note=None, preferred_interval_max=5):
        # Prefers notes closer to the last note, within the scale
        if not scale_notes: return random.randint(55, 75) # Fallback if scale is empty
        if last_note is None: return random.choice(scale_notes)

        candidates = [n for n in scale_notes if abs(n - last_note) <= preferred_interval_max]
        if not candidates: # If no notes within preferred interval, expand search
            candidates = [n for n in scale_notes if abs(n - last_note) <= 12] # Within octave
        
        return random.choice(candidates if candidates else scale_notes)


    def _generate_rhythmic_pattern(self, num_steps, base_density=0.6):
        # base_density: 0 (sparse) to 1 (dense)
        # density influenced by arousal
        density = np.clip(base_density + (self.qecc.state["vad"]["arousal"] - 0.5) * 0.4, 0.2, 0.9)
        pattern = [] # (duration_in_steps, is_rest)
        steps_remaining = num_steps
        while steps_remaining > 0:
            is_rest = random.random() > density # Higher density = less rests
            # Duration: 1 to 4 steps (16th to quarter note if 4 subdivisions/beat)
            # Shorter notes more likely with high arousal/complexity
            max_dur = 4 - int(self.qecc.state["vad"]["arousal"] * 2 + self.qecc.state["complexity_level"] * 2)
            max_dur = max(1, max_dur)
            
            duration = random.randint(1, min(max_dur, steps_remaining))
            pattern.append((duration, is_rest))
            steps_remaining -= duration
        return pattern

    def _generate_chord_progression(self, num_chords=4):
        # Very basic chord progression generator based on key/mode
        # I, IV, V, vi are common in major. i, VI, VII, iv, v in minor.
        # This needs a proper music theory rule engine.
        key = self.qecc.state["current_key"]
        mode = self.qecc.state["current_mode"]
        scale_root_midi = self.qecc.get_scale_notes(key, mode)[0] % 12 # 0-11 relative to C

        # Roman numeral mapping to scale degrees (0-indexed)
        # For simplicity, just using scale degrees for now
        chords_in_key_major = [0, 3, 4, 7] # I, IV, V, (vi relative to root) -> Using 0,5,7,9 semitones for I,IV,V,vi from root of scale
                                        # This is simplified, usually it's based on scale degrees
        chords_in_key_minor = [0, 5, 7, 3] # i, VI, VII, (iv relative to root) -> using 0,8,10,5 for i,VI,VII,iv

        possible_chord_roots_relative = chords_in_key_major if "major" in mode else chords_in_key_minor
        progression_midi_roots = []
        for _ in range(num_chords):
            # Choose a root note for the chord from the scale
            # This is a placeholder for a more musically intelligent selection
            chord_root_degree_idx = random.choice(possible_chord_roots_relative)
            # Get the actual MIDI note from the scale based on the degree
            # Example: degree 0 is scale_notes[0], degree 1 is scale_notes[1], etc.
            # This is not standard music theory for chord roots directly, but works for this procedural example
            # A better way: use the scale intervals to find chord roots.
            # If scale is Cmaj (0,2,4,5,7,9,11), then I is on 0, IV on 5, V on 7.
            # For simplicity, let's just pick notes from the current scale as chord "centers"
            chord_center_note = random.choice(self.qecc.state["current_scale_notes"])
            progression_midi_roots.append(chord_center_note)

            # For explainability, translate MIDI root to note name (approximate)
            # For now, just store the MIDI root.
            
        self.qecc.update_state(current_chord_progression=progression_midi_roots)
        self.log_message("HMME", "DEBUG", f"Generated chord progression (centers): {progression_midi_roots}")
        return progression_midi_roots


    def generate_melody_sequence(self):
        self.log_message("HMME", "INFO", "Generating melody sequence.")
        scale_notes = self.qecc.state["current_scale_notes"]
        complexity = self.qecc.state["complexity_level"]
        vad = self.qecc.state["vad"]

        # Generate a simple chord progression to guide melody
        # Each chord lasts for one measure (beats_per_measure * subdivisions_per_beat steps)
        chord_prog_roots = self._generate_chord_progression(num_chords=self.measures) 
        
        melody_midi = [] # List of (pitch, duration_in_steps, velocity) tuples
        current_step = 0
        last_note_pitch = random.choice(scale_notes) if scale_notes else 60

        # Retrieve or generate melodic motifs
        active_motifs = self.qecc.retrieve_musical_idea("melodic_motif", vad, num_ideas=1)
        
        for measure_idx in range(self.measures):
            # Get current chord context (simplified: just the root of the "chord area")
            current_chord_root = chord_prog_roots[measure_idx % len(chord_prog_roots)]
            # Notes around the chord root are more likely
            # This is a simplification; real harmony is more complex.
            measure_target_notes = [n for n in scale_notes if abs(n - current_chord_root) < 7] # Notes within a 5th of chord root
            if not measure_target_notes: measure_target_notes = scale_notes


            # Generate rhythm for this measure
            rhythmic_pattern_steps = self._generate_rhythmic_pattern(self.beats_per_measure * self.subdivisions_per_beat)
            
            measure_melody_pos = 0
            for duration_steps, is_rest in rhythmic_pattern_steps:
                if is_rest:
                    pitch = 0 # MIDI rest
                else:
                    # Use motif? (Conceptual)
                    if active_motifs and random.random() < 0.3 * complexity: # Higher complexity = more motif use
                        # Simplified: pick a note from the motif if it fits the current harmonic context
                        motif_note_info = random.choice(active_motifs[0]) # active_motifs[0] is list of (pitch, dur)
                        pitch = motif_note_info[0] # Use pitch from motif
                        # Check if motif pitch is in scale or close to target notes
                        if pitch not in measure_target_notes: # If not, pick a new note
                            pitch = self._select_note_from_scale(measure_target_notes if measure_target_notes else scale_notes, last_note_pitch)
                    else:
                        pitch = self._select_note_from_scale(measure_target_notes if measure_target_notes else scale_notes, last_note_pitch)
                    last_note_pitch = pitch

                # Velocity influenced by arousal and dominance
                velocity = int(scale_value(vad["arousal"] * 0.6 + vad["dominance"] * 0.4, 0, 1, 60, 120))
                melody_midi.append((pitch, duration_steps, velocity))
                
                self.logs.append({
                    "step_abs": current_step + measure_melody_pos, "measure": measure_idx, 
                    "pitch": pitch, "duration_steps": duration_steps, "velocity": velocity,
                    "chord_context_root": current_chord_root, "emotion_at_gen": dict(vad)
                })
                measure_melody_pos += duration_steps
            current_step += self.beats_per_measure * self.subdivisions_per_beat
        
        # Store the generated melody as a motif idea
        # To be a proper motif, it should be shorter; this stores the whole melody
        self.qecc.store_musical_idea("melodic_motif", melody_midi, ["generated_melody_main"])

        self.qecc.update_state(memory_melody=melody_midi)
        self.log_message("HMME", "INFO", "Melody sequence generation complete.")
        return melody_midi

    def explain(self):
        explanation = ["HMME Explainability Log:"]
        # Summarize, don't dump all notes
        if self.logs:
            explanation.append(f"  Generated {len(self.logs)} melodic events across {self.measures} measures.")
            last_event = self.logs[-1]
            explanation.append(f"  Example event (last): Pitch {last_event['pitch']}, Duration {last_event['duration_steps']} steps, Velocity {last_event['velocity']}. Chord Context (Root): {last_event['chord_context_root']}.")
            explanation.append(f"  Emotion (V/A/D at last event): {last_event['emotion_at_gen']['valence']:.2f}/{last_event['emotion_at_gen']['arousal']:.2f}/{last_event['emotion_at_gen']['dominance']:.2f}")
        return "\n".join(explanation)

    def log_message(self, level, message, data=None):
        log_message("HMME", level, message, data)


# ========== [NEURAL VOICE MODULATOR (NVM) - Conceptual] ==========
class NeuralVoiceModulator:
    def __init__(self, qecc_ref, sample_rate=SR):
        self.qecc = qecc_ref
        self.sample_rate = sample_rate
        self.logs = []
        self.base_voiceprints = { # Parameters: [formant_shift_factor, noise_amount, vibrato_rate_hz, vibrato_depth_semitones]
            "QuantumBando": [1.0, 0.05, 5.0, 0.2],
            "SereneOracle": [1.1, 0.02, 3.0, 0.1],
            "GlitchWraith": [0.85, 0.2, 8.0, 0.5]
        }
        self.log_message("NVM", "INFO", "Initialized.")

    def midi_to_freq(self, midi_note):
        if midi_note == 0: return 0 # Rest
        return 440 * (2 ** ((midi_note - 69) / 12))

    def synthesize_voice(self, lyrics_text_array, melody_midi_sequence, persona="QuantumBando"):
        self.log_message("NVM", "INFO", f"Synthesizing voice for {len(lyrics_text_array)} lines, persona: {persona}.")
        voiceprint_params = self.base_voiceprints.get(persona, self.base_voiceprints["QuantumBando"])
        
        total_audio_duration_steps = sum(m[1] for m in melody_midi_sequence)
        step_duration_sec = 0.5 / self.qecc.state.get("tempo_bpm", 120) * (SR / self.sample_rate) # Duration of one 16th note at 120bpm is 0.125s. Let's adjust step_duration_sec.
                                                                                                # Assume a step is a 16th note. Beats per minute. A beat is a quarter note.
                                                                                                # So, 120bpm means 120 quarter notes per minute. Or 2 quarter notes per second.
                                                                                                # A 16th note is 1/4 of a quarter note. So, 0.5s / 4 = 0.125s per 16th.
        # Simplified: let step_duration_sec be tied to complexity/arousal.
        # Higher arousal/complexity = shorter phonetic segments.
        step_duration_sec = 0.15 * (1.0 - self.qecc.state["vad"]["arousal"] * 0.5) # Shorter for higher arousal
        step_duration_sec = max(0.05, step_duration_sec)


        output_waveform = np.zeros(int(total_audio_duration_steps * step_duration_sec * self.sample_rate))
        current_sample_pos = 0
        
        lyric_idx = 0 # Which line of lyrics are we on

        for i, (pitch_midi, duration_steps, velocity) in enumerate(melody_midi_sequence):
            current_lyric_line = lyrics_text_array[lyric_idx % len(lyrics_text_array)] if lyrics_text_array else "la"
            # Conceptual: map part of lyric line to this note's duration
            # For simplicity, just use "ah" or "oo" based on VAD or pitch
            phoneme_sound = "a" if self.qecc.state["vad"]["valence"] > 0.5 else "u"
            
            freq_hz = self.midi_to_freq(pitch_midi)
            if freq_hz == 0: # Rest
                current_sample_pos += int(duration_steps * step_duration_sec * self.sample_rate)
                continue

            note_duration_sec = duration_steps * step_duration_sec
            num_samples_note = int(note_duration_sec * self.sample_rate)
            t = np.linspace(0, note_duration_sec, num_samples_note, endpoint=False)

            # Base waveform: Additive synthesis with more partials
            signal = np.zeros_like(t)
            num_partials = 3 + int(self.qecc.state["complexity_level"] * 4) # More complex = more partials
            
            # Formant conceptual influence (very simplified)
            formant_center = freq_hz * (voiceprint_params[0] + (self.qecc.state["vad"]["valence"]-0.5)*0.2) # Shift formant based on VAD
            
            for k in range(1, num_partials + 1):
                partial_freq = freq_hz * k
                # Simple filter envelope for formants: boost frequencies near formant_center
                formant_gain = 1.0 / (1.0 + ((partial_freq - formant_center)/ (formant_center*0.5))**2 ) # Basic bell curve
                formant_gain = np.clip(formant_gain, 0.1, 1.0)
                
                amplitude = (1.0 / k) * formant_gain * (velocity / 127.0)
                signal += amplitude * np.sin(2 * np.pi * partial_freq * t + random.uniform(0, np.pi/2)) # Random phase

            # Noise component
            noise_amp = voiceprint_params[1] * (self.qecc.state["vad"]["arousal"] * 0.5) # More noise with arousal
            signal += noise_amp * (np.random.rand(num_samples_note) * 2 - 1)
            
            # Vibrato
            vibrato_rate = voiceprint_params[2]
            vibrato_depth_hz = (2**(voiceprint_params[3]/12)-1) * freq_hz # Semitones to Hz deviation
            if vibrato_rate > 0 and vibrato_depth_hz > 0:
                phase_mod = (vibrato_depth_hz / vibrato_rate) * np.sin(2 * np.pi * vibrato_rate * t)
                signal_vib = np.zeros_like(t) # Reconstruct with vibrato
                for k_vib in range(1, num_partials + 1): # Apply vibrato to all partials consistently
                    partial_freq_vib = freq_hz * k_vib
                    formant_gain_vib = 1.0 / (1.0 + ((partial_freq_vib - formant_center)/ (formant_center*0.5))**2 )
                    amplitude_vib = (1.0 / k_vib) * formant_gain_vib * (velocity / 127.0)
                    signal_vib += amplitude_vib * np.sin(2 * np.pi * partial_freq_vib * t + phase_mod * k_vib) # Modulate phase of each partial
                signal = signal_vib + noise_amp * (np.random.rand(num_samples_note) * 2 - 1)


            # Simple attack/decay envelope per note
            attack_len = min(num_samples_note // 10, int(0.02 * self.sample_rate)) # Short attack
            decay_len = min(num_samples_note // 10, int(0.03 * self.sample_rate))  # Short decay
            if num_samples_note > attack_len + decay_len:
                env = np.ones(num_samples_note)
                env[:attack_len] = np.linspace(0, 1, attack_len)
                env[-decay_len:] = np.linspace(1, 0, decay_len)
                signal *= env

            end_sample_for_note = current_sample_pos + num_samples_note
            if end_sample_for_note <= len(output_waveform):
                output_waveform[current_sample_pos : end_sample_for_note] += signal
            else: # Truncate if it exceeds total allocated length
                fit_samples = len(output_waveform) - current_sample_pos
                if fit_samples > 0:
                    output_waveform[current_sample_pos:] += signal[:fit_samples]

            current_sample_pos = end_sample_for_note
            
            # Conceptual: Advance lyric line based on some heuristic (e.g., every few notes or on longer notes)
            if duration_steps > self.qecc.state.get("subdivisions_per_beat",4) : # If note is longer than a beat
                lyric_idx = (lyric_idx + 1)

            self.logs.append({"note_idx": i, "lyric_segment": current_lyric_line[:10]+"...", "pitch_midi": pitch_midi, "freq_hz": freq_hz, "duration_sec": note_duration_sec, "persona": persona, "VAD": dict(self.qecc.state["vad"])})

        if np.max(np.abs(output_waveform)) > 0:
            output_waveform /= np.max(np.abs(output_waveform)) # Normalize

        self.qecc.update_state(memory_voice_audio_stats={"length_samples": len(output_waveform), "max_amplitude": np.max(np.abs(output_waveform))})
        self.log_message("NVM", "INFO", f"Voice synthesis complete. Output samples: {len(output_waveform)}.")
        return output_waveform

    def explain(self):
        explanation = ["NVM Explainability Log:"]
        if self.logs:
            explanation.append(f"  Synthesized {len(self.logs)} vocal events.")
            last_event = self.logs[-1]
            explanation.append(f"  Example event (last): Lyric '{last_event['lyric_segment']}', MIDI {last_event['pitch_midi']} ({last_event['freq_hz']:.0f} Hz), Duration {last_event['duration_sec']:.2f}s. Persona: {last_event['persona']}.")
            explanation.append(f"  Emotion (V/A/D at last event): {last_event['VAD']['valence']:.2f}/{last_event['VAD']['arousal']:.2f}/{last_event['VAD']['dominance']:.2f}")
        return "\n".join(explanation)

    def log_message(self, level, message, data=None):
        log_message("NVM", level, message, data)

# ========== [ADAPTIVE INSTRUMENTAL ENSEMBLE (AIE)] ==========
class AdaptiveInstrumentalEnsemble:
    def __init__(self, qecc_ref, sample_rate=SR):
        self.qecc = qecc_ref
        self.sample_rate = sample_rate
        self.logs = []
        # Instrument sounds (conceptual: parameters for simple synthesis)
        self.instrument_synthesis_params = {
            "kick": {"base_freq": 60, "decay_time": 0.2, "harmonics": 1},
            "snare": {"noise_bw": (800, 5000), "decay_time": 0.15, "tone_freq": 180}, # (min_freq, max_freq) for filtered noise
            "hihat": {"noise_bw": (6000, 15000), "decay_time": 0.05},
            "bass_synth": {"waveform": "saw", "filter_cutoff_base": 400, "filter_q": 2.0, "decay_time": 0.4},
            "pad_synth": {"waveform": "sine_stack", "num_oscs": 5, "detune_factor": 0.01, "attack_time": 0.5, "release_time": 1.0}
        }
        self.log_message("AIE", "INFO", "Initialized.")

    def _synthesize_sound(self, instrument_name, freq_hz, duration_sec, velocity_norm): # velocity_norm 0-1
        params = self.instrument_synthesis_params.get(instrument_name)
        if not params: return np.zeros(int(duration_sec * self.sample_rate))

        num_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        signal = np.zeros(num_samples)

        if instrument_name == "kick":
            env = np.exp(-t / params["decay_time"])
            signal = np.sin(2 * np.pi * freq_hz * t * np.exp(-t * 5)) * env # Pitch envelope
        elif instrument_name == "snare":
            # Noise component
            noise = np.random.uniform(-1, 1, num_samples)
            # Conceptual bandpass filter for noise - for simplicity, just scale by envelope
            # A real filter is complex. Here, we simulate its effect on decay.
            noise_env = np.exp(-t / params["decay_time"])
            signal += noise * noise_env * 0.7
            # Tone component
            tone_env = np.exp(-t / (params["decay_time"]*0.5))
            signal += np.sin(2 * np.pi * params["tone_freq"] * t) * tone_env * 0.3
        elif instrument_name == "hihat":
            noise = np.random.uniform(-1, 1, num_samples)
            # Conceptual high-pass filter - very simple envelope
            hihat_env = np.exp(-t / params["decay_time"])
            signal = noise * hihat_env
        elif instrument_name == "bass_synth":
            # Basic sawtooth wave (sum of sines)
            for k in range(1, 6): # First 5 harmonics for saw
                signal += ((-1)**(k+1)) * (1/k) * np.sin(2 * np.pi * freq_hz * k * t)
            # Conceptual filter: simple envelope shaping based on VAD (arousal closes filter)
            filter_env_decay = self.qecc.state["vad"]["arousal"] * 10.0 # Faster decay for high arousal
            filter_env = np.exp(-t * filter_env_decay)
            signal *= filter_env
            # Amplitude envelope
            amp_env = np.exp(-t / params["decay_time"])
            signal *= amp_env
        elif instrument_name == "pad_synth":
            base_amp = 0.6 / params["num_oscs"]
            for k in range(params["num_oscs"]):
                osc_freq = freq_hz * (1 + random.uniform(-params["detune_factor"], params["detune_factor"]))
                signal += base_amp * np.sin(2 * np.pi * osc_freq * t + random.uniform(0, 2*np.pi))
            # Attack/Release envelope
            attack_samples = int(params["attack_time"] * self.sample_rate)
            release_samples = int(params["release_time"] * self.sample_rate)
            if num_samples > attack_samples:
                signal[:attack_samples] *= np.linspace(0,1,attack_samples)
            if num_samples > release_samples:
                signal[-release_samples:] *= np.linspace(1,0,release_samples)
        
        signal *= velocity_norm # Apply velocity
        if np.max(np.abs(signal)) > 0: signal /= (np.max(np.abs(signal)) / velocity_norm + 1e-5) # Normalize but respect velocity scaling
        return signal


    def generate_drum_track(self, total_steps, subdivisions_per_beat, tempo_bpm):
        self.log_message("AIE", "INFO", "Generating drum track.")
        step_duration_sec = 60.0 / (tempo_bpm * subdivisions_per_beat)
        drum_track_len_samples = int(total_steps * step_duration_sec * self.sample_rate)
        
        kick_audio = np.zeros(drum_track_len_samples)
        snare_audio = np.zeros(drum_track_len_samples)
        hihat_audio = np.zeros(drum_track_len_samples)

        # Genre and VAD influence patterns
        genre = self.qecc.state.get("current_genre", "hybrid") # Assume QECC has this
        vad = self.qecc.state["vad"]

        for step in range(total_steps):
            current_sample_pos = int(step * step_duration_sec * self.sample_rate)
            
            # Kick: Often on 1 and 3, or more syncopated in complex genres/high arousal
            kick_prob = 0.0
            if step % (subdivisions_per_beat * 2) == 0: kick_prob = 0.9 # Strong beat
            elif step % subdivisions_per_beat == 0 : kick_prob = 0.5 # On beat
            if genre == "trap": # Trap often has sparser strong kicks but more 808-like sounds
                if step % (subdivisions_per_beat * 2) == 0: kick_prob = 0.7
                # Conceptual: if trap, kick might be lower, longer (more like 808) - handled in synth
            kick_prob *= (1.0 + vad["dominance"]*0.5 - vad["arousal"]*0.2) # Dominance boosts, arousal makes it less predictable

            if random.random() < np.clip(kick_prob,0.1,0.9):
                sound = self._synthesize_sound("kick", 60 + vad["arousal"]*20, 0.2, 0.9)
                end_s = current_sample_pos + len(sound)
                if end_s <= drum_track_len_samples: kick_audio[current_sample_pos:end_s] += sound

            # Snare: Often on 2 and 4
            snare_prob = 0.0
            if (step + subdivisions_per_beat) % (subdivisions_per_beat * 2) == 0: snare_prob = 0.85 # Backbeat
            if genre == "electronic" and vad["complexity_level"] > 0.6: # More off-beat snares
                if step % (subdivisions_per_beat // 2) != 0 and random.random() < 0.1: snare_prob = 0.6
            snare_prob *= (1.0 + vad["arousal"]*0.3)

            if random.random() < np.clip(snare_prob,0,0.85):
                sound = self._synthesize_sound("snare", 200, 0.15, 0.8)
                end_s = current_sample_pos + len(sound)
                if end_s <= drum_track_len_samples: snare_audio[current_sample_pos:end_s] += sound

            # Hi-hat: Varies a lot by genre and arousal
            hihat_prob = 0.0
            if genre == "trap": # Fast hi-hats
                hihat_prob = 0.7 + vad["arousal"] * 0.25
                if step % (subdivisions_per_beat // 4) == 0 and random.random() < 0.3: # Rolls/triplets conceptual
                    hihat_prob = 0.9 
            elif genre == "pop" or genre == "hybrid":
                if step % (subdivisions_per_beat // 2) == 0: hihat_prob = 0.6 # 8th notes
            hihat_prob = np.clip(hihat_prob,0.2,0.95)
            
            if random.random() < hihat_prob:
                sound = self._synthesize_sound("hihat", 8000, 0.05, 0.5 + vad["arousal"]*0.3)
                end_s = current_sample_pos + len(sound)
                if end_s <= drum_track_len_samples: hihat_audio[current_sample_pos:end_s] += sound
            
            self.logs.append({"step":step, "genre":genre, "kick_p":kick_prob, "snare_p":snare_prob, "hihat_p":hihat_prob, "VAD":dict(vad)})

        # Mix drum elements (simple sum for now)
        # Normalize each drum sound before adding to prevent clipping if they are loud
        if np.max(np.abs(kick_audio)) > 0: kick_audio /= np.max(np.abs(kick_audio))
        if np.max(np.abs(snare_audio)) > 0: snare_audio /= np.max(np.abs(snare_audio))
        if np.max(np.abs(hihat_audio)) > 0: hihat_audio /= np.max(np.abs(hihat_audio))
        
        drum_mix = kick_audio * 0.9 + snare_audio * 0.7 + hihat_audio * 0.5
        if np.max(np.abs(drum_mix)) > 0: drum_mix /= np.max(np.abs(drum_mix))

        self.log_message("AIE", "INFO", "Drum track generation complete.")
        return drum_mix


    def generate_instrumental_part(self, instrument_name, midi_sequence, tempo_bpm):
        # midi_sequence: list of (pitch_midi, duration_steps, velocity)
        self.log_message("AIE", "INFO", f"Generating track for {instrument_name}.")
        step_duration_sec = 60.0 / (tempo_bpm * self.qecc.state.get("subdivisions_per_beat",4)) # Use subdivisions from QECC
        
        total_duration_steps = sum(s[1] for s in midi_sequence)
        track_audio = np.zeros(int(total_duration_steps * step_duration_sec * self.sample_rate))
        current_sample_pos = 0

        for pitch_midi, duration_steps, velocity_midi in midi_sequence:
            if pitch_midi == 0: # Rest
                current_sample_pos += int(duration_steps * step_duration_sec * self.sample_rate)
                continue

            freq_hz = 440 * (2 ** ((pitch_midi - 69) / 12))
            note_duration_sec = duration_steps * step_duration_sec
            velocity_norm = velocity_midi / 127.0
            
            sound = self._synthesize_sound(instrument_name, freq_hz, note_duration_sec, velocity_norm)
            
            end_s = current_sample_pos + len(sound)
            if end_s <= len(track_audio):
                track_audio[current_sample_pos:end_s] += sound
            else: # Truncate if needed
                 fit_len = len(track_audio) - current_sample_pos
                 if fit_len > 0: track_audio[current_sample_pos:] += sound[:fit_len]

            current_sample_pos = end_s
            self.logs.append({"instrument":instrument_name, "pitch_midi":pitch_midi, "duration_sec":note_duration_sec, "VAD_at_gen":dict(self.qecc.state["vad"])})
        
        if np.max(np.abs(track_audio)) > 0: track_audio /= np.max(np.abs(track_audio))
        self.log_message("AIE", "INFO", f"Track for {instrument_name} generation complete.")
        return track_audio


    def explain(self):
        explanation = ["AIE Explainability Log:"]
        num_drum_events = sum(1 for log in self.logs if 'kick_p' in log) # Count drum log entries
        num_instr_events = len(self.logs) - num_drum_events
        explanation.append(f"  Generated {num_drum_events} drum pattern decision points and {num_instr_events} instrumental note events.")
        if self.logs:
            example_log = random.choice(self.logs) # Show a random example
            if "instrument" in example_log:
                explanation.append(f"  Example Instr Event: {example_log['instrument']}, MIDI {example_log['pitch_midi']}, Duration {example_log['duration_sec']:.2f}s.")
            else:
                explanation.append(f"  Example Drum Event (Step {example_log['step']}): Genre '{example_log['genre']}', KickProb {example_log['kick_p']:.2f}.")
            explanation.append(f"  Emotion (V/A/D at event): {example_log.get('VAD_at_gen', example_log.get('VAD', {}))['valence']:.2f}/{example_log.get('VAD_at_gen', example_log.get('VAD', {}))['arousal']:.2f}/{example_log.get('VAD_at_gen', example_log.get('VAD', {}))['dominance']:.2f}")

        return "\n".join(explanation)

    def log_message(self, level, message, data=None):
        log_message("AIE", level, message, data)


# ========== [HOLISTIC ARRANGEMENT & QUANTUM MIXER (HAQM)] ==========
class HolisticArrangementQuantumMixer:
    def __init__(self, qecc_ref, sample_rate=SR, num_sections=3, measures_per_section=4):
        self.qecc = qecc_ref
        self.sample_rate = sample_rate
        self.num_sections = num_sections
        self.measures_per_section = measures_per_section
        self.logs = []
        self.log_message("HAQM", "INFO", f"Initialized for {num_sections} sections.")

    def arrange_song_structure(self, tracks_dict, total_measures):
        # tracks_dict: {"vocals": audio_data, "melody_lead": audio_data, "drums": audio_data, "bass": audio_data, "pads": audio_data}
        # total_measures: total measures in the piece, used to determine section lengths if not fixed
        # This is a simplified arranger. It decides which tracks play in which section.
        # Sections could be "intro", "verse", "chorus", "bridge", "outro".
        # For simplicity, just number sections and vary density/instrumentation.
        
        arrangement_plan = [] # List of dicts: {"section_idx": i, "active_tracks": ["vocals", "drums", ...], "energy_level": 0-1}
        
        # Energy curve based on VAD or defined progression
        # Example: low -> medium -> high -> medium -> low
        base_energy_curve = [0.3, 0.6, 0.9, 0.7, 0.4] # For 5 sections, adapt for self.num_sections
        
        for i in range(self.num_sections):
            section_energy = base_energy_curve[i % len(base_energy_curve)] * (0.5 + self.qecc.state["vad"]["arousal"] * 0.5) # Arousal boosts overall energy
            section_energy = np.clip(section_energy, 0.1, 1.0)
            
            active_tracks_this_section = []
            if section_energy > 0.2: active_tracks_this_section.append("drums")
            if section_energy > 0.3: active_tracks_this_section.append("bass")
            if section_energy > 0.4: active_tracks_this_section.append("pads")
            if section_energy > 0.5: active_tracks_this_section.append("melody_lead")
            if section_energy > 0.6 or (i > 0 and i < self.num_sections -1) : # Vocals in middle sections or high energy
                active_tracks_this_section.append("vocals")
            
            # Ensure essential tracks are there if available, e.g., melody if no vocals
            if "vocals" not in active_tracks_this_section and "melody_lead" not in active_tracks_this_section and "melody_lead" in tracks_dict:
                active_tracks_this_section.append("melody_lead")
            if not active_tracks_this_section and "pads" in tracks_dict: # Ensure something plays
                active_tracks_this_section.append("pads")

            arrangement_plan.append({"section_idx": i, "active_tracks": list(set(active_tracks_this_section)), "energy_level": section_energy})
            self.logs.append(f"Arrangement Plan - Section {i}: Energy {section_energy:.2f}, Tracks: {', '.join(active_tracks_this_section)}")
        
        self.log_message("HAQM", "INFO", "Song arrangement plan created.")
        return arrangement_plan

    def mix_and_master(self, tracks_dict, arrangement_plan, total_length_samples, tempo_bpm):
        self.log_message("HAQM", "INFO", "Mixing and mastering process started.")
        master_output = np.zeros(total_length_samples)
        if STEREO: master_output = np.zeros((total_length_samples, 2))

        measures_duration_steps = self.qecc.state.get("beats_per_measure",4) * self.qecc.state.get("subdivisions_per_beat",4)
        step_duration_sec = 60.0 / (tempo_bpm * self.qecc.state.get("subdivisions_per_beat",4))
        section_length_samples = int(self.measures_per_section * measures_duration_steps * step_duration_sec * self.sample_rate)

        # Track gain levels (conceptual)
        track_gains = {"vocals": 1.0, "melody_lead": 0.8, "drums": 0.9, "bass": 0.85, "pads": 0.7}
        # Track panning (conceptual: -1 L, 0 C, 1 R)
        track_panning = {"vocals": 0, "melody_lead": random.uniform(-0.2, 0.2), 
                         "drums": 0, "bass": 0, "pads": random.uniform(-0.6, 0.6)}


        for section_idx, section_data in enumerate(arrangement_plan):
            start_sample = section_idx * section_length_samples
            end_sample = min(start_sample + section_length_samples, total_length_samples)
            
            section_mix = np.zeros(end_sample - start_sample)
            if STEREO: section_mix = np.zeros((end_sample - start_sample, 2))

            for track_name in section_data["active_tracks"]:
                if track_name in tracks_dict and tracks_dict[track_name] is not None:
                    audio_data = tracks_dict[track_name]
                    # Ensure audio_data is 1D if master is mono, or can be made stereo
                    if STEREO and audio_data.ndim == 1:
                        audio_data_stereo = np.vstack((audio_data, audio_data)).T # Convert mono to stereo
                    elif not STEREO and audio_data.ndim == 2:
                        audio_data_stereo = audio_data.mean(axis=1) # Convert stereo to mono
                    else:
                        audio_data_stereo = audio_data # Already matching or will be handled

                    # Get the segment for this section
                    track_segment = audio_data_stereo[start_sample:end_sample] if len(audio_data_stereo) >= end_sample else \
                                    np.pad(audio_data_stereo, ((0, max(0, end_sample - len(audio_data_stereo))), (0,0) if STEREO and audio_data_stereo.ndim==2 else (0,0)))[start_sample:end_sample]


                    gain = track_gains.get(track_name, 0.7) * section_data["energy_level"] # Energy affects gain
                    
                    if STEREO:
                        pan = track_panning.get(track_name, 0)
                        left_gain = gain * math.cos((pan + 1) * math.pi / 4) # Pan law
                        right_gain = gain * math.sin((pan + 1) * math.pi / 4)
                        if track_segment.ndim == 1: # if for some reason still mono
                             section_mix[:, 0] += track_segment * left_gain
                             section_mix[:, 1] += track_segment * right_gain
                        else: # stereo track segment
                             section_mix[:, 0] += track_segment[:,0] * left_gain
                             section_mix[:, 1] += track_segment[:,1] * right_gain
                    else: # Mono mixing
                        section_mix += track_segment * gain
            
            if STEREO: master_output[start_sample:end_sample, :] += section_mix
            else: master_output[start_sample:end_sample] += section_mix
            self.logs.append({"section": section_idx, "energy": section_data["energy_level"], "active_tracks_count": len(section_data["active_tracks"])})

        # Conceptual Mastering Effects
        # 1. Normalization (already done per track mostly, one final pass)
        if np.max(np.abs(master_output)) > 0:
            master_output /= (np.max(np.abs(master_output)) + 1e-5) # Avoid div by zero
        
        # 2. Limiter (clip to -1, 1)
        master_output = np.clip(master_output, -0.98, 0.98) # Leave a tiny bit of headroom

        # 3. Subtle EQ based on overall VAD (conceptual)
        vad = self.qecc.state["vad"]
        if vad["valence"] > 0.6: # Brighter for positive valence (conceptual high-shelf boost)
            # This would require FFT. For now, it's a conceptual step.
            pass # master_output = self._apply_conceptual_eq(master_output, "high_shelf_boost")
        if vad["arousal"] > 0.7: # More bass/impact for high arousal (conceptual low-shelf boost)
            pass # master_output = self._apply_conceptual_eq(master_output, "low_shelf_boost")

        self.log_message("HAQM", "INFO", "Mixing and mastering complete.")
        return master_output

    def explain(self):
        explanation = ["HAQM Explainability Log:"]
        explanation.append(f"  Arranged song into {self.num_sections} sections.")
        if self.logs:
            avg_energy = np.mean([log['energy'] for log in self.logs if 'energy' in log])
            avg_tracks = np.mean([log['active_tracks_count'] for log in self.logs if 'active_tracks_count' in log])
            explanation.append(f"  Average section energy: {avg_energy:.2f}, Average active tracks per section: {avg_tracks:.1f}.")
            explanation.append(f"  Mastering involved normalization and limiting. Conceptual EQ applied based on VAD.")
        return "\n".join(explanation)
    
    def log_message(self, level, message, data=None):
        log_message("HAQM", level, message, data)


# ========== [COGNITIVE EVOLUTION LOOP (CEL)] ==========
class CognitiveEvolutionLoop(threading.Thread):
    def __init__(self, qecc_ref, all_modules_refs, evolution_interval_sec=60): # Faster evolution for demo
        super().__init__(daemon=True, name="CELThread")
        self.qecc = qecc_ref
        self.modules = all_modules_refs # Dict: {"lyrics": slw_ref, "melody": hmme_ref, ...}
        self.interval = evolution_interval_sec
        self._stop_event = threading.Event()
        self.evolution_cycles = 0
        self.log_message("CEL", "INFO", f"Initialized. Evolution cycle every {self.interval}s.")

    def stop(self):
        self._stop_event.set()

    def run(self):
        self._stop_event.wait(10) # Initial delay
        while not self._stop_event.is_set():
            self.evolve_strategies()
            self._stop_event.wait(self.interval)
        self.log_message("CEL", "INFO", "Stopped.")

    def evolve_strategies(self):
        self.evolution_cycles += 1
        self.log_message("CEL", "INFO", f"Evolution Cycle #{self.evolution_cycles} started.")
        
        # Analyze recent QECC state timeline and memory bank (simplified analysis)
        # Example: If arousal is often high but complexity is low, try increasing complexity factor.
        recent_arousals = [s['vad']['arousal'] for s in self.qecc.state_timeline if 'vad' in s]
        avg_arousal = np.mean(recent_arousals) if recent_arousals else 0.5
        
        if avg_arousal > 0.7 and self.qecc.state["complexity_level"] < 0.8:
            new_complexity = np.clip(self.qecc.state["complexity_level"] + 0.05, 0.1, 0.9)
            self.qecc.update_state(complexity_level=new_complexity)
            self.log_message("CEL", "DEBUG", f"High arousal detected, increased complexity_level to {new_complexity:.2f}.")

        # Example: If a lyrical theme hasn't been used much, increase its "chance" or add a new one.
        # This requires tracking theme usage, which isn't implemented. Conceptual:
        if random.random() < 0.1: # Low chance to add a new conceptual theme
            new_themes = list(self.qecc.state["lyrical_themes"])
            potential_new_themes = ["nature", "technology", "mythology", "dreams"]
            chosen_new = random.choice(potential_new_themes)
            if chosen_new not in new_themes:
                new_themes.append(chosen_new)
                self.qecc.update_state(lyrical_themes=new_themes[-3:]) # Keep last 3
                self.log_message("CEL", "DEBUG", f"Introduced/refreshed lyrical theme: {chosen_new}. Current: {self.qecc.state['lyrical_themes']}")

        # Example: Modify probabilities or parameters within modules (conceptual)
        # e.g., if melodies are too static, increase preferred_interval_max in HMME (needs direct access or messaging)
        # This is where a more complex parameter management system would be beneficial.
        # For now, CEL influences QECC state, which in turn influences modules.

        self.log_message("CEL", "INFO", f"Evolution Cycle #{self.evolution_cycles} complete.")

    def log_message(self, level, message, data=None):
        log_message("CEL", level, message, data)

# ========== [ADVANCED EXPLAINABILITY CORE (AEC)] ==========
class AdvancedExplainabilityCore:
    def __init__(self, qecc_ref, module_refs_dict): # module_refs is a dict {"name": module_instance}
        self.qecc = qecc_ref
        self.modules = module_refs_dict
        self.log_message("AEC", "INFO", "Initialized.")

    def generate_full_explanation(self, song_title="Untitled Quantum Piece"):
        print(f"\n===== Explanation Report for '{song_title}' =====")
        print(f"\n--- QECC Initial State Summary ---")
        print(self.qecc.explain_state()) # Assumes qecc has an explain_state method

        for module_name, module_instance in self.modules.items():
            if hasattr(module_instance, 'explain') and callable(module_instance.explain):
                print(f"\n--- {module_name.upper()} Module Explanation ---")
                print(module_instance.explain())
            else:
                print(f"\n--- {module_name.upper()} Module (No custom explanation) ---")
        
        print("\n===== End of Report =====")
        self.log_message("AEC", "INFO", "Full explanation report generated.")

    def log_message(self, level, message, data=None):
        log_message("AEC", level, message, data)


# ========== [MAIN ORCHESTRATOR V5] ==========
if __name__ == "__main__":
    print(f"=== VictorAudioGenesis QUANTUMHARMONY v5.0.0 — Initiating 2029 GODCORE ===\n")
    time.sleep(0.5)

    # --- Initialize Core Systems ---
    qec_core = QuantumEmotionCognitionCore()
    
    # Modules expect QECC reference
    lyric_engine = SemanticLyricWeaver(qec_core, topic="cosmic echoes", persona="StardustMuse")
    melody_engine = HarmonicMelodyMorphingEngine(qec_core, measures=8, beats_per_measure=4, subdivisions_per_beat=4) # Longer piece
    voice_synth_engine = NeuralVoiceModulator(qec_core, sample_rate=SR)
    instrument_ensemble_engine = AdaptiveInstrumentalEnsemble(qec_core, sample_rate=SR)
    arranger_mixer_engine = HolisticArrangementQuantumMixer(qec_core, sample_rate=SR, num_sections=5, measures_per_section=4) # 5 sections for more structure

    all_modules = {
        "lyric_engine": lyric_engine, "melody_engine": melody_engine, 
        "voice_synth": voice_synth_engine, "instrument_ensemble": instrument_ensemble_engine,
        "arranger_mixer": arranger_mixer_engine
    }
    
    cognitive_evolver = CognitiveEvolutionLoop(qec_core, all_modules, evolution_interval_sec=45) # Faster for demo
    explainability_core = AdvancedExplainabilityCore(qec_core, all_modules)

    # --- Start Background Processes ---
    cognitive_evolver.start()
    log_message("Orchestrator", "INFO", "All core modules initialized. CEL started.")

    # --- Simulate Dynamic Context Change & Generation ---
    # This loop can be adapted for interactive input or pre-set scenarios
    for cycle in range(2): # Generate a couple of pieces with evolving context
        print(f"\n\n--- Generation Cycle {cycle + 1} ---")
        # 1. Update QECC state (simulating external influence or internal shift)
        new_valence = random.uniform(0.2, 0.8)
        new_arousal = random.uniform(0.3, 0.9)
        new_dominance = random.uniform(0.4, 0.7)
        new_key = random.choice(["C", "G", "D", "A", "E", "F", "Bb", "Eb"])
        new_mode = random.choice(["major", "minor", "dorian"])
        new_complexity = random.uniform(0.3, 0.8)
        new_themes = random.sample(["cosmos", "introspection", "transformation", "nature", "technology", "dreams", "mythology"], k=2)
        
        qec_core.update_state(
            vad_update={"valence_delta": new_valence - qec_core.state["vad"]["valence"], 
                        "arousal_delta": new_arousal - qec_core.state["vad"]["arousal"],
                        "dominance_delta": new_dominance - qec_core.state["vad"]["dominance"]},
            current_key=new_key, current_mode=new_mode, 
            complexity_level=new_complexity, lyrical_themes=new_themes,
            current_genre=random.choice(["hybrid", "electronic_chill", "ambient_space", "glitch_hop_infused"]) # Add genre to QECC
        )
        log_message("Orchestrator", "INFO", f"Cycle {cycle+1}: QECC state updated for new piece.", {"new_key":new_key, "new_mode":new_mode, "new_VAD_v":new_valence})
        current_tempo_bpm = int(scale_value(qec_core.state["vad"]["arousal"], 0, 1, 70, 160)) # Tempo from arousal
        qec_core.update_state(tempo_bpm=current_tempo_bpm, beats_per_measure=4, subdivisions_per_beat=4) # Store tempo info

        # 2. Generate Musical Components
        log_message("Orchestrator", "INFO", "Starting lyrical generation...")
        lyrics_array = lyric_engine.generate_lyrics(num_lines=16, structure="ABAB rhyme_alternate_length") # More lines, new structure concept
        print(f"\n[Generated Lyrics - Cycle {cycle+1} ({lyric_engine.persona} on '{lyric_engine.topic}')]")
        for line_idx, line_text in enumerate(lyrics_array): print(f"  {line_idx+1}: {line_text}")

        log_message("Orchestrator", "INFO", "Starting melodic generation...")
        # Melody engine now uses measures, not just length
        melody_sequence_midi = melody_engine.generate_melody_sequence()
        # print(f"\n[Generated Melody MIDI (first 5 events) - Cycle {cycle+1}]")
        # for m_event in melody_sequence_midi[:5]: print(f"  Pitch: {m_event[0]}, Duration: {m_event[1]} steps, Velocity: {m_event[2]}")

        # 3. Synthesize Audio Components
        log_message("Orchestrator", "INFO", "Starting voice synthesis...")
        voice_audio_data = voice_synth_engine.synthesize_voice(lyrics_array, melody_sequence_midi, persona=lyric_engine.persona)

        log_message("Orchestrator", "INFO", "Starting drum synthesis...")
        total_steps_for_piece = melody_engine.total_steps # Use total steps from melody engine
        drum_audio_data = instrument_ensemble_engine.generate_drum_track(total_steps_for_piece, 
                                                                         qec_core.state["subdivisions_per_beat"], 
                                                                         current_tempo_bpm)
        
        # Generate Bass part based on melody's chord progression (conceptual)
        # Bass usually plays roots or harmonically relevant notes from the chord progression
        bass_midi_sequence = []
        for m_idx in range(melody_engine.measures):
            chord_root = qec_core.state["current_chord_progression"][m_idx % len(qec_core.state["current_chord_progression"])]
            # Bass plays simpler rhythm, often on beat 1 and 3, or following kick
            # For simplicity, make bass play root of chord for half the measure, then maybe a 5th or octave
            bass_pitch1 = chord_root - 12 if chord_root > 48 else chord_root # Lower octave
            bass_duration1 = qec_core.state["beats_per_measure"] * qec_core.state["subdivisions_per_beat"] // 2
            bass_midi_sequence.append((bass_pitch1, bass_duration1, 100))
            
            # Optional second note in measure
            # This is very basic. Real basslines are much more complex.
            if random.random() < 0.5: # Chance to play another note
                bass_pitch2 = bass_pitch1 + random.choice([0, 7, -5]) # Octave, 5th up, or 4th down
                bass_pitch2 = np.clip(bass_pitch2, 24, 60) # Keep in bass range
                bass_duration2 = qec_core.state["beats_per_measure"] * qec_core.state["subdivisions_per_beat"] - bass_duration1
                bass_midi_sequence.append((bass_pitch2, bass_duration2, 90))
            else: # Rest of measure is the same note or rest
                 bass_midi_sequence.append((bass_pitch1, qec_core.state["beats_per_measure"] * qec_core.state["subdivisions_per_beat"] - bass_duration1, 100))


        bass_audio_data = instrument_ensemble_engine.generate_instrumental_part("bass_synth", bass_midi_sequence, current_tempo_bpm)

        # Generate Pad part (long notes following chord progression)
        pad_midi_sequence = []
        for m_idx in range(melody_engine.measures):
            chord_root = qec_core.state["current_chord_progression"][m_idx % len(qec_core.state["current_chord_progression"])]
            # Pad plays root, 3rd, 5th of the chord (conceptual, just use root for synth for now)
            pad_pitch = chord_root - 12 # Lower octave for pad base
            pad_duration = qec_core.state["beats_per_measure"] * qec_core.state["subdivisions_per_beat"] # Whole measure
            pad_midi_sequence.append((pad_pitch, pad_duration, 70))
        pad_audio_data = instrument_ensemble_engine.generate_instrumental_part("pad_synth", pad_midi_sequence, current_tempo_bpm)


        # 4. Arrange and Mix
        log_message("Orchestrator", "INFO", "Starting arrangement and mixing...")
        tracks_for_mixing = {
            "vocals": voice_audio_data, 
            "melody_lead": instrument_ensemble_engine.generate_instrumental_part("bass_synth", melody_sequence_midi, current_tempo_bpm), # Use bass_synth sound for melody lead example
            "drums": drum_audio_data,
            "bass": bass_audio_data,
            "pads": pad_audio_data
        }
        
        # Determine total length from the longest generated audio track (usually drums or lead melody parts)
        max_len_samples = 0
        for track_name, audio_d in tracks_for_mixing.items():
            if audio_d is not None:
                 max_len_samples = max(max_len_samples, len(audio_d.T if STEREO and audio_d.ndim==2 else audio_d))
        
        if max_len_samples == 0:
            log_message("Orchestrator", "ERROR", "All audio tracks are empty. Skipping mix.")
            master_audio_final = np.zeros(SR * 10) # 10s of silence
        else:
            arrangement = arranger_mixer_engine.arrange_song_structure(tracks_for_mixing, melody_engine.measures)
            master_audio_final = arranger_mixer_engine.mix_and_master(tracks_for_mixing, arrangement, max_len_samples, current_tempo_bpm)

        # 5. Output & Explain
        song_title = f"QuantumHarmony_Piece_{cycle+1}_{qec_core.state['current_key']}{qec_core.state['current_mode']}_{int(time.time())}"
        try:
            import soundfile as sf
            output_filename = f"{song_title}.wav"
            sf.write(output_filename, master_audio_final, SR)
            log_message("Orchestrator", "INFO", f"Master audio saved to {output_filename}")
            print(f"\n[OUTPUT] Master audio for Cycle {cycle+1} saved to {output_filename}")
        except ImportError:
            log_message("Orchestrator", "WARNING", "Module 'soundfile' not found. Skipping .wav export.")
            print("\n[OUTPUT] Install 'soundfile' (pip install soundfile) to save .wav audio.")
        except Exception as e:
            log_message("Orchestrator", "ERROR", f"Failed to write WAV file: {e}")
            print(f"\n[OUTPUT] Error saving WAV file: {e}")

        if cycle == 0: # Explain only the first piece in detail for brevity in demo
             explainability_core.generate_full_explanation(song_title)
        
        time.sleep(1) # Pause between cycles

    # --- Shutdown ---
    log_message("Orchestrator", "INFO", "Generation cycles complete. Shutting down CEL.")
    cognitive_evolver.stop()
    cognitive_evolver.join(timeout=5) # Wait for thread to finish
    log_message("Orchestrator", "INFO", "VictorAudioGenesis V5 session ended.")
    print("\n=== VictorAudioGenesis QUANTUMHARMONY v5.0.0 — Session Complete ===")

# ========== END 2029 GODCORE UPGRADE ==========