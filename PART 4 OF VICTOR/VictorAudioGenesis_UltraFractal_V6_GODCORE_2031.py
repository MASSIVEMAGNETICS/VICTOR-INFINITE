#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: VictorAudioGenesis_UltraFractal_V6_GODCORE_2031.py
VERSION: v6.0.0-ULTRAFRACTAL-GODCORE_2031
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode) x Quantum Cypher Crew
PURPOSE: 2031-level, modular, recursive, emotion/cognition-driven, plugin-powered,
         syllable/rhyme-dense, neural-upgradable, explainable, fractal-mutation AI music engine.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network / Quantum Harmony Labs
"""

import random, math, numpy as np, time, threading, json, sys, os
from collections import deque
import mido # pip install mido
import soundfile as sf # pip install soundfile

SR = 44100  # Sample Rate
STEREO = True
LOG_LEVEL = "INFO"
INFINITE_RECURSION = False # Set True to keep outputting recursively (dangerous fun)

def scale_value(value, old_min, old_max, new_min, new_max):
    if old_max == old_min: return new_min
    return (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

def log_message(module_name, level, message, data=None):
    if LOG_LEVEL == "DEBUG" or (LOG_LEVEL == "INFO" and level != "DEBUG"):
        log_entry = f"[{time.strftime('%H:%M:%S')}] [{level}] [{module_name}] {message}"
        if data:
            log_entry += f" | Data: {json.dumps(data, sort_keys=True, default=lambda o: '<not serializable>')[:200]}"
        print(log_entry)

# ======================= PLUGIN/AGENT REGISTRY ========================
module_registry = {}

def register_module(name, cls):
    module_registry[name] = cls

def get_module(name, *args, **kwargs):
    return module_registry[name](*args, **kwargs)

# ======================= FRACTAL NEURAL LAYER ========================
class FractalPatternLearner:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
    def predict(self, x):
        return np.tanh(x @ self.W)
    def train(self, x, y, lr=0.01):
        y_pred = self.predict(x)
        grad = (y_pred - y) * (1 - y_pred**2)
        self.W -= lr * (x.T @ grad)

# ======================= SYLLABLE/RHYME HELPERS =======================
def syllable_count(word):
    return sum(1 for c in word.lower() if c in 'aeiouy')

def rhyme_density(line1, line2):
    w1 = line1.strip().split()
    w2 = line2.strip().split()
    if not w1 or not w2: return 0
    last1 = w1[-1]
    count = 0
    for w in w2:
        if w[-2:] == last1[-2:]: count += syllable_count(w)
    total = sum([syllable_count(w) for w in w2])
    return count / total if total > 0 else 0

# ==================== QUANTUM EMOTION-COGNITION CORE ===================
class QuantumEmotionCognitionCore:
    def __init__(self):
        self.state = {
            "vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            "current_key": "C",
            "current_mode": "major",
            "current_scale_notes": self.get_scale_notes("C", "major"),
            "current_chord_progression": [],
            "active_melodic_motifs": [],
            "active_rhythmic_patterns": [],
            "lyrical_themes": ["introspection", "cosmos"],
            "complexity_level": 0.5,
            "recursion_depth_factor": 1.0
        }
        self.state_timeline = deque(maxlen=20)
        self.memory_bank = {}
        self.log_message("QECC", "INFO", "Initialized.")

    def get_scale_notes(self, key_root_note="C", mode="major"):
        notes = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11}
        root_midi = notes.get(key_root_note.upper(), 0) + 60
        intervals_map = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "blues": [0, 3, 5, 6, 7, 10]
        }
        intervals = intervals_map.get(mode, intervals_map["major"])
        return sorted(list(set([(root_midi + i) % 12 + (octave * 12) for i in intervals for octave in range(3, 7)])))

    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            if key == "vad_update":
                for vad_key, delta in value.items():
                    self.state["vad"][vad_key] = np.clip(self.state["vad"].get(vad_key,0.5) + delta, 0, 1)
            elif key in self.state:
                self.state[key] = value
        if "current_key" in kwargs or "current_mode" in kwargs:
            self.state["current_scale_notes"] = self.get_scale_notes(self.state["current_key"], self.state["current_mode"])
        self.state_timeline.append(json.loads(json.dumps(self.state, default=lambda o: '<state_obj>')))
        self.log_message("QECC", "DEBUG", "State updated", kwargs)

    def store_musical_idea(self, idea_type, content, related_tags=None):
        idea_id = f"{idea_type}_{int(time.time()*1000)}_{random.randint(0,999)}"
        self.memory_bank[idea_id] = {
            "type": idea_type,
            "content": content,
            "vad_at_creation": dict(self.state["vad"]),
            "tags": related_tags or [],
            "timestamp": time.time(),
            "usage_count": 0
        }
        self.log_message("QECC", "DEBUG", f"Stored musical idea: {idea_id}")

    def retrieve_musical_idea(self, idea_type, current_vad, num_ideas=1, similarity_threshold=0.3):
        candidates = []
        for idea_id, idea_data in self.memory_bank.items():
            if idea_data["type"] == idea_type:
                vad_c = idea_data["vad_at_creation"]
                dist = math.sqrt(sum((vad_c[x]-current_vad[x])**2 for x in ['valence','arousal','dominance']))
                similarity = np.clip(1.0 - (dist / math.sqrt(3)), 0, 1)
                if similarity >= similarity_threshold:
                    candidates.append((idea_data, similarity))
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates:
            for idea_data, _ in candidates[:num_ideas]:
                idea_data["usage_count"] += 1
            return [c[0]['content'] for c in candidates[:num_ideas]]
        return []

    def explain_state(self):
        s = self.state
        return f"Current Emotion (V/A/D): {s['vad']['valence']:.2f}/{s['vad']['arousal']:.2f}/{s['vad']['dominance']:.2f}. Key: {s['current_key']} {s['current_mode']}. Complexity: {s['complexity_level']:.2f}."

    def log_message(self, module_name, level, message, data=None):
        log_message(module_name, level, message, data)

register_module('qecc', QuantumEmotionCognitionCore)

# ======================= FRACTAL LYRIC ENGINE V6 =======================
class FractalLyricEngine:
    def __init__(self, qecc_ref, topic="transformation", persona="QuantumBando"):
        self.qecc = qecc_ref
        self.topic = topic
        self.persona = persona
        self.logs = []
        self.log_message("FLE", "INFO", f"Initialized with topic: {topic}, persona: {persona}.")

    def _get_themed_vocab(self):
        base_vocab = {"time":1, "space":1, "mind":1, "light":1, "dark":1, "dream":1, "real":1, "echo":1, "fractal":1, "code":1, "flow":1}
        for theme in self.qecc.state["lyrical_themes"]:
            if theme == "cosmos": base_vocab.update({"star":1, "nebula":1, "galaxy":1, "void":1, "pulse":1})
            if theme == "introspection": base_vocab.update({"soul":1, "self":1, "thought":1, "within":1, "reflect":1})
            if theme == "transformation": base_vocab.update({"change":1, "shift":1, "evolve":1, "become":1, "new":1})
            if theme == "mythology": base_vocab.update({"legend":1, "titan":1, "oracle":1})
            if theme == "technology": base_vocab.update({"signal":1, "quantum":1, "matrix":1})
            if theme == "nature": base_vocab.update({"root":1, "river":1, "sky":1, "stone":1})
        vad = self.qecc.state["vad"]
        if vad['valence'] > 0.7: base_vocab.update({"joy":2, "shine":2, "soar":2})
        if vad['valence'] < 0.3: base_vocab.update({"pain":2, "break":2, "fade":2})
        if vad['arousal'] > 0.7: base_vocab.update({"storm":2, "surge":2, "wild":2})
        if vad['arousal'] < 0.3: base_vocab.update({"still":2, "calm":2, "hush":2})
        retrieved_fragments = self.qecc.retrieve_musical_idea("lyric_fragment", vad, num_ideas=2)
        for fragment in retrieved_fragments:
            for word in fragment.split(): base_vocab[word.lower()] = base_vocab.get(word.lower(), 0) + 1
        return list(base_vocab.keys())

    def _generate_fractal_line(self, vocab, min_syll=8, max_syll=16, rhyme_target=None):
        # Syllable-dense, rhyme-obsessed, recursive line generator
        tries = 0
        while tries < 8:
            tries += 1
            cur_len = random.randint(min_syll, max_syll)
            line_words = []
            syllables = 0
            current_word = random.choice(vocab)
            while syllables < cur_len:
                line_words.append(current_word)
                syllables += syllable_count(current_word)
                next_cand = [w for w in vocab if w != current_word]
                if not next_cand: break
                # If we're ending and need rhyme, bias toward matching ending
                if rhyme_target and (syllables > cur_len*0.6):
                    rhymes = [w for w in next_cand if w[-2:] == rhyme_target[-2:]]
                    current_word = random.choice(rhymes) if rhymes else random.choice(next_cand)
                else:
                    current_word = random.choice(next_cand)
            line = " ".join(line_words).capitalize()
            if not rhyme_target or rhyme_density(line, rhyme_target) >= 0.5:
                self.qecc.store_musical_idea("lyric_fragment", line, self.qecc.state["lyrical_themes"])
                return line
        # Fallback if no rhyme hit
        return " ". ".join(line_words).capitalize()"

    def generate_lyrics(self, num_lines=16, structure="AABB"):
        vocab = self._get_themed_vocab()
        verse, rhyme_map = [], {}
        for i in range(num_lines):
            line_rhyme_char = structure[i % len(structure)]
            rhyme_target = rhyme_map.get(line_rhyme_char, None)
            line = self._generate_fractal_line(vocab, rhyme_target=rhyme_target)
            if (i % 2 == 1) and rhyme_density(verse[-1], line) < 0.5:
                # Force re-gen for rhyme density
                for _ in range(5):
                    new_line = self._generate_fractal_line(vocab, rhyme_target=verse[-1])
                    if rhyme_density(verse[-1], new_line) >= 0.5:
                        line = new_line
                        break
            rhyme_map[line_rhyme_char] = line
            verse.append(line)
            self.logs.append({"line_num": i+1, "text": line, "rhyme_char": line_rhyme_char, "rhyme_target": rhyme_target, "emotion_at_gen": dict(self.qecc.state["vad"])})
        self.qecc.update_state(memory_lyrics=verse)
        return verse

    def explain(self):
        explanation = ["FLE Explainability Log:"]
        for log_entry in self.logs[-10:]:
            explanation.append(f"  Line {log_entry['line_num']} ('{log_entry['text'][:30]}...'): Rhyme '{log_entry['rhyme_char']}', Target='{log_entry['rhyme_target']}'. VAD(V/A/D): {log_entry['emotion_at_gen']['valence']:.2f}/{log_entry['emotion_at_gen']['arousal']:.2f}/{log_entry['emotion_at_gen']['dominance']:.2f}")
        return "\n".join(explanation)

    def log_message(self, module_name, level, message, data=None):
        log_message(module_name, level, message, data)

register_module('lyric_engine', FractalLyricEngine)

# ======================= [FRACTAL MELODY ENGINE V6] =====================
class FractalMelodyEngine:
    def __init__(self, qecc_ref, measures=8, beats_per_measure=4, subdivisions_per_beat=4):
        self.qecc = qecc_ref
        self.measures = measures
        self.beats_per_measure = beats_per_measure
        self.subdivisions_per_beat = subdivisions_per_beat
        self.total_steps = measures * beats_per_measure * subdivisions_per_beat
        self.logs = []
        self.log_message("FME", "INFO", f"Initialized for {measures} measures.")

    def _select_note_from_scale(self, scale_notes, last_note=None, preferred_interval_max=5):
        if not scale_notes: return random.randint(55, 75)
        if last_note is None: return random.choice(scale_notes)
        candidates = [n for n in scale_notes if abs(n - last_note) <= preferred_interval_max]
        if not candidates:
            candidates = [n for n in scale_notes if abs(n - last_note) <= 12]
        return random.choice(candidates if candidates else scale_notes)

    def _generate_rhythmic_pattern(self, num_steps, base_density=0.6):
        density = np.clip(base_density + (self.qecc.state["vad"]["arousal"] - 0.5) * 0.4, 0.2, 0.9)
        pattern, steps_remaining = [], num_steps
        while steps_remaining > 0:
            is_rest = random.random() > density
            max_dur = 4 - int(self.qecc.state["vad"]["arousal"] * 2 + self.qecc.state["complexity_level"] * 2)
            max_dur = max(1, max_dur)
            duration = random.randint(1, min(max_dur, steps_remaining))
            pattern.append((duration, is_rest))
            steps_remaining -= duration
        return pattern

    def _generate_chord_progression(self, num_chords=4):
        key = self.qecc.state["current_key"]
        mode = self.qecc.state["current_mode"]
        scale_root_midi = self.qecc.get_scale_notes(key, mode)[0] % 12
        chords_in_key_major = [0, 3, 4, 7]
        chords_in_key_minor = [0, 5, 7, 3]
        possible_chord_roots_relative = chords_in_key_major if "major" in mode else chords_in_key_minor
        progression_midi_roots = []
        for _ in range(num_chords):
            chord_center_note = random.choice(self.qecc.state["current_scale_notes"])
            progression_midi_roots.append(chord_center_note)
        self.qecc.update_state(current_chord_progression=progression_midi_roots)
        return progression_midi_roots

    def generate_melody_sequence(self):
        scale_notes = self.qecc.state["current_scale_notes"]
        complexity = self.qecc.state["complexity_level"]
        vad = self.qecc.state["vad"]
        chord_prog_roots = self._generate_chord_progression(num_chords=self.measures)
        melody_midi, current_step = [], 0
        last_note_pitch = random.choice(scale_notes) if scale_notes else 60
        active_motifs = self.qecc.retrieve_musical_idea("melodic_motif", vad, num_ideas=1)
        for measure_idx in range(self.measures):
            current_chord_root = chord_prog_roots[measure_idx % len(chord_prog_roots)]
            measure_target_notes = [n for n in scale_notes if abs(n - current_chord_root) < 7]
            if not measure_target_notes: measure_target_notes = scale_notes
            rhythmic_pattern_steps = self._generate_rhythmic_pattern(self.beats_per_measure * self.subdivisions_per_beat)
            measure_melody_pos = 0
            for duration_steps, is_rest in rhythmic_pattern_steps:
                if is_rest:
                    pitch = 0
                else:
                    if active_motifs and random.random() < 0.3 * complexity:
                        motif_note_info = random.choice(active_motifs[0])
                        pitch = motif_note_info[0]
                        if pitch not in measure_target_notes:
                            pitch = self._select_note_from_scale(measure_target_notes if measure_target_notes else scale_notes, last_note_pitch)
                    else:
                        pitch = self._select_note_from_scale(measure_target_notes if measure_target_notes else scale_notes, last_note_pitch)
                    last_note_pitch = pitch
                velocity = int(scale_value(vad["arousal"] * 0.6 + vad["dominance"] * 0.4, 0, 1, 60, 120))
                melody_midi.append((pitch, duration_steps, velocity))
                self.logs.append({
                    "step_abs": current_step + measure_melody_pos, "measure": measure_idx,
                    "pitch": pitch, "duration_steps": duration_steps, "velocity": velocity,
                    "chord_context_root": current_chord_root, "emotion_at_gen": dict(vad)
                })
                measure_melody_pos += duration_steps
            current_step += self.beats_per_measure * self.subdivisions_per_beat
        self.qecc.store_musical_idea("melodic_motif", melody_midi, ["generated_melody_main"])
        self.qecc.update_state(memory_melody=melody_midi)
        return melody_midi

    def explain(self):
        explanation = ["FME Explainability Log:"]
        if self.logs:
            explanation.append(f"  Generated {len(self.logs)} melodic events across {self.measures} measures.")
            last_event = self.logs[-1]
            explanation.append(f"  Example event (last): Pitch {last_event['pitch']}, Duration {last_event['duration_steps']} steps, Velocity {last_event['velocity']}. Chord Context (Root): {last_event['chord_context_root']}.")
            explanation.append(f"  Emotion (V/A/D at last event): {last_event['emotion_at_gen']['valence']:.2f}/{last_event['emotion_at_gen']['arousal']:.2f}/{last_event['emotion_at_gen']['dominance']:.2f}")
        return "\n".join(explanation)

    def log_message(self, module_name, level, message, data=None):
        log_message(module_name, level, message, data)

register_module('melody_engine', FractalMelodyEngine)

# ======================== [FRACTAL VOCODER V6] =========================
class FractalVocoder:
    def __init__(self, qecc_ref, sample_rate=SR):
        self.qecc = qecc_ref
        self.sample_rate = sample_rate
        self.logs = []
        self.base_voiceprints = {
            "QuantumBando": [1.0, 0.05, 5.0, 0.2],
            "SereneOracle": [1.1, 0.02, 3.0, 0.1],
            "GlitchWraith": [0.85, 0.2, 8.0, 0.5]
        }
        self.log_message("VOCODER", "INFO", "Initialized.")

    def midi_to_freq(self, midi_note):
        if midi_note == 0: return 0
        return 440 * (2 ** ((midi_note - 69) / 12))

    def synthesize_voice(self, lyrics_text_array, melody_midi_sequence, persona="QuantumBando", fractal_mode=True):
        self.log_message("VOCODER", "INFO", f"Synthesizing voice for {len(lyrics_text_array)} lines, persona: {persona}.")
        voiceprint_params = self.base_voiceprints.get(persona, self.base_voiceprints["QuantumBando"])
        total_audio_duration_steps = sum(m[1] for m in melody_midi_sequence)
        step_duration_sec = 0.15 * (1.0 - self.qecc.state["vad"]["arousal"] * 0.5)
        step_duration_sec = max(0.05, step_duration_sec)
        output_waveform = np.zeros(int(total_audio_duration_steps * step_duration_sec * self.sample_rate))
        current_sample_pos = 0
        lyric_idx = 0
        for i, (pitch_midi, duration_steps, velocity) in enumerate(melody_midi_sequence):
            current_lyric_line = lyrics_text_array[lyric_idx % len(lyrics_text_array)] if lyrics_text_array else "la"
            phoneme_sound = "a" if self.qecc.state["vad"]["valence"] > 0.5 else "u"
            freq_hz = self.midi_to_freq(pitch_midi)
            if freq_hz == 0:
                current_sample_pos += int(duration_steps * step_duration_sec * self.sample_rate)
                continue
            note_duration_sec = duration_steps * step_duration_sec
            num_samples_note = int(note_duration_sec * self.sample_rate)
            t = np.linspace(0, note_duration_sec, num_samples_note, endpoint=False)
            # Fractal mode: modulate harmonics with Julia set or recursive chaos
            signal = np.zeros_like(t)
            num_partials = 3 + int(self.qecc.state["complexity_level"] * 4)
            if fractal_mode:
                for k in range(1, num_partials+1):
                    # Use a recursive phase modulation for each overtone
                    fractal_shift = math.sin((k**1.5) * t * 0.25 + np.sin(t * k) * k)
                    partial_freq = freq_hz * k * (1 + 0.01 * fractal_shift)
                    signal += (1.0 / k) * np.sin(2 * np.pi * partial_freq * t + random.uniform(0, np.pi/2))
            else:
                for k in range(1, num_partials+1):
                    partial_freq = freq_hz * k
                    signal += (1.0 / k) * np.sin(2 * np.pi * partial_freq * t + random.uniform(0, np.pi/2))
            # Add noise and vibrato
            noise_amp = voiceprint_params[1] * (self.qecc.state["vad"]["arousal"] * 0.5)
            signal += noise_amp * (np.random.rand(num_samples_note) * 2 - 1)
            vibrato_rate = voiceprint_params[2]
            vibrato_depth_hz = (2**(voiceprint_params[3]/12)-1) * freq_hz
            if vibrato_rate > 0 and vibrato_depth_hz > 0:
                signal += 0.08 * np.sin(2 * np.pi * vibrato_rate * t) * np.sin(2 * np.pi * vibrato_depth_hz * t)
            # Envelope
            attack_len = min(num_samples_note // 10, int(0.02 * self.sample_rate))
            decay_len = min(num_samples_note // 10, int(0.03 * self.sample_rate))
            if num_samples_note > attack_len + decay_len:
                env = np.ones(num_samples_note)
                env[:attack_len] = np.linspace(0, 1, attack_len)
                env[-decay_len:] = np.linspace(1, 0, decay_len)
                signal *= env
            end_sample_for_note = current_sample_pos + num_samples_note
            if end_sample_for_note <= len(output_waveform):
                output_waveform[current_sample_pos : end_sample_for_note] += signal
            else:
                fit_samples = len(output_waveform) - current_sample_pos
                if fit_samples > 0:
                    output_waveform[current_sample_pos:] += signal[:fit_samples]
            current_sample_pos = end_sample_for_note
            if duration_steps > self.qecc.state.get("subdivisions_per_beat",4) :
                lyric_idx = (lyric_idx + 1)
        if np.max(np.abs(output_waveform)) > 0:
            output_waveform /= np.max(np.abs(output_waveform))
        self.qecc.update_state(memory_voice_audio_stats={"length_samples": len(output_waveform), "max_amplitude": np.max(np.abs(output_waveform))})
        return output_waveform

    def log_message(self, module_name, level, message, data=None):
        log_message(module_name, level, message, data)

register_module('voice_synth', FractalVocoder)

# ==================== [EXPORT MIDI] ======================
def export_midi(melody_seq, filename="out.mid"):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for pitch, dur, vel in melody_seq:
        if pitch > 0:
            track.append(mido.Message('note_on', note=int(pitch), velocity=int(vel), time=0))
            track.append(mido.Message('note_off', note=int(pitch), velocity=0, time=int(dur*120)))
    mid.save(filename)

# ==================== [EXPLAINABILITY] ===================
class UltraExplainabilityCore:
    def __init__(self, qecc_ref, module_refs_dict):
        self.qecc = qecc_ref
        self.modules = module_refs_dict
        self.log_message("UEC", "INFO", "Initialized.")

    def generate_full_explanation(self, song_title="Untitled UltraFractal"):
        print(f"\n===== Explanation Report for '{song_title}' =====")
        print(f"\n--- QECC Initial State Summary ---")
        print(self.qecc.explain_state())
        for module_name, module_instance in self.modules.items():
            if hasattr(module_instance, 'explain') and callable(module_instance.explain):
                print(f"\n--- {module_name.upper()} Module Explanation ---")
                print(module_instance.explain())
            else:
                print(f"\n--- {module_name.upper()} Module (No custom explanation) ---")
        print("\n===== End of Report =====")
        self.log_message("UEC", "INFO", "Full explanation report generated.")

    def log_message(self, module_name, level, message, data=None):
        log_message(module_name, level, message, data)

register_module('explainability_core', UltraExplainabilityCore)

# ==================== [ORCHESTRATOR MAIN] ==================
if __name__ == "__main__":
    print(f"=== VictorAudioGenesis ULTRAFRACTAL v6.0.0 — Mutating GODCORE ===\n")
    time.sleep(0.4)
    # -- Init core systems
    qec_core = get_module('qecc')
    lyric_engine = get_module('lyric_engine', qec_core, topic="infinite recursion", persona="FractalOverlord")
    melody_engine = get_module('melody_engine', qec_core, measures=8, beats_per_measure=4, subdivisions_per_beat=4)
    voice_synth = get_module('voice_synth', qec_core, sample_rate=SR)
    all_modules = {
        "lyric_engine": lyric_engine,
        "melody_engine": melody_engine,
        "voice_synth": voice_synth
    }
    explainability_core = get_module('explainability_core', qec_core, all_modules)
    # --- Main infinite evolution loop (or just once if INFINITE_RECURSION==False)
    recursion_depth = 2 if not INFINITE_RECURSION else 99999
    for cycle in range(recursion_depth):
        print(f"\n--- UltraFractal Generation Cycle {cycle+1} ---")
        # 1. Evolve QECC State
        qec_core.update_state(
            vad_update={
                "valence_delta": random.uniform(-0.2,0.2),
                "arousal_delta": random.uniform(-0.2,0.2),
                "dominance_delta": random.uniform(-0.2,0.2)
            },
            current_key=random.choice(["C","G","A","Eb","F#"]),
            current_mode=random.choice(["major","minor","dorian"]),
            complexity_level=random.uniform(0.3,0.8),
            lyrical_themes=random.sample(["cosmos","transformation","introspection","nature","technology","mythology"],2),
            current_genre=random.choice(["trap","ambient","hybrid","fractalcore"]),
            tempo_bpm=int(scale_value(qec_core.state["vad"]["arousal"],0,1,70,160))
        )
        # 2. Lyrics
        lyrics_array = lyric_engine.generate_lyrics(num_lines=16, structure="AABB")
        print("\n[Generated Lyrics]")
        for idx, l in enumerate(lyrics_array): print(f" {idx+1}: {l}")
        # 3. Melody
        melody_seq = melody_engine.generate_melody_sequence()
        export_midi(melody_seq, filename=f"UltraFractal_Melody_{cycle+1}.mid")
        # 4. Vocals
        voice_audio = voice_synth.synthesize_voice(lyrics_array, melody_seq, persona=lyric_engine.persona, fractal_mode=True)
        # 5. Save .wav
        filename = f"UltraFractal_Music_{cycle+1}.wav"
        sf.write(filename, voice_audio, SR)
        print(f"[OUTPUT] Master audio saved to {filename}")
        print(f"[OUTPUT] MIDI melody saved to UltraFractal_Melody_{cycle+1}.mid")
        # 6. Explain
        explainability_core.generate_full_explanation(filename)
        print("\n--- Cycle Complete ---\n")
        if not INFINITE_RECURSION: break
        time.sleep(0.8)
    print("\n=== VictorAudioGenesis ULTRAFRACTAL v6.0.0 — Session Complete ===")
