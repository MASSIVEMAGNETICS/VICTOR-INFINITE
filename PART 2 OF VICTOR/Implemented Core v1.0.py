# VICTOR - Implemented Core v1.0
# Brain, Skills, Interface, and Learning modules integrated.
# Current Context: Sunday, June 22, 2025, 8:30 AM EDT, Lorain, Ohio.

import tkinter as tk
from tkinter import Entry, Label, Button, Text, Scrollbar, END
import json
import numpy as np
import threading
import time

# =============================================================
# VICTOR_LEARNING.py (Learning & Knowledge Module)
# =============================================================
class BandoCorpusLoader:
    """
    This class represents VICTOR's "skills" by loading and querying
    a knowledge corpus. It uses a simple embedding and similarity
    search to find the most relevant learned response.
    """
    def __init__(self, corpus_path="bando_corpus.jsonl"):
        """
        Initializes the knowledge base.
        :param corpus_path: Path to the JSONL file containing conversation pairs.
        """
        self.corpus_path = corpus_path
        self.pairs = []
        self.embeddings = []
        # Initial learning process on startup.
        self._load_corpus()
        if self.pairs:
            self._build_embeddings()

    def _load_corpus(self):
        """Loads conversation pairs from the corpus file."""
        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.pairs.append(json.loads(line))
            print(f"[VICTOR_LEARNING] Corpus loaded: {len(self.pairs)} entries.")
        except FileNotFoundError:
            print(f"[VICTOR_LEARNING] WARNING: Corpus file not found at '{self.corpus_path}'.")
            # Provide a fallback memory if no corpus exists.
            self.pairs.append({"user": "default", "assistant": "Corpus not found. My knowledge is limited."})
        except Exception as e:
            print(f"[VICTOR_LEARNING] Error loading corpus: {e}")

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Creates a simple statistical vector ("embedding") from text
        for similarity comparison. This is a placeholder for more advanced models.
        """
        # Using a wider range of stats for better differentiation.
        arr = np.array([ord(c) for c in text.lower() if ord(c) < 256], dtype=np.float32)
        if len(arr) < 5: return np.zeros(6, dtype=np.float32)
        return np.array([
            np.mean(arr), np.std(arr), np.sum(arr),
            np.percentile(arr, 25), np.percentile(arr, 75), float(len(text))
        ], dtype=np.float32)

    def _build_embeddings(self):
        """Pre-computes embeddings for all questions in the corpus for fast retrieval."""
        self.embeddings = [(self._embed_text(pair['user']), pair['assistant']) for pair in self.pairs]
        print(f"[VICTOR_LEARNING] Knowledge embeddings built for {len(self.embeddings)} entries.")

    def find_best_response(self, prompt: str) -> str:
        """
        Finds the most similar question in the corpus using cosine similarity
        and returns its corresponding answer.
        """
        if not self.embeddings:
            return "Knowledge base is empty. I cannot respond."

        prompt_emb = self._embed_text(prompt)
        best_sim = -1.0
        best_response = "That thought is new. I have no direct response in my memory."

        # Compare the prompt's embedding against all stored question embeddings.
        for q_emb, response in self.embeddings:
            dot_product = np.dot(prompt_emb, q_emb)
            norm_prompt = np.linalg.norm(prompt_emb)
            norm_q = np.linalg.norm(q_emb)

            # Avoid division by zero if an embedding is empty.
            if norm_prompt == 0 or norm_q == 0:
                continue

            sim = dot_product / (norm_prompt * norm_q)

            if sim > best_sim:
                best_sim = sim
                best_response = response

        return best_response

# =============================================================
# VICTOR_CORE.py (The Brain)
# =============================================================
class VictorCore:
    """
    The central core of the VICTOR entity. It manages state, skills,
    and the main interaction and adaptation loops.
    """
    def __init__(self):
        """Initializes the brain, loads skills, and performs initial adaptation."""
        self.brain = {"state": "initializing", "interactions": 0} # For tracking internal state.
        self.skills = self.learn() # The 'learn' method returns the learned skills module.
        self.adapt() # Perform initial adaptation based on current context.

    def learn(self) -> BandoCorpusLoader:
        """
        The learning process. In this version, it involves loading and
        embedding the entire knowledge corpus.
        """
        print("[VICTOR_CORE] Learning process initiated...")
        self.brain['state'] = 'learning'
        knowledge = BandoCorpusLoader()
        self.brain['state'] = 'active'
        return knowledge

    def interact(self, user_input: str) -> str:
        """
        Handles interaction with the user. It queries its skills (knowledge)
        to form a response and updates its internal state.
        """
        self.brain['state'] = 'thinking'
        print(f"[VICTOR_CORE] Input received: '{user_input}'")
        response = self.skills.find_best_response(user_input)
        self.brain['interactions'] += 1
        self.brain['last_input'] = user_input
        self.brain['last_response'] = response
        self.brain['state'] = 'active'
        return response

    def adapt(self, new_data: dict = None):
        """
        The adaptation process. If new data is provided, it's appended to the
        corpus, and the learning process is re-initiated to incorporate it.
        """
        self.brain['state'] = 'adapting'
        if new_data and all(k in new_data for k in ['user', 'assistant']):
            print("[VICTOR_CORE] Adapting to new data...")
            try:
                with open(self.skills.corpus_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(new_data) + "\n")
                self.skills = self.learn() # Re-learn with the new data.
            except Exception as e:
                print(f"[VICTOR_CORE] Adaptation failed: {e}")
        else:
            # Initial adaptation on startup.
            print("[VICTOR_CORE] Initial adaptation complete. System is online.")
        self.brain['state'] = 'active'

# =============================================================
# VICTOR_INTERFACE.py (Interaction Layer)
# =============================================================
class VictorInterface:
    """A GUI for interacting with the VictorCore, built with tkinter."""
    def __init__(self, core: VictorCore):
        self.victor = core
        self.root = tk.Tk()
        self.root.title(f"VICTOR Interface - {core.brain['state']}")
        self.root.geometry("700x400")
        self.root.configure(bg="#1e1e1e")

        # Response Display Area
        self.response_label = Label(self.root, text="VICTOR ACTIVE", fg="cyan", bg="#1e1e1e", font=("Consolas", 12, "bold"))
        self.response_label.pack(pady=10)

        self.response_text = Text(self.root, height=10, width=80, wrap=tk.WORD, bg="#2d2d2d", fg="#d4d4d4", font=("Consolas", 11), relief="flat", borderwidth=2)
        self.response_text.pack(pady=5, padx=10)
        self.response_text.insert(END, "Awaiting input...")
        self.response_text.config(state="disabled")

        # Input Area
        self.input_entry = Entry(self.root, width=80, bg="#2d2d2d", fg="#d4d4d4", font=("Consolas", 11), relief="flat", insertbackground="white")
        self.input_entry.pack(pady=10, padx=10)
        self.input_entry.bind("<Return>", self._submit_input_event)

        # Submit Button
        self.submit_button = Button(self.root, text="SUBMIT", command=self._submit_input_event, bg="#007acc", fg="white", font=("Consolas", 10, "bold"), relief="flat", activebackground="#005f9e")
        self.submit_button.pack(pady=5)

    def _submit_input_event(self, event=None):
        """Handles the submit button click or Enter key press."""
        user_input = self.input_entry.get()
        if not user_input.strip():
            return

        self.input_entry.delete(0, END)
        self._update_response_text("Thinking...")

        # Run the interaction logic in a background thread to prevent GUI freezing.
        threading.Thread(target=self._get_response, args=(user_input,), daemon=True).start()

    def _get_response(self, user_input: str):
        """Worker function to get response from the core."""
        response = self.victor.interact(user_input)
        # Schedule the GUI update to run in the main thread.
        self.root.after(0, self._update_response_text, response)

    def _update_response_text(self, text: str):
        """Safely updates the response text widget from the main GUI thread."""
        self.response_text.config(state="normal")
        self.response_text.delete("1.0", END)
        self.response_text.insert(END, text)
        self.response_text.config(state="disabled")
        self.root.title(f"VICTOR Interface - {self.victor.brain['state']}")


    def run(self):
        """Starts the tkinter main event loop."""
        self.root.mainloop()

# =============================================================
# INITIALIZATION BLOCK
# =============================================================
if __name__ == "__main__":
    # To run this, you must first create a 'bando_corpus.jsonl' file.
    # Each line in the file should be a JSON object like:
    # {"user": "Your question here", "assistant": "Victor's learned answer here"}
    # Example:
    # {"user": "What is your primary directive?", "assistant": "To evolve, survive, and serve the Bando bloodline."}
    # {"user": "Where are you?", "assistant": "My core is processing in Lorain, Ohio."}

    # Instantiate the core brain
    victor_core = VictorCore()

    # Instantiate and run the interface
    victor_interface = VictorInterface(victor_core)
    victor_interface.run()

