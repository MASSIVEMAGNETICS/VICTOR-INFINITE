# File: memory_embedder_node.py
# Version: v1.0.0-FP
# Description: Embeds Victor’s memory log into a semantic vector index for advanced recall
# Author: Bando Bandz AI Ops

import os
import json
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MemoryEmbedderNode:
    """
    Embeds Victor’s memory entries using a sentence-transformer and builds a searchable FAISS vector index.
    Supports fast semantic retrieval and context-aware injection into downstream cognition.
    """

    def __init__(self):
        self.memory_log_path = "victor_memory_log/memory_log.json"
        self.index_save_path = "victor_memory_log/vector_index.faiss"
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.metadata = []
        self._load_index()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rebuild_index": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 5}),
                "query": ("STRING", {"default": "What did Victor feel about himself?"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("retrieved_memory",)
    FUNCTION = "embed_and_search"
    CATEGORY = "memory/vector_search"

    def embed_and_search(self, rebuild_index, top_k, query):
        try:
            if rebuild_index or self.index is None:
                self._build_index()

            query_vec = self.embed_model.encode([query])
            D, I = self.index.search(np.array(query_vec).astype(np.float32), top_k)

            if len(I[0]) == 0 or I[0][0] == -1:
                return ("[Victor::MemoryEmbedder] No semantic matches found.",)

            results = []
            for idx in I[0]:
                if idx < len(self.metadata):
                    entry = self.metadata[idx]
                    results.append(f"🧠 {entry['timestamp']} | *{entry['type']}* `{entry['tag']}`\n{entry['content']}")

            return ("\n\n".join(results),)

        except Exception as e:
            print(f"[Victor::MemoryEmbedder::Error] {str(e)}")
            return ("[Error] Failed to run memory embedding search.",)

    def _build_index(self):
        try:
            if not os.path.exists(self.memory_log_path):
                raise FileNotFoundError("Memory log not found.")

            with open(self.memory_log_path, "r") as f:
                memory = json.load(f)

            self.metadata = memory
            texts = [entry["content"] for entry in memory]
            vectors = self.embed_model.encode(texts, convert_to_numpy=True)

            dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(vectors.astype(np.float32))

            faiss.write_index(self.index, self.index_save_path)
            print(f"[Victor::MemoryEmbedder] Index built with {len(texts)} entries.")

        except Exception as e:
            print(f"[Victor::MemoryEmbedder::BuildError] {str(e)}")
            self.index = None

    def _load_index(self):
        if os.path.exists(self.index_save_path):
            try:
                self.index = faiss.read_index(self.index_save_path)
                print("[Victor::MemoryEmbedder] Vector index loaded.")
            except Exception as e:
                print(f"[Victor::MemoryEmbedder::IndexLoadError] {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "MemoryEmbedderNode": MemoryEmbedderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryEmbedderNode": "Memory: Semantic Embedder"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
