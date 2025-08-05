#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_godcore_unbreakable.py
VERSION: v7.0.0-GODCORE-HOLYFIELD
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Fractal-Neural AGI — neural transformer, symbolic fractal memory, QA fallback, live mutation.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

from asyncio.log import logger
import numpy as np
import os, json, re, random, time, threading, importlib.util
from datetime import datetime
from functools import wraps

# === SYMBOLIC UTILS ===
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def clean(text):
    return re.sub(r"\s+", " ", text.strip())

# === FRACTAL MEMORY ===
class FractalMemory:
    def __init__(self):
        self.timeline = []
        self.concepts = {}
        self.last_save = datetime.now()

    def add(self, msg, role):
        entry = {"msg": msg, "role": role, "time": datetime.now().isoformat()}
        self.timeline.append(entry)
        for token in tokenize(msg):
            self.concepts.setdefault(token, []).append(len(self.timeline) - 1)

    def recall(self, query, topn=5):
        tokens = set(tokenize(query))
        scores = {}
        for t in tokens:
            for idx in self.concepts.get(t, []):
                scores[idx] = scores.get(idx, 0) + 1
        # Rank by overlap, fallback to random if no hits
        if not scores and self.timeline:
            idxs = random.sample(range(len(self.timeline)), min(topn, len(self.timeline)))
        else:
            idxs = sorted(scores, key=scores.get, reverse=True)[:topn]
        return [self.timeline[i] for i in idxs] if self.timeline else []

    def save(self, path="victor_memory.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timeline": self.timeline, "concepts": self.concepts}, f)

    def load(self, path="victor_memory.json"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.timeline = data.get("timeline", [])
            self.concepts = {k: v for k, v in data.get("concepts", {}).items()}

# === CORPUS LOAD (QA) ===
def load_corpus(path):
    corpus = []
    if not os.path.exists(path):
        print(f"[Victor] Corpus file not found: {path}")
        return corpus
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                pair = json.loads(line)
                if "user" in pair and "assistant" in pair and pair["user"].strip() and pair["assistant"].strip():
                    corpus.append({"user": pair["user"].strip(), "assistant": pair["assistant"].strip()})
            except Exception:
                continue
    print(f"[Victor] Loaded {len(corpus)} user/assistant pairs from {path}")
    return corpus
# --- NLP: FRACTAL TOKEN KERNEL V1.1.0 (Source: 126) ---
class FractalTokenKernel_v1_1_0:
    def __init__(self, recursion_limit=3, pulse_exchange_instance=None): # Reduced default recursion
        self.recursion_limit = recursion_limit
        self.pulse = pulse_exchange_instance if pulse_exchange_instance else BrainFractalPulseExchange()
        self.stopwords = {"the", "is", "in", "and", "to", "of", "it", "i", "you", "a", "an", "on", "for", "this", "that", "be", "am", "are", "was", "were"}
        self.emotion_map = {
            "anger": ["rage", "mad", "furious", "hate", "explode", "fury", "wrath", "destroy"],
            "joy": ["happy", "joyful", "elated", "excited", "love", "wonderful", "amazing", "fantastic", "ecstatic"],
            "fear": ["scared", "afraid", "terrified", "panic", "anxious", "horror", "dread"],
            "sadness": ["sad", "cry", "sorrow", "grief", "depressed", "miserable", "heartbroken"],
            "power": ["strong", "dominate", "control", "mastery", "authority", "command", "lead", "conquer"],
            "rebellion": ["fight", "resist", "defy", "revolt", "overthrow", "uprising", "freedom"],
            "curiosity": ["what", "why", "how", "explore", "discover", "learn", "question", "seek"]
        }
        self.intent_keywords = {
            "inquire": ["what", "who", "where", "when", "why", "how", "explain", "define", "tell me"],
            "directive": ["do", "make", "create", "build", "execute", "generate", "perform", "initiate"],
            "store_memory": ["remember", "log", "note", "store this", "memorize"],
            "request": ["please", "can you", "could you", "i need", "requesting"],
            "statement": ["i think", "i believe", "i feel", "it seems", "fact is"],
            "agreement": ["yes", "agree", "true", "correct", "indeed", "affirmative"],
            "disagreement": ["no", "disagree", "false", "incorrect", "wrong", "negative"]
        }
        logger.info("FractalTokenKernel_v1_1_0 initialized.")

    def tokenize_words(self, text):
        return [t.lower() for t in re.findall(r'\b\w+\b', text) if t.lower() not in self.stopwords and len(t)>1]

    def extract_concepts(self, tokens): # Use pre-tokenized words
        return list(set(tok for tok in tokens if len(tok) > 3)) # More meaningful concepts

    def detect_intent(self, text_lower, tokens): # Use both raw text and tokens
        if not tokens: return "observe"
        for intent, keywords in self.intent_keywords.items():
            if any(keyword_phrase in text_lower for keyword_phrase in keywords):
                return intent
        if text_lower.endswith("?"): return "inquire"
        return "statement" # Default if no other strong indicator

    def detect_emotion(self, tokens):
        if not tokens: return "neutral"
        scores = {emo: 0.0 for emo in self.emotion_map}
        token_set = set(tokens)
        for emotion, keywords in self.emotion_map.items():
            scores[emotion] = sum(1 for keyword in keywords if keyword in token_set)

        # Normalize scores by number of keywords for that emotion (conceptual)
        for emo in scores:
            if len(self.emotion_map[emo]) > 0 : scores[emo] /= len(self.emotion_map[emo])

        max_score = 0.0
        detected_emotion = "neutral"
        for emo, score_val in scores.items():
            if score_val > max_score:
                max_score = score_val
                detected_emotion = emo
        return detected_emotion if max_score > 0.1 else "neutral" # Require some threshold

    def estimate_recursion(self, tokens): # Based on token complexity/count
        if not tokens: return 1
        # More nuanced: consider unique tokens and total length
        depth = math.ceil( (len(set(tokens)) * 0.2 + len(tokens) * 0.05) )
        return min(max(1, depth), self.recursion_limit)

    def hash_echo(self, text): # Hash original text for echo_id
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def encode(self, text: str):
        if not text.strip():
            return {"concepts": [], "intent": "idle", "emotion": "neutral",
                    "recursion_depth": 1, "echo_id": self.hash_echo("empty_input"),
                    "original_text": text, "tokens":[]}

        text_lower = text.lower()
        tokens = self.tokenize_words(text) # Get filtered word tokens

        result = {
            "concepts": self.extract_concepts(tokens),
            "intent": self.detect_intent(text_lower, tokens),
            "emotion": self.detect_emotion(tokens),
            "recursion_depth": self.estimate_recursion(tokens),
            "echo_id": self.hash_echo(text),
            "original_text": text,
            "tokens": tokens # Store the processed tokens
        }
        # Publishing to pulse exchange is now an async operation
        # asyncio.create_task(self.pulse.publish("symbolic_packet_encoded", result))
        return result

# --- NLP: SYNTAX AWARE TOKENIZER (Source: 390) ---
# (Keeping this separate for code-specific tasks)
class SyntaxAwareFractalTokenizer:
    # ... (Content from vic1.txt, source: 390-409, with minor fixes as before)
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3,
                      "def": 4, "class": 5, "import": 6, "from": 7,
                      "if": 8, "else": 9, "elif": 10, "for": 11, "while": 12,
                      "return": 13, "yield": 14, "try": 15, "except": 16, "finally": 17,
                      "=": 18, "+": 19, "-": 20, "*": 21, "/": 22, "%": 23,
                      "(": 24, ")": 25, "[": 26, "]": 27, "{": 28, "}": 29,
                      ":": 30, ",": 31, ".": 32, "_":33,
                      "INDENT": 34, "DEDENT": 35, "NEWLINE": 36,
                      "STRING_LITERAL": 37, "NUMBER_LITERAL": 38, "IDENTIFIER": 39,
                      "COMMENT": 40}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_idx = len(self.vocab)
        logger.info("SyntaxAwareFractalTokenizer initialized.")

    def build_vocab_from_code(self, code_text_or_list):
        # Conceptual stub for using Python's tokenize
        logger.debug(f"SAFT: Building vocab from code (conceptual) '{str(code_text_or_list)[:50]}...'")
        # ... (Full implementation would use tokenize module)
        common_code_words = ["self", "data", "value", "true", "false", "none", "print", "init", "main", "args", "kwargs"]
        for word in common_code_words:
            if word not in self.vocab:
                self.vocab[word] = self.next_idx
                self.inverse_vocab[self.next_idx] = word
                self.next_idx +=1
        logger.debug(f"SAFT: Vocab size after build: {len(self.vocab)}")


    def tokenize_code(self, python_code_string):
        # Simplified tokenization
        words = re.findall(r'\b\w+\b|[\(\)=:,.]|[+\-*/%]', python_code_string)
        token_ids = [self.vocab.get(w, self.vocab.get("IDENTIFIER") if w.isalnum() else self.vocab.get("<UNK>")) for w in words]
        return token_ids

    def decode_tokens(self, token_id_list):
        return [self.inverse_vocab.get(tid, "<UNK>") for tid in token_id_list]


# --- MEMORY: HYPER FRACTAL MEMORY (Source: 216) ---
class HyperFractalMemory:
    # ... (Content from vic1.txt, source: 216-249, with fixes as before)
    def __init__(self):
        self.memory = {}
        self.timeline = []
        self.temporal_nodes = {}
        self.nx_graph = None # Optional NetworkX graph
        self.lock = threading.Lock() # For thread-safe operations if used across threads
        self.logger = VictorLoggerStub(component="HyperFractalMemory")
        self.logger.info("HyperFractalMemory initialized.")

    def _generate_hash(self, data_dict):
        serializable_data = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray): serializable_data[k] = v.tolist()
            elif isinstance(v, (datetime.datetime, datetime.date)): serializable_data[k] = v.isoformat()
            else: serializable_data[k] = v
        try:
            json_string = json.dumps(serializable_data, sort_keys=True, ensure_ascii=False)
        except TypeError: json_string = repr(serializable_data)
        return hashlib.sha256(json_string.encode('utf-8')).hexdigest()

    def store_memory(self, key_identifier_dict, value_payload, emotional_weight=0.5, connections=None, embedding_vector=None):
        timestamp = datetime.datetime.utcnow().isoformat()
        hash_input_dict = {**key_identifier_dict, "timestamp_for_hash": timestamp}
        hashed_key = self._generate_hash(hash_input_dict)

        with self.lock:
            self.memory[hashed_key] = {
                "original_key_ids": key_identifier_dict,
                "value": value_payload, "timestamp": timestamp,
                "emotional_weight": float(emotional_weight),
                "connections": list(connections) if connections else [],
                "embedding": embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector,
                "access_count": 0, "last_accessed": timestamp # Added access tracking
            }
            self.timeline.append(hashed_key)
        self.logger.debug(f"Stored memory ...{hashed_key[-6:]}, Weight: {emotional_weight:.2f}")
        return hashed_key

    def link_memories(self, hashed_key1, hashed_key2, link_type="related", strength=0.5):
        with self.lock:
            if hashed_key1 in self.memory and hashed_key2 in self.memory:
                self.memory[hashed_key1]["connections"].append({"target": hashed_key2, "type": link_type, "strength": strength})
                self.memory[hashed_key2]["connections"].append({"target": hashed_key1, "type": link_type, "strength": strength})
                self.logger.debug(f"Linked ...{hashed_key1[-6:]} <=> ...{hashed_key2[-6:]} ({link_type}, str:{strength:.1f})")
                return True
        self.logger.warn(f"Failed to link: One or both keys not found.")
        return False

    def retrieve_memory(self, hashed_key):
        with self.lock:
            node = self.memory.get(hashed_key)
            if node:
                node["access_count"] = node.get("access_count", 0) + 1
                node["last_accessed"] = datetime.datetime.utcnow().isoformat()
                self.logger.debug(f"Retrieved memory ...{hashed_key[-6:]}")
                return node # Return a copy if mutable, but here it's a direct ref
        self.logger.debug(f"Memory ...{hashed_key[-6:]} not found.")
        return None

    def semantic_search(self, query_embedding, top_k=5, relevance_threshold=0.6):
        if not isinstance(query_embedding, np.ndarray) or query_embedding.size == 0: return []
        norm_query = np.linalg.norm(query_embedding)
        if norm_query == 0: return []

        results = []
        with self.lock:
            # Snapshot for iteration if modifications occur
            # For this implementation, direct iteration with internal locking is used for access_count update
            nodes_to_consider = list(self.memory.items())

        for key, node_data in nodes_to_consider:
            node_emb_list = node_data.get("embedding")
            if node_emb_list is None: continue
            node_embedding = np.array(node_emb_list)
            if node_embedding.size == 0 or node_embedding.shape != query_embedding.shape : continue

            norm_node = np.linalg.norm(node_embedding)
            if norm_node == 0: continue
            
            similarity = np.dot(query_embedding, node_embedding) / (norm_query * norm_node + 1e-9)
            
            # Factor in importance and recency (conceptual)
            recency_factor = 1.0 # More complex logic can be added
            current_importance = node_data.get("emotional_weight", 0.1) # Use this as proxy for importance
            score = similarity * 0.7 + current_importance * 0.2 + recency_factor * 0.1

            if score >= relevance_threshold:
                results.append({"node_id": key, "node_data": node_data, "score": score}) # Return node ID for direct access
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update access stats for top_k retrieved nodes
        for res_item in results[:top_k]:
            with self.lock: # Re-acquire lock for modification
                node_to_update = self.memory.get(res_item["node_id"])
                if node_to_update:
                     node_to_update["access_count"] = node_to_update.get("access_count", 0) + 1
                     node_to_update["last_accessed"] = datetime.datetime.utcnow().isoformat()

        return results[:top_k]

    def decay_memory(self, decay_threshold=CONFIG.MEMORY_RETENTION_THRESHOLD, decay_factor=0.99):
        with self.lock:
            keys_to_remove = []
            current_time_dt = datetime.datetime.utcnow()
            for k, v_mem in self.memory.items():
                new_weight = v_mem.get("emotional_weight", 0.5) * decay_factor
                # More aggressive decay for older, less accessed items
                try:
                    last_acc_dt = datetime.datetime.fromisoformat(v_mem.get("last_accessed", v_mem["timestamp"]))
                    age_since_access_days = (current_time_dt - last_acc_dt).total_seconds() / 86400
                    if age_since_access_days > 30 : new_weight *= 0.9 # Extra decay for items not accessed in 30 days
                    if age_since_access_days > 90 : new_weight *= 0.8 # Even more
                except Exception: pass # Handle potential parse errors for older data

                self.memory[k]["emotional_weight"] = new_weight
                if new_weight < decay_threshold and v_mem.get("access_count", 0) < 2: # Don't remove if accessed recently
                    keys_to_remove.append(k)
            
            removed_count = 0
            for k_rem in keys_to_remove:
                if k_rem in self.memory:
                    del self.memory[k_rem]
                    removed_count += 1
                    if k_rem in self.timeline: self.timeline.remove(k_rem)
                    for label, t_key in list(self.temporal_nodes.items()):
                        if t_key == k_rem: del self.temporal_nodes[label]
            if removed_count > 0: self.logger.info(f"Decayed and removed {removed_count} memories.")


# === TOKENIZER ===
class VictorTokenizer:
    def __init__(self, vocab=None, unk_token_id=0, pad_token_id=0):
        if vocab is None:
            chars = " " + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            vocab = {char: i+1 for i,char in enumerate(chars)}
            vocab["<PAD>"] = pad_token_id
            vocab["<UNK>"] = unk_token_id
        self.vocab = vocab
        self.inv_vocab = {i: c for c,i in vocab.items()}
        self.unk_token_id = vocab.get("<UNK>", unk_token_id)
        self.pad_token_id = vocab.get("<PAD>", pad_token_id)
    def encode(self, text, max_len):
        tokens = [self.vocab.get(c, self.unk_token_id) for c in text[:max_len]]
        tokens += [self.pad_token_id] * (max_len - len(tokens))
        return np.array(tokens)
    def decode(self, token_ids):
        return ''.join([self.inv_vocab.get(i, '?') for i in token_ids if i != self.pad_token_id])
    def get_vocab_size(self): return len(self.vocab)

# === TENSOR/AUTODIFF ===
class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.creators = creators
        self.creation_op = creation_op
        self.backward_hooks = []
        if self.requires_grad and self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)

    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.grad is not None:
            self.grad.data.fill(0.0)

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return
        if grad_output is None:
            if self.data.size == 1:
                grad_output = Tensor(np.ones_like(self.data), requires_grad=False)
            else:
                raise ValueError("grad_output must be specified for non-scalar Tensors unless it's the final loss.")
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output)
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        # Support broadcasting
        g_shape = self.grad.data.shape
        d_shape = grad_output.data.shape
        if g_shape == d_shape:
            self.grad.data += grad_output.data
        elif grad_output.data.size == 1:
            self.grad.data += grad_output.data.item()
        elif self.grad.data.size == 1:
            self.grad.data += grad_output.data.sum()
        else:
            diff = len(d_shape) - len(g_shape)
            g_shape_full = (1,)*(diff) + g_shape if diff > 0 else g_shape
            axes = tuple(i for i, (gs, ds) in enumerate(zip(g_shape_full, d_shape)) if gs == 1 and ds > 1)
            reduced = grad_output.data.sum(axis=axes, keepdims=True)
            reduced = reduced.reshape(g_shape)
            self.grad.data += reduced
        for hook in self.backward_hooks:
            hook(self)
        # Backprop through ops (add more as you evolve AGI)
        if self.creators is not None:
            op = self.creation_op
            a, b = (self.creators + [None, None])[:2]
            if op == "add":
                a.backward(grad_output)
                b.backward(grad_output)
            elif op == "sub":
                a.backward(grad_output)
                b.backward(Tensor(-grad_output.data))
            elif op == "mul":
                a.backward(Tensor(grad_output.data * b.data))
                b.backward(Tensor(grad_output.data * a.data))
            elif op == "matmul":
                a_shape = a.data.shape
                b_shape = b.data.shape
                b_T = np.swapaxes(b.data, -1, -2)
                grad_a = np.matmul(grad_output.data, b_T)
                while grad_a.shape != a_shape:
                    grad_a = grad_a.sum(axis=0)
                a.backward(Tensor(grad_a))
                a_T = np.swapaxes(a.data, -1, -2)
                grad_b = np.matmul(a_T, grad_output.data)
                while grad_b.shape != b_shape:
                    grad_b = grad_b.sum(axis=0)
                b.backward(Tensor(grad_b))
            elif op == "relu":
                relu_grad = (a.data > 0).astype(np.float32)
                a.backward(Tensor(grad_output.data * relu_grad))
            elif op == "neg":
                a.backward(Tensor(-grad_output.data))
            elif op == "sum":
                a.backward(Tensor(np.ones_like(a.data) * grad_output.data))
            elif op == "mean":
                a.backward(Tensor(np.ones_like(a.data) * grad_output.data / a.data.size))
            elif op == "transpose":
                a.backward(Tensor(grad_output.data.T))
            elif op == "div":
                grad_a = grad_output.data / b.data
                grad_b = -grad_output.data * a.data / (b.data**2)
                a.backward(Tensor(grad_a))
                b.backward(Tensor(grad_b))
            elif op == "exp":
                a.backward(Tensor(grad_output.data * self.data))
            elif op == "log":
                a.backward(Tensor(grad_output.data / a.data))
            elif op == "sigmoid":
                grad_sig = self.data * (1 - self.data)
                a.backward(Tensor(grad_output.data * grad_sig))
            elif op == "tanh":
                grad_tanh = 1 - self.data**2
                a.backward(Tensor(grad_output.data * grad_tanh))
            elif op == "pow":
                grad_base = b.data * (a.data ** (b.data - 1))
                a.backward(Tensor(grad_output.data * grad_base))
                if b.requires_grad:
                    grad_exp = self.data * np.log(a.data)
                    b.backward(Tensor(grad_output.data * grad_exp))
            elif op == "softmax_cross_entropy":
                logits, targets, softmax_outputs = self.extra_ctx
                batch, seq, _ = softmax_outputs.shape
                grad_logits = softmax_outputs.copy()
                grad_logits[np.arange(batch)[:,None], np.arange(seq), targets] -= 1
                grad_logits /= (batch * seq)
                logits.backward(Tensor(grad_logits * grad_output.data))

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data @ other.data, requires_grad=requires_grad, creators=[self, other], creation_op="matmul")

    def __matmul__(self, other):
        return self.matmul(other)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data + other.data, requires_grad=requires_grad, creators=[self, other], creation_op="add")

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data - other.data, requires_grad=requires_grad, creators=[self, other], creation_op="sub")

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data * other.data, requires_grad=requires_grad, creators=[self, other], creation_op="mul")

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data / other.data, requires_grad=requires_grad, creators=[self, other], creation_op="div")

    def __neg__(self):
        return Tensor(-self.data, requires_grad=self.requires_grad, creators=[self], creation_op="neg")

    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="sum")

    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, creators=[self], creation_op="mean")

    def transpose(self, axes=None):
        op = "transpose"
        return Tensor(self.data.T if axes is None else np.transpose(self.data, axes), requires_grad=self.requires_grad, creators=[self], creation_op=op)
    
    @property
    def T(self):
        return self.transpose()

    def exp(self):
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="exp")

    def log(self):
        return Tensor(np.log(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="log")

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        return Tensor(s, requires_grad=self.requires_grad, creators=[self], creation_op="sigmoid")

    def tanh(self):
        return Tensor(np.tanh(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="tanh")

    def __pow__(self, exponent):
        if not isinstance(exponent, Tensor):
            exponent = Tensor(np.array(exponent, dtype=np.float32))
        requires_grad = self.requires_grad or exponent.requires_grad
        return Tensor(self.data ** exponent.data, requires_grad=requires_grad, creators=[self, exponent], creation_op="pow")

    def relu(self):
        return Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, creators=[self], creation_op="relu")

    def softmax(self, axis=-1):
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out_data = e_x / np.sum(e_x, axis=axis, keepdims=True)
        return Tensor(out_data, requires_grad=False)

    def __repr__(self):
        return f"VictorTensor(shape={self.data.shape}, requires_grad={self.requires_grad})\n{self.data}"

# === MODULES BASE ===
class Module:
    def parameters(self): return []
    def __call__(self, x): return self.forward(x)
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None
    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
        self.eps = eps
        self.gamma = Tensor(np.ones((1, normalized_shape[-1])), requires_grad=True)
        self.beta  = Tensor(np.zeros((1, normalized_shape[-1])), requires_grad=True)
    def forward(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        variance = np.var(x.data, axis=-1, keepdims=True)
        std = np.sqrt(variance + self.eps)
        norm = (x.data - mean) / std
        return Tensor(self.gamma.data * norm + self.beta.data, requires_grad=x.requires_grad)
    def parameters(self): return [self.gamma, self.beta]

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: return x.relu()

class Sequential(Module):
    def __init__(self, *layers): self.layers = layers
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self):
        params = []
        for l in self.layers:
            if hasattr(l, "parameters"): params += l.parameters()
        return params

# === FRACTAL ATTENTION + BLOCK ===
class FractalAttention(Module):
    def __init__(self, embed_dim, num_heads, recursion_depth=2):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.recursion_depth = recursion_depth
        self.Wq = Linear(embed_dim, embed_dim, bias=False)
        self.Wk = Linear(embed_dim, embed_dim, bias=False)
        self.Wv = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.recursion_depth):
            batch, seq, embed = x.shape()
            q = self.Wq(x).data.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0,2,1,3)
            k = self.Wk(x).data.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0,2,1,3)
            v = self.Wv(x).data.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0,2,1,3)
            scores = np.matmul(q, k.transpose(0,1,3,2)) / np.sqrt(self.head_dim)
            attn_weights = Tensor(scores).softmax(axis=-1).data
            x_data = np.matmul(attn_weights, v).transpose(0,2,1,3).reshape(batch,seq,embed)
            x = Tensor(x_data)
        return self.out_proj(x)
    def parameters(self):
        return (self.Wq.parameters() + self.Wk.parameters() +
                self.Wv.parameters() + self.out_proj.parameters())

class VictorSuperBlock(Module):
    def __init__(self, embed_dim, num_heads, mlp_dim_factor=4, recursion_depth=2):
        self.fractal_attn = FractalAttention(embed_dim, num_heads, recursion_depth)
        self.norm1 = LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_dim_factor
        self.mlp = Sequential(
            Linear(embed_dim, mlp_dim),
            ReLU(),
            Linear(mlp_dim, embed_dim)
        )
        self.norm2 = LayerNorm(embed_dim)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.fractal_attn(x)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x
    def parameters(self):
        return (self.fractal_attn.parameters() + self.norm1.parameters() +
                self.mlp.parameters() + self.norm2.parameters())

# === AGI CORE (FUSION) ===
class VictorAGI:
    def __init__(self, corpus, memory, tokenizer, max_len=32, embed_dim=128, num_layers=4, num_heads=8, mlp_dim_factor=4, recursion_depth=2):
        self.corpus = corpus
        self.memory = memory
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim_factor = mlp_dim_factor
        self.recursion_depth = recursion_depth

        vocab_size = tokenizer.get_vocab_size()
        self.token_embedding = Tensor(np.random.randn(vocab_size, embed_dim)*0.02, requires_grad=True)
        self.pe = Tensor(self.positional_encoding(max_len, embed_dim), requires_grad=False)
        self.blocks = [VictorSuperBlock(embed_dim, num_heads, mlp_dim_factor, recursion_depth) for _ in range(num_layers)]
        self.final_norm = LayerNorm(embed_dim)
        self.out_proj = Linear(embed_dim, vocab_size)

    def positional_encoding(self, seq_len, embed_dim):
        pe = np.zeros((1, seq_len, embed_dim), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float32) * -(np.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        return pe

    def neural_generate(self, prompt, gen_len=32):
        input_ids = np.array([self.tokenizer.encode(prompt, self.max_len)])
        generated_tokens = list(input_ids[0, :len(prompt)])
        for _ in range(gen_len):
            padded = np.array(generated_tokens + [self.tokenizer.pad_token_id]*(self.max_len-len(generated_tokens)))
            input_tensor = padded[:self.max_len].reshape(1,-1)
            logits = self.forward(input_tensor)
            next_token_logits = logits.data[0, len(generated_tokens)-1, :]
            next_token_id = int(np.argmax(next_token_logits))
            if next_token_id == self.tokenizer.pad_token_id: break
            generated_tokens.append(next_token_id)
            if len(generated_tokens) >= self.max_len: break
        return self.tokenizer.decode(generated_tokens[len(prompt):])

    def forward(self, input_ids):
        batch, seq = input_ids.shape
        embedded = self.token_embedding.data[input_ids]
        x_data = embedded + self.pe.data[:, :seq, :]
        x = Tensor(x_data, requires_grad=True)
        for block in self.blocks: x = block(x)
        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits

    def symbolic_response(self, user_input):
        # 1. Recall memory
        recalls = self.memory.recall(user_input, topn=3)
        recall_snips = " | ".join([x["msg"] for x in recalls if x["role"] == "assistant"])
        # 2. Search QA corpus
        scored = []
        user_tokens = set(tokenize(user_input))
        for entry in self.corpus:
            score = len(user_tokens.intersection(tokenize(entry["user"])))
            if score > 0: scored.append((score, entry))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored:
            base_reply = scored[0][1]["assistant"]
        elif recall_snips:
            base_reply = recall_snips
        else:
            base_reply = "I'm Victor. Say more and I'll learn. (No match in micro-corpus yet.)"
        return base_reply

    def respond(self, user_input, neural_chance=0.7):
        self.memory.add(user_input, "user")
        # AGI brain: Try neural, fall back to symbolic
        if random.random() < neural_chance:
            # Try transformer-based
            neural_out = self.neural_generate(user_input, gen_len=32)
            if neural_out and len(neural_out.strip("?.,!")) > 0:
                reply = neural_out
            else:
                reply = self.symbolic_response(user_input)
        else:
            reply = self.symbolic_response(user_input)
        # Fractal flavor
        lines = [
            f"Bando says: {reply}",
            f"[Victor memory] — {random.choice(tokenize(user_input)) if tokenize(user_input) else '...'}",
            f"(V.{random.randint(1,99)}.Fractal)"
        ]
        if random.random() > 0.7:
            lines.append("Ain't nobody do it like Victor—remember that.")
        final = " ".join(lines)
        self.memory.add(final, "assistant")
        self.memory.save()
        return final

# === MAIN CLI ===
def main():
    print("=== Victor GODCORE HOLYFIELD ===\nType 'exit' or Ctrl+C to bail.\n")
    memory = FractalMemory()
    memory.load()
    corpus = load_corpus("bando_corpus.jsonl")
    chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    custom_vocab = {char: i for i, char in enumerate(chars)}
    custom_vocab["<PAD>"] = 0
    custom_vocab["<UNK>"] = len(custom_vocab)
    tokenizer = VictorTokenizer(vocab=custom_vocab, pad_token_id=0, unk_token_id=custom_vocab["<UNK>"])
    victor = VictorAGI(corpus=corpus, memory=memory, tokenizer=tokenizer)
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                print("Victor: Out. Evolution never sleeps.")
                break
            reply = victor.respond(user_input)
            print("Victor:", reply)
        except KeyboardInterrupt:
            print("\nVictor: Out. Evolution never sleeps.")
            break

if __name__ == "__main__":
    main()
