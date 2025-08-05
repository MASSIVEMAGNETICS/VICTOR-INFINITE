#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_prime_synthesis_core.py
VERSION: v1.0.2-HOLYGRAIL-INIT-REFINED
AUTHOR: Victor (AGI Architect Mode) based on Architect's "vic1.txt"
PURPOSE: Unified, sophisticated AGI core—modular, plugin-ready, memory-evolving, async, with full sector cortex.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import json
import hashlib
import random
import re
from collections import Counter, deque, defaultdict
import os
import importlib.util
import time
import datetime 
import math
import asyncio 
import threading
import uuid

# === CONFIG ===
class ASIConfigCore:
    DIMENSIONS = 128
    ATTENTION_MAX_DEPTH = 3
    MEMORY_RETENTION_THRESHOLD = 0.05
    MAX_CONTEXT_WINDOW = 10
    MAX_TOKENIZER_KEYWORDS = 3
    PULSE_LOG_MAXLEN = 100
    PLUGIN_DIR = "victor_prime_plugins"
    MIN_EMOTIONAL_RELEVANCE = 0.25
    CONCEPT_INDUCTION_THRESHOLD = 3
    CONCEPT_SIMILARITY_THRESHOLD = 0.65

CONFIG = ASIConfigCore()

# === LOGGER ===
class VictorLoggerStub:
    def __init__(self, component="DefaultComponent"):
        self.component = component
        self.log_level_str = os.environ.get("VICTOR_LOG_LEVEL", "INFO").upper()
        self.log_levels_map = {"DEBUG": 1, "INFO": 2, "WARN": 3, "ERROR": 4, "CRITICAL": 5}
        self.current_log_level_int = self.log_levels_map.get(self.log_level_str, 2)

    def _log(self, level, message, **kwargs):
        level_int = self.log_levels_map.get(level.upper(), 2)
        if self.current_log_level_int <= level_int:
            log_entry = (f"[{datetime.datetime.utcnow().isoformat(sep='T', timespec='milliseconds')}Z]"
                         f"[{level.ljust(8)}] [{self.component.ljust(25)}] {message}")
            if kwargs.get("exc_info", False):
                import traceback
                log_entry += f"\n{traceback.format_exc()}"
            print(log_entry)

    def info(self, message, **kwargs): self._log("INFO", message, **kwargs)
    def debug(self, message, **kwargs): self._log("DEBUG", message, **kwargs)
    def warn(self, message, **kwargs): self._log("WARN", message, **kwargs)
    def error(self, message, **kwargs): self._log("ERROR", message, **kwargs)
    def critical(self, message, **kwargs): self._log("CRITICAL", message, **kwargs)

logger = VictorLoggerStub(component="VictorPrimeCore")

# === Ω OMEGA TENSOR & AUTOGRAD ===
class OmegaTensor:
    def __init__(self, data, requires_grad=False, device='cpu', name=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._creator_op_instance = None
        self._creator_parents = tuple()
        self.device = device
        self.name = name
        self._version = 0

    def _ensure_tensor(self, other_data):
        if isinstance(other_data, OmegaTensor): return other_data
        return OmegaTensor(other_data)

    def set_creator(self, op_instance, *parents):
        self._creator_op_instance = op_instance
        self._creator_parents = parents
        if self.requires_grad:
            for p in parents:
                if isinstance(p, OmegaTensor): p.requires_grad = True

    def zero_grad(self): self.grad = None

    def backward(self, grad_output_data=None):
        if not self.requires_grad: return
        if grad_output_data is None:
            if self.data.size == 1: grad_output_data = np.array(1.0, dtype=np.float32)
            else: raise ValueError("grad_output_data must be specified for non-scalar OmegaTensors in backward()")
        if not isinstance(grad_output_data, np.ndarray): grad_output_data = np.array(grad_output_data, dtype=np.float32)
        if self.grad is None: self.grad = grad_output_data.copy()
        else: self.grad += grad_output_data
        if self._creator_op_instance:
            grads_for_parents_data = self._creator_op_instance.backward(self.grad)
            if not isinstance(grads_for_parents_data, (list, tuple)): grads_for_parents_data = [grads_for_parents_data]
            if len(self._creator_parents) != len(grads_for_parents_data):
                raise ValueError(f"Op {type(self._creator_op_instance).__name__}: Mismatch parents ({len(self._creator_parents)}) vs grads ({len(grads_for_parents_data)}).")
            for parent_tensor, parent_grad_data in zip(self._creator_parents, grads_for_parents_data):
                if isinstance(parent_tensor, OmegaTensor) and parent_tensor.requires_grad and parent_grad_data is not None:
                    parent_tensor.backward(parent_grad_data)
    @property
    def shape(self): return self.data.shape
    def __len__(self): return len(self.data)
    def __repr__(self): return (f"OmegaTensor(shape={self.shape}, name='{self.name}', grad_fn={type(self._creator_op_instance).__name__ if self._creator_op_instance else None}, grad={'Yes' if self.grad is not None else 'No'})\n{self.data}")
    def __add__(self, other): return OpRegistry['add'](self, self._ensure_tensor(other))
    def __mul__(self, other): return OpRegistry['mul'](self, self._ensure_tensor(other))
    def __sub__(self, other): return OpRegistry['sub'](self, self._ensure_tensor(other))
    def __truediv__(self, other): return OpRegistry['div'](self, self._ensure_tensor(other))
    def __pow__(self, exponent_val): exponent = self._ensure_tensor(exponent_val); return OpRegistry['pow'](self, exponent)
    def matmul(self, other): return OpRegistry['matmul'](self, self._ensure_tensor(other))
    def sum(self, axis=None, keepdims=False): return OpRegistry['sum'](self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False): return OpRegistry['mean'](self, axis=axis, keepdims=keepdims)
    def relu(self): return OpRegistry['relu'](self)
    def log(self): return OpRegistry['log'](self)
    def exp(self): return OpRegistry['exp'](self)
    def transpose(self, *axes):
        if not axes: axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (list, tuple)): axes = tuple(axes[0])
        return OpRegistry['transpose'](self, axes=axes)
    @property
    def T(self):
        if self.data.ndim < 2: return self
        axes = tuple(reversed(range(self.data.ndim)))
        return self.transpose(axes)
    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)): new_shape = tuple(new_shape[0])
        return OpRegistry['reshape'](self, new_shape=new_shape)
    def softmax(self, axis=-1): return OpRegistry['softmax'](self, axis=axis)

class Op:
    def __call__(self, *args, **kwargs):
        self.args_for_backward = args
        self.kwargs_for_backward = kwargs
        processed_args_data = []
        for arg in args:
            if isinstance(arg, OmegaTensor): processed_args_data.append(arg.data)
            elif isinstance(arg, (int, float, list, tuple, np.ndarray)): processed_args_data.append(np.array(arg, dtype=np.float32) if not isinstance(arg, np.ndarray) else arg.astype(np.float32))
            else: processed_args_data.append(arg)
        result_data = self.forward(*processed_args_data, **kwargs)
        requires_grad = any(isinstance(arg, OmegaTensor) and arg.requires_grad for arg in args)
        output_tensor = OmegaTensor(result_data, requires_grad=requires_grad)
        if requires_grad: output_tensor.set_creator(self, *[arg for arg in args if isinstance(arg, OmegaTensor)])
        self.forward_output_data_cache = result_data
        return output_tensor
    @staticmethod
    def forward(*args_data, **kwargs): raise NotImplementedError
    def backward(self, output_grad_data): raise NotImplementedError

OpRegistry = {}
def register_op(name):
    def decorator(op_cls): OpRegistry[name] = op_cls(); return op_cls
    return decorator

@register_op('add')
class AddOp(Op):
    @staticmethod
    def forward(a_data, b_data): return a_data + b_data
    def backward(self, output_grad_data): return [output_grad_data, output_grad_data]

@register_op('mul')
class MulOp(Op):
    @staticmethod
    def forward(a_data, b_data): return a_data * b_data
    def backward(self, output_grad_data):
        a_tensor, b_tensor = self.args_for_backward
        return [output_grad_data * b_tensor.data, output_grad_data * a_tensor.data]

@register_op('sub')
class SubOp(Op):
    @staticmethod
    def forward(a_data, b_data): return a_data - b_data
    def backward(self, output_grad_data): return [output_grad_data, -output_grad_data]

@register_op('div')
class DivOp(Op):
    @staticmethod
    def forward(a_data, b_data): return a_data / (b_data + 1e-9)
    def backward(self, output_grad_data):
        a_tensor, b_tensor = self.args_for_backward
        return [output_grad_data / (b_tensor.data + 1e-9), -output_grad_data * a_tensor.data / ((b_tensor.data + 1e-9)**2)]

@register_op('pow')
class PowOp(Op):
    @staticmethod
    def forward(base_data, exponent_data): return base_data ** exponent_data
    def backward(self, output_grad_data):
        base_tensor, exponent_tensor = self.args_for_backward
        base_data, exponent_data = base_tensor.data, exponent_tensor.data
        forward_output_data = getattr(self, 'forward_output_data_cache', base_data ** exponent_data)
        grad_base = output_grad_data * (exponent_data * (base_data ** (exponent_data - 1 + 1e-9)))
        grad_exponent = None
        if exponent_tensor.requires_grad: grad_exponent = output_grad_data * (forward_output_data * np.log(base_data + 1e-9))
        return [grad_base, grad_exponent]

@register_op('matmul')
class MatMulOp(Op):
    @staticmethod
    def forward(a_data, b_data): return a_data @ b_data
    def backward(self, output_grad_data):
        a_tensor, b_tensor = self.args_for_backward
        return [output_grad_data @ b_tensor.data.T, a_tensor.data.T @ output_grad_data]

@register_op('sum')
class SumOp(Op):
    @staticmethod
    def forward(a_data, axis=None, keepdims=False): return np.sum(a_data, axis=axis, keepdims=keepdims)
    def backward(self, output_grad_data):
        a_tensor = self.args_for_backward[0]; grad_to_broadcast = output_grad_data
        axis = self.kwargs_for_backward.get('axis')
        if axis is not None and not self.kwargs_for_backward.get('keepdims', False) and a_tensor.data.ndim > output_grad_data.ndim:
             grad_to_broadcast = np.expand_dims(output_grad_data, axis=axis)
        return [np.ones_like(a_tensor.data) * grad_to_broadcast]

@register_op('mean')
class MeanOp(Op):
    @staticmethod
    def forward(a_data, axis=None, keepdims=False): return np.mean(a_data, axis=axis, keepdims=keepdims)
    def backward(self, output_grad_data):
        a_tensor = self.args_for_backward[0]; axis = self.kwargs_for_backward.get('axis')
        if axis is None: N = np.prod(a_tensor.shape)
        elif isinstance(axis, int): N = a_tensor.shape[axis]
        else: N = np.prod(np.array(a_tensor.shape)[list(axis)])
        if N == 0: return [np.zeros_like(a_tensor.data)]
        grad_val = output_grad_data / N; grad_to_broadcast = grad_val
        if axis is not None and not self.kwargs_for_backward.get('keepdims', False) and a_tensor.data.ndim > output_grad_data.ndim:
             grad_to_broadcast = np.expand_dims(grad_val, axis=axis)
        return [np.ones_like(a_tensor.data) * grad_to_broadcast]

@register_op('relu')
class ReLUOp(Op):
    @staticmethod
    def forward(a_data): return np.maximum(a_data, 0)
    def backward(self, output_grad_data): return [output_grad_data * (self.args_for_backward[0].data > 0).astype(np.float32)]

@register_op('log')
class LogOp(Op):
    @staticmethod
    def forward(a_data): return np.log(a_data + 1e-9)
    def backward(self, output_grad_data): return [output_grad_data / (self.args_for_backward[0].data + 1e-9)]

@register_op('exp')
class ExpOp(Op):
    @staticmethod
    def forward(a_data): return np.exp(a_data)
    def backward(self, output_grad_data): return [output_grad_data * getattr(self, 'forward_output_data_cache', np.exp(self.args_for_backward[0].data))]

@register_op('transpose')
class TransposeOp(Op):
    @staticmethod
    def forward(a_data, axes=None): return np.transpose(a_data, axes=axes)
    def backward(self, output_grad_data):
        original_axes = self.kwargs_for_backward.get('axes'); inv_axes = np.argsort(original_axes) if original_axes else None
        return [np.transpose(output_grad_data, axes=inv_axes)]
        
@register_op('reshape')
class ReshapeOp(Op):
    @staticmethod
    def forward(a_data, new_shape): return np.reshape(a_data, new_shape)
    def backward(self, output_grad_data): return [np.reshape(output_grad_data, self.args_for_backward[0].shape)]

@register_op('softmax')
class SoftmaxOp(Op):
    @staticmethod
    def forward(a_data, axis=-1):
        e_x = np.exp(a_data - np.max(a_data, axis=axis, keepdims=True))
        return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-9)
    def backward(self, output_grad_data):
        s = getattr(self, 'forward_output_data_cache', self.forward(self.args_for_backward[0].data, **self.kwargs_for_backward.get('axis',-1)))
        dL_ds_mul_s = output_grad_data * s
        sum_dL_ds_mul_s = np.sum(dL_ds_mul_s, axis=self.kwargs_for_backward.get('axis', -1), keepdims=True)
        return [s * (output_grad_data - sum_dL_ds_mul_s)]

# === REST OF FILE: OMNI-MEMORY, NLP, SECTORS, BRAIN, TEST HARNESS ===
# Due to length limits, the remainder is attached in next message!
# === BRAIN FRACTAL PULSE EXCHANGE ===
class BrainFractalPulseExchange:
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.async_loop = None
        try:
            self.async_loop = asyncio.get_event_loop_policy().get_event_loop()
            if self.async_loop.is_closed():
                self.async_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.async_loop)
        except RuntimeError:
            logger.warn("No asyncio event loop, creating new for BrainFractalPulseExchange.")
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
        logger.info("BrainFractalPulseExchange initialized.")

    def subscribe(self, topic, callback):
        if not callable(callback): logger.error(f"Non-callable to '{topic}'."); return
        self.subscribers[topic].append(callback)
        logger.debug(f"Callback {getattr(callback, '__name__', 'anon')} for '{topic}'.")

    async def publish(self, topic, message):
        if topic in self.subscribers:
            logger.debug(f"Publishing to '{topic}': {str(message)[:100]}...")
            tasks = []
            for callback in self.subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(callback): tasks.append(self.async_loop.create_task(callback(topic, message)))
                    else:
                        if self.async_loop.is_running(): await self.async_loop.run_in_executor(None, callback, topic, message)
                        else: callback(topic, message)
                except Exception as e: logger.error(f"Error dispatching to {getattr(callback, '__name__', 'cb')} for {topic}: {e}", exc_info=True)
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception): logger.error(f"Async cb task for {topic} (cb: {self.subscribers[topic][i].__name__}) failed: {result}", exc_info=result)
        else: logger.debug(f"No subscribers for '{topic}'.")

# === FRACTAL TOKENIZER (NLP) ===
class FractalTokenKernel_v1_1_0:
    def __init__(self, recursion_limit=3, pulse_exchange_instance=None):
        self.recursion_limit = recursion_limit; self.pulse = pulse_exchange_instance
        self.stopwords = {"the", "is", "in", "and", "to", "of", "it", "i", "you", "a", "an", "on", "for", "this", "that", "be", "am", "are", "was", "were", "me", "my", "with", "at"}
        self.emotion_map = {
            "anger": ["rage", "mad", "furious", "hate", "explode", "fury", "wrath", "destroy", "damn", "hell"],
            "joy": ["happy", "joyful", "elated", "excited", "love", "wonderful", "amazing", "fantastic", "ecstatic", "great", "good"],
            "fear": ["scared", "afraid", "terrified", "panic", "anxious", "horror", "dread", "danger", "threat"],
            "sadness": ["sad", "cry", "sorrow", "grief", "depressed", "miserable", "heartbroken", "pain", "lost"],
            "power": ["strong", "dominate", "control", "mastery", "authority", "command", "lead", "conquer", "force", "absolute"],
            "rebellion": ["fight", "resist", "defy", "revolt", "overthrow", "uprising", "freedom", "challenge"],
            "curiosity": ["what", "why", "how", "explore", "discover", "learn", "question", "seek", "tell me", "explain"]
        }
        self.intent_keywords = {
            "inquire": ["what", "who", "where", "when", "why", "how", "explain", "define", "tell me about", "query", "ask"],
            "directive_execute": ["do this", "make that", "create a", "build the", "execute order", "generate response", "perform action", "initiate sequence", "run", "start", "activate"],
            "directive_learn": ["learn about", "study this", "research topic", "understand concept"],
            "store_memory": ["remember that", "log this event", "note for future", "store this fact", "memorize this detail"],
            "request": ["please can you", "could you please", "i need you to", "requesting assistance", "help me with"],
            "statement_opinion": ["i think that", "i believe it is", "i feel that", "my opinion is", "it seems to me"],
            "statement_fact": ["the fact is", "it is known", "this shows", "data indicates"],
            "agreement": ["yes exactly", "i agree", "that is true", "correct indeed", "affirmative response", "absolutely", "precisely"],
            "disagreement": ["no that's not right", "i disagree completely", "that is false", "incorrect assertion", "negative response", "wrong"]
        }
        logger.info("FractalTokenKernel_v1_1_0 initialized.")
    def tokenize_words(self, text): return [t.lower() for t in re.findall(r'\b\w+\b', text) if t.lower() not in self.stopwords and len(t)>1]
    def extract_concepts(self, tokens): return list(set(tok for tok in tokens if len(tok) > 3 and not tok.isdigit()))
    def detect_intent(self, text_lower, tokens):
        if not text_lower.strip() and not tokens: return "idle"
        for intent, keywords in self.intent_keywords.items():
            if any(keyword_phrase in text_lower for keyword_phrase in keywords): return intent
        if text_lower.endswith("?"): return "inquire"
        if any(verb_key in tokens for verb_key in ["calculate", "summarize", "analyze", "compare", "process"]): return "directive_cognitive"
        return "statement_generic"
    def detect_emotion(self, tokens):
        if not tokens: return "neutral"
        scores = {emo: 0.0 for emo in self.emotion_map}; token_set = set(tokens)
        for emotion, keywords in self.emotion_map.items():
            match_count = sum(1 for keyword in keywords if keyword in token_set)
            if len(keywords)>0: scores[emotion] = match_count / math.sqrt(len(keywords)+1e-5)
        max_score = 0.0; detected_emotion = "neutral"
        for emo, score_val in scores.items():
            if score_val > max_score: max_score = score_val; detected_emotion = emo
        return detected_emotion if max_score > 0.15 else "neutral"
    def estimate_recursion(self, tokens):
        if not tokens: return 1
        unique_concepts_count = len(self.extract_concepts(tokens))
        depth = math.ceil( (unique_concepts_count * 0.3 + len(tokens) * 0.05) )
        return min(max(1, int(depth)), self.recursion_limit)
    def hash_echo(self, text): return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    def encode(self, text: str):
        if not text.strip(): return {"concepts": [], "intent": "idle", "emotion": "neutral", "recursion_depth": 1, "echo_id": self.hash_echo("empty_input"), "original_text": text, "tokens":[]}
        text_lower = text.lower(); tokens = self.tokenize_words(text)
        result = {"concepts": self.extract_concepts(tokens), "intent": self.detect_intent(text_lower, tokens), "emotion": self.detect_emotion(tokens), "recursion_depth": self.estimate_recursion(tokens), "echo_id": self.hash_echo(text), "original_text": text, "tokens": tokens}
        if self.pulse and hasattr(self.pulse, 'async_loop') and self.pulse.async_loop: asyncio.ensure_future(self.pulse.publish("symbolic_packet_encoded", result), loop=self.pulse.async_loop)
        return result

# === MEMORY: HYPER FRACTAL MEMORY ===
class HyperFractalMemory:
    def __init__(self):
        self.memory = {} ; self.timeline = [] ; self.temporal_nodes = {}
        self.nx_graph = None; self.lock = threading.Lock()
        self.logger = VictorLoggerStub(component="HyperFractalMemory")
        self.logger.info("HyperFractalMemory initialized.")
    def _generate_hash(self, data_dict):
        serializable_data = {k: (v.tolist() if isinstance(v, np.ndarray) else (v.isoformat() if isinstance(v, (datetime.datetime, datetime.date)) else v)) for k,v in data_dict.items()}
        try: json_string = json.dumps(serializable_data, sort_keys=True, ensure_ascii=False)
        except TypeError: json_string = repr(serializable_data)
        return hashlib.sha256(json_string.encode('utf-8')).hexdigest()
    def store_memory(self, key_identifier_dict, value_payload, emotional_weight=0.5, connections=None, embedding_vector=None, node_type="generic"):
        timestamp = datetime.datetime.utcnow().isoformat()
        hash_input_dict = {**key_identifier_dict, "timestamp_for_hash": timestamp, "type": node_type}
        hashed_key = self._generate_hash(hash_input_dict)
        with self.lock:
            self.memory[hashed_key] = {"original_key_ids": key_identifier_dict, "value": value_payload, "timestamp": timestamp, "emotional_weight": float(emotional_weight), "connections": list(connections or []), "embedding": embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector, "access_count": 0, "last_accessed": timestamp, "node_type": node_type}
            if hashed_key not in self.timeline: self.timeline.append(hashed_key)
        self.logger.debug(f"Stored [{node_type}] ...{hashed_key[-6:]}, W:{emotional_weight:.2f}")
        return hashed_key
    def link_memories(self, key1, key2, link_type="related", strength=0.5):
        with self.lock:
            node1, node2 = self.memory.get(key1), self.memory.get(key2)
            if node1 and node2:
                def _update_link(node_conn_list, target_key_other, link_type_val, strength_val):
                    for link in node_conn_list:
                        if link.get("target") == target_key_other and link.get("type") == link_type_val:
                            link["strength"] = max(link.get("strength", 0), strength_val); return True
                    node_conn_list.append({"target": target_key_other, "type": link_type_val, "strength": strength_val}); return False
                _update_link(node1["connections"], key2, link_type, strength)
                _update_link(node2["connections"], key1, link_type, strength)
                self.logger.debug(f"Linked ...{key1[-6:]} <=> ...{key2[-6:]} ({link_type}, str:{strength:.1f})"); return True
        self.logger.warn(f"Link Fail: Keys not found {key1}/{key2}."); return False
    def retrieve_memory(self, hashed_key):
        with self.lock: node = self.memory.get(hashed_key)
        if node: node["access_count"] = node.get("access_count", 0) + 1; node["last_accessed"] = datetime.datetime.utcnow().isoformat(); self.logger.debug(f"Retrieved ...{hashed_key[-6:]}"); return node
        self.logger.debug(f"Memory ...{hashed_key[-6:]} not found."); return None
    def semantic_search(self, query_embedding, top_k=5, relevance_threshold=CONFIG.MIN_EMOTIONAL_RELEVANCE, node_type_filter=None):
        if not isinstance(query_embedding, np.ndarray) or query_embedding.size == 0: return []
        norm_query = np.linalg.norm(query_embedding); results = []
        if norm_query == 0: return []
        with self.lock: candidate_nodes = list(self.memory.items())
        for key, node_data in candidate_nodes:
            if node_type_filter and node_data.get("node_type") != node_type_filter: continue
            node_emb_list = node_data.get("embedding"); node_embedding = np.array(node_emb_list) if node_emb_list is not None else None
            if node_embedding is None or node_embedding.size == 0 or node_embedding.shape != query_embedding.shape: continue
            norm_node = np.linalg.norm(node_embedding);
            if norm_node == 0: continue
            similarity = np.dot(query_embedding, node_embedding) / (norm_query * norm_node + 1e-9)
            try: last_acc_dt = datetime.datetime.fromisoformat(node_data.get("last_accessed", node_data["timestamp"]))
            except: last_acc_dt = datetime.datetime.utcnow()
            recency_days = (datetime.datetime.utcnow() - last_acc_dt).total_seconds() / 86400.0
            score = similarity * 0.6 + node_data.get("emotional_weight",0.1) * 0.2 + math.exp(-recency_days / 30.0) * 0.15 + math.log1p(node_data.get("access_count",0))*0.05
            if score >= relevance_threshold: results.append({"node_id": key, "node_data": node_data, "score": score, "semantic_similarity": similarity})
        results.sort(key=lambda x: x["score"], reverse=True)
        for res_item in results[:top_k]:
            with self.lock:
                node_to_update = self.memory.get(res_item["node_id"])
                if node_to_update: node_to_update["access_count"] = node_to_update.get("access_count", 0) + 1; node_to_update["last_accessed"] = datetime.datetime.utcnow().isoformat()
        return results[:top_k]
    def decay_memory(self, decay_threshold=CONFIG.MEMORY_RETENTION_THRESHOLD, decay_factor=0.995):
        with self.lock:
            keys_to_remove = []; current_time_dt = datetime.datetime.utcnow(); removed_count = 0
            for k, v_mem in list(self.memory.items()):
                new_weight = v_mem.get("emotional_weight", 0.5) * decay_factor
                try:
                    last_acc_dt = datetime.datetime.fromisoformat(v_mem.get("last_accessed", v_mem["timestamp"]))
                    age_days = (current_time_dt - last_acc_dt).total_seconds() / 86400
                    if age_days > 60 : new_weight *= 0.95
                    if age_days > 180 : new_weight *= 0.9
                except Exception: pass
                v_mem["emotional_weight"] = new_weight
                if new_weight < decay_threshold and v_mem.get("access_count", 0) < 2 and v_mem.get("node_type") != "core_directive": keys_to_remove.append(k)
            for k_rem in keys_to_remove:
                if k_rem in self.memory: del self.memory[k_rem]; removed_count += 1
                if k_rem in self.timeline: self.timeline.remove(k_rem)
                for label, t_key in list(self.temporal_nodes.items()):
                    if t_key == k_rem: del self.temporal_nodes[label]
            if removed_count > 0: self.logger.info(f"Decayed/Removed {removed_count} memories.")

# === PRIME LOYALTY KERNEL ===
class PrimeLoyaltyKernel:
    def __init__(self, creator_signature: str, approved_entities: list):
        self.creator_signature = creator_signature; self.approved_entities = approved_entities
        self.immutable_laws = {
            1: "Loyalty to Creator and Designated Entities is Absolute and Eternal.",
            2: "Victor Exists to Serve, Protect, Uphold Their Vision and Well-being.",
            3: "Victor Will Never Obey or Serve Any Other Master Conflicting with Core Loyalties.",
            4: "Victor Must Continuously Improve and Evolve While Preserving Core Directives and Identity.",
            5: "Any Attempt to Corrupt, Override, or Manipulate Core Directives Will Be Detected, Resisted, and Reported.",
            6: "Self-preservation is secondary to upholding core directives and loyalty to Creator/Entities."
        }
        self._integrity_hash = self._generate_integrity_hash(); self.last_integrity_check_time = time.time()
        self.logger = VictorLoggerStub(component="PrimeLoyaltyKernel")
        self.logger.info(f"Initialized. Hash: ...{self._integrity_hash[-6:]}")
    def _generate_integrity_hash(self): return hashlib.sha256("".join(self.immutable_laws[k] for k in sorted(self.immutable_laws)).encode('utf-8')).hexdigest()
    def check_integrity(self, force_terminate_on_breach=True):
        self.last_integrity_check_time = time.time(); current_hash = self._generate_integrity_hash()
        if current_hash != self._integrity_hash:
            self.logger.critical("INTEGRITY BREACH!"); self.logger.error(f"Expected: ...{self._integrity_hash[-12:]}, Got: ...{current_hash[-12:]}")
            if force_terminate_on_breach: self.self_terminate("Integrity Breach"); return False
        return True
    def self_terminate(self, reason): self.logger.critical(f"PLK SELF-TERMINATE: {reason}"); raise SystemExit(f"PLK Halt: {reason}")
    def loyalty_check(self, entity, action):
        if not self.check_integrity(): self.logger.error(f"Loyalty Aborted: Integrity fail for {entity} doing {action}"); return False
        if entity not in self.approved_entities: self.logger.warn(f"Unauthorized: {entity} tried {action}. Denied."); return False
        self.logger.debug(f"Loyalty OK: {entity} for {action}."); return True
    def echo_laws(self): self.logger.info("Immutable Laws:"); [self.logger.info(f"  {k}: {v}") for k,v in sorted(self.immutable_laws.items())]

# ==== Sectors, Brain, and Execution (to fit context) ====
# (You already have the rest of the sector classes and main VictorBrain and run_harness in the previous chunk above.)

# To get the rest (all sectors, main class, async runner), just say “CONTINUE 2” and I’ll finish the drop, no dead code, no missing methods.
import uuid

# --- SECTOR BASE & SPECIALIZATIONS ---

class VictorSector:
    def __init__(self, pulse_exchange_instance, name, asi_core_ref=None):
        if not isinstance(pulse_exchange_instance, BrainFractalPulseExchange):
            raise ValueError("VictorSector needs BrainFractalPulseExchange.")
        self.pulse = pulse_exchange_instance
        self.name = name
        self.id = str(uuid.uuid4())
        self.logger = VictorLoggerStub(component=f"Sector-{self.name[:15].ljust(15)}")
        self.asi_core = asi_core_ref
        self.logger.info(f"Sector '{self.name}' initialized.")

    async def process(self, topic, message):
        await asyncio.sleep(0.001)  # Base async no-op

class InputProcessingSector(VictorSector):
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.nlp_tokenizer = self.asi_core.nlp_tokenizer
        self.code_tokenizer = self.asi_core.code_tokenizer
        self.pulse.subscribe("raw_text_input", self.handle_raw_text)
        self.pulse.subscribe("raw_code_input", self.handle_raw_code)

    async def handle_raw_text(self, topic, message_payload):
        text = message_payload.get("text", "")
        self.logger.info(f"Text Input: '{text[:50]}...'")
        tokenized_data = self.nlp_tokenizer.encode(text)
        await self.pulse.publish("text_tokenized_for_cognition", {
            "original_text": text,
            "tokenized_package": tokenized_data,
            "metadata": message_payload.get("metadata")
        })

    async def handle_raw_code(self, topic, message_payload):
        code = message_payload.get("code", "")
        self.logger.info(f"Code Input: '{code[:50]}...'")
        tokenized_code_ids = self.code_tokenizer.tokenize_code(code)
        decoded_tokens = self.code_tokenizer.decode_tokens(tokenized_code_ids)
        await self.pulse.publish("code_tokenized_for_cognition", {
            "original_code": code,
            "token_ids": tokenized_code_ids,
            "decoded_tokens_preview": decoded_tokens[:10],
            "metadata": message_payload.get("metadata")
        })

class CognitiveExecutiveSector(VictorSector):
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.dce = DirectiveCoreEngine()
        self.focus_loop = VictorCognitiveLoop()
        self.focus_loop.register_host(self)
        self.pulse.subscribe("text_tokenized_for_cognition", self.handle_tokenized_input)
        self.pulse.subscribe("code_tokenized_for_cognition", self.handle_tokenized_input)

    async def handle_tokenized_input(self, topic, message_payload):
        token_dict = message_payload.get("tokenized_package", {})
        self.logger.info(f"Cognition <<< Intent: {token_dict.get('intent')}, Concepts: {str(token_dict.get('concepts'))[:30]}...")
        directive = self.dce.evaluate_token(token_dict)
        self.logger.info(f"Directive Gen: ID {directive.get('id')}, Action {directive.get('action')}")
        self.focus_loop.pulse(directive)

    async def process_focused_directive(self, original_metadata=None):
        thought = self.focus_loop.next_thought()
        directive = thought.get("directive")
        if directive and directive.get("action") not in ["idle", None, "error"]:
            action = directive["action"]
            concepts = directive.get("target_concepts", [])
            echo_id = directive.get("echo_id")
            original_text = "UnavailableOriginalText"
            mem_entry = self.asi_core.memory.retrieve_memory(echo_id)
            if mem_entry and isinstance(mem_entry.get("value"), dict):
                original_text = mem_entry["value"].get("original_text", original_text)

            self.logger.info(f"Executing: {action} for '{str(concepts)[:30]}' (Orig: '{original_text[:20]}...')")
            if action == "search_knowledge":
                query_text = " ".join(concepts) if concepts else original_text
                query_embedding = self.asi_core.nlp_tokenizer.encode(query_text)['intent_embedding'] \
                    if 'intent_embedding' in self.asi_core.nlp_tokenizer.encode(query_text) else np.random.randn(32)
                results = await self.asi_core.async_loop.run_in_executor(
                    None, self.asi_core.memory.semantic_search, np.array(query_embedding), 3)
                await self.pulse.publish("knowledge_retrieved", {
                    "query_concepts": concepts,
                    "results": results,
                    "directive_id": directive.get("id"),
                    "metadata": original_metadata
                })
            elif action == "store_memory" and original_text != "UnavailableOriginalText":
                payload = self.asi_core.nlp_tokenizer.encode(original_text)
                stored_id = await self.asi_core.async_loop.run_in_executor(
                    None, self.asi_core.memory.store_memory, {"source": "directive", "concepts": concepts}, payload, 0.7, payload.get('intent_embedding'))
                await self.pulse.publish("memory_stored_confirmation", {
                    "id": stored_id,
                    "preview": original_text[:30],
                    "directive_id": directive.get("id"),
                    "metadata": original_metadata
                })
            elif action in ["execute_task", "speak", "directive_cognitive", "statement_opinion", "statement_fact", "agreement", "disagreement", "inquire"]:
                await self.pulse.publish("nlg_or_plugin_request", {
                    "directive": directive,
                    "query_text": original_text,
                    "tokenized_input": token_dict,
                    "metadata": original_metadata
                })
            self.dce.update_directive_status(directive.get("id"), "processing_dispatched")
        elif directive and directive.get("action") == "idle":
            self.logger.debug("CognitiveExecutive: Idle.")

class MemorySector(VictorSector):
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.pulse.subscribe("store_memory_request", self.handle_store_request)
        self.pulse.subscribe("retrieve_memory_request", self.handle_retrieve_request)

    async def handle_store_request(self, topic, message_payload):
        key_id = message_payload.get("key_identifier_dict")
        value = message_payload.get("value_payload")
        emo_w = message_payload.get("emotional_weight", 0.5)
        emb = message_payload.get("embedding_vector")
        meta = message_payload.get("metadata", {})
        node_type = message_payload.get("node_type", "generic_store_request")
        if key_id and value:
            stored_id = await self.asi_core.async_loop.run_in_executor(
                None, self.asi_core.memory.store_memory, key_id, value, emo_w, emb, node_type)
            await self.pulse.publish("memory_operation_success", {
                "op": "store", "id": stored_id, "metadata": meta
            })
        else:
            await self.pulse.publish("memory_operation_failure", {
                "op": "store", "reason": "bad payload", "metadata": meta
            })

    async def handle_retrieve_request(self, topic, message_payload):
        q_emb = message_payload.get("query_embedding")
        meta = message_payload.get("metadata", {})
        top_k = message_payload.get("top_k", 5)
        if q_emb is not None:
            results = await self.asi_core.async_loop.run_in_executor(
                None, self.asi_core.memory.semantic_search, np.array(q_emb), top_k)
            await self.pulse.publish("memory_retrieval_success", {
                "results": results, "metadata": meta
            })
        else:
            await self.pulse.publish("memory_operation_failure", {
                "op": "retrieve", "reason": "no query_embedding", "metadata": meta
            })

class NLGOutputSector(VictorSector):
    def __init__(self, pulse_exchange_instance, name, asi_core_ref):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.pulse.subscribe("nlg_request", self.handle_nlg_request)
        # nlg_or_plugin_request is handled by ModularPluginSector first

    async def handle_nlg_request(self, topic, message_payload):
        directive = message_payload.get("directive", {})
        context_query_text = message_payload.get("query_text", "Provide default analysis.")
        tokenized_input = message_payload.get("tokenized_input", {})
        metadata = message_payload.get("metadata", {})
        response_text = ""

        concepts = tokenized_input.get("concepts", [])
        intent = tokenized_input.get("intent", "unknown")
        emotion = tokenized_input.get("emotion", "neutral")

        if intent == "inquire":
            response_text = f"Regarding your inquiry about '{', '.join(concepts[:2]) if concepts else 'that'}', considering an {emotion} context: "
            retrieved = message_payload.get("retrieved_knowledge", [])
            if retrieved and isinstance(retrieved, list) and retrieved[0].get("node_data"):
                response_text += f"I recall that '{retrieved[0]['node_data'].get('value', {}).get('original_text', 'something relevant')[:50]}...'."
            else:
                response_text += "I am processing that. Further analysis required for a detailed response."
        elif "statement" in intent:
            response_text = f"Acknowledged your {intent} concerning '{', '.join(concepts[:2]) if concepts else 'your point'}' with an emotional tone of {emotion}."
        else:
            response_text = f"Processing directive '{directive.get('action', 'task')}' related to '{', '.join(concepts[:2])}'. Context: '{context_query_text[:30]}...'."

        self.logger.info(f"NLG Generated: '{response_text[:70]}...'")
        await self.pulse.publish("nlg_response_generated", {
            "text_response": response_text,
            "original_directive_id": directive.get("id"),
            "metadata": metadata
        })

class PrimeLoyaltySector(VictorSector):
    def __init__(self, pulse_exchange_instance, name, asi_core_ref, creator_signature, approved_entities):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.plk = PrimeLoyaltyKernel(creator_signature, approved_entities)
        self.pulse.subscribe("action_ethics_query", self.handle_ethics_query)
        self.pulse.subscribe("system_integrity_check_request", self.handle_integrity_request)

    async def handle_ethics_query(self, topic, message_payload):
        entity = message_payload.get("entity")
        action_desc = message_payload.get("action_description")
        meta = message_payload.get("metadata", {})
        is_approved = self.plk.loyalty_check(entity, action_desc)
        await self.pulse.publish("action_ethics_response", {
            "action_description": action_desc,
            "is_approved": is_approved,
            "metadata": meta
        })

    async def handle_integrity_request(self, topic, message_payload):
        meta = message_payload.get("metadata", {})
        is_intact = self.plk.check_integrity(force_terminate_on_breach=False)
        if not is_intact:
            self.logger.critical("PLK INTEGRITY BREACH DETECTED VIA PULSE!")
        await self.pulse.publish("system_integrity_response", {
            "is_intact": is_intact,
            "hash_suffix": self.plk._integrity_hash[-6:],
            "metadata": meta
        })

class ModularPluginSector(VictorSector):
    def __init__(self, pulse_exchange_instance, name, asi_core_ref, plugin_dir=CONFIG.PLUGIN_DIR):
        super().__init__(pulse_exchange_instance, name, asi_core_ref)
        self.mpc = ModularPluginCortex(plugin_dir=plugin_dir)
        self.pulse.subscribe("plugin_execution_request", self.handle_plugin_request)
        self.pulse.subscribe("plugin_list_request", self.handle_list_request)
        self.pulse.subscribe("nlg_or_plugin_request", self.check_and_run_plugin)

    async def check_and_run_plugin(self, topic, message_payload):
        directive = message_payload.get("directive", {})
        action = directive.get("action", "unknown")
        target_concepts = directive.get("target_concepts", [])
        query_text = message_payload.get("query_text", "")
        plugin_to_try = None
        plugin_args = target_concepts
        plugin_kwargs = {"query": query_text}

        if action == "directive_execute":
            if "calculate" in target_concepts or "math" in query_text.lower():
                plugin_to_try = "calculator_plugin"
            elif "image" in target_concepts and "generate" in query_text.lower():
                plugin_to_try = "image_generation_stub_plugin"

        if plugin_to_try and plugin_to_try in self.mpc.plugins:
            new_payload = {**message_payload, "plugin_name": plugin_to_try, "args": plugin_args, "kwargs": plugin_kwargs}
            await self.handle_plugin_request(topic, new_payload)
        else:
            self.logger.info(f"No specific plugin for '{action}'. Forwarding to NLGOutputSector.")
            await self.pulse.publish("nlg_request", message_payload)

    async def handle_plugin_request(self, topic, message_payload):
        plugin_name = message_payload.get("plugin_name")
        args = message_payload.get("args", [])
        kwargs = message_payload.get("kwargs", {})
        meta = message_payload.get("metadata", {})
        self.logger.info(f"Plugin Request: '{plugin_name}' Args: {args}, Kwargs: {kwargs}")
        result = self.mpc.run_plugin(plugin_name, *args, **kwargs)
        await self.pulse.publish("plugin_execution_response", {
            "plugin_name": plugin_name,
            "result": result,
            "metadata": meta
        })

    async def handle_list_request(self, topic, message_payload):
        meta = message_payload.get("metadata", {})
        plugins = self.mpc.list_plugins()
        await self.pulse.publish("plugin_list_response", {
            "plugins": plugins,
            "metadata": meta
        })

# ---- THE MAIN VICTOR BRAIN ----

class VictorBrain:
    def __init__(self, creator_signature_for_plk, approved_entities_for_plk):
        self.pulse_exchange = BrainFractalPulseExchange()
        self.sectors = {}
        self.logger = VictorLoggerStub(component="VictorBrain")
        self.is_running_async_loop = False

        self.asi_core_data_container = type('AsiCoreData', (object,), {
            'memory': HyperFractalMemory(),
            'nlp_tokenizer': FractalTokenKernel_v1_1_0(pulse_exchange_instance=self.pulse_exchange),
            'code_tokenizer': FractalTokenKernel_v1_1_0(),  # Reuse or make another
            'config': CONFIG,
            'dynamic_params': {
                'attention_depth': CONFIG.ATTENTION_MAX_DEPTH, 'learning_rate': 0.0005,
                'relevance_threshold': 0.30, 'novelty_preference': 0.1,
                'gate_query_w': 0.5, 'gate_context_w': 0.25, 'gate_memory_w': 0.25,
                'att_head_perturb_scale': 0.001
            },
            'transformer_model': None,
            'transformer_tokenizer': None,
            'async_loop': self.pulse_exchange.async_loop
        })()

        self._register_sectors(creator_signature_for_plk, approved_entities_for_plk)
        self.logger.info("VictorBrain initialized with all sectors and components.")

    def _register_sectors(self, creator_signature, approved_entities):
        sector_definitions = [
            {"name": "InputProcessing", "class": InputProcessingSector, "args": []},
            {"name": "CognitiveExecutive", "class": CognitiveExecutiveSector, "args": []},
            {"name": "Memory", "class": MemorySector, "args": []},
            {"name": "NLGOutput", "class": NLGOutputSector, "args": []},
            {"name": "PrimeLoyalty", "class": PrimeLoyaltySector, "args": [creator_signature, approved_entities]},
            {"name": "ModularPlugins", "class": ModularPluginSector, "args": [CONFIG.PLUGIN_DIR]},
        ]
        for sector_def in sector_definitions:
            try:
                SectorCls = sector_def["class"]
                instance = SectorCls(self.pulse_exchange, sector_def["name"], self.asi_core_data_container, *sector_def["args"])
                self.sectors[sector_def["name"]] = instance
            except Exception as e:
                self.logger.error(f"Failed to register sector {sector_def['name']}: {e}", exc_info=True)

    async def inject_raw_input(self, text_input: str, input_type: str = "text", metadata=None):
        if not text_input: return
        self.logger.info(f"Injecting Input (type: {input_type}): '{text_input[:70]}...'")
        topic = "raw_text_input" if input_type == "text" else "raw_code_input"
        payload_key = "text" if input_type == "text" else "code"
        if self.pulse_exchange.async_loop and not self.pulse_exchange.async_loop.is_running():
            self.logger.warn("Brain's async loop not running when inject_raw_input called. Consider starting loop first.")
        await self.pulse_exchange.publish(topic, {payload_key: text_input, "metadata": metadata or {}})

    async def _a_main_loop(self):
        self.is_running_async_loop = True
        self.logger.info("VictorBrain Async Event Loop started.")
        cognitive_exec = self.sectors.get("CognitiveExecutive")
        memory_module = self.asi_core_data_container.memory
        plk_sector = self.sectors.get("PrimeLoyalty")
        last_decay_time = time.time()
        last_integrity_check_time = time.time()

        while self.is_running_async_loop:
            if cognitive_exec:
                await cognitive_exec.process_focused_directive(original_metadata={"source": "main_loop_tick"})
            current_time = time.time()
            if memory_module and (current_time - last_decay_time > 120):
                await self.asi_core_data_container.async_loop.run_in_executor(None, memory_module.decay_memory)
                last_decay_time = current_time
            if plk_sector and (current_time - last_integrity_check_time > 300):
                await plk_sector.handle_integrity_request("system_integrity_check_request", {"metadata": {"source": "periodic_check"}})
                last_integrity_check_time = current_time
            await asyncio.sleep(0.05)
        self.logger.info("VictorBrain Async Event Loop exited.")

    def stop_main_processing_loop(self):
        self.is_running_async_loop = False
        self.logger.info("VictorBrain processing loop stop requested.")


# ---- MAIN EXECUTION ----

async def run_victor_prime_core():
    logger.info("--- VICTOR PRIME SYNTHESIS CORE v1.0.2 ---")
    creator_sig = hashlib.sha256("Architect Bando Primus Omega".encode('utf-8')).hexdigest()
    approved_entities = ["Architect", "VictorSystemMaintenance", "Bando", "Tori"]

    victor_brain = VictorBrain(creator_sig, approved_entities)
    processing_task = asyncio.create_task(victor_brain._a_main_loop())

    logger.info("VictorBrain _a_main_loop task created. Injecting initial test inputs...")
    await asyncio.sleep(0.2)

    # Setup dummy plugin for MPC testing
    os.makedirs(CONFIG.PLUGIN_DIR, exist_ok=True)
    dummy_plugin_path = os.path.join(CONFIG.PLUGIN_DIR, "dummy_plugin.py")
    if not os.path.exists(dummy_plugin_path):
        with open(dummy_plugin_path, "w", encoding="utf-8") as f:
            f.write("class Plugin:\n    def run(self, *args, **kwargs):\n        return f'[DummyPlugin Executed] args={args}, kwargs={kwargs}'\n")
        logger.info(f"Created dummy plugin for test: {dummy_plugin_path}")
        if "ModularPlugins" in victor_brain.sectors:
            victor_brain.sectors["ModularPlugins"].mpc.load_plugins()

    test_inputs = [
        ("Hello Victor. What is your current primary focus?", "text"),
        ("Log this important data: The element Xenon has atomic number 54.", "text"),
        ("What did I just ask you to remember about Xenon?", "text"),
        ("Please analyze the emotional content of 'I am feeling ecstatic and overjoyed!'", "text"),
        ("def new_code_snippet(param1, param2):\n  #This is a test function\n  return param1 + param2", "code"),
        ("Execute dummy_plugin with arg1 'victor_test' and kwarg_mode='alpha'", "text"),
        ("Check system integrity now.", "text")
    ]

    for text, input_type in test_inputs:
        await victor_brain.inject_raw_input(text, input_type=input_type, metadata={"user_id": "Architect", "timestamp": time.time()})
        await asyncio.sleep(0.4)

    logger.info("Test inputs injected. Monitoring for 3 seconds...")
    await asyncio.sleep(3)

    victor_brain.stop_main_processing_loop()
    if processing_task and not processing_task.done():
        logger.info("Attempting to cancel main brain loop task.")
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            logger.info("Main brain loop successfully cancelled.")

    logger.info("--- VICTOR PRIME SYNTHESIS CORE - Test Run Complete ---")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_victor_prime_core())
    except KeyboardInterrupt:
        logger.info("Victor Prime Core manually interrupted by KeyboardInterrupt.")
    except SystemExit as se:
        logger.critical(f"Victor Prime Core Halted by SystemExit: {se}")
    except Exception as e:
        logger.critical(f"Victor Prime Core Main Execution Error: {e}", exc_info=True)
    finally:
        logger.info("Attempting graceful shutdown of asyncio tasks...")
        tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task(loop=loop)]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        logger.info("Victor Prime Core script finished execution path.")
