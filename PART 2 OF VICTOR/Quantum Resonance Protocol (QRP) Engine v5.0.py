#!/usr/bin/env python3
# Quantum Resonance Protocol (QRP) Engine v5.0
# Author: Upgrade Overlord
# Purpose: Secure, fault-tolerant quantum state synchronization for distributed AI nodes

import asyncio
import hashlib
import hmac
import json
import logging
import os
import socket
import struct
import sys
import time
import zlib
from collections import deque
from logging.handlers import RotatingFileHandler
from secrets import token_bytes

import numpy as np

# ---------------- CONFIG ----------------
STATE_DIM = 256
BASE_SYNC_INTERVAL = 0.05
DRIFT_THRESHOLD = 0.03
ENTANGLEMENT_ALPHA = 0.75
UDP_PORT = 5005
SECRET_KEY = token_bytes(32)  # Shared HMAC key for message signing
ROLLBACK_HISTORY = 10  # Number of states to keep for rollback
MAX_PACKET_SIZE = 65507  # UDP max safe size
HEARTBEAT_TIMEOUT = 2.0  # Seconds before peer considered unhealthy

# Logging setup
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("QRP")
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("logs/qrp.log", maxBytes=5_000_000, backupCount=5)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------- UTILS ----------------
def sign_message(payload: bytes) -> str:
    return hmac.new(SECRET_KEY, payload, hashlib.sha256).hexdigest()

def verify_signature(payload: bytes, signature: str) -> bool:
    return hmac.compare_digest(sign_message(payload), signature)

def compress_state(state_vector: np.ndarray) -> bytes:
    return zlib.compress(state_vector.astype(np.float32).tobytes())

def decompress_state(data: bytes) -> np.ndarray:
    raw = zlib.decompress(data)
    return np.frombuffer(raw, dtype=np.float32)

# ---------------- QRP CORE ----------------
class QuantumState:
    def __init__(self, dim=STATE_DIM):
        self.state_vector = self._init_random(dim)
        self.lock = asyncio.Lock()
        self.rollback_buffer = deque(maxlen=ROLLBACK_HISTORY)

    def _init_random(self, dim):
        vec = np.random.rand(dim)
        return vec / np.linalg.norm(vec)

    async def collapse(self, observation_vector, alpha=ENTANGLEMENT_ALPHA):
        async with self.lock:
            self.rollback_buffer.append(self.state_vector.copy())
            obs = observation_vector / np.linalg.norm(observation_vector)
            new_state = alpha * obs + (1 - alpha) * self.state_vector
            self.state_vector = new_state / np.linalg.norm(new_state)

    def entropy(self):
        p = np.abs(self.state_vector)
        p /= np.sum(p)
        return -np.sum(p * np.log(p + 1e-12))

    def hash_state(self):
        return hashlib.sha256(self.state_vector.tobytes()).hexdigest()

    def rollback(self):
        if self.rollback_buffer:
            self.state_vector = self.rollback_buffer.pop()
            logger.warning("[SELF-HEAL] Rolled back to previous state after anomaly")

# ---------------- NODE ----------------
class QRPNode:
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers
        self.quantum_state = QuantumState()
        self.last_heartbeat = {peer: time.time() for peer in peers}

    async def start(self):
        loop = asyncio.get_event_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: QRPProtocol(self),
            local_addr=("0.0.0.0", UDP_PORT)
        )
        asyncio.create_task(self.sync_loop())
        asyncio.create_task(self.health_monitor())
        logger.info(f"[INIT] Node {self.node_id} online on UDP {UDP_PORT}")

    async def handle_message(self, data, addr):
        try:
            payload = json.loads(data.decode())
            sig = payload.get("sig")
            compressed_state = bytes.fromhex(payload["state"])
            core_data = json.dumps({
                "id": payload["id"], "state": payload["state"]
            }).encode()

            if not verify_signature(core_data, sig):
                logger.error(f"[SECURITY] Invalid signature from {addr}")
                return

            peer_state = decompress_state(compressed_state)
            local_entropy = self.quantum_state.entropy()
            peer_entropy = -np.sum(np.abs(peer_state) * np.log(np.abs(peer_state) + 1e-12))

            drift = abs(local_entropy - peer_entropy)
            if drift > DRIFT_THRESHOLD:
                alpha = min(1.0, ENTANGLEMENT_ALPHA + drift)
                await self.quantum_state.collapse(peer_state, alpha)
                logger.info(f"[SYNC] Collapse applied from peer {payload['id']} (drift={drift:.4f})")

            self.last_heartbeat[addr] = time.time()

        except Exception as e:
            logger.exception(f"[ERROR] Failed to process message: {e}")
            self.quantum_state.rollback()

    async def sync_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        while True:
            state_bytes = compress_state(self.quantum_state.state_vector)
            state_hex = state_bytes.hex()
            core_data = json.dumps({"id": self.node_id, "state": state_hex}).encode()
            sig = sign_message(core_data)
            packet = json.dumps({"id": self.node_id, "state": state_hex, "sig": sig}).encode()

            for peer in self.peers:
                await asyncio.get_event_loop().sock_sendall(sock, packet)

            await asyncio.sleep(BASE_SYNC_INTERVAL)

    async def health_monitor(self):
        while True:
            now = time.time()
            for peer, last_seen in self.last_heartbeat.items():
                if now - last_seen > HEARTBEAT_TIMEOUT:
                    logger.warning(f"[HEALTH] Peer {peer} unresponsive > {HEARTBEAT_TIMEOUT}s")
            await asyncio.sleep(HEARTBEAT_TIMEOUT)

class QRPProtocol(asyncio.DatagramProtocol):
    def __init__(self, node):
        self.node = node

    def datagram_received(self, data, addr):
        asyncio.create_task(self.node.handle_message(data, addr))

# ---------------- DEMO ----------------
if __name__ == "__main__":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    peers = [("127.0.0.1", UDP_PORT)]
    node = QRPNode(node_id="Node_A", peers=peers)
    asyncio.run(node.start())
