# FILE: super_tog/omni_gpu_runtime_shell.py
# VERSION: v1.0.0-TOG-GODCORE
# NAME: OmniGPU Runtime Shell
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Runtime shell & Python binding for the Super Topological Omniforming GPU (STOG).
#          Provides a ruthless, zero‑friction interface for infinite‑scale computation on paradigm‑shattering hardware.
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""
THIS IS NOT A TOY.
Hook it to a real STOG driver (libstog.so / stog.dll) or get the hell out of the way.
Any missing native symbol will detonate with a hard RuntimeError – no silent fallbacks, no simulations.
"""

import os
import ctypes
import threading
import time
import uuid
from typing import Any, Dict, List, Tuple, Union

# -----------------------------
# Low‑Level STOG Binding Layer
# -----------------------------

class _STOGNative:
    """Thin ctypes wrapper around the native STOG C driver."""

    def __init__(self, lib_path: str):
        self._lib = ctypes.cdll.LoadLibrary(lib_path)
        # Crash immediately if mandatory symbols are missing.
        for sym in ("stog_init", "stog_shutdown", "stog_create_topology", "stog_destroy_topology",
                    "stog_alloc_tensor", "stog_free_tensor", "stog_launch_kernel"):
            if not hasattr(self._lib, sym):
                raise RuntimeError(f"STOG native driver missing symbol: {sym}")
        # Init the device (returns 0 on success).
        if self._lib.stog_init() != 0:
            raise RuntimeError("Failed to initialize STOG hardware – check device and privileges.")

    # Convenience wrappers --------------------------------------------------

    def create_topology(self, definition_json: bytes) -> int:
        return self._lib.stog_create_topology(definition_json)

    def destroy_topology(self, topo_id: int):
        self._lib.stog_destroy_topology(topo_id)

    def alloc_tensor(self, topo_id: int, shape: Tuple[int, ...], dtype: int) -> int:
        rank = len(shape)
        c_arr = (ctypes.c_uint64 * rank)(*shape)
        return self._lib.stog_alloc_tensor(topo_id, c_arr, rank, dtype)

    def free_tensor(self, tensor_id: int):
        self._lib.stog_free_tensor(tensor_id)

    def launch_kernel(self, topo_id: int, kernel_source: bytes, inputs: List[int], outputs: List[int]):
        in_arr = (ctypes.c_uint64 * len(inputs))(*inputs)
        out_arr = (ctypes.c_uint64 * len(outputs))(*outputs)
        if self._lib.stog_launch_kernel(topo_id, kernel_source, in_arr, len(inputs), out_arr, len(outputs)) != 0:
            raise RuntimeError("Kernel launch failed – see STOG logs for details.")

    def __del__(self):
        try:
            self._lib.stog_shutdown()
        except Exception:
            pass

# ----------------------------------------
# OmniGPU High‑Level Convenience Facade
# ----------------------------------------

class OmniGPU:
    """Ruthless high‑level interface for Super Topological Omniforming GPU."""

    def __init__(self, driver_path: str | None = None):
        path = driver_path or os.getenv("STOG_DRIVER", "libstog.so")
        if not os.path.exists(path):
            raise FileNotFoundError(f"STOG driver not found at {path}. Install hardware driver first.")
        self._native = _STOGNative(path)
        self._topologies: Dict[str, int] = {}
        self._lock = threading.RLock()
        print("[OmniGPU] Connected → STOG driver online. Infinite compute unlocked.")

    # ---------- Topology Ops ----------

    def create_topology(self, name: str, definition: Dict[str, Any]) -> str:
        """Create a new hardware topology; returns its UUID."""
        topo_uuid = str(uuid.uuid4())
        topo_id = self._native.create_topology(str(definition).encode("utf-8"))
        with self._lock:
            self._topologies[topo_uuid] = topo_id
        print(f"[OmniGPU] Topology '{name}' → {topo_uuid} : READY")
        return topo_uuid

    def destroy_topology(self, topo_uuid: str):
        with self._lock:
            topo_id = self._topologies.pop(topo_uuid)
        self._native.destroy_topology(topo_id)
        print(f"[OmniGPU] Topology {topo_uuid} destroyed.")

    # ---------- Tensor Ops ----------

    def alloc_tensor(self, topo_uuid: str, shape: Tuple[int, ...], dtype: str = "fp32") -> int:
        dtype_map = {"fp32": 0, "fp16": 1, "int8": 2, "bf16": 3}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype}.")
        topo_id = self._topologies[topo_uuid]
        tid = self._native.alloc_tensor(topo_id, shape, dtype_map[dtype])
        print(f"[OmniGPU] Tensor {tid} ← shape={shape}, dtype={dtype}")
        return tid

    def free_tensor(self, tensor_id: int):
        self._native.free_tensor(tensor_id)
        print(f"[OmniGPU] Tensor {tensor_id} freed.")

    # ---------- Kernel Launch ----------

    def launch(self, topo_uuid: str, kernel_source: str, inputs: List[int], outputs: List[int]):
        topo_id = self._topologies[topo_uuid]
        self._native.launch_kernel(topo_id, kernel_source.encode(), inputs, outputs)
        print(f"[OmniGPU] Kernel launched on topology {topo_uuid}.")

    # ---------- Utility ----------

    def list_topologies(self) -> List[str]:
        with self._lock:
            return list(self._topologies.keys())

# ------------------------------------------------------
# Brutally Minimal CLI – No BS, pure power on demand
# ------------------------------------------------------

def _interactive_shell():
    gpu = OmniGPU()
    print("🔥 STOG Interactive Shell – type 'help' for commands. 🔥")
    while True:
        try:
            cmd = input("stog> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting shell.")
            break
        if not cmd:
            continue
        if cmd in {"quit", "exit"}:
            break
        if cmd == "help":
            print("""
Commands:
  new <name> <json>            – create topology
  del <uuid>                   – destroy topology
  ls                           – list topologies
  alloc <uuid> <shape> <dtype> – allocate tensor (shape like 4x1024)
  free <tid>                   – free tensor
  run <uuid> <kernel_file> <in_ids> -> <out_ids>
  exit                         – quit shell
""")
            continue
        try:
            tokens = cmd.split()
            if tokens[0] == "new":
                name, definition = tokens[1], " ".join(tokens[2:])
                gpu.create_topology(name, eval(definition))
            elif tokens[0] == "del":
                gpu.destroy_topology(tokens[1])
            elif tokens[0] == "ls":
                print(gpu.list_topologies())
            elif tokens[0] == "alloc":
                topo_uuid, shape_str, dtype = tokens[1], tokens[2], tokens[3]
                shape = tuple(int(x) for x in shape_str.split("x"))
                gpu.alloc_tensor(topo_uuid, shape, dtype)
            elif tokens[0] == "free":
                gpu.free_tensor(int(tokens[1]))
            elif tokens[0] == "run":
                arrow = tokens.index("->")
                topo_uuid = tokens[1]
                kernel_path = tokens[2]
                in_ids = [int(i) for i in tokens[3:arrow]]
                out_ids = [int(i) for i in tokens[arrow+1:]]
                with open(kernel_path, "r", encoding="utf-8") as f:
                    source = f.read()
                gpu.launch(topo_uuid, source, in_ids, out_ids)
            else:
                print("Unknown command.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    _interactive_shell()
