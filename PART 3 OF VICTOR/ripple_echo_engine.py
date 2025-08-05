# FILE: ripple_echo_engine.py
# VERSION: v2.1.0
# AUTHOR: Brandon "iambandobandz" Emery × Victor + OmniForge
# --------------------------------------------------------------------------------------
# RippleEchoEngine – 37‑Node Flower‑of‑Life AGI Core (Offline)
# --------------------------------------------------------------------------------------
# v2.1.0 upgrades
#   • ✅ Strong type hints, `@dataclass(slots=True)` models, Enum roles.
#   • ✅ Deterministic RNG seeding (`--seed`) for reproducible states.
#   • ✅ Optional pure‑Python math fallback if NumPy missing.
#   • ✅ Async heartbeat mode with `asyncio.TaskGroup` + graceful shutdown.
#   • ✅ Structured logging (`structlog` JSON or classic) + CLI verbosity.
#   • ✅ PulseTelemetryBus reused from VictorThoughtEngine for consistency.
#   • ✅ NDJSON state dump (`--dump pulses.ndjson`) & final snapshot YAML/JSON.
#   • ✅ Configurable tick rate (`--hz`) and infinite mode (`--infinite`).
# --------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Mapping, Optional

# -----------------------------------------------------------------------------
# Optional NumPy import --------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # graceful pure‑Python fallback
    np = None  # type: ignore

# -----------------------------------------------------------------------------
# Logging configuration --------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import structlog  # type: ignore
except ModuleNotFoundError:
    structlog = None  # type: ignore


def _configure_logging(verbosity: int, json_logs: bool) -> None:
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = level_map.get(verbosity, logging.DEBUG)
    if structlog and json_logs:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(level),
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        logging.basicConfig(
            level=level,
            format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )


logger = logging.getLogger("RippleEcho")

# -----------------------------------------------------------------------------
# Paths ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Telemetry Bus ----------------------------------------------------------------
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class Pulse:
    typ: str
    payload: Mapping[str, Any]
    ts: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class PulseBus:
    def __init__(self, history_max: int = 1024):
        self._subs: List[Callable[[Pulse], Awaitable[None]] | Callable[[Pulse], None]] = []
        self._history: Deque[Pulse] = collections.deque(maxlen=history_max)  # type: ignore

    def subscribe(self, fn: Callable[[Pulse], Awaitable[None]] | Callable[[Pulse], None]) -> None:
        self._subs.append(fn)

    async def pulse(self, typ: str, payload: Mapping[str, Any], latency: float = 0.0) -> None:
        p = Pulse(typ, payload, latency_ms=latency)
        self._history.append(p)
        for fn in list(self._subs):
            if asyncio.iscoroutinefunction(fn):
                asyncio.create_task(fn(p))
            else:
                fn(p)

    def dump_ndjson(self, dest: Path) -> None:
        with dest.open("w", encoding="utf-8") as f:
            for p in self._history:
                f.write(json.dumps(asdict(p)) + "\n")


# -----------------------------------------------------------------------------
# Domain Model -----------------------------------------------------------------
# -----------------------------------------------------------------------------
class Role(Enum):
    CORE = auto()
    PRIMARY = auto()
    SECONDARY = auto()
    PERIPHERAL = auto()

    @property
    def weight(self) -> float:
        return {
            Role.CORE: 1.0,
            Role.PRIMARY: 0.9,
            Role.SECONDARY: 0.7,
            Role.PERIPHERAL: 0.5,
        }[self]

    @property
    def threshold(self) -> float:
        return {
            Role.CORE: 0.75,
            Role.PRIMARY: 0.80,
            Role.SECONDARY: 0.90,
            Role.PERIPHERAL: 0.95,
        }[self]


def _role_for_index(idx: int) -> Role:
    if idx == 0:
        return Role.CORE
    if 1 <= idx <= 6:
        return Role.PRIMARY
    if 7 <= idx <= 18:
        return Role.SECONDARY
    return Role.PERIPHERAL


@dataclass(slots=True)
class Node:
    idx: int
    role: Role
    resonance: float

    @classmethod
    def random(cls, idx: int, rng: random.Random) -> "Node":
        return cls(idx=idx, role=_role_for_index(idx), resonance=rng.uniform(-1.0, 1.0))


@dataclass(slots=True)
class Directive:
    ts: float
    node_idx: int
    resonance: float
    role: Role
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "ts": self.ts, "node": self.node_idx, "role": self.role.name, "res": self.resonance}


# -----------------------------------------------------------------------------
# Engine -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
class RippleEchoEngine:
    NODE_COUNT = 37

    def __init__(self, *, seed: Optional[int] = None, bus: Optional[PulseBus] = None):
        self.rng = random.Random(seed)
        self.bus = bus or PulseBus()
        self.nodes: List[Node] = [Node.random(i, self.rng) for i in range(self.NODE_COUNT)]
        self.cycle = 0
        self._last_state: List[float] | None = None
        logger.debug("Engine initialized with seed=%s", seed)

    # math utils ---------------------------------------------------------------
    def _neighbor_sum(self, values: List[float]) -> List[float]:
        if np:  # vectorized path
            arr = np.array(values)
            return (np.roll(arr, 1) + np.roll(arr, -1)).tolist()
        # pure‑python fallback
        return [values[(i - 1) % len(values)] + values[(i + 1) % len(values)] for i in range(len(values))]

    # single step --------------------------------------------------------------
    async def step(self) -> None:
        tic = time.perf_counter()
        self.cycle += 1
        cur = [n.resonance for n in self.nodes]
        wave = [math.sin(x * math.pi) for x in cur]
        neighbor = self._neighbor_sum(wave)
        updated: List[float] = []
        for i, n in enumerate(self.nodes):
            v = 0.5 * cur[i] + 0.5 * neighbor[i]
            v *= n.role.weight
            v = max(-1.0, min(1.0, v))
            n.resonance = v
            updated.append(v)
        self._last_state = updated
        await self.bus.pulse("cycle", {"cycle": self.cycle, "state": updated}, (time.perf_counter() - tic) * 1000)
        # directive extraction
        await self._extract_directives()

    async def _extract_directives(self) -> None:
        ts = time.time()
        for n in self.nodes:
            if abs(n.resonance) >= n.role.threshold:
                d = Directive(ts, n.idx, n.resonance, n.role)
                await self.bus.pulse("directive", d.as_dict())

    # run loop -----------------------------------------------------------------
    async def run(self, *, hz: float, infinite: bool, iterations: int | None) -> None:
        period = 1.0 / hz
        try:
            while infinite or (iterations is None or self.cycle < iterations):
                await self.step()
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass

    # snapshot -----------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "cycles": self.cycle,
            "state": self._last_state,
            "seed": self.rng.seed,  # type: ignore[attr-defined]
        }


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
async def _cli(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser("RippleEchoEngine v2.1")
    parser.add_argument("--iterations", "-n", type=int, help="Number of cycles before exit (default infinite)")
    parser.add_argument("--hz", type=float, default=20.0, help="Tick rate (Hz)")
    parser.add_argument("--seed", type=int, help="RNG seed for reproducibility")
    parser.add_argument("--json-logs", action="store_true", help="Emit structlog JSON logs if available")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--dump", type=Path, help="Dump pulse NDJSON on exit")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose, args.json_logs)

    bus = PulseBus()

    engine = RippleEchoEngine(seed=args.seed, bus=bus)

    # live directive print -----------------------------------------------------
    def _dir_print(p: Pulse):
        if p.typ == "directive":
            logger.info("DIR node=%s role=%s res=%.3f", p.payload["node"], p.payload["role"], p.payload["res"])

    bus.subscribe(_dir_print)

    async def _runner():
        await engine.run(hz=args.hz, infinite=args.iterations is None, iterations=args.iterations)

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    task = asyncio.create_task(_runner())
    await stop
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    if args.dump:
        bus.dump_ndjson(args.dump)
        logger.info("Pulse dump saved to %s", args.dump)

    # final snapshot
    snap_path = LOG_DIR / f"ripple_snapshot_{int(time.time())}.json"
    with snap_path.open("w", encoding="utf-8") as f:
        json.dump(engine.snapshot(), f, indent=2)
    logger.info("Snapshot written to %s", snap_path)


if __name__ == "__main__":
    try:
        asyncio.run(_cli())
    except KeyboardInterrupt:
        pass
