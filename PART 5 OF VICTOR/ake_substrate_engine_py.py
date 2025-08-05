# FILE: ake_substrate_engine.py
# VERSION: v3.1.0
# AUTHOR: Brandon "iambandobandz" Emery × Victor (Fractal Architect Mode) + OmniForge
# --------------------------------------------------------------------------------------
# AKE: Neuro‑Symbolic Substrate Engine — **OFFLINE EDITION**
# --------------------------------------------------------------------------------------
# Victor’s directive: **NO third‑party cloud hooks.** This build purges every external
# generative dependency (OpenAI, Anthropic, etc.) while keeping the architecture pluggable
# for future *local* LLMs you might run on‑device (e.g., llama‑cpp, Mistral‑inference).
#
# What changed (v3.0.0 → v3.1.0)
# • 🧹 Removed `openai`, `httpx`, and any outbound network calls
# • 🔒 `StaticProvider` is now the default and only baked‑in TextureProvider
# • 🔌 A `LocalCmdProvider` stub shows how to wrap an offline binary (disabled by default)
# • ⚙️ ENV `AKE_PROVIDER` still exists but only accepts `static` or `localcmd`
# • 📦 Dependency list trimmed to FastAPI + SQLModel + PyYAML (optional)
# --------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlmodel import Field, SQLModel, create_engine, select, Session

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None  # type: ignore

# --------------------------------------------------------------------------------------
# Logging -----------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
LOG_LEVEL_ENV = "AKE_LOG_LEVEL"
DEFAULT_LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=os.getenv(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL),
    format="[%(levelname)s] %(asctime)s ‑ %(name)s: %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AKE‑Engine‑Offline")

# --------------------------------------------------------------------------------------
# Domain Model (Symbolic) --------------------------------------------------------------
# --------------------------------------------------------------------------------------
class Shape(str, Enum):
    SPHERE = "sphere"
    SLAB = "slab"
    TOROID = "toroid"
    HUMANOID = "humanoid"
    UNKNOWN = "unknown"


class Material(str, Enum):
    CRYSTAL = "crystal"
    STONE = "stone"
    PLASMA = "plasma"
    SILICON = "silicon"
    UNDEFINED = "undefined"


class State(str, Enum):
    ANCIENT = "ancient"
    NEW = "new"
    BROKEN = "broken"
    ACTIVE = "active"
    DEFAULT = "neutral"


@dataclass(slots=True, frozen=True)
class SymbolicObject:
    name: str
    shape: Shape = Shape.UNKNOWN
    material: Material = Material.UNDEFINED
    state: State = State.DEFAULT
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def symbolic_str(self) -> str:
        return (
            f"Symbolic ID: {self.name} | Shape: {self.shape.value} | "
            f"Material: {self.material.value} | State: {self.state.value}"
        )

# --------------------------------------------------------------------------------------
# Persistence Layer (SQLModel) ---------------------------------------------------------
# --------------------------------------------------------------------------------------
class ObjectORM(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    shape: str
    material: str
    state: str
    metadata: str  # JSON string

    @classmethod
    def from_symbolic(cls, obj: SymbolicObject) -> "ObjectORM":
        return cls(
            name=obj.name,
            shape=obj.shape.value,
            material=obj.material.value,
            state=obj.state.value,
            metadata=json.dumps(obj.metadata),
        )

    def to_symbolic(self) -> SymbolicObject:
        return SymbolicObject(
            name=self.name,
            shape=Shape(self.shape),
            material=Material(self.material),
            state=State(self.state),
            metadata=json.loads(self.metadata or "{}"),
        )


DB_URL_DEFAULT = "sqlite:///ake_substrate.db"
engine = create_engine(DB_URL_DEFAULT, echo=False, connect_args={"check_same_thread": False})
SQLModel.metadata.create_all(engine)

# --------------------------------------------------------------------------------------
# Texture Provider Protocol (OFFLINE) --------------------------------------------------
# --------------------------------------------------------------------------------------
class TextureProvider(Protocol):
    async def generate_texture(self, obj: SymbolicObject) -> str:  # noqa: D401
        ...


class StaticProvider:
    """Deterministic / pseudo‑random textures, zero external calls."""

    _fallbacks: Dict[Material, List[str]] = {
        Material.CRYSTAL: [
            "shimmers with captured light",
            "fractures the air with prismatic ghosts",
        ],
        Material.STONE: [
            "is rough and time‑scarred",
            "bears lichen like ancient insignia",
        ],
        Material.PLASMA: [
            "whips in luminous spirals",
            "hums with caged lightning",
        ],
        Material.SILICON: [
            "gleams with artificial sheen",
            "ticks with nanoscopic logic",
        ],
        Material.UNDEFINED: ["defies description"],
    }

    async def generate_texture(self, obj: SymbolicObject) -> str:  # noqa: D401
        choices = self._fallbacks.get(obj.material, ["has an indescribable texture"])
        return random.choice(choices)


class LocalCmdProvider:
    """Example wrapper for an offline command‑line model (disabled by default)."""

    def __init__(self, cmd: str = "./generate_description.sh"):
        self._cmd = cmd

    async def generate_texture(self, obj: SymbolicObject) -> str:  # noqa: D401
        # Placeholder: implement subprocess call to local binary returning 1‑line output
        return "[local‑cmd output missing]"


# Provider selection (static / localcmd) ----------------------------------------------
_provider_choice = os.getenv("AKE_PROVIDER", "static").lower()
if _provider_choice == "localcmd":
    PROVIDER: TextureProvider = LocalCmdProvider(os.getenv("AKE_CMD", "./gen.sh"))
else:
    PROVIDER = StaticProvider()
    if _provider_choice not in {"static", ""}:
        logger.warning("Unknown AKE_PROVIDER='%s'; defaulting to static", _provider_choice)

# --------------------------------------------------------------------------------------
# Scene Renderer ----------------------------------------------------------------------
# --------------------------------------------------------------------------------------
class SceneRenderer:
    def __init__(self, *, rng_seed: Optional[int] = None):
        self.rng = random.Random(rng_seed)
        self.seed = rng_seed

    async def describe_object(self, obj: SymbolicObject) -> str:
        symbolic = obj.symbolic_str()
        neural = await PROVIDER.generate_texture(obj)
        sentence = f"[LOGIC CORE]: {symbolic}\n[NEURAL TEXTURE]: It {neural}.\n"
        return sentence

    async def describe_scene(self, objs: Iterable[SymbolicObject]) -> str:
        if not objs:
            return "The scene is empty."
        parts = await asyncio.gather(*(self.describe_object(o) for o in objs))
        return "--- SCENE REPORT ---\n" + "\n".join(parts) + "--- END REPORT ---"

# --------------------------------------------------------------------------------------
# FastAPI (REST / WS) ------------------------------------------------------------------
# --------------------------------------------------------------------------------------
app = FastAPI(title="AKE Neuro‑Symbolic Engine ‑ Offline", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymbolicObjectIn(BaseModel):
    name: str
    shape: Shape = Shape.UNKNOWN
    material: Material = Material.UNDEFINED
    state: State = State.DEFAULT
    metadata: Dict[str, Any] = {}

    @validator("metadata", pre=True)
    def ensure_dict(cls, v: Any) -> Dict[str, Any]:  # noqa: N805
        return dict(v or {})


@app.on_event("startup")
async def startup_db() -> None:  # pragma: no cover
    logger.info("DB ready at %s (offline edition)", DB_URL_DEFAULT)


@app.post("/objects", response_model=SymbolicObjectIn)
async def create_object(obj: SymbolicObjectIn):
    sym = SymbolicObject(
        name=obj.name,
        shape=obj.shape,
        material=obj.material,
        state=obj.state,
        metadata=obj.metadata,
    )
    with Session(engine) as session:
        orm = ObjectORM.from_symbolic(sym)
        session.add(orm)
        session.commit()
        session.refresh(orm)
    return obj


@app.get("/scene", response_model=str)
async def render_scene(seed: Optional[int] = None):
    renderer = SceneRenderer(rng_seed=seed)
    with Session(engine) as session:
        objs = [row.to_symbolic() for row in session.exec(select(ObjectORM)).all()]
    return await renderer.describe_scene(objs)


# WebSocket broadcast (unchanged) -----------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, msg: str):
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except WebSocketDisconnect:
                self.disconnect(ws)


autobroker = ConnectionManager()


@app.websocket("/ws/scene")
async def scene_ws(ws: WebSocket):
    await autobroker.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("cmd")
            if cmd == "render":
                seed = data.get("seed")
                desc = await render_scene(seed)  # type: ignore[arg-type]
                await ws.send_text(desc)
    except WebSocketDisconnect:
        autobroker.disconnect(ws)

# --------------------------------------------------------------------------------------
# CLI ---------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
async def cli_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser("AKE Engine v3.1 ‑ Offline")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, help="Render a one‑off scene and exit")
    parser.add_argument("--config", type=Path, help="Load objects from config into DB then exit")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.getLogger().setLevel(level_map.get(args.verbose, logging.DEBUG))

    # Preload objects if requested ----------------------------------------------------
    if args.config:
        objs = load_objects_from_file(args.config)
        with Session(engine) as session:
            session.add_all([ObjectORM.from_symbolic(o) for o in objs])
            session.commit()
        logger.info("Loaded %d objects from %s", len(objs), args.config)
        if not args.serve and args.seed is None:
            return

    if args.serve:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    # One‑off render ------------------------------------------------------------------
    if args.seed is not None:
        renderer = SceneRenderer(rng_seed=args.seed)
        with Session(engine) as session:
            objs = [row.to_symbolic() for row in session.exec(select(ObjectORM)).all()]
        print(await renderer.describe_scene(objs))
        return

    parser.print
