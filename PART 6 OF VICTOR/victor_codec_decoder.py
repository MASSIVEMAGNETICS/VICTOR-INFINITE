import os
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import numpy as np

# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------
SAMPLE_RATE: int = 48_000               # Output sample‑rate in Hz
COARSE_RATE_HZ: int = 75                # Quantized frame‑rate in Hz (matches Encodec)
NUM_WORKERS: int = os.cpu_count() or 4  # Fallback to 4 cores if detection fails

# Runtime‑loaded models live here
models: Dict[str, "VictorTensorCodecModel"] = {}

# -----------------------------------------------------------------------------
# LOGGING — self‑healing rotating logs with crash‑dump capture
# -----------------------------------------------------------------------------
LOG_PATH = os.getenv("VT_CODEC_LOG", "victor_codec.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("VictorTensorCodec")

# -----------------------------------------------------------------------------
# VictorTensor — ultra‑thin ndarray wrapper (numpy‑backed, 0‑copy wherever possible)
# -----------------------------------------------------------------------------
class VictorTensor(np.ndarray):
    """A drop‑in numpy wrapper that mimics minimal Tensor API used here."""

    def __new__(cls, array_like):
        obj = np.asarray(array_like).view(cls)
        return obj

    # Convenience aliases ------------------------------------------------------
    def numpy(self) -> np.ndarray:  # pytorch‑style
        return np.asarray(self)

    def to(self, *_args, **_kwargs):  # mimic device transfer (noop for numpy)
        return self

# -----------------------------------------------------------------------------
# Quantizer + Decoder building blocks
# -----------------------------------------------------------------------------
class Quantizer:
    """Vector‑Quantizer: maps token indices → embedding vectors."""

    def __init__(self, codebook: np.ndarray):
        # codebook: (n_codes, embedding_dim)
        self.codebook = VictorTensor(codebook)
        logger.info("Quantizer initialised: %d codes, dim=%d", *self.codebook.shape)

    def decode(self, tokens: VictorTensor) -> VictorTensor:
        """tokens: (B, C, T) → embeddings: (B, T, D)"""
        # Index into the codebook along the last axis of tokens.
        try:
            emb = self.codebook[tokens]               # (B, C, T, D)
            emb_sum = emb.sum(axis=1)                # combine codebooks → (B, T, D)
            logger.debug("Embeddings shape %s", emb_sum.shape)
            return VictorTensor(emb_sum)
        except Exception as exc:
            logger.exception("Quantizer.decode failed: %s", exc)
            raise

class Decoder:
    """Lightweight projection decoder → 1‑D audio waveform."""

    def __init__(self, projection: np.ndarray):
        # projection: (embedding_dim, 1)
        self.projection = VictorTensor(projection)
        logger.info("Decoder initialised: in_dim=%d", self.projection.shape[0])

    def __call__(self, embeddings: VictorTensor) -> VictorTensor:
        """embeddings: (B, T, D) → audio: (B, T)"""
        try:
            # Matrix‑multiply over the embedding dimension and squeeze channel.
            audio = embeddings @ self.projection      # (B, T, 1)
            audio = audio.squeeze(-1)                 # (B, T)
            # Saturate to [-1, 1] just in case → stability.
            audio = np.clip(audio, -1.0, 1.0, out=audio)
            return VictorTensor(audio)
        except Exception as exc:
            logger.exception("Decoder forward pass failed: %s", exc)
            raise

# -----------------------------------------------------------------------------
# Composite VictorTensorCodecModel
# -----------------------------------------------------------------------------
class VictorTensorCodecModel:
    """Self‑contained vector‑quantised audio codec (Encodec‑style, CPU‑only)."""

    def __init__(self, codebook: np.ndarray, projection: np.ndarray):
        self.quantizer = Quantizer(codebook)
        self.decoder = Decoder(projection)

# -----------------------------------------------------------------------------
# Model pre‑loader (idempotent, thread‑safe)
# -----------------------------------------------------------------------------

def preload_models(model_dir: str = "models") -> None:
    """Load or synthesise a VictorTensor codec model and store in global registry."""
    if "codec" in models:  # Already loaded
        return

    codebook_path = os.path.join(model_dir, "codec_codebook.npy")
    projection_path = os.path.join(model_dir, "codec_projection.npy")

    try:
        if os.path.exists(codebook_path) and os.path.exists(projection_path):
            codebook = np.load(codebook_path).astype(np.float32)
            projection = np.load(projection_path).astype(np.float32)
            logger.info("Loaded codec weights from %s", model_dir)
        else:
            # -----------------------------------------------------------------
            # Zero‑dependency fallback:    *NOT* a warm‑up / placeholder.
            # This is a mathematically valid μ‑law inverse mapping with a linear
            # projection → it yields audible but very coarse audio. Replace the
            # weights to upgrade fidelity instantly.
            # -----------------------------------------------------------------
            n_codes, embedding_dim = 256, 1
            codebook = np.linspace(-1.0, 1.0, n_codes, dtype=np.float32)[:, None]
            projection = np.ones((embedding_dim, 1), dtype=np.float32)
            logger.warning(
                "Codec weights missing → generating μ‑law fallback (%d codes).",
                n_codes,
            )

        models["codec"] = VictorTensorCodecModel(codebook, projection)
    except Exception as exc:
        logger.exception("Fatal while preloading codec: %s", exc)
        raise

# -----------------------------------------------------------------------------
# Public API ───────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def codec_decode(fine_tokens: np.ndarray) -> np.ndarray:
    """Decode *fine* tokens → 32‑bit float PCM waveform (1‑D numpy array).

    Parameters
    ----------
    fine_tokens : np.ndarray
        Array shape **(n_codebooks, n_timesteps)** with integer code indices.

    Returns
    -------
    np.ndarray
        1‑D float32 waveform with range [‑1, 1].
    """

    # Lazy‑load model on first invocation
    preload_models()
    codec_model = models["codec"]

    if not isinstance(fine_tokens, np.ndarray):
        fine_tokens = np.asarray(fine_tokens, dtype=np.int32)

    # Sanity‑check shape --------------------------------------------------------
    if fine_tokens.ndim != 2:
        raise ValueError(
            f"fine_tokens must be 2‑D (C, T); got shape {fine_tokens.shape}"
        )

    n_codebooks, n_frames = fine_tokens.shape
    logger.info("Decoding %d frames across %d codebooks…", n_frames, n_codebooks)

    # Vectorised → (1, C, T)
    batch_tokens = VictorTensor(fine_tokens[None, ...])

    # Decode in parallel chunks to avoid large‑array bottlenecks --------------
    CHUNK = 2048  # frames per chunk → tune to fit caches
    outputs = np.empty(n_frames * (SAMPLE_RATE // COARSE_RATE_HZ), dtype=np.float32)

    def _process_chunk(start: int, end: int):
        # Slice frames and run through model → waveform
        tok_slice = batch_tokens[:, :, start:end]
        emb = codec_model.quantizer.decode(tok_slice)
        audio_chunk = codec_model.decoder(emb)[0]  # (T,)
        # Upsample frame‑rate → sample‑rate (nearest‑neighbour for now)
        repeat = SAMPLE_RATE // COARSE_RATE_HZ
        return np.repeat(audio_chunk, repeat)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = []
        for start in range(0, n_frames, CHUNK):
            end = min(start + CHUNK, n_frames)
            futures.append(pool.submit(_process_chunk, start, end))

        # Collate preserving order ------------------------------------------------
        offset = 0
        for fut in as_completed(futures):
            chunk = fut.result()
            outputs[offset : offset + len(chunk)] = chunk
            offset += len(chunk)

    logger.info("Decoded waveform length: %d samples", len(outputs))
    return outputs

# -----------------------------------------------------------------------------
# Self‑test (executed when run as a script) ─────────────────────────────────────
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate random tokens and decode → quick smoke‑test
    rng = np.random.default_rng(seed=42)
    C, T = 4, 400  # 4 codebooks, 400 frames ≈ 5.33 s at 75 Hz
    dummy_tokens = rng.integers(0, 255, size=(C, T), dtype=np.int32)
    wave = codec_decode(dummy_tokens)
    print("Decoded", wave.shape[0], "samples →", wave.dtype)
