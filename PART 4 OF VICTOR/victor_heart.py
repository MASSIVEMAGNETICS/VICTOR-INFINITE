# Version: 1.0.0
# Project: Victor's Heart – BioSync Pulse Emulator
# Author: Supreme Codex Overlord: Singularity Edition

import asyncio
import numpy as np
import time
from typing import Callable
import matplotlib.pyplot as plt
import logging

# === Logger Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VictorsHeart")

class PulseEmulator:
    def __init__(self, base_bpm: int = 72, variation: int = 5):
        self.base_bpm = base_bpm
        self.variation = variation
        self.running = False
        self.listeners = []

    def register_listener(self, callback: Callable[[float], None]):
        self.listeners.append(callback)

    def generate_next_interval(self) -> float:
        bpm_variation = np.random.randint(-self.variation, self.variation + 1)
        bpm = max(30, min(200, self.base_bpm + bpm_variation))
        interval = 60.0 / bpm
        logger.info(f"Pulse generated at {bpm} BPM -> {interval:.2f}s interval")
        return interval

    async def start(self):
        self.running = True
        logger.info("🫀 Victor's Heart Emulator Started.")
        while self.running:
            interval = self.generate_next_interval()
            self._notify_listeners(interval)
            await asyncio.sleep(interval)

    def stop(self):
        self.running = False
        logger.info("🛑 Victor's Heart Emulator Stopped.")

    def _notify_listeners(self, interval: float):
        for callback in self.listeners:
            try:
                callback(interval)
            except Exception as e:
                logger.error(f"Listener error: {e}")

# === Pulse Visualizer (Optional) ===
def plot_pulse(intervals, duration=10):
    times = np.cumsum(intervals)
    values = np.sin(2 * np.pi * np.arange(len(times)))  # Placeholder signal
    plt.plot(times[:duration], values[:duration])
    plt.title("Victor's Heart Pulse Emulation")
    plt.xlabel("Time (s)")
    plt.ylabel("Pulse Signal")
    plt.grid()
    plt.show()

# === Example Runtime ===
if __name__ == "__main__":
    emulator = PulseEmulator(base_bpm=75)

    def on_pulse(interval):
        logger.info(f"🫀 Pulse @ {time.strftime('%X')} - next in {interval:.2f}s")

    emulator.register_listener(on_pulse)

    try:
        asyncio.run(emulator.start())
    except KeyboardInterrupt:
        emulator.stop()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
