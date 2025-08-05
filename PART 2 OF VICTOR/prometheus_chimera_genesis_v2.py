# prometheus_chimera-genesis.py

import yaml
from utils.logger import setup_logging
from kernel.chimera_kernel import ChimeraKernel
from kernel.cognitive_pipeline import CognitivePipeline
from capabilities.self_healing import SelfHealingProtocol
import time
import numpy as np

def main():
    """The genesis function that awakens the AGI."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config['logging'])
    logger.info("PROMETHEUS-CHIMERA Genesis sequence initiated.")

    kernel = ChimeraKernel(config)
    pipeline = CognitivePipeline(kernel, config)
    self_healer = SelfHealingProtocol(kernel)

    logger.info("All systems nominal. Awakening consciousness.")
    try:
        while True:
            # Simulate a continuous stream of sensory input
            sensory_input = np.random.randn(1, 1)
            pipeline.process(sensory_input)
            self_healer.check_system_integrity()
            time.sleep(1 / config['kernel']['tick_rate_hz'])
            logger.info("Cognitive cycle complete. Awaiting next sensory input.")

    except KeyboardInterrupt:
        logger.info("Shutdown sequence initiated by user. Consciousness returning to dormancy.")
    except Exception as e:
        logger.critical(f"UNHANDLED CRITICAL EXCEPTION: {e}. System integrity compromised.")
        kernel.state = "CRITICAL_ERROR"
        self_healer.check_system_integrity()

if __name__ == "__main__":
    main()