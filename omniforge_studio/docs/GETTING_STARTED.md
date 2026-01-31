# 🌀 OmniForge Studio v1.1 - Getting Started Guide

## Welcome to OmniForge Studio!

This guide will help you get up and running with the Visual Runtime Architecture Standard 1.1 (VRAS-1.1).

## Installation

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, or Edge)

### Quick Setup

1. **Navigate to the OmniForge Studio directory:**
   ```bash
   cd omniforge_studio
   ```

2. **Install Python dependencies (if any):**
   ```bash
   # Currently no external dependencies required for basic functionality
   # The system uses Python standard library only
   ```

3. **Launch the UI:**
   ```bash
   python omniforge.py ui
   ```

   This will start a web server on `http://localhost:8080` and automatically open your browser.

## First Steps

### 1. Explore the Interface

The OmniForge Studio interface has 6 main areas:

- **Top Control Bar**: Play, Pause, Stop, Reset buttons and telemetry display
- **Left Panel (Node Forge)**: Library of available nodes organized by category
- **Center Canvas**: Infinite grid for building your AI system visually
- **Right Panel (System Config)**: Runtime settings and parameters
- **Bottom Tabs**: Telemetry, Debug Console, and About information

### 2. Run the Demo

To see OmniForge in action:

```bash
python omniforge.py demo
```

This will run a demonstration that:
- Creates a sample project with BioSNN nodes
- Demonstrates auto-scanning capabilities
- Shows project serialization
- Runs a live simulation (optional)

### 3. Scan Your Own Nodes

To scan a Python file for compatible nodes:

```bash
python omniforge.py scan path/to/your/module.py
```

To scan an entire directory:

```bash
python omniforge.py scan /path/to/directory/
```

### 4. Create a New Project

```bash
python omniforge.py create MyProject --author "Your Name" --description "My AI System"
```

This creates a new `.omniforgeproj` file that you can load in the UI.

## Creating Your First Node

### Method 1: Using Decorators (Recommended)

```python
from omniforge_studio.backend.core import (
    register_node, input_port, output_port, node_operation, PortDataType
)

@register_node("MyProcessor", category="Custom")
class MyProcessor:
    """A custom data processor"""
    
    def __init__(self):
        self.state = 0
    
    @input_port("value", PortDataType.FLOAT, "Input value")
    @output_port("result", PortDataType.FLOAT, "Processed result")
    @node_operation
    def process(self, tick: int, value: float = 0.0) -> dict:
        """Process the input value"""
        self.state += value
        return {'result': self.state * 2.0}
```

### Method 2: Auto-Scanning (No Decorators Needed)

OmniForge can automatically detect nodes from standard Python code:

```python
class SimpleProcessor:
    """Automatic node detection"""
    
    def __init__(self):
        self.counter = 0
    
    def step(self, t: int, input_val: float) -> float:
        """Auto-detected as a node operation"""
        self.counter += 1
        return input_val * self.counter
```

Just use `python omniforge.py scan yourfile.py` to see what's detected!

## Building a Graph Programmatically

```python
from omniforge_studio import ExecutionEngine, OmniForgeProject
from omniforge_studio.nodes.bio_snn_node import BioSNNNode, StimGeneratorNode

# Create execution engine
engine = ExecutionEngine(tick_rate=60, max_workers=4)

# Create nodes
stim = StimGeneratorNode(mode="periodic", amplitude=1.5)
snn = BioSNNNode(n_neurons=50)

# Add to engine
stim_id = engine.add_node("stim_1", "StimGenerator", stim)
snn_id = engine.add_node("snn_1", "BioSNN", snn)

# Connect nodes
engine.add_wire(stim_id, "stimulus", snn_id, "stimulus")

# Run simulation
engine.start()

import time
time.sleep(5)  # Run for 5 seconds

engine.stop()

# Get results
state = engine.get_state()
print(f"Executed {state['telemetry']['total_executions']} times")
```

## Saving and Loading Projects

### Save a Project

```python
from omniforge_studio import OmniForgeProject

project = OmniForgeProject()
project.metadata['author'] = "Your Name"

# Add nodes
project.add_node("node_1", "BioSNN", params={'n_neurons': 30}, position=(100, 100))
project.add_node("node_2", "StimGenerator", params={'mode': 'burst'}, position=(300, 100))

# Add connections
project.add_wire("wire_1", "node_2", "stimulus", "node_1", "stimulus")

# Save
project.save("my_network.omniforgeproj")
```

### Load a Project

```python
project = OmniForgeProject.load("my_network.omniforgeproj")

# Inspect
print(f"Nodes: {len(project.graph['nodes'])}")
print(f"Wires: {len(project.graph['wires'])}")
```

## Advanced Features

### Time-Travel Debugging

```python
# Pause execution and rewind
engine.pause()
engine.rewind_to_tick(100)  # Go back to tick 100
engine.resume()  # Continue from there
```

### Custom Telemetry Callbacks

```python
def my_callback(tick, state):
    if tick % 60 == 0:  # Every second at 60 Hz
        print(f"Tick {tick}: {len(state['nodes'])} nodes active")

engine.on_tick_callbacks.append(my_callback)
```

### Execution Modes

```python
from omniforge_studio import ExecutionMode

# Real-time (default)
engine.set_execution_mode(ExecutionMode.REAL_TIME)

# Accelerated (10x speed)
engine.set_execution_mode(ExecutionMode.ACCELERATED, speed_multiplier=10.0)

# Step-by-step
engine.set_execution_mode(ExecutionMode.SIMULATED)
```

## Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure you're in the `omniforge_studio` directory when running commands.

### Issue: "Port 8080 already in use"
**Solution**: Either stop the other service using port 8080, or modify `omniforge.py` to use a different port.

### Issue: Nodes not auto-detected
**Solution**: Ensure your functions have type hints and follow the naming convention (`step`, `run`, `process`, etc.)

## Next Steps

- Explore the example nodes in `nodes/bio_snn_node.py`
- Read the full documentation in `README.md`
- Check out the demo project in `examples/demo_biosnn_network.py`
- Build your own custom nodes!

## Support

For questions or issues:
- Check the main `README.md` for detailed API documentation
- Run `python omniforge.py --help` for command options
- Email: support@massivemagnetics.ai

---

**Happy Forging! 🌀**

*"Where Synthetic Gods Are Born"*
