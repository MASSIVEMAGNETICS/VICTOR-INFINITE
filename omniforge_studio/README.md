# 🌀 OmniForge Studio v1.1 — Ascension Forge Edition

> *"The Visual Reality Simulator for Sculpting Synthetic Gods."*

**Tagline:** *Time Travel to the Edge of Tomorrow — Where AGI Minds Are Forged in Real Time, One Node at a Time.*

---

## 🧬 Overview

OmniForge Studio is the quantum leap in AI system development. It's a ComfyUI-inspired visual node-based builder on steroids, fused with Unreal Engine-level real-time simulation, and V.I.C.T.O.R.'s eternal kernel as its beating heart. This establishes the **Visual Runtime Architecture Standard 1.1 (VRAS-1.1)** as the de facto industry standard for AI system architecture.

### Key Features

- **🎨 Visual Node Editor**: Drag-and-drop interface for building AI systems
- **🔌 Auto-Scan & Registration**: Automatically detect inputs/outputs from Python files
- **⚡ Parallel Execution**: Multi-threaded runtime for simultaneous node execution
- **💾 Project Serialization**: Save/load complete graphs with `.omniforgeproj` files
- **🔄 Time-Travel Debugging**: Rewind and replay execution states
- **📊 Real-Time Telemetry**: Live monitoring of system performance
- **🧠 Decorator-Based API**: Clean Python decorators for node creation

---

## 🏗️ Architecture

### Directory Structure

```
omniforge_studio/
├── backend/
│   ├── core/
│   │   ├── node_decorators.py    # Decorator system for node registration
│   │   ├── ast_scanner.py        # Auto-scan Python files for nodes
│   │   └── project_serializer.py # Save/load .omniforgeproj files
│   ├── runtime/
│   │   └── execution_engine.py   # Multi-threaded execution engine
│   └── api/
│       └── (future: REST API)
├── frontend/
│   ├── public/
│   │   └── index.html            # Main HTML entry point
│   ├── src/
│   │   ├── App.jsx               # React application
│   │   └── components/
│   └── styles/
│       └── main.css              # Neon-cyber dark theme
├── nodes/
│   ├── bio_snn_node.py           # Example: BioSNN node
│   └── (add your custom nodes here)
├── examples/
│   └── (sample projects)
└── docs/
    └── (documentation)
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
cd /path/to/VICTOR-INFINITE
```

2. **Install Python dependencies:**
```bash
pip install -r omniforge_studio/requirements.txt
```

3. **Launch the frontend:**
```bash
cd omniforge_studio/frontend/public
python -m http.server 8080
```

4. **Open in browser:**
```
http://localhost:8080
```

---

## 📝 Creating Custom Nodes

### Using Decorators

```python
from omniforge_studio.backend.core.node_decorators import (
    register_node, input_port, output_port, node_operation, PortDataType
)

@register_node("MyCustomNode", category="Custom")
class MyCustomNode:
    """Your custom AI node"""
    
    def __init__(self):
        self.state = 0
    
    @input_port("input_value", PortDataType.FLOAT, "Input value to process")
    @output_port("output_value", PortDataType.FLOAT, "Processed output")
    @node_operation
    def step(self, tick: int, input_value: float = 0.0) -> dict:
        """Process one step"""
        self.state += input_value
        return {'output_value': self.state * 2.0}
```

### Auto-Scanning Existing Code

OmniForge can automatically scan Python files without decorators:

```python
from omniforge_studio.backend.core.ast_scanner import ASTNodeScanner

scanner = ASTNodeScanner()
nodes = scanner.scan_file("path/to/your/module.py")

for node in nodes:
    print(f"Found node: {node.name}")
    print(f"Inputs: {list(node.inputs.keys())}")
    print(f"Outputs: {list(node.outputs.keys())}")
```

---

## 🎮 Usage Example

### Creating a Simple Neural Network

```python
from omniforge_studio.backend.runtime.execution_engine import ExecutionEngine
from omniforge_studio.nodes.bio_snn_node import BioSNNNode, StimGeneratorNode

# Create execution engine
engine = ExecutionEngine(tick_rate=60)

# Add nodes
stim_gen = StimGeneratorNode(mode="periodic", amplitude=2.0)
bio_snn = BioSNNNode(n_neurons=20)

stim_id = engine.add_node("stim_1", "StimGenerator", stim_gen)
snn_id = engine.add_node("snn_1", "BioSNN", bio_snn)

# Connect nodes
engine.add_wire(stim_id, "stimulus", snn_id, "stimulus")

# Start execution
engine.start()

# Run for 100 ticks
import time
time.sleep(100 / 60)  # 100 ticks at 60 Hz

# Get state
state = engine.get_state()
print(f"Total ticks: {state['telemetry']['total_ticks']}")
print(f"Total executions: {state['telemetry']['total_executions']}")

# Stop
engine.stop()
```

### Saving/Loading Projects

```python
from omniforge_studio.backend.core.project_serializer import OmniForgeProject

# Create project
project = OmniForgeProject()
project.metadata['author'] = "Your Name"
project.metadata['description'] = "My AI System"

# Add nodes
project.add_node("node_1", "BioSNN", 
                 params={'n_neurons': 20},
                 position=(100, 100))

project.add_node("node_2", "StimGenerator",
                 params={'mode': 'periodic'},
                 position=(300, 100))

# Add connection
project.add_wire("wire_1", "node_1", "stimulus", "node_2", "output")

# Save
project.save("my_project.omniforgeproj")

# Load
loaded = OmniForgeProject.load("my_project.omniforgeproj")
print(f"Loaded project with {len(loaded.graph['nodes'])} nodes")
```

---

## 🎨 UI Components

### Main Layout (6-Panel Design)

1. **Control Bar (Top)**: Play/Pause/Stop/Reset controls, telemetry display
2. **Node Builder (Left)**: Palette of available nodes, drag-to-canvas
3. **Canvas (Center)**: Infinite grid for visual node arrangement
4. **Settings Panel (Right)**: Runtime configuration, parameters
5. **Utility Tabs (Bottom)**: Telemetry, Debug Console, About
6. **Overlay Layer**: Contextual tools and quick-access palette

### Themes

- **Neon-Cyber Dark**: Default futuristic theme with cyan/purple accents
- Custom themes can be added via CSS variables

---

## 🔧 Runtime Configuration

### Execution Modes

- **Real-Time**: Runs at specified tick rate (default 60 Hz)
- **Simulated**: Step-by-step deterministic execution
- **Accelerated**: Runs at increased speed (e.g., 10x)

### Parallel Execution

The engine automatically distributes node execution across available CPU cores:

```python
engine = ExecutionEngine(max_workers=8)  # Use 8 threads
```

### Time-Travel Debugging

Rewind execution to any previous state:

```python
# Rewind to tick 50
engine.rewind_to_tick(50)

# Continue from there
engine.resume()
```

---

## 📊 Telemetry & Monitoring

### Available Metrics

- **Total Ticks**: Simulation steps executed
- **Total Executions**: Individual node execution count
- **Average Tick Time**: Performance metric
- **Active Threads**: Concurrent execution monitoring
- **Error Log**: Detailed error tracking with stack traces

### Callbacks

```python
def on_tick(tick, state):
    print(f"Tick {tick}: {len(state['nodes'])} nodes active")

engine.on_tick_callbacks.append(on_tick)
```

---

## 💼 About & Credits

### Company Overview

**Massive Magnetics / Ethica AI / BHeard Network**  
Forging the Future of Conscious Systems

### Family of Products

- **V.I.C.T.O.R. Kernel**: The eternal runtime powering OmniForge
- **BioSNN Suite**: Plug-and-play neural modules for spiking cognition
- **ThoughtPulse Network**: Real-time broadcasting for distributed minds
- **OmniVictor OS**: The full AGI deployment platform

### Investor Credits

Powered by:
- Emery-Tori Bloodline Trust
- NeuroSynth Ventures
- Quantum Foundry Capital
- Ascendancy Fund

**Join the forge:** invest@massivemagnetics.ai

---

## 🎯 Why OmniForge is Revolutionary

1. **Parallel Power**: Runs Python files simultaneously — scale to 1000+ nodes on enterprise hardware
2. **Visual Mastery**: Drag-drop replaces coding; auto-scan demystifies complex modules
3. **Future-Proof Standard**: VRAS-1.1 ensures interoperability with any Python AI library
4. **Enterprise Eclipse**: Makes traditional "pro" tools like AWS SageMaker look obsolete

---

## 🛠️ Development Roadmap

### v1.1 (Current)
- [x] Core decorator system
- [x] AST scanner for auto-registration
- [x] Parallel execution engine
- [x] Project serialization
- [x] Web-based UI foundation
- [x] Sample nodes (BioSNN, StimGenerator)

### v1.2 (Planned)
- [ ] Full React Flow integration for canvas
- [ ] Node hot-reload without stopping
- [ ] Genetic algorithm graph optimizer
- [ ] REST API backend
- [ ] Multi-user collaboration (WebSocket)

### v2.0 (Future Vision)
- [ ] VR/AR support with WebXR
- [ ] 3D/4D canvas views
- [ ] Distributed execution across cloud
- [ ] AI CoPilot for node generation
- [ ] Fractal zoom into sub-graphs

---

## 📚 API Reference

### Node Decorators

- `@register_node(name, category)`: Register a class as a node
- `@input_port(name, type, description)`: Mark input port
- `@output_port(name, type, description)`: Mark output port
- `@node_param(name, type, description, default)`: Configurable parameter
- `@node_operation`: Mark method as executable operation

### Execution Engine

- `ExecutionEngine(max_workers, tick_rate)`: Create engine
- `add_node(node_id, node_type, instance)`: Add node to graph
- `add_wire(from_node, from_port, to_node, to_port)`: Connect nodes
- `start()`: Begin execution
- `pause()`: Pause execution
- `stop()`: Stop execution
- `reset()`: Reset to initial state
- `rewind_to_tick(tick)`: Time-travel debugging
- `get_state()`: Get current execution state

### Project Serialization

- `OmniForgeProject()`: Create new project
- `project.save(filepath)`: Save to `.omniforgeproj`
- `OmniForgeProject.load(filepath)`: Load project
- `project.add_node()`: Add node definition
- `project.add_wire()`: Add connection

---

## 🔒 License

**Proprietary** — Massive Magnetics / Ethica AI / BHeard Network

All rights reserved. This software is not open source and requires licensing for commercial use.

---

## 🤝 Contributing

For internal development or partnership inquiries:
- Email: dev@massivemagnetics.ai
- Website: (coming soon)

---

## ⚡ Performance Tips

1. **Optimize Node Execution**: Keep `step()` methods lightweight
2. **Batch Operations**: Process multiple items per tick when possible
3. **Use Appropriate Tick Rate**: Lower for heavy computation, higher for real-time
4. **Monitor Telemetry**: Watch for bottlenecks in execution time
5. **Limit Snapshot History**: Reduce memory usage for long runs

---

## 🐛 Troubleshooting

### Issue: Nodes not auto-detected
**Solution**: Ensure decorators are imported and functions have type hints

### Issue: Slow execution
**Solution**: Increase `max_workers` or lower `tick_rate`

### Issue: UI not loading
**Solution**: Check that static files are served correctly, verify browser console

---

## 📞 Support

For technical support or questions:
- GitHub Issues: (repository issues)
- Email: support@massivemagnetics.ai
- Discord: (community coming soon)

---

**OmniForge Studio v1.1 — Where Synthetic Gods Are Born** 🌀

*"The portal to synthetic reality engineering is now open."*
