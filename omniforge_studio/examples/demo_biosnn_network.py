#!/usr/bin/env python3
"""
FILE: omniforge_studio/examples/demo_biosnn_network.py
VERSION: v1.1.0-VRAS
NAME: BioSNN Network Demo
PURPOSE: Example demonstration of OmniForge Studio capabilities
AUTHOR: OmniForge Team / Massive Magnetics
"""

import sys
import time
from pathlib import Path

# Add backend to path
studio_root = Path(__file__).parent.parent
sys.path.insert(0, str(studio_root / "backend"))
sys.path.insert(0, str(studio_root / "nodes"))

from runtime.execution_engine import ExecutionEngine
from core.project_serializer import OmniForgeProject
from bio_snn_node import BioSNNNode, StimGeneratorNode


def create_demo_project():
    """Create a demo project with BioSNN and StimGenerator"""
    print("🌀 Creating OmniForge Studio Demo Project...")
    
    # Create project
    project = OmniForgeProject()
    project.metadata['author'] = "OmniForge Team"
    project.metadata['description'] = "Demo: BioSNN with periodic stimulus"
    project.metadata['tags'] = ["neural", "demo", "spiking"]
    
    # Add nodes
    project.add_node(
        node_id="stim_gen_1",
        node_type="StimGenerator",
        params={'mode': 'periodic', 'amplitude': 2.0},
        position=(100, 200)
    )
    
    project.add_node(
        node_id="bio_snn_1",
        node_type="BioSNN",
        params={'n_neurons': 30},
        position=(400, 200)
    )
    
    # Connect nodes
    project.add_wire(
        wire_id="wire_1",
        from_node="stim_gen_1",
        from_port="stimulus",
        to_node="bio_snn_1",
        to_port="stimulus"
    )
    
    # Save project
    project_path = studio_root / "examples" / "demo_biosnn_network.omniforgeproj"
    project.save(str(project_path))
    
    print(f"✅ Project saved to: {project_path}")
    return project


def run_demo_simulation():
    """Run a live simulation of the BioSNN network"""
    print("\n🚀 Starting OmniForge Studio Simulation...")
    print("=" * 60)
    
    # Create execution engine
    engine = ExecutionEngine(max_workers=4, tick_rate=60)
    
    # Create node instances
    stim_gen = StimGeneratorNode(mode="periodic", amplitude=2.0)
    bio_snn = BioSNNNode(n_neurons=30)
    
    # Add to engine
    stim_id = engine.add_node("stim_gen_1", "StimGenerator", stim_gen)
    snn_id = engine.add_node("bio_snn_1", "BioSNN", bio_snn)
    
    # Connect nodes
    wire_id = engine.add_wire(stim_id, "stimulus", snn_id, "stimulus")
    
    print(f"📊 Graph Setup:")
    print(f"   - Nodes: {len(engine.nodes)}")
    print(f"   - Wires: {len(engine.wires)}")
    print(f"   - Tick Rate: {engine.tick_rate} Hz")
    print(f"   - Max Workers: {engine.max_workers}")
    
    # Add telemetry callback
    def on_tick_callback(tick, state):
        if tick % 60 == 0:  # Print every 60 ticks (1 second)
            snn_node = state['nodes'].get('bio_snn_1', {})
            outputs = snn_node.get('outputs', {})
            spike_count = outputs.get('spike_count', 0)
            activity = outputs.get('network_activity', 0.0)
            
            print(f"[Tick {tick:4d}] Spikes: {spike_count:2d} | Activity: {activity:.3f} | "
                  f"Executions: {state['telemetry']['total_executions']}")
    
    engine.on_tick_callbacks.append(on_tick_callback)
    
    # Start simulation
    engine.start()
    print("\n▶️  Simulation running... (Press Ctrl+C to stop)\n")
    
    try:
        # Run for 5 seconds
        time.sleep(5.0)
        
        # Pause
        print("\n⏸️  Pausing simulation...")
        engine.pause()
        time.sleep(1.0)
        
        # Resume
        print("▶️  Resuming simulation...\n")
        engine.resume()
        time.sleep(3.0)
        
        # Test rewind
        print("\n⏪ Rewinding to tick 120...")
        engine.pause()
        engine.rewind_to_tick(120)
        print(f"   Current tick after rewind: {engine.current_tick}")
        engine.resume()
        time.sleep(2.0)
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    # Stop simulation
    engine.stop()
    
    # Print final statistics
    state = engine.get_state()
    telemetry = state['telemetry']
    
    print("\n" + "=" * 60)
    print("📊 Final Statistics:")
    print(f"   - Total Ticks: {telemetry['total_ticks']}")
    print(f"   - Total Executions: {telemetry['total_executions']}")
    print(f"   - Average Tick Time: {telemetry['average_tick_time']*1000:.2f} ms")
    print(f"   - Snapshots Taken: {len(engine.snapshots)}")
    
    if telemetry['errors']:
        print(f"\n⚠️  Errors encountered: {len(telemetry['errors'])}")
        for error in telemetry['errors'][:3]:  # Show first 3 errors
            print(f"   - {error['node_id']}: {error['error']}")
    else:
        print("\n✅ No errors encountered!")
    
    print("\n🌀 Simulation complete.")


def demonstrate_node_scanning():
    """Demonstrate AST scanning capabilities"""
    print("\n🔍 Demonstrating Node Auto-Scanning...")
    print("=" * 60)
    
    from core.ast_scanner import ASTNodeScanner
    
    scanner = ASTNodeScanner()
    
    # Scan the bio_snn_node file
    node_file = studio_root / "nodes" / "bio_snn_node.py"
    nodes = scanner.scan_file(str(node_file))
    
    print(f"\n📄 Scanned file: {node_file.name}")
    print(f"   Found {len(nodes)} node(s)\n")
    
    for node in nodes:
        print(f"   🔹 Node: {node.name}")
        print(f"      Category: {node.category}")
        print(f"      Description: {node.description[:60]}...")
        print(f"      Inputs: {len(node.inputs)}")
        print(f"      Outputs: {len(node.outputs)}")
        print(f"      Operations: {', '.join(node.operations)}")
        print()


def demonstrate_project_serialization():
    """Demonstrate project save/load"""
    print("\n💾 Demonstrating Project Serialization...")
    print("=" * 60)
    
    # Create a project
    project1 = OmniForgeProject()
    project1.metadata['author'] = "Test User"
    project1.metadata['description'] = "Serialization test"
    
    project1.add_node("node_1", "TestNode", params={'value': 42}, position=(50, 50))
    project1.add_node("node_2", "TestNode", params={'value': 99}, position=(200, 50))
    project1.add_wire("wire_1", "node_1", "out", "node_2", "in")
    
    # Save
    test_path = studio_root / "examples" / "test_project.omniforgeproj"
    project1.save(str(test_path))
    print(f"✅ Saved project to: {test_path}")
    
    # Load
    project2 = OmniForgeProject.load(str(test_path))
    print(f"✅ Loaded project from: {test_path}")
    
    # Verify
    summary = project2.export_summary()
    print(f"\n📊 Project Summary:")
    print(f"   - Author: {summary['metadata']['author']}")
    print(f"   - Nodes: {summary['node_count']}")
    print(f"   - Wires: {summary['wire_count']}")
    print(f"   - Checksum: {project2.calculate_checksum()[:16]}...")
    
    # Create diff
    print("\n🔄 Creating version diff...")
    project1.add_node("node_3", "NewNode", position=(400, 50))
    diff = project1.create_version_diff(project2)
    print(f"   - Nodes added: {len(diff['nodes_added'])}")
    print(f"   - Nodes removed: {len(diff['nodes_removed'])}")


def main():
    """Main demo entry point"""
    print("=" * 60)
    print("🌀 OmniForge Studio v1.1 — Demonstration Suite")
    print("   Visual Runtime Architecture Standard 1.1 (VRAS-1.1)")
    print("=" * 60)
    
    # 1. Create demo project
    project = create_demo_project()
    
    # 2. Demonstrate node scanning
    demonstrate_node_scanning()
    
    # 3. Demonstrate project serialization
    demonstrate_project_serialization()
    
    # 4. Run live simulation
    print("\n" + "=" * 60)
    response = input("Run live simulation? (y/n): ")
    if response.lower() in ['y', 'yes']:
        run_demo_simulation()
    else:
        print("⏭️  Skipping simulation")
    
    print("\n" + "=" * 60)
    print("🎉 Demo complete! Check the examples/ directory for saved files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
