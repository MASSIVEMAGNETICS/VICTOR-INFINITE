#!/usr/bin/env python3
"""
FILE: omniforge_studio/omniforge.py
VERSION: v1.1.0-VRAS
NAME: OmniForge Studio Launcher
PURPOSE: Main entry point for OmniForge Studio
AUTHOR: OmniForge Team / Massive Magnetics
"""

import sys
import argparse
from pathlib import Path


def print_banner():
    """Print OmniForge Studio banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        🌀  OmniForge Studio v1.1 — Ascension Forge Edition       ║
║                                                                   ║
║     "The Visual Reality Simulator for Sculpting Synthetic Gods"  ║
║                                                                   ║
║           Visual Runtime Architecture Standard 1.1 (VRAS-1.1)    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_demo():
    """Run the demo suite"""
    print("🚀 Launching Demo Suite...\n")
    from examples.demo_biosnn_network import main
    main()


def run_ui():
    """Launch the web UI"""
    import http.server
    import socketserver
    import webbrowser
    import threading
    
    PORT = 8080
    DIRECTORY = Path(__file__).parent / "frontend" / "public"
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    print(f"🌐 Starting OmniForge Studio Web UI on port {PORT}...")
    print(f"📂 Serving from: {DIRECTORY}")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"✅ Server ready at: {url}")
        print("   Press Ctrl+C to stop\n")
        
        # Open browser after a short delay
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down server...")


def scan_nodes(path):
    """Scan a Python file or directory for nodes"""
    from backend.core.ast_scanner import ASTNodeScanner
    
    scanner = ASTNodeScanner()
    path = Path(path)
    
    if path.is_file():
        print(f"🔍 Scanning file: {path}\n")
        nodes = scanner.scan_file(str(path))
        
        if not nodes:
            print("❌ No nodes found in file")
            return
        
        print(f"✅ Found {len(nodes)} node(s):\n")
        for node in nodes:
            print(f"  🔹 {node.name}")
            print(f"     Category: {node.category}")
            print(f"     Inputs: {len(node.inputs)}")
            print(f"     Outputs: {len(node.outputs)}")
            print(f"     Operations: {', '.join(node.operations)}")
            print()
    
    elif path.is_dir():
        print(f"🔍 Scanning directory: {path}\n")
        results = scanner.scan_directory(str(path))
        
        if not results:
            print("❌ No nodes found in directory")
            return
        
        print(f"✅ Found nodes in {len(results)} file(s):\n")
        for file_path, nodes in results.items():
            print(f"  📄 {Path(file_path).name}: {len(nodes)} node(s)")
    else:
        print(f"❌ Path not found: {path}")


def create_project(name, description="", author=""):
    """Create a new OmniForge project"""
    from backend.core.project_serializer import OmniForgeProject
    
    project = OmniForgeProject()
    project.metadata['author'] = author or "Unknown"
    project.metadata['description'] = description or f"OmniForge Project: {name}"
    
    filename = f"{name}.omniforgeproj"
    project.save(filename)
    
    print(f"✅ Created new project: {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="OmniForge Studio v1.1 - Visual AI System Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omniforge.py ui                     # Launch web UI
  omniforge.py demo                   # Run demo suite
  omniforge.py scan nodes/            # Scan directory for nodes
  omniforge.py scan bio_snn.py        # Scan specific file
  omniforge.py create MyProject       # Create new project
        """
    )
    
    parser.add_argument(
        'command',
        choices=['ui', 'demo', 'scan', 'create', 'version'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Additional arguments for the command'
    )
    
    parser.add_argument(
        '--author',
        help='Author name for new projects',
        default=""
    )
    
    parser.add_argument(
        '--description',
        help='Description for new projects',
        default=""
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'ui':
        run_ui()
    
    elif args.command == 'demo':
        run_demo()
    
    elif args.command == 'scan':
        if not args.args:
            print("❌ Error: Please provide a path to scan")
            print("   Usage: omniforge.py scan <path>")
            sys.exit(1)
        scan_nodes(args.args[0])
    
    elif args.command == 'create':
        if not args.args:
            print("❌ Error: Please provide a project name")
            print("   Usage: omniforge.py create <name>")
            sys.exit(1)
        create_project(args.args[0], args.description, args.author)
    
    elif args.command == 'version':
        print("OmniForge Studio v1.1.0-VRAS")
        print("Visual Runtime Architecture Standard 1.1")
        print("Copyright © 2025 Massive Magnetics / Ethica AI / BHeard Network")


if __name__ == "__main__":
    main()
