"""A2A Agent Launcher — Start Edge, Cloud, and RAG agent services.

Usage:
    # Start all agents (development — one command, multiple processes)
    python -m a2a.launcher --all

    # Start individual agents (production — separate machines)
    python -m a2a.launcher --edge --port 8001
    python -m a2a.launcher --cloud --port 8002
    python -m a2a.launcher --rag --port 8003
"""

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def start_edge(port: int, model_id: str, device: str, quant_mode: str):
    """Start the Edge Vision agent service."""
    import uvicorn
    from a2a.edge_agent import app, init_edge_vision

    print(f"\n🔧 [LAUNCHER] Starting Edge Agent on port {port}...")
    init_edge_vision(model_id=model_id, device=device, quant_mode=quant_mode)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def start_cloud(port: int, api_key: str, model_id: str):
    """Start the Cloud AI agent service."""
    import uvicorn
    from a2a.cloud_agent import app, init_cloud_ai

    print(f"\n🔧 [LAUNCHER] Starting Cloud Agent on port {port}...")
    init_cloud_ai(api_key=api_key, model_id=model_id)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def start_rag(port: int, db_dir: str, embedding_model: str):
    """Start the RAG agent service."""
    import uvicorn
    from rag.rag_agent import app, init_rag

    print(f"\n🔧 [LAUNCHER] Starting RAG Agent on port {port}...")
    init_rag(persist_dir=db_dir, embedding_model=embedding_model)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description="Launch A2A agent services")
    parser.add_argument("--all", action="store_true",
                        help="Start all agents (Edge, Cloud, RAG)")
    parser.add_argument("--edge", action="store_true",
                        help="Start Edge Vision agent")
    parser.add_argument("--cloud", action="store_true",
                        help="Start Cloud AI agent")
    parser.add_argument("--rag", action="store_true",
                        help="Start RAG agent")
    parser.add_argument("--port", type=int, default=None,
                        help="Port for single-agent mode")
    parser.add_argument("--edge-port", type=int, default=8001,
                        help="Edge agent port (default: 8001)")
    parser.add_argument("--cloud-port", type=int, default=8002,
                        help="Cloud agent port (default: 8002)")
    parser.add_argument("--rag-port", type=int, default=8003,
                        help="RAG agent port (default: 8003)")

    args = parser.parse_args()

    # Load config
    from config import Config

    if args.all:
        print("\n" + "=" * 60)
        print("🚀 A2A AGENT LAUNCHER — Starting all agents")
        print("=" * 60)

        processes = []

        # Edge agent
        p_edge = multiprocessing.Process(
            target=start_edge,
            args=(
                args.edge_port,
                Config.EDGE_MODEL_ID,
                Config.DEVICE,
                Config.EDGE_QUANT_MODE,
            ),
            daemon=True,
        )
        processes.append(("Edge", p_edge, args.edge_port))

        # Cloud agent
        p_cloud = multiprocessing.Process(
            target=start_cloud,
            args=(args.cloud_port, Config.GROQ_API_KEY, Config.CLOUD_MODEL_ID),
            daemon=True,
        )
        processes.append(("Cloud", p_cloud, args.cloud_port))

        # RAG agent
        rag_db_dir = str(getattr(Config, "RAG_DB_DIR", Config.BASE_DIR / "rag_db"))
        rag_model = getattr(Config, "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        p_rag = multiprocessing.Process(
            target=start_rag,
            args=(args.rag_port, rag_db_dir, rag_model),
            daemon=True,
        )
        processes.append(("RAG", p_rag, args.rag_port))

        # Start all
        for name, proc, port in processes:
            proc.start()
            print(f"   ✅ {name} agent starting on port {port} (PID {proc.pid})")

        print(f"\n📡 Agent endpoints:")
        print(f"   Edge:  http://localhost:{args.edge_port}")
        print(f"   Cloud: http://localhost:{args.cloud_port}")
        print(f"   RAG:   http://localhost:{args.rag_port}")
        print(f"\nPress Ctrl+C to stop all agents.\n")

        try:
            # Wait for processes
            for name, proc, _ in processes:
                proc.join()
        except KeyboardInterrupt:
            print("\n⚠️  Shutting down agents...")
            for name, proc, _ in processes:
                proc.terminate()
            for name, proc, _ in processes:
                proc.join(timeout=5)
            print("✅ All agents stopped.")

    elif args.edge:
        port = args.port or args.edge_port
        start_edge(port, Config.EDGE_MODEL_ID, Config.DEVICE, Config.EDGE_QUANT_MODE)

    elif args.cloud:
        port = args.port or args.cloud_port
        start_cloud(port, Config.GROQ_API_KEY, Config.CLOUD_MODEL_ID)

    elif args.rag:
        port = args.port or args.rag_port
        rag_db_dir = str(getattr(Config, "RAG_DB_DIR", Config.BASE_DIR / "rag_db"))
        rag_model = getattr(Config, "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        start_rag(port, rag_db_dir, rag_model)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m a2a.launcher --all")
        print("  python -m a2a.launcher --edge --port 8001")
        print("  python -m a2a.launcher --cloud --port 8002")
        print("  python -m a2a.launcher --rag --port 8003")


if __name__ == "__main__":
    main()
