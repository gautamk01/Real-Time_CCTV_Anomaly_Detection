
"""Application entrypoint for the violence detection system."""

import sys

from app.orchestrator import run_application
from config import Config


def print_banner() -> None:
    """Print the startup banner."""

    print("\n" + "=" * 60)
    print("REAL-TIME VIOLENCE DETECTION SYSTEM")
    print("   Distributed GPU Inference Architecture")
    print("=" * 60)


def main() -> None:
    """Main entry point for the distributed multi-camera runtime."""

    print_banner()

    if not Config.validate():
        sys.exit(1)

    Config.print_info()
    run_application(Config)


if __name__ == "__main__":
    main()
