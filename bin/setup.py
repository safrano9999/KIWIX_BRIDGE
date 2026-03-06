#!/usr/bin/env python3
"""
Setup for KIWIX_BRIDGE — creates local venv and installs dependencies.

Usage:
    python3 bin/setup.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV        = PROJECT_DIR / "venv"


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    if not VENV.exists():
        print(f"Creating venv at {VENV} ...")
        run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print(f"Using existing venv: {VENV}")

    pip = VENV / "bin" / "pip"
    run([str(pip), "install", "--upgrade", "pip", "-q"])

    req = PROJECT_DIR / "requirements.txt"
    print(f"Installing {req} ...")
    run([str(pip), "install", "-r", str(req)])

    print(f"\nDone.")
    print(f"Start: python web.py")


if __name__ == "__main__":
    main()
