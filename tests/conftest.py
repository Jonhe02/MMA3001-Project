"""Test setup for MMA3001: import scripts from the repository's src directory."""

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if not SRC_DIR.is_dir():
    raise RuntimeError("Expected an src/ folder beside tests/. Copy tests/ to the project root.")

sys.path.insert(0, str(SRC_DIR))
