"""Export a trained policy: python -m src.tasks.supermimic.scripts.export."""

# ruff: noqa: E402 -- support direct execution before importing repository packages.

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import tyro

from src.tasks.supermimic.eval.export import export_checkpoint


def main(checkpoint_file: Path, output_dir: Path):
    """Write the complete history policy as TorchScript, ONNX, and JSON metadata."""
    for name, path in export_checkpoint(checkpoint_file, output_dir).items():
        print(f"[INFO] Exported {name}: {path}")


if __name__ == "__main__":
    tyro.cli(main)
