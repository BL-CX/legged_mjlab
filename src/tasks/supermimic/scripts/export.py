# Copyright (c) 2026-2027 zh
"""Export a trained policy: python -m src.tasks.supermimic.scripts.export."""

from pathlib import Path

import tyro

from ..eval.export import export_checkpoint


def main(checkpoint_file: Path, output_dir: Path):
    """Write the complete history policy as TorchScript, ONNX, and JSON metadata."""
    for name, path in export_checkpoint(checkpoint_file, output_dir).items():
        print(f"[INFO] Exported {name}: {path}")


if __name__ == "__main__":
    tyro.cli(main)
