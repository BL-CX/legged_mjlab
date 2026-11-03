# Copyright (c) 2026-2027 zh
"""
训练 Go2 SuperMimic。

用法：
    python scripts/supermimic/train.py go2-supermimic-flat --device cuda:0
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tasks.supermimic.cli import train_main


if __name__ == "__main__":
    train_main()
