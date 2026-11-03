# Copyright (c) 2026-2027 zh
"""
导出包含 MoE 编码器和归一化的完整策略。

用法：
    python scripts/supermimic/export.py --checkpoint-file /path/model_1000.pt --output-dir /path/export
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tasks.supermimic.cli import export_main


if __name__ == "__main__":
    export_main()
