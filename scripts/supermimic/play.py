# Copyright (c) 2026-2027 zh
"""
播放 Go2 SuperMimic 策略或检查环境。

用法：
    python scripts/supermimic/play.py go2-supermimic-flat --agent zero --viewer viser
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tasks.supermimic.cli import play_main


if __name__ == "__main__":
    play_main()
