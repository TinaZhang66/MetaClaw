#!/usr/bin/env python3
"""
Run the paper-aligned benchmark suite sequentially.

This avoids cross-run interference and upstream overload from parallel runs.
Default order:
  1. static skills
  2. skills only
  3. RL
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

RUNNERS: list[tuple[str, Path]] = [
    ("static_skills", SCRIPT_DIR / "paper_static_skills_run.py"),
    ("skills_only", SCRIPT_DIR / "paper_skills_only_run.py"),
    ("rl", SCRIPT_DIR / "paper_rl_run.py"),
]


def _format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _validate_env() -> None:
    missing = []
    if not os.environ.get("TINKER_KEY"):
        missing.append("TINKER_KEY")
    if not os.environ.get("METACLAW_ROOT"):
        missing.append("METACLAW_ROOT")
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


def main() -> int:
    _validate_env()

    print("Sequential paper benchmark suite")
    print("Order: static_skills -> skills_only -> rl")
    print()

    overall_start = time.time()
    for name, script_path in RUNNERS:
        start = time.time()
        print(f"[suite] starting {name}: {script_path.name}")
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(SCRIPT_DIR.parent))
        elapsed = time.time() - start
        if result.returncode != 0:
            print(
                f"[suite] {name} failed after {_format_elapsed(elapsed)} "
                f"(exit={result.returncode})"
            )
            return result.returncode
        print(f"[suite] {name} finished in {_format_elapsed(elapsed)}")
        print()

    total_elapsed = time.time() - overall_start
    print(f"[suite] all runs finished in {_format_elapsed(total_elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
