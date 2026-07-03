"""Application entry for source runs and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _ensure_on_path(*dirs: Path) -> None:
    for directory in dirs:
        path_str = str(directory)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def main() -> None:
    view_dir = ROOT / "View"
    algo_dir = ROOT / "Algorithm"
    ctrl_dir = ROOT / "Controller"
    _ensure_on_path(ROOT, view_dir, algo_dir, ctrl_dir)

    from handle_event import run_app

    run_app()

if __name__ == "__main__":
    main()
