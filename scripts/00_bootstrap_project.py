"""
Bootstrap AL-QNN project directories.

Run:
    python scripts/00_bootstrap_project.py
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_DIRS = [
    "configs",
    "src/synthetic_bp",
    "scripts",
    "outputs",
    "outputs/resolved_configs",
    "outputs/logs",
    "outputs/stage1a",
    "outputs/stage1b",
    "outputs/stage1c",
    "outputs/training",
]

REQUIRED_INIT_FILES = [
    "src/synthetic_bp/__init__.py"
]

def main() -> None:
    '''Create project directories and __init__.py files.'''
    root = Path(__file__).resolve().parents[1]

    for directory in REQUIRED_DIRS:
        (root / directory).mkdir(parents=True, exist_ok= True)
    
    for file in REQUIRED_INIT_FILES:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok= True)
        path.touch(exist_ok=True)
    
    print("Stage 0 bootstrap completed")

if __name__ == "__main__":
    main()