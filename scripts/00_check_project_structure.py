"""
Check Stage 0 project structure.

Run:
    python scripts/00_check_project_structure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_FILES = [
    "src/synthetic_bp/constants.py",
    "src/synthetic_bp/path_utils.py",
    "src/synthetic_bp/config_loader.py",
    "src/synthetic_bp/config_resolver.py",
    "src/synthetic_bp/resource_guards.py",
    "src/synthetic_bp/config_validation.py",
    "src/synthetic_bp/schema.py",
    "src/synthetic_bp/schema_validation.py",
    "scripts/00_validate_and_resolve_config.py",
]

REQUIRED_DIRS = [
    "configs",
    "src/synthetic_bp",
    "outputs/resolved_configs",
    "outputs/logs",
]

def main() -> None:
    '''Validate expected project structure'''
    root = Path(__file__).resolve().parents[1]

    missing_files = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    missing_dirs = [path for path in REQUIRED_DIRS if not (root / path).is_dir()]

    if missing_dirs:
        print("Missing directories:")
        for item in missing_dirs:
            print(f"    - {item}")
        
    if missing_files:
        print("Missing files:")
        for item in missing_files:
            print(f"    - {item}")
    
    if missing_files or missing_files:
        sys.exit(1)
    
    print("Project structure check passed")

if __name__ == "__main__":
    main()