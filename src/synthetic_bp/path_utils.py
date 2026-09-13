'''
Path and filesystem utilities for AL-QNN

These utilities are safe for Stage 0:
- create directories,
- validate paths,
- write files atomically.

They do not perform any quantum computation
'''

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

def project_root_from_file(file_path: PathLike, parents_up: int = 2) -> Path:
    """
    Infer project root from a file path.

    Args:
        file_path: Usually __file__ from a script.
        parents_up: Number of parent levels to move upward.

    Returns:
        Inferred project root path.
    """
    path = Path(file_path).resolve()
    for _ in range(parents_up):
        path = path.parent
    return path

def ensure_dir(path: PathLike) -> Path:
    '''
    Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object for the created/existing directory.
    '''
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def ensure_parent_dir(path: PathLike) -> Path:
    '''
    Create the parent directory of a file path.

    Args: 
        path: File path
    
    Returns:
        Parent directory path
    '''
    file_path = Path(path)
    parent = file_path.parent
    parent.mkdir(parents= True, exist_ok= True)
    return parent

def validate_non_empty_path(path: PathLike, field_name: str) -> None:
    '''
    Validate that a path-llke field is non-empty

    Args:
        path: Path-like value.
        field_name: Human-readable field name.
    
    Raises:
        ValueError: If path is empty
    '''
    if path is None or str(path).strip() == "":
        raise ValueError(f"{field_name} must be a non-empty path")

def atomic_write_text(path: PathLike, text: str, encoding: str = 'utf-8') -> None:
    '''
    Atomically write text to a file

    This reduces the chance of partially written config_resolved.yaml

    Args:
        path: Destination file path.
        text: Text content.
        encoding: File encoding
    '''
    output_path = Path(path)
    ensure_parent_dir(output_path)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete = False,
            dir = str(output_path.parent),
            encoding= encoding
        ) as tmp: 
            tmp.write(text)
            tmp_path = tmp.name
        
        os.replace(tmp_path, output_path)
        tmp_path = None

    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise

def require_file_exists(path: PathLike, field_name: str) -> None:
    '''
    Ensure that a file exists.
    
    Args: 
        path: File path
        field_name: Human-readable field name.

    Raises:
        FileNotFoundError: If file does not exist.
    '''
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"{field_name} does not exists: {file_path}")