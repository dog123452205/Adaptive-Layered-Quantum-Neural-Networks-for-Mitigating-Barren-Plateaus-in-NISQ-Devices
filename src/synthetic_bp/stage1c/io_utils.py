"""
Stage 1C artifact I/O utilities.

This module loads Stage 1A/1B artifacts and writes Stage 1C calibration
artifacts.

Supported tabular formats:
    - .csv
    - .parquet

Supported JSON output:
    - .json

Stage 1C intentionally does not import PennyLane and does not instantiate
quantum circuits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.synthetic_bp.path_utils import ensure_parent_dir

def load_table(path: str | Path, required: bool = True) -> pd.DataFrame:
    '''
    Load a tabular artifact from CSV or Parquet.

    Args:
        path:
            Input file path.
        required:
            Whether missing file should raise an error.

    Returns:
        Loaded DataFrame. If required=False and the file does not exist,
        returns an empty DataFrame.

    Raises:
        FileNotFoundError:
            If required=True and the file does not exist.
        ValueError:
            If file extension is unsupported.
    '''
    table_path = Path(path)

    if not table_path.exists():
        if required:
            raise FileNotFoundError(f"Required artifact not found: {table_path}")
        return pd.DataFrame()
    
    if table_path.suffix == ".csv": return pd.read_csv(table_path)

    if table_path.suffix == '.parquet': return pd.read_parquet(table_path)

    raise ValueError(f"Unsupported table format: {table_path}")

def save_table(df: pd.DataFrame, path: str | Path) -> None:
    '''
    Save DataFrame as CSV or Parquet.

    Args:
        df:
            DataFrame to save.
        path:
            Output path.

    Raises:
        ValueError:
            If output extension is unsupported.
    '''
    output_path = Path(path)
    ensure_parent_dir(output_path)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index= False)
        return
    
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
        return
    
    raise ValueError(f"Unsupported output table format: {output_path}")

def write_json(data: dict[str, Any], path: str | Path) -> None:
    """
    Write dictionary as pretty JSON.

    Args:
        data:
            JSON-serializable dictionary.
        path:
            Output path.
    """
    output_path = Path(path)
    ensure_parent_dir(output_path)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent= 2, ensure_ascii= False)

def dataframe_empty_or_missing(df: pd.DataFrame) -> bool:
    """
    Check whether a DataFrame is empty.

    Args:
        df:
            DataFrame.

    Returns:
        True if empty, otherwise False.
    """
    return df is None or df.empty