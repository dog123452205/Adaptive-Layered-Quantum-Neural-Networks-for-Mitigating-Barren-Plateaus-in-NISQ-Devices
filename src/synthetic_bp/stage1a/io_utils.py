"""
Stage 1A artifact I/O utilities.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.synthetic_bp.path_utils import ensure_dir, ensure_parent_dir

def save_table(df: pd.DataFrame, path: str) -> None:
    '''
    Save a DataFrame as CSV or Parquet.

    Args:
        df:
            DataFrame to save.
        path:
            Output path ending with .csv or .parquet.

    Raises:
        ValueError:
            If file extension is unsupported.
    '''
    output_path = Path(path)
    ensure_parent_dir(output_path)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path, index= False)
        return
    
    raise ValueError(f"Unsupported table extension: {output_path}")

def ensure_plot_dir(path: str) -> None:
    '''
    Create plot directory.

    Args:  
        path:
            Plot directory
    '''
    ensure_dir(path)

