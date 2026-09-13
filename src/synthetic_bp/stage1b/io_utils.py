"""
Stage 1B artifact I/O utilities.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.synthetic_bp.path_utils import ensure_dir, ensure_parent_dir

def save_table(df: pd.DataFrame, path: str) -> None:
    '''
    Save DataFrame as CSV or Parquet

    Args:
        df:
            DataFrame to save.
        path:
            Destination path.

    Raises:
        ValueError:
            If file extension is unsupported.
    '''
    output_path = Path(path)
    ensure_parent_dir(output_path)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index= False)
        return
    
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
        return
    
    raise ValueError(f"Unsupported output extension: {output_path}")

def ensure_plot_dir(path: str) -> None:
    """
    Create plot directory if missing.

    Args:
        path:
            Plot directory path.
    """
    ensure_dir(path)
    