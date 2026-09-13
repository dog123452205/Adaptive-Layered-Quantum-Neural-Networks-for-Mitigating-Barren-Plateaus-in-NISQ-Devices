"""
DataFrame and artifact schema validators.

These validators inspect tabular/scalar artifacts only.
They do not validate quantum states.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

def validate_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    """
    Validate required columns.

    Args:
        df: DataFrame to validate.
        required: Required column names.
        table_name: Human-readable table name.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing columns: {missing}")
    
    if df.empty:
        raise ValueError(f"{table_name} is empty. Cannot validate artifacts.")

def validate_probability_column(df: pd.DataFrame, column: str, table_name: str) -> None:
    """
    Validate that a column lies in [0, 1].

    Args:
        df: DataFrame.
        column: Column name.
        table_name: Human-readable table name.
    """
    if column not in df.columns:
        raise KeyError(f"{table_name} missing column: {column} for probability validation")
    
    values = df[column].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{table_name}.{column} contains NaN or Inf values.")
    
    if ((df[column] < 0) | (df[column] > 1)).any():
        raise ValueError(f"{table_name}.{column} must be in [0,1]")

def validate_finite_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    '''
    Validate that numeric columns contain finite values.

    Args:
        df: DataFrame
        columns: Numeric columns.
        table_name: Human-readable table name.
    '''
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"{table_name} missing expected column for finiteness check: '{column}'")

        values = df[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{table_name}.{column} contains NaN or Inf")
    
def validate_non_negative_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    table_name: str
) -> None:
    """
    Validate that columns are non-negative.

    Args:
        df: DataFrame.
        columns: Column names.
        table_name: Human-readable table name.
    """
    for column in columns:
        if (df[column] < 0).any():
            raise ValueError(f"{table_name}.{column} must be non-negative.")

