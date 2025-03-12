"""
Anonymizer module for protecting player identities in poker datasets.
"""

import hashlib
import pandas as pd
from typing import List, Union, Optional


def hash_value(value: str, algorithm: str = 'md5') -> str:
    """
    Hash a value using the specified algorithm.
    
    Args:
        value: String value to hash
        algorithm: Hashing algorithm to use
        
    Returns:
        Hashed string
    """
    if pd.isna(value) or value == "":
        return ""
    
    if algorithm == 'md5':
        return hashlib.md5(str(value).encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(str(value).encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(str(value).encode()).hexdigest()
    else:
        raise ValueError(f"Unknown hashing algorithm: {algorithm}")


def anonymize_dataset(
    df: pd.DataFrame,
    columns_to_mask: List[str] = ['name'],
    algorithm: str = 'md5',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Anonymize sensitive columns in a poker dataset.
    
    Args:
        df: DataFrame to anonymize
        columns_to_mask: List of column names to anonymize
        algorithm: Hashing algorithm to use
        inplace: Whether to modify the DataFrame in-place
        
    Returns:
        Anonymized DataFrame
    """
    if not inplace:
        df = df.copy()
    
    for col in columns_to_mask:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: hash_value(x, algorithm))
        else:
            print(f"Warning: Column {col} not found in DataFrame")
    
    return df
