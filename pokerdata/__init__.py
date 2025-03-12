"""
PokerData: A Python package for processing poker hand histories
"""

from .parser import load_data, build_dataset
from .anonymizer import anonymize_dataset

__version__ = '1.0.0'
