"""
metrics.py

Bias and (optionally) entropy helpers.
"""
import numpy as np
from typing import List

def compute_bias(bits: List[int]) -> float:
    """
    Return absolute bias: |p(1) - 0.5|.
    """
    return abs(np.mean(bits) - 0.5)