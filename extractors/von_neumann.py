"""
von_neumann.py
Classical von Neumann randomness extractor

"""
from typing import List

def von_neumann(bits: List[int]) -> List[int]:
    """
    Pair up bits and emit 0 for [0,1], 1 for [1,0].
    Discard [0,0] and [1,1].
    """
    out: List[int] = []
    for i in range(0, len(bits) - 1, 2):
        a, b = bits[i], bits[i+1]
        if a != b:
            out.append(a)
    return out