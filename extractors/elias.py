"""
elias.py
Elias–Wagner randomness extractor (variable block size

"""
import math
from typing import List

def elias(bits: List[int]) -> List[int]:
    """
    Elias extractor (Ann. Math. Stat., 1972):
      1. While |S| > 1, let n = |S|, k = floor(log2 n), L = 2**k.
      2. Read first k bits as integer x.
      3. If x < 2**k - (n - L): emit x in k-bit form.
      4. Remove those k bits from S and repeat.
    """
    out: List[int] = []
    S = bits[:]  
    while True:
        n = len(S)
        if n <= 1:
            break
        k = math.floor(math.log2(n))
        L = 1 << k
        discard = n - L
        # read k bits -> x
        x = 0
        for bit in S[:k]:
            x = (x << 1) | bit
        S = S[k:]
        if x < (L - discard):
            # emit k-bit big-endian of x
            for shift in range(k-1, -1, -1):
                out.append((x >> shift) & 1)
    return out