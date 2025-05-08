import math
from typing import List

def elias(bits: List[int]) -> List[int]:
    """
    Corrected Elias–Wagner extractor:
       Let m = len(bits). While m > 1:
       k = floor(log2 m), t = 2**(k+1) - m.
      Read the first k+1 bits and form integer y.
      If y < t*2:  (i.e. y < 2*t)
            Drop the first k bits (no output).
         Else:
           Compute z = y - 2*t  (range [0 .. 2**(k+1)-2*t -1]).
           Output the lower k bits of z.
           Drop the first k+1 bits.
      5.   Update m = len(bits) and repeat.
    """
    out: List[int] = []
    S = bits[:]
    while len(S) > 1:
        m = len(S)
        k = math.floor(math.log2(m))
        # Number of values we'd have to discard if we just used k bits
        t = (1 << (k+1)) - m
        
        # Need k+1 bits for the test
        if len(S) < k+1:
            break
        
        # Read k+1 bits as integer y
        y = 0
        for b in S[:k+1]:
            y = (y << 1) | b
        
        if y < 2*t:
            # reject case: drop first k bits only
            S = S[k:]
        else:
            # accept case: produce output
            z = y - 2*t
            # take lower k bits of z
            for shift in range(k-1, -1, -1):
                out.append((z >> shift) & 1)
            # drop k+1 bits
            S = S[k+1:]
    return out


