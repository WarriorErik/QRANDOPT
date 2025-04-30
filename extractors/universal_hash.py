"""
universal_hash.py

Auto-generated stub.
"""
import hashlib
from typing import List

def universal_hash(bits: List[int], seed: str = "seed") -> List[int]:
    """
    Pack bits into bytes, hash with SHA256(seed || data),
    and unpack the 256-bit digest.
    """
    # pack into bytes
    data = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i+j < len(bits) and bits[i+j] == 1:
                byte |= 1 << (7-j)
        data.append(byte)
    # hash
    digest = hashlib.sha256(seed.encode() + bytes(data)).digest()
    # unpack
    out: List[int] = []
    for byte in digest:
        for bit in range(8):
            out.append((byte >> (7-bit)) & 1)
    return out