
import hashlib
from typing import List

def maurer_wolf_extractor(raw_bits: bytes, seed: bytes, output_len: int) -> bytes:
    """
    Maurer–Wolf extractor via two-universal hashing (SHAKE256):
      - raw_bits: input bytes from bit‐list
      - seed: random bytes
      - output_len: desired output byte length
    """
    shake = hashlib.shake_256()
    shake.update(raw_bits)
    shake.update(seed)
    return shake.digest(output_len)
