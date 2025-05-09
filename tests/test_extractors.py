"""
test_extractors.py
Pytest suite for classical extractors and Maurer–Wolf extractor.
"""
import pytest
import numpy as np

from extractors.von_neumann      import von_neumann
from extractors.elias            import elias
from extractors.universal_hash   import universal_hash
from extractors.maurer_wolf      import maurer_wolf_extractor
from utils                       import compute_bias

def bits_to_bytes(bits):
    """Pack list of 0/1 bits into a bytes object (MSB-first)."""
    pad = (-len(bits)) % 8
    bits = bits + [0] * pad
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i : i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)

def bytes_to_bits(b):
    """Unpack bytes into list of bits (MSB-first)."""
    bits = []
    for byte in b:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits

@pytest.fixture
def alternating_bits():
    return [0, 1] * 50

@pytest.fixture
def biased_bits():
    return [0] * 800 + [1] * 200

def test_von_neumann_reduction(alternating_bits):
    out = von_neumann(alternating_bits)
    assert out == [0] * 50

def test_universal_hash_length(alternating_bits):
    out = universal_hash(alternating_bits, seed="abc")
    assert len(out) == 256
    assert all(bit in (0, 1) for bit in out)

def test_elias_basic_properties(biased_bits, alternating_bits):
    for inp in (biased_bits, alternating_bits):
        out = elias(inp)
        assert all(bit in (0, 1) for bit in out)
        assert 0 < len(out) < len(inp)

def test_compute_bias_range():
    for bits in ([0, 1] * 10, [0] * 10, [1] * 10):
        b = compute_bias(bits)
        assert 0.0 <= b <= 0.5

def test_maurer_wolf_basic_properties(biased_bits):
    # Inline generation of a pseudo-random stream
    rng = np.random.default_rng(seed=123)
    random_bits = rng.choice([0, 1], size=1000, p=[0.6, 0.4]).tolist()

    for inp in (biased_bits, random_bits):
        raw_bytes = bits_to_bytes(inp)
        seed = b"fixed-seed"
        out_bytes = maurer_wolf_extractor(raw_bytes, seed, output_len=len(raw_bytes)//2)
        out = bytes_to_bits(out_bytes)

        # Output is a list of bits
        assert isinstance(out, list)
        assert all(bit in (0, 1) for bit in out)

        #  Output length is shorter than input
        assert 0 < len(out) < len(inp)

        # Deterministic for same seed
        out_bytes2 = maurer_wolf_extractor(raw_bytes, seed, output_len=len(raw_bytes)//2)
        out2 = bytes_to_bits(out_bytes2)
        assert out == out2

def test_maurer_wolf_multiple_seeds(biased_bits):
    raw_bytes = bits_to_bytes(biased_bits)
    out_a = maurer_wolf_extractor(raw_bytes, b"seed-A", output_len=len(raw_bytes)//2)
    out_b = maurer_wolf_extractor(raw_bytes, b"seed-B", output_len=len(raw_bytes)//2)
    # Different seeds should produce different outputs
    assert out_a != out_b
