"""
test_extractors.py
Pytest suite for classical extractors.
"""
import pytest
import numpy as np

from extractors import von_neumann, elias, universal_hash
from utils      import compute_bias

@pytest.fixture
def alternating_bits():
    return [0,1] * 50

@pytest.fixture
def biased_bits():
    return [0]*800 + [1]*200

def test_von_neumann_reduction(alternating_bits):
    out = von_neumann(alternating_bits)
    assert out == [0] * 50  # every [0,1] → 0

def test_universal_hash_length(alternating_bits):
    out = universal_hash(alternating_bits, seed="abc")
    assert len(out) == 256
    assert all(bit in (0,1) for bit in out)

def test_elias_basic_properties(biased_bits, alternating_bits):
    # Elias should output only bits and shorter lists
    for inp in (biased_bits, alternating_bits):
        out = elias(inp)
        assert all(bit in (0,1) for bit in out)
        assert 0 < len(out) < len(inp)

def test_compute_bias_range():
    # Bias is always between 0 and 0.5
    for bits in ([0,1]*10, [0]*10, [1]*10):
        b = compute_bias(bits)
        assert 0.0 <= b <= 0.5
