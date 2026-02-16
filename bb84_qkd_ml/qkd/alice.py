"""Alice (sender) utilities for BB84 QKD."""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

from utils.constants import BASES

Bit = int
Basis = int
State = Tuple[Bit, Basis]


def generate_bits(n: int, rng: random.Random | None = None) -> List[Bit]:
    """Generate a list of n random bits."""
    rng = rng or random
    return [rng.getrandbits(1) for _ in range(n)]


def generate_bases(n: int, rng: random.Random | None = None) -> List[Basis]:
    """Generate a list of n random bases."""
    rng = rng or random
    return [rng.choice(BASES) for _ in range(n)]


def encode(bits: Sequence[Bit], bases: Sequence[Basis]) -> List[State]:
    """Encode bits with bases as (bit, basis) tuples."""
    return list(zip(bits, bases))
