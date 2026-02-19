"""Bob (receiver) utilities for BB84 QKD."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from utils.constants import BASES

Bit = int
Basis = int
State = Tuple[Bit, Basis]


def generate_bases(n: int, rng: random.Random | None = None) -> List[Basis]:
    """Generate a list of n random bases."""
    rng = rng or random
    return [rng.choice(BASES) for _ in range(n)]


def measure(states: Sequence[State], bob_bases: Sequence[Basis], rng: random.Random | None = None) -> List[Bit]:
    """Measure incoming states with Bob's bases."""
    rng = rng or random
    result = []

    for (bit, sender_basis), bob_basis in zip(states, bob_bases):
        if sender_basis == bob_basis:
            result.append(bit)
        else:
            result.append(rng.getrandbits(1))

    return result
