"""Eve (attacker) intercept-resend simulation for BB84 QKD."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from utils.constants import BASES

Bit = int
Basis = int
State = Tuple[Bit, Basis]


def intercept_resend(states: Sequence[State], rng: random.Random | None = None) -> List[State]:
    """Intercept each state, measure in a random basis, and resend."""
    rng = rng or random
    attacked = []

    for bit, basis in states:
        eve_basis = rng.choice(BASES)

        if eve_basis == basis:
            measured_bit = bit
        else:
            measured_bit = rng.getrandbits(1)

        attacked.append((measured_bit, eve_basis))

    return attacked
