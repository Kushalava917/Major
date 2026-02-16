"""Classical basis reconciliation for BB84 QKD."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Bit = int
Basis = int


def sift_keys(
    alice_bits: Sequence[Bit],
    bob_bits: Sequence[Bit],
    alice_bases: Sequence[Basis],
    bob_bases: Sequence[Basis],
) -> Tuple[List[Bit], List[Bit]]:
    """Sift the key by keeping positions where bases match."""
    alice_key = []
    bob_key = []

    for i, (a_basis, b_basis) in enumerate(zip(alice_bases, bob_bases)):
        if a_basis == b_basis:
            alice_key.append(alice_bits[i])
            bob_key.append(bob_bits[i])

    return alice_key, bob_key
