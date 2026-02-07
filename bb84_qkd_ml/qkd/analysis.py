"""QKD analysis utilities."""

from __future__ import annotations

from typing import List, Sequence

Bit = int
Basis = int


def compute_qber(alice_key: Sequence[Bit], bob_key: Sequence[Bit]) -> float:
    """Compute the quantum bit error rate (QBER)."""
    if not alice_key:
        return 0.0
    errors = sum(a != b for a, b in zip(alice_key, bob_key))
    return errors / len(alice_key)


def extract_features(
    alice_key: Sequence[Bit],
    bob_key: Sequence[Bit],
    alice_bases: Sequence[Basis],
    bob_bases: Sequence[Basis],
) -> List[float]:
    """Extract features for ML detection."""
    if not alice_bases:
        return [0.0, 0.0, float(len(alice_key))]

    qber = compute_qber(alice_key, bob_key)
    basis_match_rate = sum(a == b for a, b in zip(alice_bases, bob_bases)) / len(
        alice_bases
    )
    key_length = len(alice_key)

    return [qber, basis_match_rate, float(key_length)]
