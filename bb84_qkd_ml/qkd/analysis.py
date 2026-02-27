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


<<<<<<< HEAD
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
=======
def compute_error_variance(alice_key: Sequence[Bit], bob_key: Sequence[Bit]) -> float:
    """Variance of error indicator sequence as a stability measure."""
    if not alice_key:
        return 0.0
    indicators = [1.0 if a != b else 0.0 for a, b in zip(alice_key, bob_key)]
    mean = sum(indicators) / len(indicators)
    return sum((x - mean) ** 2 for x in indicators) / len(indicators)


def compute_burst_error_frequency(alice_key: Sequence[Bit], bob_key: Sequence[Bit]) -> float:
    """Frequency of consecutive error bursts in sifted key."""
    if len(alice_key) < 2:
        return 0.0
    indicators = [1 if a != b else 0 for a, b in zip(alice_key, bob_key)]
    bursts = sum(
        1 for i in range(1, len(indicators)) if indicators[i] == 1 and indicators[i - 1] == 1
    )
    return bursts / (len(indicators) - 1)


def extract_features(
    alice_key: Sequence[Bit],
    bob_key: Sequence[Bit],
    loss_rate: float,
    intercept_ratio: float,
    channel_mu_ch: float,
) -> List[float]:
    """Extract required ML features for eavesdropping detection."""
    qber = compute_qber(alice_key, bob_key)
    error_variance = compute_error_variance(alice_key, bob_key)
    burst_error_frequency = compute_burst_error_frequency(alice_key, bob_key)

    return [
        qber,
        loss_rate,
        error_variance,
        burst_error_frequency,
        intercept_ratio,
        channel_mu_ch,
    ]
>>>>>>> codex/create-project-folder-structure-and-files
