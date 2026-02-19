"""Main controller for BB84 QKD simulation and ML pipeline."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from ml.dataset import init_csv, write_row
from ml.model import train_and_evaluate
from qkd import alice, analysis, bob, eve, reconciliation


@dataclass(frozen=True)
class SimulationConfig:
    bits: int
    runs: int
    seed: int | None


def _apply_channel(states, loss_rate: float, channel_mu_ch: float, rng: random.Random):
    """Apply channel loss and true channel noise."""
    channel_states = []
    for bit, basis in states:
        if rng.random() < loss_rate:
            bit = rng.getrandbits(1)
        elif rng.random() < channel_mu_ch:
            bit = 1 - bit
        channel_states.append((bit, basis))
    return channel_states


def _intercept_with_ratio(states, intercept_ratio: float, rng: random.Random):
    intercepted_states = []
    for state in states:
        if rng.random() < intercept_ratio:
            intercepted_states.extend(eve.intercept_resend([state], rng))
        else:
            intercepted_states.append(state)
    return intercepted_states


def run_simulation(config: SimulationConfig) -> None:
    if config.runs != 10000:
        raise ValueError("runs must be exactly 10000 to meet dataset requirements.")

    rng = random.Random(config.seed)
    init_csv(reset=True)

    labels = [0] * 5000 + [1] * 5000
    rng.shuffle(labels)

    for label in labels:
        eve_present = label == 1

        alice_bits = alice.generate_bits(config.bits, rng)
        alice_bases = alice.generate_bases(config.bits, rng)
        states = alice.encode(alice_bits, alice_bases)

        channel_mu_ch = rng.uniform(0.005, 0.06)
        loss_rate = rng.uniform(0.01, 0.2)
        intercept_ratio = rng.uniform(0.45, 1.0) if eve_present else 0.0

        if eve_present:
            states = _intercept_with_ratio(states, intercept_ratio, rng)

        states = _apply_channel(states, loss_rate, channel_mu_ch, rng)

        bob_bases = bob.generate_bases(config.bits, rng)
        bob_bits = bob.measure(states, bob_bases, rng)

        alice_key, bob_key = reconciliation.sift_keys(
            alice_bits, bob_bits, alice_bases, bob_bases
        )

        features = analysis.extract_features(
            alice_key,
            bob_key,
            loss_rate=loss_rate,
            intercept_ratio=intercept_ratio,
            channel_mu_ch=channel_mu_ch,
        )
        write_row(features, label)

    train_and_evaluate()


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description="BB84 QKD simulation and ML pipeline")
    parser.add_argument("--bits", type=int, default=128, help="Number of bits per run")
    parser.add_argument(
        "--runs",
        type=int,
        default=10000,
        help="Number of simulation runs (must be exactly 10000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    return SimulationConfig(bits=args.bits, runs=args.runs, seed=args.seed)


def main() -> None:
    config = parse_args()
    run_simulation(config)


if __name__ == "__main__":
    main()
