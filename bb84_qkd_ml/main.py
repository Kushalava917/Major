"""Main controller for BB84 QKD simulation and ML pipeline."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from qkd import alice, analysis, bob, eve, reconciliation
from ml.dataset import init_csv, write_row
from ml.model import train_and_evaluate


@dataclass(frozen=True)
class SimulationConfig:
    bits: int
    runs: int
    seed: int | None


def run_simulation(config: SimulationConfig) -> None:
    rng = random.Random(config.seed)
    init_csv()

    for run in range(config.runs):
        eve_present = run % 2 == 0

        alice_bits = alice.generate_bits(config.bits, rng)
        alice_bases = alice.generate_bases(config.bits, rng)
        states = alice.encode(alice_bits, alice_bases)

        if eve_present:
            states = eve.intercept_resend(states, rng)

        bob_bases = bob.generate_bases(config.bits, rng)
        bob_bits = bob.measure(states, bob_bases, rng)

        alice_key, bob_key = reconciliation.sift_keys(
            alice_bits, bob_bits, alice_bases, bob_bases
        )

        features = analysis.extract_features(
            alice_key, bob_key, alice_bases, bob_bases
        )

        label = 1 if eve_present else 0
        write_row(features, label)

    train_and_evaluate()


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description="BB84 QKD simulation and ML pipeline")
    parser.add_argument("--bits", type=int, default=128, help="Number of bits per run")
    parser.add_argument("--runs", type=int, default=400, help="Number of simulation runs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    return SimulationConfig(bits=args.bits, runs=args.runs, seed=args.seed)


def main() -> None:
    config = parse_args()
    run_simulation(config)


if __name__ == "__main__":
    main()
