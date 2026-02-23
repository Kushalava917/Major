"""Lightweight UI server for BB84 QKD eavesdropping demo."""

from __future__ import annotations

import json
import random
import sys
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ml.model import predict_label, train_model_from_dataset
from qkd import alice, analysis, bob, eve, reconciliation


def _apply_channel(states, loss_rate: float, channel_mu_ch: float, rng: random.Random):
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


def _simulate_runs(bits: int, runs: int, mode: str) -> dict:
    rng = random.Random(42)
    qber_values = []
    loss_values = []
    var_values = []
    burst_values = []
    intercept_values = []
    channel_values = []

    for run in range(runs):
        if mode == "always":
            eve_present = True
        elif mode == "never":
            eve_present = False
        else:
            eve_present = run % 2 == 0

        alice_bits = alice.generate_bits(bits, rng)
        alice_bases = alice.generate_bases(bits, rng)
        states = alice.encode(alice_bits, alice_bases)

        channel_mu_ch = rng.uniform(0.005, 0.06)
        loss_rate = rng.uniform(0.01, 0.2)
        intercept_ratio = rng.uniform(0.45, 1.0) if eve_present else 0.0

        if eve_present:
            states = _intercept_with_ratio(states, intercept_ratio, rng)

        states = _apply_channel(states, loss_rate, channel_mu_ch, rng)

        bob_bases = bob.generate_bases(bits, rng)
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

        qber_values.append(features[0])
        loss_values.append(features[1])
        var_values.append(features[2])
        burst_values.append(features[3])
        intercept_values.append(features[4])
        channel_values.append(features[5])

    return {
        "qber": sum(qber_values) / len(qber_values),
        "loss_rate": sum(loss_values) / len(loss_values),
        "error_variance": sum(var_values) / len(var_values),
        "burst_error_frequency": sum(burst_values) / len(burst_values),
        "intercept_ratio": sum(intercept_values) / len(intercept_values),
        "channel_mu_ch": sum(channel_values) / len(channel_values),
    }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            try:
                bits = int(query.get("bits", [128])[0])
                runs = int(query.get("runs", [50])[0])
                mode = query.get("mode", ["alternate"])[0]
            except ValueError:
                self._send_json({"error": "Invalid query parameters."}, status=400)
                return

            try:
                metrics = _simulate_runs(bits, runs, mode)
                weights, bias, means, stds = train_model_from_dataset()
                features = [
                    metrics["qber"],
                    metrics["loss_rate"],
                    metrics["error_variance"],
                    metrics["burst_error_frequency"],
                    metrics["channel_mu_ch"],
                ]
                prediction = predict_label(features, weights, bias, means, stds)

                self._send_json({"metrics": metrics, "prediction": prediction})
            except Exception as exc:  # pragma: no cover
                self._send_json({"error": str(exc)}, status=500)
            return

        if parsed.path.startswith("/data/"):
            self.path = str(BASE_DIR / parsed.path.lstrip("/"))
            return super().do_GET()

        if parsed.path in {"/", "/index.html"}:
            self.path = str(UI_DIR / "index.html")
            return super().do_GET()

        return super().do_GET()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    with TCPServer(("0.0.0.0", 8000), Handler) as httpd:
        print("Serving UI on http://0.0.0.0:8000")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
