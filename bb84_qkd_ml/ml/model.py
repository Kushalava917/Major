"""Model training and evaluation for BB84 QKD ML pipeline.

This module avoids external dependencies by using the Python standard library.
It produces numeric outputs plus visualization data as CSV/TXT artifacts.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
PLOTS_DIR = BASE_DIR / "data" / "plots"
<<<<<<< HEAD
FEATURE_COLUMNS = ["QBER", "BasisMatchRate", "KeyLength"]
=======
FEATURE_COLUMNS = [
    "QBER",
    "LossRate",
    "ErrorVariance",
    "BurstErrorFrequency",
    "ChannelMuCh",
]
>>>>>>> codex/create-project-folder-structure-and-files


def _ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_dataset(path: Path) -> Tuple[List[List[float]], List[int]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty after loading.")

    X: List[List[float]] = []
    y: List[int] = []
    for row in rows:
        X.append([float(row[col]) for col in FEATURE_COLUMNS])
        y.append(int(row["Label"]))
    return X, y


def _train_test_split_stratified(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    test_size: float,
    seed: int,
) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    rng = random.Random(seed)
    indices_by_label = {0: [], 1: []}
    for idx, label in enumerate(y):
        indices_by_label[label].append(idx)

    train_indices: List[int] = []
    test_indices: List[int] = []
    for label, indices in indices_by_label.items():
        rng.shuffle(indices)
        split = max(1, int(len(indices) * test_size))
        test_indices.extend(indices[:split])
        train_indices.extend(indices[split:])

    X_train = [list(X[i]) for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [list(X[i]) for i in test_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _predict_proba_row(weights: Sequence[float], bias: float, row: Sequence[float]) -> float:
    z = sum(w * x for w, x in zip(weights, row)) + bias
    return _sigmoid(z)


def _standardize(
    X: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[float], List[float]]:
    columns = list(zip(*X))
    means = [sum(col) / len(col) for col in columns]
    stds = []
    for col, mean in zip(columns, means):
        variance = sum((v - mean) ** 2 for v in col) / len(col)
        std = math.sqrt(variance) or 1.0
        stds.append(std)
    standardized = [
        [(val - mean) / std for val, mean, std in zip(row, means, stds)]
        for row in X
    ]
    return standardized, means, stds


def _apply_standardize(
    X: Sequence[Sequence[float]],
    means: Sequence[float],
    stds: Sequence[float],
) -> List[List[float]]:
    return [
        [(val - mean) / std for val, mean, std in zip(row, means, stds)]
        for row in X
    ]


def _train_logistic_regression(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    lr: float = 0.2,
    epochs: int = 1500,
) -> Tuple[List[float], float]:
    if not X:
        raise ValueError("Training data is empty.")

    weights = [0.0 for _ in range(len(X[0]))]
    bias = 0.0

    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for row, label in zip(X, y):
            pred = _predict_proba_row(weights, bias, row)
            error = pred - label
            for i in range(len(weights)):
                grad_w[i] += error * row[i]
            grad_b += error

        n = float(len(X))
        weights = [w - lr * (g / n) for w, g in zip(weights, grad_w)]
        bias -= lr * (grad_b / n)

    return weights, bias


<<<<<<< HEAD
=======
def train_model_from_dataset() -> Tuple[List[float], float, List[float], List[float]]:
    """Train a model on the dataset and return weights, bias, means, and stds."""
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset not found or empty: {DATA_PATH}")

    X, y = _read_dataset(DATA_PATH)
    X_train, _, y_train, _ = _train_test_split_stratified(X, y, test_size=0.3, seed=42)

    X_train_scaled, means, stds = _standardize(X_train)
    weights, bias = _train_logistic_regression(X_train_scaled, y_train)
    return weights, bias, means, stds


def predict_label(
    features: Sequence[float],
    weights: Sequence[float],
    bias: float,
    means: Sequence[float],
    stds: Sequence[float],
) -> int:
    """Predict a label for a single feature row."""
    scaled = _apply_standardize([features], means, stds)[0]
    prob = _predict_proba_row(weights, bias, scaled)
    return 1 if prob >= 0.5 else 0


>>>>>>> codex/create-project-folder-structure-and-files
def _confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> List[List[int]]:
    tn = fp = fn = tp = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
    return [[tn, fp], [fn, tp]]


def _accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def _roc_curve(y_true: Sequence[int], y_prob: Sequence[float]) -> List[Tuple[float, float]]:
    thresholds = sorted(set(y_prob), reverse=True)
    curve: List[Tuple[float, float]] = []
    for threshold in thresholds:
        preds = [1 if p >= threshold else 0 for p in y_prob]
        cm = _confusion_matrix(y_true, preds)
        tn, fp = cm[0]
        fn, tp = cm[1]
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        curve.append((fpr, tpr))
    return curve


def _write_confusion_matrix(cm: Sequence[Sequence[int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Predicted 0", "Predicted 1"])
        writer.writerow(["Actual 0", cm[0][0], cm[0][1]])
        writer.writerow(["Actual 1", cm[1][0], cm[1][1]])


def _write_roc_curve(curve: Sequence[Tuple[float, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FPR", "TPR"])
        writer.writerows(curve)


def _write_feature_summary(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    path: Path,
) -> None:
    stats = {0: [], 1: []}
    for row, label in zip(X, y):
        stats[label].append(row)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Feature", "Mean", "StdDev"])
        for label in (0, 1):
            rows = stats[label]
            for idx, feature in enumerate(FEATURE_COLUMNS):
                values = [row[idx] for row in rows]
                mean = sum(values) / len(values) if values else 0.0
                variance = (
                    sum((v - mean) ** 2 for v in values) / len(values)
                    if values
                    else 0.0
                )
                std = math.sqrt(variance)
                writer.writerow([label, feature, f"{mean:.6f}", f"{std:.6f}"])


def _write_feature_correlation(X: Sequence[Sequence[float]], path: Path) -> None:
    columns = list(zip(*X))
    means = [sum(col) / len(col) for col in columns]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature"] + FEATURE_COLUMNS)
        for i, feature in enumerate(FEATURE_COLUMNS):
            row = [feature]
            for j in range(len(FEATURE_COLUMNS)):
                cov = sum(
                    (columns[i][k] - means[i]) * (columns[j][k] - means[j])
                    for k in range(len(X))
                )
                denom = math.sqrt(
                    sum((columns[i][k] - means[i]) ** 2 for k in range(len(X)))
                    * sum((columns[j][k] - means[j]) ** 2 for k in range(len(X)))
                )
                corr = cov / denom if denom else 0.0
                row.append(f"{corr:.6f}")
            writer.writerow(row)


<<<<<<< HEAD
=======
def _write_diagnostics(train_acc: float, test_acc: float, path: Path) -> None:
    gap = train_acc - test_acc
    status = "balanced"
    if train_acc > 0.98 and test_acc < 0.90:
        status = "overfitting_risk"
    elif train_acc < 0.75 and test_acc < 0.75:
        status = "underfitting_risk"

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["TrainAccuracy", f"{train_acc:.6f}"])
        writer.writerow(["TestAccuracy", f"{test_acc:.6f}"])
        writer.writerow(["AccuracyGap", f"{gap:.6f}"])
        writer.writerow(["FitStatus", status])


def _write_accuracy_bar_svg(train_acc: float, test_acc: float, path: Path) -> None:
    max_h = 200
    train_h = int(max_h * train_acc)
    test_h = int(max_h * test_acc)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="320">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<text x="20" y="30" font-size="20" font-family="Arial">Train vs Test Accuracy</text>'
        '<line x1="70" y1="250" x2="470" y2="250" stroke="#333"/>'
        f'<rect x="140" y="{250-train_h}" width="90" height="{train_h}" fill="#3182ce"/>'
        f'<rect x="300" y="{250-test_h}" width="90" height="{test_h}" fill="#38a169"/>'
        '<text x="145" y="270" font-size="14" font-family="Arial">Train</text>'
        '<text x="305" y="270" font-size="14" font-family="Arial">Test</text>'
        f'<text x="145" y="{245-train_h}" font-size="12" font-family="Arial">{train_acc:.3f}</text>'
        f'<text x="305" y="{245-test_h}" font-size="12" font-family="Arial">{test_acc:.3f}</text>'
        '</svg>'
    )
    path.write_text(svg)


def _write_feature_mean_bar_svg(X: Sequence[Sequence[float]], y: Sequence[int], path: Path) -> None:
    stats = {0: [], 1: []}
    for row, label in zip(X, y):
        stats[label].append(row)

    means0: List[float] = []
    means1: List[float] = []
    for idx in range(len(FEATURE_COLUMNS)):
        v0 = [r[idx] for r in stats[0]]
        v1 = [r[idx] for r in stats[1]]
        means0.append(sum(v0) / len(v0) if v0 else 0.0)
        means1.append(sum(v1) / len(v1) if v1 else 0.0)

    max_val = max(means0 + means1) if (means0 or means1) else 1.0
    max_val = max(max_val, 1e-9)
    scale = 220 / max_val

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="420">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="20" y="30" font-size="20" font-family="Arial">Feature Means by Label (Blue=No Eve, Red=Eve)</text>',
        '<line x1="40" y1="330" x2="880" y2="330" stroke="#333"/>',
    ]

    x = 60
    for idx, feature in enumerate(FEATURE_COLUMNS):
        h0 = int(means0[idx] * scale)
        h1 = int(means1[idx] * scale)
        parts.append(f'<rect x="{x}" y="{330-h0}" width="24" height="{h0}" fill="#3182ce"/>')
        parts.append(f'<rect x="{x+28}" y="{330-h1}" width="24" height="{h1}" fill="#e53e3e"/>')
        parts.append(f'<text x="{x-5}" y="352" font-size="10" font-family="Arial">{feature}</text>')
        x += 140

    parts.append('</svg>')
    path.write_text("".join(parts))


def _write_roc_svg(curve: Sequence[Tuple[float, float]], path: Path) -> None:
    points = []
    for fpr, tpr in curve:
        x = 40 + int(fpr * 340)
        y = 380 - int(tpr * 340)
        points.append(f"{x},{y}")

    poly = " ".join(points) if points else "40,380"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420">'
        '<rect width="100%" height="100%" fill="#fff"/>'
        '<text x="20" y="25" font-size="18" font-family="Arial">ROC Curve</text>'
        '<line x1="40" y1="380" x2="380" y2="380" stroke="#333"/>'
        '<line x1="40" y1="380" x2="40" y2="40" stroke="#333"/>'
        '<line x1="40" y1="380" x2="380" y2="40" stroke="#bbb" stroke-dasharray="4,4"/>'
        f'<polyline fill="none" stroke="#2b6cb0" stroke-width="2" points="{poly}"/>'
        '</svg>'
    )
    path.write_text(svg)


>>>>>>> codex/create-project-folder-structure-and-files
def train_and_evaluate() -> None:
    """Train a logistic regression model and report metrics/visualizations."""
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset not found or empty: {DATA_PATH}")

    X, y = _read_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = _train_test_split_stratified(
        X, y, test_size=0.3, seed=42
    )

    X_train_scaled, means, stds = _standardize(X_train)
    X_test_scaled = _apply_standardize(X_test, means, stds)

    weights, bias = _train_logistic_regression(X_train_scaled, y_train)

<<<<<<< HEAD
    probs = [_predict_proba_row(weights, bias, row) for row in X_test_scaled]
    preds = [1 if p >= 0.5 else 0 for p in probs]

    accuracy = _accuracy(y_test, preds)
    cm = _confusion_matrix(y_test, preds)

    print("Accuracy:", f"{accuracy:.4f}")
=======
    train_probs = [_predict_proba_row(weights, bias, row) for row in X_train_scaled]
    train_preds = [1 if p >= 0.5 else 0 for p in train_probs]
    probs = [_predict_proba_row(weights, bias, row) for row in X_test_scaled]
    preds = [1 if p >= 0.5 else 0 for p in probs]

    train_accuracy = _accuracy(y_train, train_preds)
    accuracy = _accuracy(y_test, preds)
    cm = _confusion_matrix(y_test, preds)

    print("Train Accuracy:", f"{train_accuracy:.4f}")
    print("Test Accuracy:", f"{accuracy:.4f}")
>>>>>>> codex/create-project-folder-structure-and-files
    print("Confusion Matrix:\n", cm)

    _ensure_plots_dir()
    _write_confusion_matrix(cm, PLOTS_DIR / "confusion_matrix.csv")
<<<<<<< HEAD
    _write_roc_curve(_roc_curve(y_test, probs), PLOTS_DIR / "roc_curve.csv")
    _write_feature_summary(X, y, PLOTS_DIR / "feature_summary.csv")
    _write_feature_correlation(X, PLOTS_DIR / "feature_correlation.csv")
=======
    roc = _roc_curve(y_test, probs)
    _write_roc_curve(roc, PLOTS_DIR / "roc_curve.csv")
    _write_feature_summary(X, y, PLOTS_DIR / "feature_summary.csv")
    _write_feature_correlation(X, PLOTS_DIR / "feature_correlation.csv")
    _write_diagnostics(train_accuracy, accuracy, PLOTS_DIR / "model_diagnostics.csv")
    _write_accuracy_bar_svg(train_accuracy, accuracy, PLOTS_DIR / "accuracy_bar.svg")
    _write_feature_mean_bar_svg(X, y, PLOTS_DIR / "feature_means_bar.svg")
    _write_roc_svg(roc, PLOTS_DIR / "roc_curve.svg")
>>>>>>> codex/create-project-folder-structure-and-files
