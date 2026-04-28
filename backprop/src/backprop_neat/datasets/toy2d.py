from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def _shuffle(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(x))
    return x[order], y[order]


def generate_xor(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-5.0, 5.0, size=(n, 2)) + rng.normal(0.0, noise, size=(n, 2))
    y = (((x[:, 0] > 0.0) & (x[:, 1] > 0.0)) | ((x[:, 0] < 0.0) & (x[:, 1] < 0.0))).astype(np.float32)
    return x.astype(np.float32), y.reshape(-1, 1)


def generate_spiral(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    half = n // 2
    points = []
    labels = []
    for delta_t, label in ((0.0, 0.0), (np.pi, 1.0)):
        for i in range(half):
            r = i / max(1, half) * 6.0
            t = 1.75 * i / max(1, half) * 2.0 * np.pi + delta_t
            x = r * np.sin(t) + rng.uniform(-1.0, 1.0) * noise
            y = r * np.cos(t) + rng.uniform(-1.0, 1.0) * noise
            points.append((x, y))
            labels.append(label)
    if n % 2:
        points.append((0.0, 0.0))
        labels.append(0.0)
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.float32).reshape(-1, 1)


def generate_gaussian(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    half = n // 2
    std = noise + 1.0
    pos = rng.normal(2.0, std, size=(half, 2))
    neg = rng.normal(-2.0, std, size=(n - half, 2))
    x = np.vstack([pos, neg]).astype(np.float32)
    y = np.vstack([np.ones((half, 1)), np.zeros((n - half, 1))]).astype(np.float32)
    return x, y


def generate_circle(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    half = n // 2
    radius = 5.0
    points = []
    labels = []
    for _ in range(half):
        r = rng.uniform(0.0, radius * 0.5)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        point = np.asarray([r * np.sin(angle), r * np.cos(angle)])
        point += rng.uniform(-radius, radius, size=2) * noise / 3.0
        points.append(point)
        labels.append(float(np.sum(point * point) < (radius * 0.5) ** 2))
    for _ in range(n - half):
        r = rng.uniform(radius * 0.75, radius)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        point = np.asarray([r * np.sin(angle), r * np.cos(angle)])
        point += rng.uniform(-radius, radius, size=2) * noise / 3.0
        points.append(point)
        labels.append(float(np.sum(point * point) < (radius * 0.5) ** 2))
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.float32).reshape(-1, 1)


GENERATORS = {
    "circle": generate_circle,
    "xor": generate_xor,
    "gaussian": generate_gaussian,
    "spiral": generate_spiral,
}


def generate_dataset(name: str, n: int = 200, noise: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if name not in GENERATORS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(GENERATORS)}")
    rng = np.random.default_rng(seed)
    x, y = GENERATORS[name](n, noise, rng)
    return _shuffle(x, y, rng)


def generate_split(
    name: str,
    train_size: int = 200,
    test_size: int = 200,
    noise: float = 0.5,
    seed: int = 0,
) -> DatasetSplit:
    train_x, train_y = generate_dataset(name, train_size, noise, seed)
    test_x, test_y = generate_dataset(name, test_size, noise, seed + 10_000)
    return DatasetSplit(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

