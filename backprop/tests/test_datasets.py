from __future__ import annotations

import numpy as np

from backprop_neat.datasets import generate_dataset, generate_split


def test_dataset_generation_is_deterministic():
    x1, y1 = generate_dataset("xor", n=32, noise=0.5, seed=123)
    x2, y2 = generate_dataset("xor", n=32, noise=0.5, seed=123)
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y1, y2)


def test_all_dataset_shapes():
    for name in ["circle", "xor", "gaussian", "spiral"]:
        split = generate_split(name, train_size=31, test_size=17, seed=5)
        assert split.train_x.shape == (31, 2)
        assert split.train_y.shape == (31, 1)
        assert split.test_x.shape == (17, 2)
        assert split.test_y.shape == (17, 1)
        assert set(np.unique(split.train_y)).issubset({0.0, 1.0})

