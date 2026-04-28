from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def softplus(x: float) -> float:
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def select_activation(name: str):
    if name == "tanh":
        return math.tanh
    if name == "sigmoid":
        return sigmoid
    if name == "softplus":
        return softplus
    if name == "silu":
        return silu
    raise ValueError(f"Unknown or unsupported activation: {name}")

