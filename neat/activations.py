import math


def relu(x):
    return max(0.0, x)


def sigmoid(x):
    clamped_x = max(-700.0, min(700.0, x))  # Values outside this range are effectively 0 or 1
    return 1 / (1 + math.exp(-clamped_x))


def tanh(x):
    return math.tanh(x)


def select_activation(name):
    if name == 'relu':
        return relu
    elif name == 'sigmoid':
        return sigmoid
    elif name == 'tanh':
        return tanh
    else:
        raise ValueError(f'Unknown activation function: {name}')
