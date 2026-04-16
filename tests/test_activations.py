from neat.activations import relu, sigmoid, tanh, select_activation


def test_relu():
    assert relu(-1.0) == 0.0
    assert relu(2.0) == 2.0


def test_sigmoid():
    y = sigmoid(0.0)
    assert 0.49 < y < 0.51


def test_tanh():
    assert tanh(0.0) == 0.0


def test_select_activation():
    assert select_activation("relu") is relu
