from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from backprop_neat import BackpropNEATConfig, Population
from backprop_neat.config import SMOOTH_ACTIVATIONS
from backprop_neat.datasets import generate_dataset
from backprop_neat.jax.training import (
    TrainState,
    batched_forward_state,
    evaluate_and_train_genome,
    genome_to_jax,
    loss_fn,
    train_genome,
    write_state_to_genome,
)


def test_smooth_activation_validation_rejects_relu():
    with pytest.raises(ValueError):
        BackpropNEATConfig(allowed_activations=SMOOTH_ACTIVATIONS + ("relu",))


def test_jax_forward_matches_python_forward_before_training():
    pop = Population(BackpropNEATConfig(population_size=2, genome_shape=(2, 1)))
    genome = pop.members[0]
    x = np.asarray([[0.2, -0.4], [1.0, 1.0]], dtype=np.float32)
    jax_genome = genome_to_jax(genome)
    state = TrainState(jax_genome.initial_weights, jax_genome.initial_biases)
    actual = np.asarray(batched_forward_state(jax_genome, state, x))
    expected = np.asarray([genome.forward(row) for row in x], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_training_reduces_loss_on_xor_like_data():
    pop = Population(BackpropNEATConfig(population_size=2, genome_shape=(2, 1)))
    genome = pop.members[0]
    genome.add_node(next(iter(genome.connections)))
    x, y = generate_dataset("xor", n=80, noise=0.05, seed=1)
    jax_genome = genome_to_jax(genome)
    initial_state = TrainState(jax_genome.initial_weights, jax_genome.initial_biases)
    initial_loss = float(loss_fn(jax_genome, initial_state, x, y, 0.0))
    trained_state, metrics = train_genome(genome, x, y, steps=20, learning_rate=0.05, seed=9)
    final_loss = float(loss_fn(jax_genome, trained_state, x, y, 0.0))
    assert final_loss <= initial_loss
    assert metrics["loss"] <= initial_loss


def test_trained_params_are_written_back_to_genome():
    pop = Population(BackpropNEATConfig(population_size=2, genome_shape=(2, 1)))
    genome = pop.members[0]
    original = [conn.weight for conn in genome.enabled_connections()]
    x, y = generate_dataset("gaussian", n=64, noise=0.2, seed=2)
    state, _ = train_genome(genome, x, y, steps=5, learning_rate=0.05, seed=2)
    write_state_to_genome(genome, state)
    updated = [conn.weight for conn in genome.enabled_connections()]
    assert any(abs(a - b) > 1e-8 for a, b in zip(original, updated))


def test_complexity_penalty_lowers_fitness():
    config = BackpropNEATConfig(
        population_size=2,
        genome_shape=(2, 1),
        backprop_steps=1,
        complexity_conn_penalty=1.0,
        complexity_node_penalty=1.0,
    )
    pop = Population(config)
    small = pop.members[0]
    large = small.copy()
    large.add_node(next(iter(large.connections)))
    x, y = generate_dataset("circle", n=32, noise=0.1, seed=3)
    small_metrics = evaluate_and_train_genome(
        small,
        x,
        y,
        x,
        y,
        steps=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=None,
        conn_penalty=1.0,
        node_penalty=1.0,
        seed=1,
    )
    large_metrics = evaluate_and_train_genome(
        large,
        x,
        y,
        x,
        y,
        steps=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=None,
        conn_penalty=1.0,
        node_penalty=1.0,
        seed=1,
    )
    assert large_metrics["complexity_penalty"] > small_metrics["complexity_penalty"]

