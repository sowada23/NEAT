import numpy as np
import pytest

from neat import NEATConfig, Population

jax = pytest.importorskip("jax")

from neat.jax import compile_genome_forward, forward_jax, genome_to_jax


def test_jax_forward_matches_python_forward():
    config = NEATConfig(population_size=2, genome_shape=(2, 1))
    pop = Population(config=config)
    genome = pop.members[0]
    x = np.asarray([0.25, -0.75], dtype=np.float32)

    expected = np.asarray(genome.forward(x), dtype=np.float32)
    actual = np.asarray(forward_jax(genome_to_jax(genome), x))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_compiled_jax_forward_supports_batches():
    config = NEATConfig(population_size=2, genome_shape=(2, 1))
    pop = Population(config=config)
    genome = pop.members[0]
    _, _forward_fn, batched_fn, _action_fn = compile_genome_forward(genome)

    xs = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    outputs = np.asarray(batched_fn(xs))

    assert outputs.shape == (2, 1)
    assert jax.devices()
