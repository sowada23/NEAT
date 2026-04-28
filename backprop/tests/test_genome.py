from __future__ import annotations

from backprop_neat import BackpropNEATConfig, Population


def test_initial_genome_is_acyclic_and_forward_runs():
    pop = Population(BackpropNEATConfig(population_size=3, genome_shape=(2, 1)))
    genome = pop.members[0]
    order = genome.topological_sort()
    assert len(order) == len(genome.nodes)
    output = genome.forward([0.25, -0.75])
    assert len(output) == 1
    assert 0.0 <= output[0] <= 1.0


def test_add_node_preserves_feed_forward_topology():
    pop = Population(BackpropNEATConfig(population_size=2, genome_shape=(2, 1)))
    genome = pop.members[0]
    conn_id = next(iter(genome.connections))
    assert genome.add_node(conn_id)
    assert len(genome.hidden_nodes()) == 1
    assert len(genome.topological_sort()) == len(genome.nodes)


def test_crossover_returns_valid_child():
    pop = Population(BackpropNEATConfig(population_size=2, genome_shape=(2, 1)))
    parent1, parent2 = pop.members
    parent1.fitness = 2.0
    parent2.fitness = 1.0
    parent1.add_node(next(iter(parent1.connections)))
    child = pop.crossover(parent1, parent2)
    assert len(child.topological_sort()) == len(child.nodes)
    assert child.nodes
    assert child.connections


def test_compatibility_is_zero_for_identical_copy():
    pop = Population(BackpropNEATConfig(population_size=2, genome_shape=(2, 1)))
    genome = pop.members[0]
    assert pop.compatibility(genome, genome.copy()) == 0.0

