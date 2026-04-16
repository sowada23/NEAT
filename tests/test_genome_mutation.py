from neat import NEATConfig, Population


def test_add_connection_or_node_mutation_runs():
    config = NEATConfig(population_size=2, genome_shape=(2, 1))
    pop = Population(config=config)
    genome = pop.members[0]

    genome.add_connection_mutation(prob=1.0)
    genome.add_node_mutation(prob=1.0)

    assert genome is not None
