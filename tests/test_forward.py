from neat import NEATConfig, Population


def test_forward_output_length():
    config = NEATConfig(population_size=2, genome_shape=(2, 1))
    pop = Population(config=config)
    genome = pop.members[0]

    output = genome.forward([0.0, 1.0])
    assert len(output) == 1
