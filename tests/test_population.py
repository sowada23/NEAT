from neat import NEATConfig, Population


def test_population_initializes():
    config = NEATConfig(population_size=5, genome_shape=(2, 1))
    pop = Population(config=config)

    assert len(pop.members) == 5
    assert len(pop.species) >= 1
