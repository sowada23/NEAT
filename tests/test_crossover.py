from neat import NEATConfig, Population


def test_crossover_runs():
    config = NEATConfig(population_size=2, genome_shape=(2, 1))
    pop = Population(config=config)

    g1 = pop.members[0]
    g2 = pop.members[1]
    g1.fitness = 2.0
    g2.fitness = 1.0

    child = pop.cross_over(g1, g2)
    assert child is not None
