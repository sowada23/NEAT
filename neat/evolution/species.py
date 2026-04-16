class Species:
    """
    Represents a species in the population.
    """
    def __init__(self, config=None):
        """
        Initializes a new, empty species.
        """
        self.config = config
        self.members = []
        self.representative = None

    def adjust_fitness(self):
        num_members = len(self.members)
        if num_members == 0:
            return []

        for genome in self.members:
            genome.fitness /= float(num_members)

        return self.members

    def linear_scale_fitness(self, c=1.5):
        if not self.members:
            return []

        fitnesses = [g.fitness for g in self.members]
        f_avg = sum(fitnesses) / len(fitnesses)
        f_max = max(fitnesses)

        if f_max <= f_avg:
            for g in self.members:
                g.fitness = max(1.0, g.fitness)
            return self.members

        if f_max - f_avg == 0:
            a = 1.0
            b = 0.0
        else:
            a = (c - 1.0) * f_avg / (f_max - f_avg)
            b = f_avg * (1.0 - a)

        for g in self.members:
            scaled_fitness = a * g.fitness + b
            g.fitness = max(0.0, scaled_fitness)

        return self.members

    def offset_fitness(self):
        if not self.members:
            return []

        f_min = min(g.fitness for g in self.members)
        epsilon = 1e-7
        offset = -f_min + epsilon if f_min < epsilon else 0.0

        for genome in self.members:
            genome.fitness += offset

        return self.members

    def rank(self):
        return sorted(self.members, key=lambda g: g.fitness, reverse=True)
