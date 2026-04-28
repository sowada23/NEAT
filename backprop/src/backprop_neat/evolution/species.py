from __future__ import annotations

import random


class Species:
    def __init__(self, representative=None):
        self.representative = representative
        self.members = []
        if representative is not None:
            self.members.append(representative)

    def rank(self):
        return sorted(self.members, key=lambda genome: genome.fitness, reverse=True)

    def choose_representative(self) -> None:
        self.representative = random.choice(self.members) if self.members else None

