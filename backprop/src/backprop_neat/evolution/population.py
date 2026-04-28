from __future__ import annotations

import random

from backprop_neat.config import BackpropNEATConfig
from backprop_neat.evolution.species import Species
from backprop_neat.genetics.genome import Genome


class Population:
    def __init__(self, config: BackpropNEATConfig):
        self.config = config
        self.size = config.population_size
        self.conn_id = -1
        self.node_id = sum(config.genome_shape) - 1
        self.conn_genes: dict[tuple[int, int], int] = {}
        self.members: list[Genome] = []
        self.species: list[Species] = []
        self.species_threshold = config.species_threshold
        self.initialize()

    def initialize(self) -> None:
        self.members = [Genome(self, self.config) for _ in range(self.size)]
        self.respeciate()

    def set_conn_id(self, in_node_id: int, out_node_id: int) -> int:
        key = (in_node_id, out_node_id)
        if key not in self.conn_genes:
            self.conn_id += 1
            self.conn_genes[key] = self.conn_id
        return self.conn_genes[key]

    def set_node_id(self) -> int:
        self.node_id += 1
        return self.node_id

    def categorize_genes(self, genome1: Genome, genome2: Genome):
        genes1 = genome1.sorted_connections
        genes2 = genome2.sorted_connections
        ids1 = {conn.id: conn for conn in genes1}
        ids2 = {conn.id: conn for conn in genes2}
        all_ids = sorted(set(ids1) | set(ids2))
        max1 = max(ids1, default=-1)
        max2 = max(ids2, default=-1)
        matching = []
        disjoint = []
        excess = []
        for conn_id in all_ids:
            c1 = ids1.get(conn_id)
            c2 = ids2.get(conn_id)
            if c1 is not None and c2 is not None:
                matching.append((c1, c2))
            elif conn_id > min(max1, max2):
                excess.append(c1 or c2)
            else:
                disjoint.append(c1 or c2)
        return matching, disjoint, excess

    def compatibility(self, genome1: Genome, genome2: Genome) -> float:
        matching, disjoint, excess = self.categorize_genes(genome1, genome2)
        n = max(1, len(genome1.connections), len(genome2.connections))
        if matching:
            weight_diff = sum(abs(c1.weight - c2.weight) for c1, c2 in matching) / len(matching)
        else:
            weight_diff = 0.0
        return (
            self.config.c1 * len(excess) / n
            + self.config.c2 * len(disjoint) / n
            + self.config.c3 * weight_diff
        )

    def respeciate(self) -> None:
        old_species = self.species
        self.species = []
        for species in old_species:
            species.members = []
        for genome in self.members:
            placed = False
            for species in self.species:
                if species.representative and self.compatibility(genome, species.representative) < self.species_threshold:
                    species.members.append(genome)
                    placed = True
                    break
            if not placed:
                self.species.append(Species(genome))
        self.species = [species for species in self.species if species.members]
        for species in self.species:
            species.choose_representative()
        self._adapt_species_threshold()

    def _adapt_species_threshold(self) -> None:
        if self.config.adaptive_threshold <= 0.0:
            return
        if len(self.species) > self.config.target_species_number:
            self.species_threshold += self.config.adaptive_threshold
        elif len(self.species) < self.config.target_species_number:
            self.species_threshold = max(0.2, self.species_threshold - self.config.adaptive_threshold)

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1
        child_nodes = {node_id: node.copy() for node_id, node in parent1.nodes.items()}
        child_connections = {}
        p2_connections = {conn.id: conn for conn in parent2.connections.values()}
        for conn in parent1.sorted_connections:
            other = p2_connections.get(conn.id)
            chosen = random.choice((conn, other)) if other is not None else conn
            copied = chosen.copy()
            if copied.in_node.id in child_nodes and copied.out_node.id in child_nodes:
                copied.in_node = child_nodes[copied.in_node.id]
                copied.out_node = child_nodes[copied.out_node.id]
                if other is not None and (not conn.enabled or not other.enabled) and random.random() < 0.75:
                    copied.enabled = False
                child_connections[copied.id] = copied
        child = Genome(self, self.config, nodes=child_nodes, connections=child_connections)
        child.topological_sort()
        return child

    def reproduce(self) -> None:
        new_members: list[Genome] = []
        living = [species for species in self.species if species.members]
        if not living:
            self.initialize()
            return
        adjusted_totals = []
        for species in living:
            ranked = species.rank()
            adjusted_totals.append(sum(max(0.0, g.fitness) for g in ranked) / max(1, len(ranked)))
        total_adjusted = sum(adjusted_totals)
        allocations = []
        for adjusted in adjusted_totals:
            if total_adjusted > 0:
                allocations.append(max(0, int(round(self.size * adjusted / total_adjusted))))
            else:
                allocations.append(self.size // len(living))
        while sum(allocations) < self.size:
            allocations[allocations.index(max(allocations))] += 1
        while sum(allocations) > self.size:
            idx = allocations.index(max(allocations))
            allocations[idx] -= 1

        for species, allocation in zip(living, allocations):
            ranked = species.rank()
            if allocation <= 0 or not ranked:
                continue
            for elite in ranked[: min(self.config.num_elites, allocation, len(ranked))]:
                new_members.append(elite.copy())
            remaining = allocation - min(self.config.num_elites, allocation, len(ranked))
            pool_size = max(1, int(len(ranked) * self.config.selection_share))
            pool = ranked[:pool_size]
            for _ in range(remaining):
                p1 = random.choice(pool)
                p2 = random.choice(pool)
                try:
                    child = self.crossover(p1, p2)
                except Exception:
                    child = (p1 if p1.fitness >= p2.fitness else p2).copy()
                child.mutate()
                new_members.append(child)
        while len(new_members) < self.size:
            new_members.append(Genome(self, self.config))
        self.members = new_members[: self.size]
        self.respeciate()

    def best(self) -> Genome:
        return max(self.members, key=lambda genome: genome.fitness)

