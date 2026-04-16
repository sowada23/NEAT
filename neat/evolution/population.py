import pickle as pkl
from random import choice, random

from neat.evolution.species import Species
from neat.genetics.genome import Genome


class Population:
    """
    Manages the entire population of genomes, including speciation, reproduction,
    and tracking of innovation numbers.
    """
    def __init__(self, config=None, genome_shape=(1, 1), size=100, adaptive_threshold=0.0, target_species_number=15):
        """
        Initializes the population.
        """
        self.config = config
        self.genome_shape = config.genome_shape if config is not None else genome_shape
        self.species_threshold = config.species_threshold if config is not None else 3.0
        self.adaptive_threshold = config.adaptive_threshold if config is not None else adaptive_threshold
        self.target_species_number = config.target_species_number if config is not None else target_species_number
        self.size = config.population_size if config is not None else size
        self.species = []
        self.conn_id = -1
        self.node_id = sum(self.genome_shape) - 1
        self.conn_genes = {}
        self.members = []

        self.initialize(self.genome_shape)

    def set_conn_id(self, in_node_id, out_node_id):
        key = (in_node_id, out_node_id)
        existing_id = self.conn_genes.get(key)

        if existing_id is not None:
            return existing_id
        else:
            self.conn_id += 1
            self.conn_genes[key] = self.conn_id
            return self.conn_id

    def set_node_id(self):
        self.node_id += 1
        return self.node_id

    def initialize(self, shape):
        self.members = []
        for _ in range(self.size):
            genome = Genome(self, shape=shape, config=self.config)
            self.members.append(genome)

        self.species = []
        for genome in self.members:
            self.speciate(genome)

    def categorize_genes(self, genome1, genome2):
        genes1 = genome1.sorted_conns
        genes2 = genome2.sorted_conns

        idx1, idx2 = 0, 0
        matching1, matching2 = [], []
        disjoint1, disjoint2 = [], []
        excess1, excess2 = [], []

        max_id1 = genes1[-1].id if genes1 else -1
        max_id2 = genes2[-1].id if genes2 else -1

        while idx1 < len(genes1) or idx2 < len(genes2):
            conn1 = genes1[idx1] if idx1 < len(genes1) else None
            conn2 = genes2[idx2] if idx2 < len(genes2) else None

            id1 = conn1.id if conn1 else float('inf')
            id2 = conn2.id if conn2 else float('inf')

            if id1 == id2:
                matching1.append(conn1)
                matching2.append(conn2)
                idx1 += 1
                idx2 += 1
            elif id1 < id2:
                if id1 > max_id2:
                    excess1.append(conn1)
                else:
                    disjoint1.append(conn1)
                idx1 += 1
            elif id2 < id1:
                if id2 > max_id1:
                    excess2.append(conn2)
                else:
                    disjoint2.append(conn2)
                idx2 += 1

        return {
            'genome1': (matching1, disjoint1, excess1),
            'genome2': (matching2, disjoint2, excess2)
        }

    def cross_over(self, genome1, genome2):
        if genome2.fitness > genome1.fitness:
            genome1, genome2 = genome2, genome1

        categorized = self.categorize_genes(genome1, genome2)
        matching1, disjoint1, excess1 = categorized['genome1']
        matching2, _, _ = categorized['genome2']

        offspring_nodes = {node.id: node.copy() for node in genome1.nodes.values()}
        offspring_connections = {}

        for conn1, conn2 in zip(matching1, matching2):
            chosen_conn_gene_original = choice((conn1, conn2))
            chosen_conn_gene = chosen_conn_gene_original.copy()

            if chosen_conn_gene.in_node.id in offspring_nodes and \
               chosen_conn_gene.out_node.id in offspring_nodes:
                chosen_conn_gene.in_node = offspring_nodes.get(chosen_conn_gene.in_node.id)
                chosen_conn_gene.out_node = offspring_nodes.get(chosen_conn_gene.out_node.id)
                offspring_connections[chosen_conn_gene.id] = chosen_conn_gene

                if not conn1.enabled or not conn2.enabled:
                    if random() < 0.75:
                        chosen_conn_gene.enabled = False

        for conn in disjoint1 + excess1:
            new_conn = conn.copy()
            if new_conn.in_node.id in offspring_nodes and \
               new_conn.out_node.id in offspring_nodes:
                new_conn.in_node = offspring_nodes.get(new_conn.in_node.id)
                new_conn.out_node = offspring_nodes.get(new_conn.out_node.id)
                offspring_connections[new_conn.id] = new_conn

        offspring = Genome(self, connections=offspring_connections, nodes=offspring_nodes, config=self.config)

        try:
            offspring.topological_sort()
        except Exception as e:
            print(f"Error during crossover validation: {e}. Offspring generation failed. Returning copy of fitter parent.")
            offspring = genome1.copy()

        return offspring

    def calculate_compatibility(self, genome1, genome2, c1=1.0, c2=1.0, c3=0.4):
        c1 = self.config.c1 if self.config is not None else c1
        c2 = self.config.c2 if self.config is not None else c2
        c3 = self.config.c3 if self.config is not None else c3

        categorized = self.categorize_genes(genome1, genome2)
        matching1, disjoint1, excess1 = categorized['genome1']
        matching2, disjoint2, excess2 = categorized['genome2']

        n1 = len(genome1.connections)
        n2 = len(genome2.connections)

        N = max(1.0, float(max(n1, n2)))

        E = float(len(excess1) + len(excess2))
        D = float(len(disjoint1) + len(disjoint2))

        num_matching = len(matching1)
        if num_matching > 0:
            weight_diff_sum = sum(abs(conn1.weight - conn2.weight) for conn1, conn2 in zip(matching1, matching2))
            W = weight_diff_sum / float(num_matching)
        else:
            W = 0.0

        delta = (c1 * E / N) + (c2 * D / N) + (c3 * W)
        return delta

    def adjust_species_threshold(self, min_species_threshold=0.15, max_species_threshold=15.0):
        min_species_threshold = self.config.min_species_threshold if self.config is not None else min_species_threshold
        max_species_threshold = self.config.max_species_threshold if self.config is not None else max_species_threshold

        num_species = len(self.species)

        if self.adaptive_threshold > 0.0:
            if num_species > self.target_species_number:
                self.species_threshold += self.adaptive_threshold
            elif num_species < self.target_species_number:
                self.species_threshold -= self.adaptive_threshold

            self.species_threshold = max(min_species_threshold, min(self.species_threshold, max_species_threshold))

        elif self.adaptive_threshold == 0.0:
            pass

        else:
            raise ValueError("Invalid adaptive threshold value. Must be >= 0.")

    def speciate(self, genome):
        assigned = False

        if not self.species:
            new_species = Species(config=self.config)
            new_species.members.append(genome)
            new_species.representative = genome
            self.species.append(new_species)
            assigned = True
        else:
            for species_obj in self.species:
                if species_obj.representative is None:
                    if species_obj.members:
                        species_obj.representative = species_obj.members[0]
                    else:
                        continue

                delta = self.calculate_compatibility(species_obj.representative, genome)

                if delta < self.species_threshold:
                    species_obj.members.append(genome)
                    assigned = True
                    break

            if not assigned:
                new_species = Species(config=self.config)
                new_species.members.append(genome)
                new_species.representative = genome
                self.species.append(new_species)

    def reproduce(self, num_elites=1, selection_share=0.2):
        num_elites = self.config.num_elites if self.config is not None else num_elites
        selection_share = self.config.selection_share if self.config is not None else selection_share

        new_pop = []
        species_data = []

        total_average_fitness = 0
        living_species = [s for s in self.species if s.members]

        if not living_species:
            print("Warning: No living species found for reproduction. Repopulating randomly.")
            while len(new_pop) < self.size:
                new_pop.append(Genome(self, self.genome_shape, config=self.config))
            self.members = new_pop[:self.size]
            self.species = []
            for genome in self.members:
                self.speciate(genome)
            return

        for species in living_species:
            species.linear_scale_fitness()
            species.offset_fitness()
            species.adjust_fitness()
            ranked_members = species.rank()
            species_total_fitness = sum(g.fitness for g in species.members)
            species_average_fitness = species_total_fitness / len(species.members) if species.members else 0.0
            total_average_fitness += species_average_fitness
            species_data.append({
                'species': species,
                'avg_fitness': species_average_fitness,
                'ranked': ranked_members
            })

        if total_average_fitness > 0:
            total_allocated = 0
            for data in species_data:
                proportion = data['avg_fitness'] / total_average_fitness
                proportion = max(0.0, proportion)
                data['num_offspring'] = int(round(proportion * self.size))
                total_allocated += data['num_offspring']

            discrepancy = self.size - total_allocated
            if discrepancy != 0 and species_data:
                species_data.sort(key=lambda x: x['num_offspring'], reverse=(discrepancy > 0))
                for i in range(abs(discrepancy)):
                    idx_to_adjust = i % len(species_data)
                    species_data[idx_to_adjust]['num_offspring'] += 1 if discrepancy > 0 else -1
                    species_data[idx_to_adjust]['num_offspring'] = max(0, species_data[idx_to_adjust]['num_offspring'])
        else:
            num_species = len(species_data)
            if num_species > 0:
                offspring_per_species = self.size // num_species
                remainder = self.size % num_species
                for i, data in enumerate(species_data):
                    data['num_offspring'] = offspring_per_species + (1 if i < remainder else 0)

        for data in species_data:
            species = data['species']
            num_offspring = data.get('num_offspring', 0)
            ranked_members = data['ranked']

            if num_offspring == 0 or not ranked_members:
                continue

            elite_count = 0
            for i in range(min(num_elites, len(ranked_members), num_offspring)):
                elite_copy = ranked_members[i].copy()
                new_pop.append(elite_copy)
                elite_count += 1

            num_remaining_offspring = num_offspring - elite_count
            if num_remaining_offspring <= 0:
                continue

            selection_pool_size = max(1, int(len(ranked_members) * selection_share))
            selection_pool = ranked_members[:selection_pool_size]

            if not selection_pool:
                parent_fallback = ranked_members[0] if ranked_members else None
                for _ in range(num_remaining_offspring):
                    if parent_fallback:
                        offspring = parent_fallback.copy()
                        offspring.mutate()
                        new_pop.append(offspring)
                    else:
                        new_pop.append(Genome(self, self.genome_shape, config=self.config))
                continue

            for _ in range(num_remaining_offspring):
                parent1 = choice(selection_pool)
                parent2 = choice(selection_pool)

                max_try_crossover = 5
                offspring = None
                for attempt in range(max_try_crossover):
                    try:
                        offspring = self.cross_over(parent1, parent2)
                        break
                    except Exception as e:
                        print(f'Error during crossover (Attempt {attempt+1}/{max_try_crossover}): {e}. Parents: P1(Fit:{parent1.fitness:.2f}), P2(Fit:{parent2.fitness:.2f})')
                        if attempt == max_try_crossover - 1:
                            print("Crossover failed after retries. Using mutated copy of fitter parent.")
                            fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
                            offspring = fitter_parent.copy()

                if offspring:
                    offspring.mutate()
                    new_pop.append(offspring)
                else:
                    print("ERROR: Offspring generation failed completely. Adding random genome.")
                    new_pop.append(Genome(self, self.genome_shape, config=self.config))

        while len(new_pop) < self.size:
            new_pop.append(Genome(self, self.genome_shape, config=self.config))

        self.members = new_pop[:self.size]

        old_representatives = {s: s.representative for s in self.species}
        for s in self.species:
            s.members = []

        for genome in self.members:
            self.speciate(genome)

        self.species = [s for s in self.species if s.members]
        for s in self.species:
            original_rep = old_representatives.get(s)
            if original_rep and original_rep in s.members:
                s.representative = original_rep
            elif s.members:
                s.representative = choice(s.members)
            else:
                s.representative = None

        self.adjust_species_threshold()

    def gather_population(self):
        all_members = []
        for species in self.species:
            all_members.extend(species.members)
        return all_members

    def get_top_genome(self):
        if not self.members:
            raise IndexError("Cannot get top genome from an empty population.")
        top_genome = sorted(self.members, key=lambda g: g.fitness, reverse=True)[0]
        return top_genome

    def save_top_genome(self, filename):
        save_path = self.config.save_path if self.config is not None else "./"

        try:
            genome_to_save = self.get_top_genome()
            full_path = f'{save_path}{filename}.pkl'
            with open(full_path, 'wb') as f:
                pkl.dump(genome_to_save, f)
            print(f"Top genome saved successfully to {full_path}")
        except IndexError:
            print("Could not save top genome: Population is empty.")
        except Exception as e:
            print(f"Error saving top genome to {filename}.pkl: {e}")
