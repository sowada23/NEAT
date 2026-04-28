from __future__ import annotations

import random
from collections import defaultdict

from backprop_neat.activations import select_activation
from backprop_neat.config import BackpropNEATConfig
from backprop_neat.genetics.genes import ConnectionGene, NodeGene


class Genome:
    def __init__(
        self,
        population,
        config: BackpropNEATConfig,
        nodes: dict[int, NodeGene] | None = None,
        connections: dict[int, ConnectionGene] | None = None,
    ):
        self.population = population
        self.config = config
        self.nodes: dict[int, NodeGene] = {} if nodes is None else nodes
        self.connections: dict[int, ConnectionGene] = {} if connections is None else connections
        self.fitness = 0.0
        self.metrics: dict[str, float] = {}
        if nodes is None and connections is None:
            self.initialize()

    @property
    def sorted_nodes(self) -> list[NodeGene]:
        return sorted(self.nodes.values(), key=lambda n: n.id)

    @property
    def sorted_connections(self) -> list[ConnectionGene]:
        return sorted(self.connections.values(), key=lambda c: c.id)

    def copy(self) -> "Genome":
        nodes = {node_id: node.copy() for node_id, node in self.nodes.items()}
        connections = {}
        for conn_id, conn in self.connections.items():
            if conn.in_node.id in nodes and conn.out_node.id in nodes:
                copied = conn.copy()
                copied.in_node = nodes[conn.in_node.id]
                copied.out_node = nodes[conn.out_node.id]
                connections[conn_id] = copied
        child = Genome(self.population, self.config, nodes=nodes, connections=connections)
        child.fitness = self.fitness
        child.metrics = dict(self.metrics)
        return child

    def initialize(self) -> None:
        n_inputs, n_outputs = self.config.genome_shape
        for node_id in range(n_inputs):
            self.nodes[node_id] = NodeGene("input", node_id, activation=None, bias=0.0)
        for out_idx in range(n_outputs):
            node_id = n_inputs + out_idx
            self.nodes[node_id] = NodeGene(
                "output",
                node_id,
                activation=self.config.output_activation,
                bias=random.uniform(self.config.min_bias, self.config.max_bias),
            )
        for in_id in range(n_inputs):
            for out_id in range(n_inputs, n_inputs + n_outputs):
                self._create_connection(in_id, out_id)
        self.population.node_id = max(self.population.node_id, n_inputs + n_outputs - 1)

    def _create_connection(self, in_id: int, out_id: int, weight: float | None = None) -> ConnectionGene:
        conn_id = self.population.set_conn_id(in_id, out_id)
        conn = ConnectionGene(
            self.nodes[in_id],
            self.nodes[out_id],
            conn_id,
            random.uniform(self.config.min_weight, self.config.max_weight) if weight is None else weight,
            True,
        )
        self.connections[conn_id] = conn
        return conn

    def enabled_connections(self) -> list[ConnectionGene]:
        return [conn for conn in self.connections.values() if conn.enabled]

    def hidden_nodes(self) -> list[NodeGene]:
        return [node for node in self.nodes.values() if node.kind == "hidden"]

    def has_connection(self, in_id: int, out_id: int) -> bool:
        return any(c.in_node.id == in_id and c.out_node.id == out_id for c in self.connections.values())

    def would_create_cycle(self, in_id: int, out_id: int) -> bool:
        adjacency: dict[int, list[int]] = defaultdict(list)
        for conn in self.enabled_connections():
            adjacency[conn.in_node.id].append(conn.out_node.id)
        adjacency[in_id].append(out_id)

        visiting: set[int] = set()
        visited: set[int] = set()

        def dfs(node_id: int) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            for nxt in adjacency.get(node_id, []):
                if dfs(nxt):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(dfs(node_id) for node_id in self.nodes)

    def add_connection(self, in_id: int, out_id: int) -> bool:
        if in_id == out_id or in_id not in self.nodes or out_id not in self.nodes:
            return False
        if self.nodes[in_id].kind == "output" or self.nodes[out_id].kind == "input":
            return False
        if self.has_connection(in_id, out_id):
            return False
        if self.would_create_cycle(in_id, out_id):
            return False
        self._create_connection(in_id, out_id)
        return True

    def add_node(self, conn_id: int) -> bool:
        old_conn = self.connections.get(conn_id)
        if old_conn is None or not old_conn.enabled:
            return False
        old_conn.enabled = False
        node_id = self.population.set_node_id()
        activation = random.choice(tuple(self.config.allowed_activations))
        node = NodeGene(
            "hidden",
            node_id,
            activation=activation,
            bias=random.uniform(self.config.min_bias, self.config.max_bias),
        )
        self.nodes[node_id] = node
        self._create_connection(old_conn.in_node.id, node_id, weight=1.0)
        self._create_connection(node_id, old_conn.out_node.id, weight=old_conn.weight)
        return True

    def remove_node(self, node_id: int) -> bool:
        node = self.nodes.get(node_id)
        if node is None or node.kind != "hidden":
            return False
        for conn in self.connections.values():
            if conn.in_node.id == node_id or conn.out_node.id == node_id:
                conn.enabled = False
        return True

    def remove_connection(self, conn_id: int) -> bool:
        conn = self.connections.get(conn_id)
        if conn is None:
            return False
        if conn.in_node.kind == "input" and conn.out_node.kind == "output":
            return False
        conn.enabled = False
        return True

    def mutate(self) -> None:
        self._mutate_weights()
        self._mutate_biases()
        self._mutate_activation()
        if random.random() < self.config.add_node_mutation_prob:
            candidates = self.enabled_connections()
            if candidates:
                self.add_node(random.choice(candidates).id)
        if random.random() < self.config.add_conn_mutation_prob:
            node_ids = list(self.nodes)
            for _ in range(40):
                if self.add_connection(random.choice(node_ids), random.choice(node_ids)):
                    break
        if random.random() < self.config.remove_conn_mutation_prob and self.connections:
            self.remove_connection(random.choice(list(self.connections)))
        if random.random() < self.config.remove_node_mutation_prob:
            hidden = self.hidden_nodes()
            if hidden:
                self.remove_node(random.choice(hidden).id)

    def _mutate_weights(self) -> None:
        for conn in self.connections.values():
            if random.random() < self.config.weight_mutation_prob:
                conn.weight += random.gauss(0.0, self.config.mutation_sigma)
                conn.weight = max(self.config.min_weight, min(self.config.max_weight, conn.weight))

    def _mutate_biases(self) -> None:
        for node in self.nodes.values():
            if node.kind != "input" and random.random() < self.config.bias_mutation_prob:
                node.bias += random.gauss(0.0, self.config.mutation_sigma)
                node.bias = max(self.config.min_bias, min(self.config.max_bias, node.bias))

    def _mutate_activation(self) -> None:
        if random.random() >= self.config.activation_mutation_prob:
            return
        hidden = self.hidden_nodes()
        if hidden:
            random.choice(hidden).activation = random.choice(tuple(self.config.allowed_activations))

    def topological_sort(self) -> list[int]:
        adjacency = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}
        for conn in self.enabled_connections():
            adjacency[conn.in_node.id].append(conn.out_node.id)
            in_degree[conn.out_node.id] += 1
        queue = sorted(node_id for node_id, degree in in_degree.items() if degree == 0)
        ordered = []
        while queue:
            node_id = queue.pop(0)
            ordered.append(node_id)
            for nxt in sorted(adjacency[node_id]):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
            queue.sort()
        if len(ordered) != len(self.nodes):
            raise ValueError("Genome contains a cycle")
        return ordered

    def forward(self, x) -> list[float]:
        values = {node_id: 0.0 for node_id in self.nodes}
        inputs = sorted([n for n in self.nodes.values() if n.kind == "input"], key=lambda n: n.id)
        if len(x) != len(inputs):
            raise ValueError(f"Expected {len(inputs)} inputs, got {len(x)}")
        for idx, node in enumerate(inputs):
            values[node.id] = float(x[idx])
        incoming: dict[int, list[ConnectionGene]] = defaultdict(list)
        for conn in self.enabled_connections():
            incoming[conn.out_node.id].append(conn)
        for node_id in self.topological_sort():
            node = self.nodes[node_id]
            if node.kind == "input":
                continue
            total = node.bias
            for conn in incoming[node_id]:
                total += values[conn.in_node.id] * conn.weight
            values[node_id] = select_activation(node.activation)(total)
        outputs = sorted([n for n in self.nodes.values() if n.kind == "output"], key=lambda n: n.id)
        return [values[node.id] for node in outputs]

