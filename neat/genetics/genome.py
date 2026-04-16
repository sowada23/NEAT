from random import uniform, choice, random, gauss

from neat.activations import select_activation
from neat.genetics.genes import Connection, Node


class Genome:
    """
    Represents a single individual (a neural network) in the population.
    """
    def __init__(self, population, shape=(1, 1), connections=None, nodes=None, config=None):
        """
        Creates a new genome.
        """
        using_config_shape = config is not None and config.genome_shape != (1, 1)
        using_param_shape = shape != (1, 1) and not using_config_shape
        default_init = (using_config_shape or using_param_shape) and connections is None and nodes is None

        using_config_default = config is not None and config.genome_shape == (1, 1)
        using_param_default = shape == (1, 1) and not using_config_default
        conn_node_init = (using_config_default or using_param_default) and connections is not None and nodes is not None

        assert default_init or conn_node_init, \
            "Provide either shape (or config with shape) for default init, or connections and nodes for explicit init."

        self.connections = {} if connections is None else connections
        self.nodes = {} if nodes is None else nodes
        self.population = population
        self.fitness = 0.0
        self.config = config
        self.shape = config.genome_shape if config is not None else shape

        if default_init:
            self.initialize(self.shape)

    def copy(self):
        """
        Creates and returns a deep copy of this genome.
        """
        new_nodes = {node_id: node.copy() for node_id, node in self.nodes.items()}
        new_conns = {}
        for conn_id, conn in self.connections.items():
            if conn.in_node.id in new_nodes and conn.out_node.id in new_nodes:
                new_conn = conn.copy()
                new_conn.in_node = new_nodes.get(conn.in_node.id)
                new_conn.out_node = new_nodes.get(conn.out_node.id)
                new_conns[conn_id] = new_conn

        copied_genome = Genome(self.population, shape=(1, 1), connections=new_conns, nodes=new_nodes, config=self.config)
        copied_genome.fitness = self.fitness
        copied_genome.shape = self.shape
        return copied_genome

    @property
    def sorted_conns(self):
        return sorted(self.connections.values(), key=lambda c: c.id)

    @property
    def sorted_nodes(self):
        return sorted(self.nodes.values(), key=lambda n: n.id)

    def add_node(self, connection_id):
        old_conn = self.connections.get(connection_id)
        if old_conn is None:
            print(f"Warning: Attempted to add node on non-existent connection ID {connection_id}. Mutation skipped.")
            return

        old_conn.enabled = False
        in_node = old_conn.in_node
        out_node = old_conn.out_node

        new_node_id = self.population.set_node_id()

        # hid_activation = self.config.hid_node_activation if self.config is not None else 'relu'
        hid_activation = choice(["relu", "tanh"])
        node = Node('hidden', new_node_id, activation=hid_activation)
        self.nodes[node.id] = node

        conn1_id = self.population.set_conn_id(in_node.id, node.id)
        conn1 = Connection(in_node, node, conn1_id)
        conn1.weight = 1.0
        self.connections[conn1.id] = conn1

        conn2_id = self.population.set_conn_id(node.id, out_node.id)
        conn2 = Connection(node, out_node, conn2_id)
        conn2.weight = old_conn.weight
        self.connections[conn2.id] = conn2

    def remove_node(self, node_id):
        node = self.nodes.get(node_id)

        if node is None or node.type != 'hidden':
            return

        for conn in list(self.connections.values()):
            if conn.in_node.id == node_id or conn.out_node.id == node_id:
                if conn.enabled:
                    conn.enabled = False

    def remove_connection(self, conn_id):
        connection = self.connections.get(conn_id)

        if connection is None:
            return

        if connection.in_node.type == 'input' and connection.out_node.type == 'output':
            return

        if connection.enabled:
            connection.enabled = False

    def check_connection(self, in_node_id, out_node_id):
        for conn in self.connections.values():
            if conn.in_node.id == in_node_id and conn.out_node.id == out_node_id:
                return True
        return False

    def check_node(self, node_id):
        return node_id in self.nodes

    def check_cycle(self, in_node_id, out_node_id):
        in_node = self.nodes.get(in_node_id) or Node('hidden', in_node_id)
        out_node = self.nodes.get(out_node_id) or Node('hidden', out_node_id)

        temp_conn = Connection(in_node, out_node, id=float('inf'))

        temp_connections = dict(self.connections)
        temp_connections[temp_conn.id] = temp_conn

        temp_nodes = dict(self.nodes)
        if not self.check_node(in_node.id):
            temp_nodes[in_node.id] = in_node
        if not self.check_node(out_node.id):
            temp_nodes[out_node.id] = out_node

        visited = {node_id: False for node_id in temp_nodes}
        rec_stack = {node_id: False for node_id in temp_nodes}

        graph = {node_id: [] for node_id in temp_nodes}
        for conn in temp_connections.values():
            if conn.enabled:
                if conn.in_node.id in graph and conn.out_node.id in graph:
                    graph[conn.out_node.id].append(conn.in_node.id)

        def dfs(node_id):
            visited[node_id] = True
            rec_stack[node_id] = True

            for neighbor in graph.get(node_id, []):
                if neighbor in visited:
                    if not visited[neighbor]:
                        if dfs(neighbor):
                            return True
                    elif rec_stack[neighbor]:
                        return True

            rec_stack[node_id] = False
            return False

        for node_id in temp_nodes:
            if node_id in visited and not visited[node_id]:
                if dfs(node_id):
                    return True

        return False

    def add_connection(self, in_node_id, out_node_id):
        in_node = self.nodes.get(in_node_id)
        out_node = self.nodes.get(out_node_id)

        if in_node is None or out_node is None:
            return False

        if out_node.type == 'input':
            return False
        if in_node.type == 'output':
            return False

        if self.check_connection(in_node_id, out_node_id):
            return False
        if self.check_connection(out_node_id, in_node_id):
            return False

        if self.check_cycle(in_node_id, out_node_id):
            return False

        conn_id = self.population.set_conn_id(in_node.id, out_node.id)
        conn = Connection(in_node, out_node, conn_id)
        self.connections[conn.id] = conn
        return True

    def add_node_mutation(self, prob=0.05):
        prob = self.config.add_node_mutation_prob if self.config is not None else prob

        if random() < prob:
            enabled_conns = [c for c in self.connections.values() if c.enabled]
            if not enabled_conns:
                return

            conn_to_split = choice(enabled_conns)
            self.add_node(conn_to_split.id)

    def add_connection_mutation(self, prob=0.08):
        prob = self.config.add_conn_mutation_prob if self.config is not None else prob

        if random() < prob:
            if len(self.nodes) < 2:
                return

            max_try = 35
            for _ in range(max_try):
                if not self.nodes:
                    return

                node1 = choice(list(self.nodes.values()))
                node2 = choice(list(self.nodes.values()))

                if node1.id == node2.id:
                    continue

                if self.add_connection(node1.id, node2.id):
                    break
                elif self.add_connection(node2.id, node1.id):
                    break

    def remove_node_mutation(self, prob=0.01):
        prob = self.config.remove_node_mutation_prob if self.config else prob

        if random() < prob:
            hidden_nodes = [n for n in self.nodes.values() if n.type == 'hidden']
            if not hidden_nodes:
                return

            node_to_remove = choice(hidden_nodes)
            self.remove_node(node_to_remove.id)

    def remove_connection_mutation(self, prob=0.02):
        prob = self.config.remove_conn_mutation_prob if self.config else prob

        if random() < prob and self.connections:
            conn_id_to_remove = choice(list(self.connections.keys()))
            self.remove_connection(conn_id_to_remove)

    def weight_mutation(self, sigma=0.1, perturb_prob=0.8, reset_prob=0.1, min_weight=-1.0, max_weight=1.0):
        sigma = self.config.sigma if self.config is not None else sigma
        perturb_prob = self.config.perturb_prob if self.config is not None else perturb_prob
        reset_prob = self.config.reset_prob if self.config is not None else reset_prob
        min_weight = self.config.min_weight if self.config is not None else min_weight
        max_weight = self.config.max_weight if self.config is not None else max_weight

        for conn in self.connections.values():
            if random() < perturb_prob:
                if random() < reset_prob:
                    conn.weight = uniform(-1, 1)
                else:
                    conn.weight += gauss(0, sigma)
                conn.weight = max(min_weight, min(max_weight, conn.weight))

    def bias_mutation(self, sigma=0.1, perturb_prob=0.8, reset_prob=0.1, min_bias=-1.0, max_bias=1.0):
        sigma = self.config.sigma if self.config is not None else sigma
        perturb_prob = self.config.perturb_prob if self.config is not None else perturb_prob
        reset_prob = self.config.reset_prob if self.config is not None else reset_prob
        min_bias = self.config.min_bias if self.config is not None else min_bias
        max_bias = self.config.max_bias if self.config is not None else max_bias

        mutable_nodes = [n for n in self.nodes.values() if n.type != 'input']

        for node in mutable_nodes:
            if random() < perturb_prob:
                if random() < reset_prob:
                    node.bias = uniform(-1, 1)
                else:
                    node.bias += gauss(0, sigma)
                node.bias = max(min_bias, min(max_bias, node.bias))

    def activation_mutation(self, prob=0.05): # new function for activation flexibility
        if random() < prob:
            hidden_nodes = [n for n in self.nodes.values() if n.type == "hidden"]
            if not hidden_nodes:
                return

            node = choice(hidden_nodes)

            if node.activation == "relu":
                node.activation = "tanh"
            else:
                node.activation = "relu"

    def mutate(self):
        self.weight_mutation()
        self.bias_mutation()
        self.activation_mutation()
        self.add_node_mutation()
        self.add_connection_mutation()
        self.remove_connection_mutation()
        self.remove_node_mutation()

    def topological_sort(self):
        adj_map = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        for conn in self.connections.values():
            if conn.enabled:
                if conn.in_node.id in in_degree and conn.out_node.id in in_degree:
                    if conn.out_node.id not in adj_map[conn.in_node.id]:
                        adj_map[conn.in_node.id].append(conn.out_node.id)
                    in_degree[conn.out_node.id] += 1

        queue = sorted([node_id for node_id, degree in in_degree.items() if degree == 0])
        sorted_nodes_ids = []

        while queue:
            node_id = queue.pop(0)
            sorted_nodes_ids.append(node_id)

            neighbors = sorted(adj_map.get(node_id, []))
            for adj_node_id in neighbors:
                if adj_node_id in in_degree:
                    in_degree[adj_node_id] -= 1
                    if in_degree[adj_node_id] == 0:
                        queue.append(adj_node_id)
            queue.sort()

        if len(sorted_nodes_ids) != len(self.nodes):
            raise ValueError("Graph contains a cycle, topological sort not possible.")

        return sorted_nodes_ids

    def forward(self, x):
        for node in self.nodes.values():
            node.value = 0.0

        try:
            sorted_node_ids = self.topological_sort()
        except ValueError as e:
            print(f"Forward pass failed: {e}. Network likely has a cycle.")
            num_output_nodes = len([n for n in self.nodes.values() if n.type == 'output'])
            return [0.0] * num_output_nodes

        input_nodes = [node for node in self.nodes.values() if node.type == 'input']
        if len(x) != len(input_nodes):
            raise ValueError(f"Input vector size {len(x)} does not match number of input nodes {len(input_nodes)}")

        input_nodes.sort(key=lambda n: n.id)
        for i, node in enumerate(input_nodes):
            if node.id in self.nodes:
                self.nodes[node.id].value = float(x[i])

        conn_map = {(conn.in_node.id, conn.out_node.id): conn for conn in self.connections.values() if conn.enabled}
        pred_map = {node_id: [] for node_id in self.nodes}
        for conn in self.connections.values():
            if conn.enabled and conn.in_node.id in self.nodes and conn.out_node.id in self.nodes:
                pred_map[conn.out_node.id].append(conn.in_node.id)

        for node_id in sorted_node_ids:
            node = self.nodes.get(node_id)
            if node and node.type != 'input':
                weighted_sum = 0.0
                predecessors = pred_map.get(node_id, [])
                for pred_node_id in predecessors:
                    conn = conn_map.get((pred_node_id, node_id))
                    if conn:
                        pred_node = self.nodes.get(pred_node_id)
                        if pred_node:
                            weighted_sum += pred_node.value * conn.weight

                if node.activation:
                    activation_function = select_activation(node.activation)
                    node.value = float(activation_function(weighted_sum + node.bias))
                else:
                    node.value = float(weighted_sum + node.bias)

        output_values = []
        output_nodes = [node for node in self.nodes.values() if node.type == 'output']
        output_nodes.sort(key=lambda n: n.id)
        for node in output_nodes:
            output_values.append(node.value)

        return output_values

    def initialize(self, shape, activation='sigmoid'):
        activation = self.config.out_node_activation if self.config is not None else activation

        self.nodes = {}
        self.connections = {}
        in_nodes = []
        out_nodes = []

        for i in range(shape[0]):
            in_node = Node('input', i, activation=None)
            in_nodes.append(in_node)
            self.nodes[in_node.id] = in_node

        for j in range(shape[1]):
            out_node_id = shape[0] + j
            out_node = Node('output', out_node_id, activation=activation)
            out_nodes.append(out_node)
            self.nodes[out_node_id] = out_node

        for in_node in in_nodes:
            for out_node in out_nodes:
                conn_id = self.population.set_conn_id(in_node.id, out_node.id)
                conn = Connection(in_node, out_node, conn_id)
                self.connections[conn.id] = conn

        self.population.node_id = max(self.population.node_id, shape[0] + shape[1] - 1)

    def print_graph(self):
        print("--- Genome Graph ---")
        print(f"Fitness: {self.fitness:.4f}")

        print("\nNodes:")
        if not self.nodes:
            print("  (No nodes defined)")
        else:
            for node in self.sorted_nodes:
                print(f"  {node}")

        print("\nConnections:")
        if not self.connections:
            print("  (No connections defined)")
        else:
            for conn in self.sorted_conns:
                print(f"  {conn}")

        print("--------------------\n")
