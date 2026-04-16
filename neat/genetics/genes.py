from random import uniform


class Connection:
    """
    Represents a connection gene in a genome.
    """
    def __init__(self, in_node, out_node, id=0):
        """
        Initializes a new connection gene.
        """
        self.in_node = in_node
        self.out_node = out_node
        self.id = id
        self.weight = uniform(-1, 1)  # Initialize weight randomly between -1 and 1
        self.enabled = True  # Connections start enabled by default

    def copy(self):
        """
        Creates and returns a deep copy of this connection.
        """
        conn = Connection(self.in_node, self.out_node, self.id)
        conn.enabled = self.enabled
        conn.weight = self.weight
        return conn

    def __repr__(self):
        """
        Returns a string representation of the connection.
        """
        return f"id: {self.id}: nodes: {self.in_node.id} -> {self.out_node.id} (W:{self.weight:.2f} E:{self.enabled})"


class Node:
    """
    Represents a node gene in a genome.
    """
    def __init__(self, type, id, activation=None):
        """
        Creates a new node gene.
        """
        self.type = type
        self.value = 0.0  # Runtime value, reset before each forward pass
        self.bias = uniform(-1, 1) if type != 'input' else 0.0
        self.activation = activation
        self.id = id

    def copy(self):
        """
        Creates and returns a deep copy of this node.
        """
        node = Node(self.type, self.id, self.activation)
        node.bias = self.bias
        return node

    def __hash__(self):
        """
        Computes the hash based on the node's unique ID.
        """
        return hash(self.id)

    def __eq__(self, other):
        """
        Checks equality based on the node's unique ID.
        """
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return f"Node(id={self.id}, type={self.type}, act={self.activation}, bias={self.bias:.2f})"
