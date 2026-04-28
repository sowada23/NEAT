from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeGene:
    kind: str
    id: int
    activation: str | None = None
    bias: float = 0.0

    def copy(self) -> "NodeGene":
        return NodeGene(self.kind, self.id, self.activation, self.bias)


@dataclass
class ConnectionGene:
    in_node: NodeGene
    out_node: NodeGene
    id: int
    weight: float
    enabled: bool = True

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(self.in_node, self.out_node, self.id, self.weight, self.enabled)

