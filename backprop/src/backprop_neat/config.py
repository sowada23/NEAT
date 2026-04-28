from __future__ import annotations

from dataclasses import dataclass, field


SMOOTH_ACTIVATIONS = ("tanh", "sigmoid", "softplus", "silu")


@dataclass
class BackpropNEATConfig:
    population_size: int = 100
    genome_shape: tuple[int, int] = (2, 1)
    allowed_activations: tuple[str, ...] = SMOOTH_ACTIVATIONS
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"
    min_weight: float = -1.0
    max_weight: float = 1.0
    min_bias: float = -1.0
    max_bias: float = 1.0
    add_node_mutation_prob: float = 0.05
    add_conn_mutation_prob: float = 0.10
    remove_conn_mutation_prob: float = 0.02
    remove_node_mutation_prob: float = 0.01
    activation_mutation_prob: float = 0.03
    weight_mutation_prob: float = 0.20
    bias_mutation_prob: float = 0.20
    mutation_sigma: float = 0.10
    num_elites: int = 2
    selection_share: float = 0.25
    species_threshold: float = 3.0
    target_species_number: int = 12
    adaptive_threshold: float = 0.05
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    backprop_steps: int = 80
    learning_rate: float = 0.03
    batch_size: int | None = None
    weight_decay: float = 1e-4
    complexity_conn_penalty: float = 0.01
    complexity_node_penalty: float = 0.03
    save_path: str = "outputs/"
    rng_seed: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        invalid = [name for name in self.allowed_activations if name not in SMOOTH_ACTIVATIONS]
        if invalid:
            raise ValueError(f"Unsupported smooth activations: {invalid}")
        if self.hidden_activation not in self.allowed_activations:
            raise ValueError("hidden_activation must be present in allowed_activations")
        if self.output_activation != "sigmoid":
            raise ValueError("Backprop-NEAT binary classification expects sigmoid output activation")
        if self.genome_shape[1] != 1:
            raise ValueError("Toy 2D binary classification expects exactly one output node")

