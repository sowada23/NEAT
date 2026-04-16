from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neat.genetics.genome import Genome

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised only when JAX is missing
    raise ImportError(
        "neat.jax requires the optional 'jax' dependency. "
        "Install JAX before importing this module."
    ) from exc


_ACTIVATION_LINEAR = 0
_ACTIVATION_RELU = 1
_ACTIVATION_SIGMOID = 2
_ACTIVATION_TANH = 3


@dataclass(frozen=True)
class JAXGenome:
    """Immutable JAX-friendly representation of a NEAT genome."""

    weights: jax.Array
    biases: jax.Array
    topo_order: jax.Array
    input_indices: jax.Array
    output_indices: jax.Array
    activation_codes: jax.Array
    input_mask: jax.Array


def _activation_code(name: str | None) -> int:
    if name is None:
        return _ACTIVATION_LINEAR
    if name == "relu":
        return _ACTIVATION_RELU
    if name == "sigmoid":
        return _ACTIVATION_SIGMOID
    if name == "tanh":
        return _ACTIVATION_TANH
    raise ValueError(f"Unsupported activation for JAX execution: {name}")


def _apply_activation(x, code):
    return jax.lax.switch(
        code,
        (
            lambda y: y,
            jax.nn.relu,
            jax.nn.sigmoid,
            jnp.tanh,
        ),
        x,
    )


def genome_to_jax(genome: Genome) -> JAXGenome:
    """Convert a mutable Genome object into dense JAX arrays."""
    node_ids = sorted(genome.nodes)
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    topo_order = np.asarray(
        [node_index[node_id] for node_id in genome.topological_sort()],
        dtype=np.int32,
    )

    num_nodes = len(node_ids)
    weights = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    biases = np.zeros(num_nodes, dtype=np.float32)
    activation_codes = np.zeros(num_nodes, dtype=np.int32)
    input_mask = np.zeros(num_nodes, dtype=bool)

    input_indices = []
    output_indices = []

    for node_id in node_ids:
        node = genome.nodes[node_id]
        idx = node_index[node_id]
        biases[idx] = float(node.bias)
        activation_codes[idx] = _activation_code(node.activation)
        if node.type == "input":
            input_mask[idx] = True
            input_indices.append(idx)
        elif node.type == "output":
            output_indices.append(idx)

    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        in_idx = node_index[conn.in_node.id]
        out_idx = node_index[conn.out_node.id]
        weights[in_idx, out_idx] = float(conn.weight)

    return JAXGenome(
        weights=jnp.asarray(weights),
        biases=jnp.asarray(biases),
        topo_order=jnp.asarray(topo_order),
        input_indices=jnp.asarray(np.asarray(input_indices, dtype=np.int32)),
        output_indices=jnp.asarray(np.asarray(output_indices, dtype=np.int32)),
        activation_codes=jnp.asarray(activation_codes),
        input_mask=jnp.asarray(input_mask),
    )


def forward_jax(jax_genome: JAXGenome, x):
    """Evaluate one observation with JAX."""
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.shape[-1] != jax_genome.input_indices.shape[0]:
        raise ValueError(
            f"Input vector size {x.shape[-1]} does not match "
            f"number of input nodes {jax_genome.input_indices.shape[0]}"
        )

    values = jnp.zeros_like(jax_genome.biases)
    values = values.at[jax_genome.input_indices].set(x)

    def body_fun(i, current_values):
        node_idx = jax_genome.topo_order[i]

        def update(vals):
            weighted_sum = jnp.dot(vals, jax_genome.weights[:, node_idx]) + jax_genome.biases[node_idx]
            activated = _apply_activation(weighted_sum, jax_genome.activation_codes[node_idx])
            return vals.at[node_idx].set(activated)

        return jax.lax.cond(
            jax_genome.input_mask[node_idx],
            lambda vals: vals,
            update,
            current_values,
        )

    values = jax.lax.fori_loop(0, jax_genome.topo_order.shape[0], body_fun, values)
    return values[jax_genome.output_indices]


def batched_forward_jax(jax_genome: JAXGenome, xs):
    """Vectorized forward pass over a batch of observations."""
    xs = jnp.asarray(xs, dtype=jnp.float32)
    return jax.vmap(lambda x: forward_jax(jax_genome, x))(xs)


def genome_to_jax_action(jax_genome: JAXGenome, obs):
    """Convert JAX network outputs into SlimeVolley button actions."""
    outputs = forward_jax(jax_genome, obs)
    return (outputs > 0.5).astype(jnp.int8)


def compile_genome_forward(genome: Genome):
    """Build JIT-compiled forward and action functions for one genome."""
    jax_genome = genome_to_jax(genome)
    forward_fn = jax.jit(lambda x: forward_jax(jax_genome, x))
    batched_fn = jax.jit(lambda xs: batched_forward_jax(jax_genome, xs))
    action_fn = jax.jit(lambda obs: genome_to_jax_action(jax_genome, obs))
    return jax_genome, forward_fn, batched_fn, action_fn
