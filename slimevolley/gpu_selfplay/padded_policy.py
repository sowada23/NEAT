from __future__ import annotations

from typing import NamedTuple

import numpy as np

from neat.genetics.genome import Genome

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "slimevolley.gpu_selfplay.padded_policy requires the optional 'jax' dependency."
    ) from exc


ACT_LINEAR = 0
ACT_RELU = 1
ACT_SIGMOID = 2
ACT_TANH = 3


class BatchedPolicyGenome(NamedTuple):
    weights: jax.Array
    biases: jax.Array
    topo_order: jax.Array
    topo_mask: jax.Array
    input_indices: jax.Array
    output_indices: jax.Array
    activation_codes: jax.Array
    input_mask: jax.Array


def _activation_code(name: str | None) -> int:
    if name is None:
        return ACT_LINEAR
    if name == "relu":
        return ACT_RELU
    if name == "sigmoid":
        return ACT_SIGMOID
    if name == "tanh":
        return ACT_TANH
    raise ValueError(f"Unsupported activation for padded JAX policy: {name}")


def genomes_to_batched_policy(genomes: list[Genome]) -> BatchedPolicyGenome:
    if not genomes:
        raise ValueError("Expected at least one genome.")

    num_genomes = len(genomes)
    input_count = len([n for n in genomes[0].nodes.values() if n.type == "input"])
    output_ids = sorted(n.id for n in genomes[0].nodes.values() if n.type == "output")
    output_count = len(output_ids)
    max_nodes = max(len(g.nodes) for g in genomes)

    weights = np.zeros((num_genomes, max_nodes, max_nodes), dtype=np.float32)
    biases = np.zeros((num_genomes, max_nodes), dtype=np.float32)
    topo_order = np.zeros((num_genomes, max_nodes), dtype=np.int32)
    topo_mask = np.zeros((num_genomes, max_nodes), dtype=bool)
    input_indices = np.zeros((num_genomes, input_count), dtype=np.int32)
    output_indices = np.zeros((num_genomes, output_count), dtype=np.int32)
    activation_codes = np.zeros((num_genomes, max_nodes), dtype=np.int32)
    input_mask = np.zeros((num_genomes, max_nodes), dtype=bool)

    for genome_idx, genome in enumerate(genomes):
        node_ids = sorted(genome.nodes)
        local_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        topo = [local_index[node_id] for node_id in genome.topological_sort()]

        topo_order[genome_idx, : len(topo)] = np.asarray(topo, dtype=np.int32)
        topo_mask[genome_idx, : len(topo)] = True

        input_ids = sorted(n.id for n in genome.nodes.values() if n.type == "input")
        output_ids_local = sorted(n.id for n in genome.nodes.values() if n.type == "output")
        input_indices[genome_idx] = np.asarray([local_index[nid] for nid in input_ids], dtype=np.int32)
        output_indices[genome_idx] = np.asarray([local_index[nid] for nid in output_ids_local], dtype=np.int32)

        for node_id in node_ids:
            node = genome.nodes[node_id]
            idx = local_index[node_id]
            biases[genome_idx, idx] = float(node.bias)
            activation_codes[genome_idx, idx] = _activation_code(node.activation)
            if node.type == "input":
                input_mask[genome_idx, idx] = True

        for conn in genome.connections.values():
            if not conn.enabled:
                continue
            in_idx = local_index[conn.in_node.id]
            out_idx = local_index[conn.out_node.id]
            weights[genome_idx, in_idx, out_idx] = float(conn.weight)

    return BatchedPolicyGenome(
        weights=jnp.asarray(weights),
        biases=jnp.asarray(biases),
        topo_order=jnp.asarray(topo_order),
        topo_mask=jnp.asarray(topo_mask),
        input_indices=jnp.asarray(input_indices),
        output_indices=jnp.asarray(output_indices),
        activation_codes=jnp.asarray(activation_codes),
        input_mask=jnp.asarray(input_mask),
    )


def gather_batched_genomes(batched: BatchedPolicyGenome, indices: jax.Array) -> BatchedPolicyGenome:
    return jax.tree_util.tree_map(lambda x: x[indices], batched)


def _apply_activation(x: jax.Array, code: jax.Array) -> jax.Array:
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


def _forward_single(genome: BatchedPolicyGenome, obs: jax.Array) -> jax.Array:
    values = jnp.zeros_like(genome.biases)
    values = values.at[genome.input_indices].set(obs)

    def body_fn(i, vals):
        node_idx = genome.topo_order[i]
        active = genome.topo_mask[i]

        def compute(v):
            weighted_sum = jnp.dot(v, genome.weights[:, node_idx]) + genome.biases[node_idx]
            activated = _apply_activation(weighted_sum, genome.activation_codes[node_idx])
            return jax.lax.cond(genome.input_mask[node_idx], lambda s: s, lambda s: s.at[node_idx].set(activated), v)

        return jax.lax.cond(active, compute, lambda v: v, vals)

    values = jax.lax.fori_loop(0, genome.topo_order.shape[0], body_fn, values)
    return values[genome.output_indices]


def forward_batched(genomes: BatchedPolicyGenome, obs_batch: jax.Array) -> jax.Array:
    return jax.vmap(_forward_single)(genomes, obs_batch)


def policy_actions_batched(genomes: BatchedPolicyGenome, obs_batch: jax.Array) -> jax.Array:
    outputs = forward_batched(genomes, obs_batch)
    return (outputs > 0.5).astype(jnp.int8)
