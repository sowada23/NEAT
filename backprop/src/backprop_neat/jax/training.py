from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from backprop_neat.genetics.genome import Genome

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install JAX to use backprop_neat.jax") from exc


ACT_LINEAR = 0
ACT_TANH = 1
ACT_SIGMOID = 2
ACT_SOFTPLUS = 3
ACT_SILU = 4


@dataclass(frozen=True)
class JAXGenome:
    node_ids: tuple[int, ...]
    conn_ids: tuple[int, ...]
    weight_sources: jax.Array
    weight_targets: jax.Array
    bias_nodes: jax.Array
    topo_order: jax.Array
    input_indices: jax.Array
    output_indices: jax.Array
    activation_codes: jax.Array
    input_mask: jax.Array
    initial_weights: jax.Array
    initial_biases: jax.Array


class TrainState(NamedTuple):
    weights: jax.Array
    biases: jax.Array


def _activation_code(name: str | None) -> int:
    if name is None:
        return ACT_LINEAR
    if name == "tanh":
        return ACT_TANH
    if name == "sigmoid":
        return ACT_SIGMOID
    if name == "softplus":
        return ACT_SOFTPLUS
    if name == "silu":
        return ACT_SILU
    raise ValueError(f"Unsupported activation for Backprop-NEAT JAX training: {name}")


def _apply_activation(x, code):
    return jax.lax.switch(
        code,
        (
            lambda y: y,
            jnp.tanh,
            jax.nn.sigmoid,
            jax.nn.softplus,
            jax.nn.silu,
        ),
        x,
    )


def genome_to_jax(genome: Genome) -> JAXGenome:
    node_ids = tuple(sorted(genome.nodes))
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    enabled = sorted(genome.enabled_connections(), key=lambda conn: conn.id)
    bias_nodes = [node_id for node_id in node_ids if genome.nodes[node_id].kind != "input"]
    return JAXGenome(
        node_ids=node_ids,
        conn_ids=tuple(conn.id for conn in enabled),
        weight_sources=jnp.asarray([node_index[c.in_node.id] for c in enabled], dtype=jnp.int32),
        weight_targets=jnp.asarray([node_index[c.out_node.id] for c in enabled], dtype=jnp.int32),
        bias_nodes=jnp.asarray([node_index[node_id] for node_id in bias_nodes], dtype=jnp.int32),
        topo_order=jnp.asarray([node_index[node_id] for node_id in genome.topological_sort()], dtype=jnp.int32),
        input_indices=jnp.asarray(
            [node_index[n.id] for n in sorted(genome.nodes.values(), key=lambda n: n.id) if n.kind == "input"],
            dtype=jnp.int32,
        ),
        output_indices=jnp.asarray(
            [node_index[n.id] for n in sorted(genome.nodes.values(), key=lambda n: n.id) if n.kind == "output"],
            dtype=jnp.int32,
        ),
        activation_codes=jnp.asarray([_activation_code(genome.nodes[node_id].activation) for node_id in node_ids], dtype=jnp.int32),
        input_mask=jnp.asarray([genome.nodes[node_id].kind == "input" for node_id in node_ids]),
        initial_weights=jnp.asarray([conn.weight for conn in enabled], dtype=jnp.float32),
        initial_biases=jnp.asarray([genome.nodes[node_id].bias for node_id in bias_nodes], dtype=jnp.float32),
    )


def forward_state(jax_genome: JAXGenome, state: TrainState, x):
    x = jnp.asarray(x, dtype=jnp.float32)
    n_nodes = len(jax_genome.node_ids)
    dense_weights = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float32)
    dense_weights = dense_weights.at[jax_genome.weight_sources, jax_genome.weight_targets].set(state.weights)
    dense_biases = jnp.zeros((n_nodes,), dtype=jnp.float32)
    dense_biases = dense_biases.at[jax_genome.bias_nodes].set(state.biases)
    values = jnp.zeros((n_nodes,), dtype=jnp.float32)
    values = values.at[jax_genome.input_indices].set(x)

    def body_fun(i, vals):
        node_idx = jax_genome.topo_order[i]

        def update(v):
            z = jnp.dot(v, dense_weights[:, node_idx]) + dense_biases[node_idx]
            return v.at[node_idx].set(_apply_activation(z, jax_genome.activation_codes[node_idx]))

        return jax.lax.cond(jax_genome.input_mask[node_idx], lambda v: v, update, vals)

    values = jax.lax.fori_loop(0, jax_genome.topo_order.shape[0], body_fun, values)
    return values[jax_genome.output_indices]


def batched_forward_state(jax_genome: JAXGenome, state: TrainState, xs):
    return jax.vmap(lambda x: forward_state(jax_genome, state, x))(xs)


def binary_cross_entropy(preds, ys):
    preds = jnp.clip(preds, 1e-6, 1.0 - 1e-6)
    return -jnp.mean(ys * jnp.log(preds) + (1.0 - ys) * jnp.log(1.0 - preds))


def loss_fn(jax_genome: JAXGenome, state: TrainState, xs, ys, weight_decay: float):
    preds = batched_forward_state(jax_genome, state, xs)
    loss = binary_cross_entropy(preds, ys)
    if weight_decay > 0.0:
        loss = loss + weight_decay * jnp.mean(jnp.square(state.weights))
    return loss


def train_genome(
    genome: Genome,
    train_x,
    train_y,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    batch_size: int | None = None,
    seed: int = 0,
) -> tuple[TrainState, dict[str, float]]:
    jax_genome = genome_to_jax(genome)
    state = TrainState(jax_genome.initial_weights, jax_genome.initial_biases)
    xs = jnp.asarray(train_x, dtype=jnp.float32)
    ys = jnp.asarray(train_y, dtype=jnp.float32).reshape((-1, 1))
    grad_fn = jax.value_and_grad(lambda s, bx, by: loss_fn(jax_genome, s, bx, by, weight_decay))
    key = jax.random.PRNGKey(seed)
    n = int(xs.shape[0])

    for step in range(max(0, steps)):
        if batch_size is None or batch_size <= 0 or batch_size >= n:
            bx, by = xs, ys
        else:
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, n, shape=(batch_size,), replace=False)
            bx, by = xs[idx], ys[idx]
        _, grads = grad_fn(state, bx, by)
        state = TrainState(
            weights=state.weights - learning_rate * grads.weights,
            biases=state.biases - learning_rate * grads.biases,
        )

    final_loss = float(loss_fn(jax_genome, state, xs, ys, weight_decay))
    preds = np.asarray(batched_forward_state(jax_genome, state, xs))
    accuracy = float(np.mean((preds >= 0.5).astype(np.float32) == np.asarray(ys)))
    return state, {"loss": final_loss, "accuracy": accuracy}


def write_state_to_genome(genome: Genome, state: TrainState) -> None:
    jax_genome = genome_to_jax(genome)
    weights = np.asarray(state.weights, dtype=np.float32)
    biases = np.asarray(state.biases, dtype=np.float32)
    for conn_id, weight in zip(jax_genome.conn_ids, weights):
        genome.connections[conn_id].weight = float(weight)
    bias_node_ids = [node_id for node_id in jax_genome.node_ids if genome.nodes[node_id].kind != "input"]
    for node_id, bias in zip(bias_node_ids, biases):
        genome.nodes[node_id].bias = float(bias)


def evaluate_state(jax_genome: JAXGenome, state: TrainState, xs, ys) -> dict[str, float]:
    xs = jnp.asarray(xs, dtype=jnp.float32)
    ys_np = np.asarray(ys, dtype=np.float32).reshape((-1, 1))
    preds = np.asarray(batched_forward_state(jax_genome, state, xs), dtype=np.float32)
    loss = float(binary_cross_entropy(jnp.asarray(preds), jnp.asarray(ys_np)))
    accuracy = float(np.mean((preds >= 0.5).astype(np.float32) == ys_np))
    return {"loss": loss, "accuracy": accuracy}


def evaluate_and_train_genome(
    genome: Genome,
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int | None,
    conn_penalty: float,
    node_penalty: float,
    seed: int,
) -> dict[str, float]:
    state, train_metrics = train_genome(
        genome,
        train_x,
        train_y,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
    )
    write_state_to_genome(genome, state)
    jax_genome = genome_to_jax(genome)
    test_metrics = evaluate_state(jax_genome, state, test_x, test_y)
    enabled_conn_count = len(genome.enabled_connections())
    hidden_node_count = len(genome.hidden_nodes())
    complexity = conn_penalty * enabled_conn_count + node_penalty * hidden_node_count
    raw_score = 10.0 * test_metrics["accuracy"] - test_metrics["loss"]
    fitness = max(0.0, raw_score - complexity)
    metrics = {
        "fitness": fitness,
        "raw_score": raw_score,
        "complexity_penalty": complexity,
        "enabled_connections": float(enabled_conn_count),
        "hidden_nodes": float(hidden_node_count),
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
    }
    genome.fitness = fitness
    genome.metrics = metrics
    return metrics
