# NEAT + JAX Integration Notes

This repository currently has:

- A NEAT implementation in Python objects.
- SlimeVolley evaluation running through Gym in Python.

That split matters for GPU work:

- You can move NEAT network inference to JAX and `jit` it.
- You cannot run full SlimeVolley episode simulation on CUDA until the environment step/reset logic is also available in JAX.

## What was added

A small JAX bridge now lives in `neat/jax/`:

- `genome_to_jax(genome)` converts a mutable genome into immutable dense JAX arrays.
- `forward_jax(...)` evaluates one observation.
- `batched_forward_jax(...)` vectorizes evaluation across many observations.
- `compile_genome_forward(genome)` returns JIT-compiled forward and action functions.

## Example usage

```python
from neat.jax import compile_genome_forward

jax_genome, forward_fn, batched_fn, action_fn = compile_genome_forward(genome)

single_output = forward_fn(obs)
batched_outputs = batched_fn(obs_batch)
action = action_fn(obs)
```

## Recommended path to GPU training on a remote Linux server

1. Install JAX with CUDA support on the server.
2. Keep NEAT evolution on CPU first.
3. Convert each genome to JAX before evaluation.
4. Batch policy inference with `vmap`/`jit`.
5. If you want true GPU episode rollout, port the SlimeVolley environment to pure JAX state transitions.

## Current blocker for full CUDA simulation

`slimevolley/selfplay/episodes.py` creates a Gym environment and steps it inside Python loops. That means episode rollout is still CPU-bound even if the policy network is JAX-compiled.

To make rollouts GPU-native, replace:

- `gym.make(...)`
- Python `while not done`
- object-based env state

with:

- pure arrays for env state
- a JAX `reset(key)` function
- a JAX `step(state, action_right, action_left)` function
- `lax.scan` or `lax.while_loop` for whole-episode rollout

## Suggested migration plan

### Phase 1

Use JAX only for policy inference. This is the lowest-risk change and is now supported by the repo.

### Phase 2

Batch many matches together. The biggest practical speedup usually comes from evaluating many observations or environments at once.

### Phase 3

Port SlimeVolley physics and reward logic into a JAX-native environment. At that point you can run thousands of environments in parallel on GPU and use `vmap` + `scan`.
