# Backprop-NEAT

Standalone Backprop-NEAT implementation for evolving feed-forward network
topologies while training each fixed topology with JAX backpropagation.

The package includes JS-playground-style 2D classification datasets:

- `circle`
- `xor`
- `gaussian`
- `spiral`

## Install

```bash
python -m pip install -e ".[test]"
```

## Run A Demo

```bash
backprop-neat-toy2d --dataset xor --generations 20 --population 40 --seed 7
```

Artifacts are written under `outputs/output_N/`:

- `best_genome.pkl`
- `history.npz`
- `summary.txt`
- `decision_boundary.png`
- `topology.png`

## Design

NEAT evolves acyclic feed-forward topology. During fitness evaluation each
genome is converted to a fixed-topology JAX representation, its weights and
biases are trained with binary cross-entropy, then the trained parameters are
copied back to the genome before reproduction.

Only smooth differentiable activations are used in Backprop-NEAT runs:
`tanh`, `sigmoid`, `softplus`, and `silu`.

