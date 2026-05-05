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

From the parent repo root, `/Users/sora/NEATJAX`, run:

```bash
python backprop/run_toy2d.py --dataset xor --generations 20 --population 40 --seed 7 --hidden-activation relu
```

Or install the standalone package and use its console script:

```bash
cd /Users/sora/NEATJAX/backprop
python -m pip install -e ".[test]"
backprop-neat-toy2d --dataset xor --generations 20 --population 40 --seed 7 --hidden-activation relu
```

Artifacts are written under `backprop/outputs/output_N/` when using the
root-friendly launcher:

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

Backprop-NEAT hidden nodes can use `tanh`, `sigmoid`, `softplus`, `silu`,
and `relu`. Binary classification output nodes use `sigmoid`.
