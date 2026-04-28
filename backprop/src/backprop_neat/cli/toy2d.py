from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np

from backprop_neat.config import BackpropNEATConfig
from backprop_neat.datasets import generate_split
from backprop_neat.evolution.population import Population
from backprop_neat.jax import evaluate_and_train_genome
from backprop_neat.visualization import save_decision_boundary, save_topology


def _next_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("output_") and path.name.removeprefix("output_").isdigit():
            indices.append(int(path.name.removeprefix("output_")))
    out_dir = root / f"output_{max(indices, default=0) + 1}"
    out_dir.mkdir()
    return out_dir


def _evaluate_population(population: Population, split, generation: int, seed: int) -> dict[str, float]:
    metrics = []
    for idx, genome in enumerate(population.members):
        metrics.append(
            evaluate_and_train_genome(
                genome,
                split.train_x,
                split.train_y,
                split.test_x,
                split.test_y,
                steps=population.config.backprop_steps,
                learning_rate=population.config.learning_rate,
                weight_decay=population.config.weight_decay,
                batch_size=population.config.batch_size,
                conn_penalty=population.config.complexity_conn_penalty,
                node_penalty=population.config.complexity_node_penalty,
                seed=seed + generation * 100_000 + idx,
            )
        )
    best = population.best()
    return {
        "best_fitness": float(best.fitness),
        "best_test_accuracy": float(best.metrics["test_accuracy"]),
        "best_test_loss": float(best.metrics["test_loss"]),
        "mean_fitness": float(np.mean([m["fitness"] for m in metrics])),
        "mean_test_accuracy": float(np.mean([m["test_accuracy"] for m in metrics])),
        "species": float(len(population.species)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Backprop-NEAT on a toy 2D classification dataset.")
    parser.add_argument("--dataset", choices=["circle", "xor", "gaussian", "spiral"], default="xor")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--backprop-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--conn-penalty", type=float, default=0.01)
    parser.add_argument("--node-penalty", type=float, default=0.03)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--stop-accuracy", type=float, default=1.01)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    split = generate_split(args.dataset, args.train_size, args.test_size, args.noise, args.seed)
    out_dir = _next_output_dir(args.outputs)
    config = BackpropNEATConfig(
        population_size=args.population,
        genome_shape=(2, 1),
        backprop_steps=args.backprop_steps,
        learning_rate=args.learning_rate,
        batch_size=None if args.batch_size <= 0 else args.batch_size,
        complexity_conn_penalty=args.conn_penalty,
        complexity_node_penalty=args.node_penalty,
        rng_seed=args.seed,
        save_path=str(out_dir),
    )
    population = Population(config)
    history = []
    best_ever = None
    best_score = (-1.0, -1.0)
    best_generation = 0

    for generation in range(args.generations + 1):
        summary = _evaluate_population(population, split, generation, args.seed)
        best = population.best()
        score = (float(best.metrics["test_accuracy"]), float(best.fitness))
        if score > best_score:
            best_score = score
            best_ever = best.copy()
            best_generation = generation
        history.append({"generation": float(generation), **summary})
        print(
            f"Generation {generation:03d} | "
            f"best_acc={summary['best_test_accuracy']:.3f} | "
            f"mean_acc={summary['mean_test_accuracy']:.3f} | "
            f"best_fitness={summary['best_fitness']:.3f} | "
            f"species={int(summary['species'])}"
        )
        if summary["best_test_accuracy"] >= args.stop_accuracy:
            break
        if generation < args.generations:
            population.reproduce()

    if best_ever is None:
        raise RuntimeError("No best genome found")

    with open(out_dir / "best_genome.pkl", "wb") as f:
        pickle.dump(best_ever, f)
    keys = sorted(history[0])
    np.savez(out_dir / "history.npz", **{key: np.asarray([row[key] for row in history]) for key in keys})
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"best_generation: {best_generation}\n")
        for key, value in sorted(best_ever.metrics.items()):
            f.write(f"{key}: {value}\n")
    save_decision_boundary(
        best_ever,
        split.test_x,
        split.test_y,
        out_dir / "decision_boundary.png",
        title=f"{args.dataset} decision boundary",
    )
    save_topology(best_ever, out_dir / "topology.png", title=f"{args.dataset} best topology")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

