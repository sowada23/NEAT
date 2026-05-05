from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np

from backprop_neat.config import SUPPORTED_ACTIVATIONS, BackpropNEATConfig
from backprop_neat.datasets import generate_split
from backprop_neat.evolution.population import Population
from backprop_neat.jax import evaluate_and_train_genome
from backprop_neat.visualization import save_all_history_svgs, save_decision_boundary, save_topology


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
    numeric_keys = [
        "fitness",
        "raw_score",
        "complexity_penalty",
        "enabled_connections",
        "hidden_nodes",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]
    summary = {
        "species": float(len(population.species)),
    }
    for key in numeric_keys:
        values = [float(m[key]) for m in metrics]
        if key in {"train_loss", "test_loss", "complexity_penalty", "enabled_connections", "hidden_nodes"}:
            summary[f"best_{key}"] = float(best.metrics[key])
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"min_{key}"] = float(np.min(values))
        else:
            summary[f"best_{key}"] = float(best.metrics[key])
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"max_{key}"] = float(np.max(values))
    summary["best_nodes"] = float(len(best.nodes))
    summary["best_connections_total"] = float(len(best.connections))
    return {
        **summary,
    }


def _print_generation(generation: int, generations: int, summary: dict[str, float]) -> None:
    print(
        f"Generation {generation:03d}/{generations:03d} | "
        f"fitness best/mean={summary['best_fitness']:.4f}/{summary['mean_fitness']:.4f} | "
        f"test loss best/mean={summary['best_test_loss']:.4f}/{summary['mean_test_loss']:.4f} | "
        f"train loss best/mean={summary['best_train_loss']:.4f}/{summary['mean_train_loss']:.4f} | "
        f"test acc best/mean={summary['best_test_accuracy']:.3f}/{summary['mean_test_accuracy']:.3f} | "
        f"complexity={summary['best_complexity_penalty']:.3f} | "
        f"hidden={int(summary['best_hidden_nodes'])} | "
        f"enabled_conn={int(summary['best_enabled_connections'])} | "
        f"species={int(summary['species'])}",
        flush=True,
    )


def _save_history_csv(history: list[dict[str, float]], out_path: Path) -> Path:
    keys = sorted(history[0])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    return out_path


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
    parser.add_argument("--hidden-activation", choices=SUPPORTED_ACTIVATIONS, default="tanh")
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
        hidden_activation=args.hidden_activation,
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
    print(f"Output dir: {out_dir.resolve()}")
    print(
        f"Training dataset={args.dataset} generations={args.generations} population={args.population} "
        f"backprop_steps={args.backprop_steps} lr={args.learning_rate} "
        f"hidden_activation={args.hidden_activation} conn_penalty={args.conn_penalty} node_penalty={args.node_penalty}",
        flush=True,
    )

    for generation in range(args.generations + 1):
        summary = _evaluate_population(population, split, generation, args.seed)
        best = population.best()
        score = (float(best.metrics["test_accuracy"]), float(best.fitness))
        if score > best_score:
            best_score = score
            best_ever = best.copy()
            best_generation = generation
        history.append({"generation": float(generation), **summary})
        _print_generation(generation, args.generations, summary)
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
    _save_history_csv(history, out_dir / "history.csv")
    history_svgs = save_all_history_svgs(history, out_dir)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"hidden_activation: {args.hidden_activation}\n")
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
    print(f"History CSV: {out_dir / 'history.csv'}")
    print(f"Fitness SVG: {history_svgs['fitness']}")
    print(f"Loss SVG: {history_svgs['loss']}")
    print(f"Accuracy SVG: {history_svgs['accuracy']}")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
