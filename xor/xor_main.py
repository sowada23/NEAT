from __future__ import annotations

import argparse
import pickle
import random

import numpy as np

from neat import NEATConfig, Population
from xor.output_utils import XOR_DIR, build_output_dir
from xor.plot import save_accuracy_svg
from xor.task import XOR_INPUTS, XOR_TARGETS, evaluate_genome_xor, evaluate_population_xor
from xor.topology import save_topology_gif


def _save_summary(best_genome, out_dir, generation, metrics):
    summary_path = out_dir / "summary.txt"
    outputs = np.asarray(metrics["outputs"], dtype=np.float32).reshape(-1)
    preds = np.asarray(metrics["predictions"], dtype=np.float32).reshape(-1)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"best_generation: {generation}\n")
        f.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"mse: {metrics['mse']:.6f}\n")
        f.write(f"fitness: {metrics['fitness']:.6f}\n")
        f.write("xor_table:\n")
        for idx, (x, target) in enumerate(zip(XOR_INPUTS, XOR_TARGETS)):
            f.write(
                f"  sample_{idx}: input={x.tolist()} target={float(target[0]):.1f} "
                f"output={float(outputs[idx]):.4f} prediction={int(preds[idx])}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Run XOR with the local NEAT implementation using the JAX execution bridge.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif-fps", type=int, default=5)
    parser.add_argument("--topology-sample-every", type=int, default=1)
    parser.add_argument("--stop-on-solve", action="store_true")
    parser.add_argument("--hid-node-activation", choices=["relu", "tanh"], default="tanh")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    out_dir = build_output_dir(XOR_DIR / "output")

    config = NEATConfig(
        population_size=args.population,
        genome_shape=(2, 1),
        hid_node_activation=args.hid_node_activation,
        out_node_activation="sigmoid",
        max_weight=3.0,
        min_weight=-3.0,
        min_bias=-3.0,
        max_bias=3.0,
        add_node_mutation_prob=0.05,
        add_conn_mutation_prob=0.10,
        remove_conn_mutation_prob=0.02,
        remove_node_mutation_prob=0.01,
        num_elites=2,
        selection_share=0.25,
        sigma=0.15,
        perturb_prob=0.90,
        reset_prob=0.10,
        species_threshold=3.0,
        min_species_threshold=0.15,
        max_species_threshold=15.0,
        target_species_number=15,
        adaptive_threshold=0.1,
        c1=1.0,
        c2=1.0,
        c3=0.4,
        save_path=str(out_dir) + "/",
    )

    population = Population(config=config)
    history = {
        "generation": [],
        "best_accuracy": [],
        "mean_accuracy": [],
        "best_fitness": [],
        "mean_fitness": [],
        "best_mse": [],
    }
    topology_history = []
    best_ever = None
    best_ever_generation = 0
    best_ever_score = (-1.0, float("inf"), -1.0)

    for generation in range(args.generations + 1):
        generation_metrics = evaluate_population_xor(population)
        best = population.get_top_genome()
        best_metrics = evaluate_genome_xor(best)

        history["generation"].append(generation)
        history["best_accuracy"].append(generation_metrics["best_accuracy"][0])
        history["mean_accuracy"].append(generation_metrics["mean_accuracy"][0])
        history["best_fitness"].append(generation_metrics["best_fitness"][0])
        history["mean_fitness"].append(generation_metrics["mean_fitness"][0])
        history["best_mse"].append(generation_metrics["best_mse"][0])
        topology_history.append((generation, best.copy(), float(best_metrics["accuracy"])))

        score_tuple = (float(best_metrics["accuracy"]), -float(best_metrics["mse"]), float(best_metrics["fitness"]))
        if score_tuple > best_ever_score:
            best_ever_score = score_tuple
            best_ever = best.copy()
            best_ever_generation = generation
            with open(out_dir / "best_xor_genome.pkl", "wb") as f:
                pickle.dump(best_ever, f)

        print(
            f"Generation {generation:03d} | "
            f"best_accuracy={best_metrics['accuracy']:.3f} | "
            f"mean_accuracy={generation_metrics['mean_accuracy'][0]:.3f} | "
            f"best_mse={best_metrics['mse']:.5f} | "
            f"best_fitness={best_metrics['fitness']:.3f} | "
            f"species={len(population.species)}"
        )

        solved = float(best_metrics["accuracy"]) >= 1.0 and float(best_metrics["mse"]) < 0.02
        if solved and args.stop_on_solve:
            break

        if generation < args.generations:
            population.reproduce()

    history_np = {key: np.asarray(value) for key, value in history.items()}
    np.savez(out_dir / "history.npz", **history_np)
    with open(out_dir / "topology_history.pkl", "wb") as f:
        pickle.dump(topology_history, f)

    if best_ever is None:
        raise RuntimeError("Training finished without a best genome.")

    best_metrics = evaluate_genome_xor(best_ever)
    _save_summary(best_ever, out_dir, best_ever_generation, best_metrics)

    svg_path = save_accuracy_svg(history_np, out_dir)
    topology_gif_path = save_topology_gif(
        topology_history,
        out_dir=out_dir,
        fps=args.gif_fps,
        sample_every=args.topology_sample_every,
    )

    print("\nXOR training complete.")
    print(f"Output dir:   {out_dir}")
    print(f"Best genome:  {out_dir / 'best_xor_genome.pkl'}")
    print(f"History:      {out_dir / 'history.npz'}")
    print(f"Accuracy SVG: {svg_path}")
    if topology_gif_path is not None:
        print(f"Topology GIF: {topology_gif_path}")
    else:
        print("Topology GIF: skipped because no frames were generated.")


if __name__ == "__main__":
    main()
