from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from neat import NEATConfig, Population
from slimevolley.gpu_selfplay.evaluation import evaluate_selfplay_population_gpu


def build_output_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("output_")]
    next_idx = 0
    if existing:
        next_idx = max(int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()) + 1
    out_dir = base / f"output_{next_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Alternative GPU self-play trainer using a JAX SlimeVolley backend and the existing NEAT implementation."
    )
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--opponents-per-genome", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="out/slimevolley_gpu_selfplay")
    args = parser.parse_args()

    out_dir = build_output_dir(Path(args.out_dir).resolve())

    config = NEATConfig(
        population_size=args.population,
        genome_shape=(12, 3),
        hid_node_activation="tanh",
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
    best_ever = None
    best_ever_score = -1e9

    for generation in range(args.generations + 1):
        generation_seed_base = args.seed + generation * 10000
        selfplay_scores, selfplay_lengths = evaluate_selfplay_population_gpu(
            population=population,
            opponents_per_genome=args.opponents_per_genome,
            seed_base=generation_seed_base,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
        )

        best = population.get_top_genome()
        best_score = max(selfplay_scores) if selfplay_scores else -1e9
        if best_score > best_ever_score:
            best_ever_score = best_score
            best_ever = best.copy()
            with open(out_dir / "best_slimevolley_gpu_selfplay_genome.pkl", "wb") as f:
                pickle.dump(best_ever, f)

        print(
            f"Generation {generation:03d} | "
            f"best_selfplay={best_score: .3f} | "
            f"mean_selfplay={float(np.mean(selfplay_scores)): .3f} | "
            f"mean_ep_len={float(np.mean(selfplay_lengths)): .1f} | "
            f"species={len(population.species)}"
        )

        if generation < args.generations:
            population.reproduce()

    print("\nGPU self-play training complete.")
    print(f"Best genome: {out_dir / 'best_slimevolley_gpu_selfplay_genome.pkl'}")


if __name__ == "__main__":
    main()
