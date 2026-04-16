
"""
Main training script for NEAT self-play in SlimeVolley.

This file is the controller of the whole program.
It reads command-line settings, creates the NEAT population,
runs training generation by generation, and saves final results.

In short:
- starts training
- calls evaluation code
- tracks the best genome
- saves plots and GIFs
"""


from __future__ import annotations

import argparse
import pickle

import numpy as np

from slimevolley.selfplay.bootstrap import EXAMPLES_DIR, NEATConfig, Population
from slimevolley.selfplay.evaluation import evaluate_selfplay_population, evaluate_vs_baseline
from slimevolley.selfplay.output import build_output_dir
from slimevolley.selfplay.viz.benchmark_plot import save_baseline_benchmark_svg
from slimevolley.selfplay.viz.gif import save_champion_gif
from slimevolley.selfplay.viz.topology import save_topology_evolution_gif


def main():
    parser = argparse.ArgumentParser(
        description="Train the local NEAT implementation on SlimeVolley using self-play only."
    )
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--opponents-per-genome", type=int, default=4)
    parser.add_argument("--benchmark-episodes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif-seed", type=int, default=1234)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--topology-gif-fps", type=int, default=6)
    parser.add_argument("--topology-sample-every", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="out/slimevolley_neat_selfplay")
    args = parser.parse_args()

    out_dir = build_output_dir((EXAMPLES_DIR / args.out_dir).resolve())

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
    best_ever_baseline_score = -1e9

    history_benchmark_mean = []
    history_benchmark_std = []
    genome_history = []


    for generation in range(args.generations + 1):
        generation_seed_base = args.seed + generation * 10000

        selfplay_scores, selfplay_lengths = evaluate_selfplay_population(
            population=population,
            opponents_per_genome=args.opponents_per_genome,
            seed_base=generation_seed_base,
        )

        best = population.get_top_genome()
        benchmark_mean, benchmark_std, benchmark_len = evaluate_vs_baseline(
            genome=best,
            episodes=args.benchmark_episodes,
            seed_base=generation_seed_base + 7777,
        )
        genome_history.append((generation, best.copy(), benchmark_mean))
        history_benchmark_mean.append(benchmark_mean)
        history_benchmark_std.append(benchmark_std)

        if benchmark_mean > best_ever_baseline_score:
            best_ever_baseline_score = benchmark_mean
            best_ever = best.copy()
            with open(out_dir / "best_slimevolley_selfplay_genome.pkl", "wb") as f:
                pickle.dump(best_ever, f)

        print(
            f"Generation {generation:03d} | "
            f"best_selfplay={max(selfplay_scores): .3f} | "
            f"mean_selfplay={float(np.mean(selfplay_scores)): .3f} | "
            f"mean_ep_len={float(np.mean(selfplay_lengths)): .1f} | "
            f"baseline_benchmark={benchmark_mean: .3f} ± {benchmark_std: .3f} | "
            f"species={len(population.species)}"
        )

        if generation < args.generations:
            population.reproduce()

    svg_path = save_baseline_benchmark_svg(history_benchmark_mean, history_benchmark_std, out_dir)
    gif_path, gif_score, gif_steps = save_champion_gif(
        best_ever,
        out_dir=out_dir,
        seed=args.gif_seed,
        fps=args.gif_fps,
    )
    topology_gif_path = save_topology_evolution_gif(
        genome_history,
        out_dir=out_dir,
        fps=args.topology_gif_fps,
        sample_every=args.topology_sample_every,
    )

    print("\nTraining complete.")
    print(f"Best genome:   {out_dir / 'best_slimevolley_selfplay_genome.pkl'}")
    print(f"SVG plot:      {svg_path}")
    if gif_path is not None:
        print(f"Gameplay GIF:  {gif_path} (score={gif_score:.3f}, steps={gif_steps})")
    else:
        print("Gameplay GIF:  skipped because frame capture was unavailable.")
    if topology_gif_path is not None:
        print(f"Topology GIF:  {topology_gif_path}")
    else:
        print("Topology GIF:  skipped because no frames were generated.")


if __name__ == "__main__":
    main()
