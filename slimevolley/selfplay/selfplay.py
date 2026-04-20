
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
import math
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
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--tournaments", type=int, default=None)
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
    default_generations = 120

    history_baseline_score = []
    history_tournament = []
    genome_history = []

    effective_opponents_per_genome = min(
        args.opponents_per_genome,
        max(0, len(population.members) - 1),
    )
    tournaments_per_generation = len(population.members) * effective_opponents_per_genome

    if args.generations is not None and args.tournaments is not None:
        parser.error("Use either --generations or --tournaments, not both.")

    if args.tournaments is not None:
        if args.tournaments < 1:
            parser.error("--tournaments must be at least 1.")
        if tournaments_per_generation < 1:
            parser.error("Tournament mode requires at least 2 population members.")
        total_generations = max(0, math.ceil(args.tournaments / tournaments_per_generation) - 1)
    elif args.generations is not None:
        if args.generations < 0:
            parser.error("--generations must be at least 0.")
        total_generations = args.generations
    else:
        total_generations = default_generations


    for generation in range(total_generations + 1):
        generation_seed_base = args.seed + generation * 10000

        selfplay_scores, selfplay_lengths = evaluate_selfplay_population(
            population=population,
            opponents_per_genome=args.opponents_per_genome,
            seed_base=generation_seed_base,
        )
        selfplay_mean = float(np.mean(selfplay_scores))

        best = population.get_top_genome()
        benchmark_mean, benchmark_std, benchmark_len, benchmark_scores = evaluate_vs_baseline(
            genome=best,
            episodes=args.benchmark_episodes,
            seed_base=generation_seed_base + 7777,
        )
        plotted_baseline_score = float(benchmark_scores[0]) if benchmark_scores else benchmark_mean
        current_tournament = (generation + 1) * tournaments_per_generation
        genome_history.append((generation, current_tournament, best.copy(), plotted_baseline_score))
        history_baseline_score.append(plotted_baseline_score)
        history_tournament.append(current_tournament)

        if benchmark_mean > best_ever_baseline_score:
            best_ever_baseline_score = benchmark_mean
            best_ever = best.copy()
            with open(out_dir / "best_slimevolley_selfplay_genome.pkl", "wb") as f:
                pickle.dump(best_ever, f)

        print(
            f"Generation {generation:03d} | "
            f"tournament={history_tournament[-1]:06d} | "
            f"best_selfplay={max(selfplay_scores): .3f} | "
            f"mean_selfplay={selfplay_mean: .3f} | "
            f"mean_ep_len={float(np.mean(selfplay_lengths)): .1f} | "
            f"baseline_score={plotted_baseline_score: .3f} | "
            f"baseline_benchmark={benchmark_mean: .3f} ± {benchmark_std: .3f} | "
            f"species={len(population.species)}"
        )

        if generation < total_generations:
            population.reproduce()

    svg_path = save_baseline_benchmark_svg(
        history_baseline_score,
        out_dir,
        x_values=history_tournament,
        x_label="Tournament",
    )
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
