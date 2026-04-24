
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
    parser.add_argument("--opponents-per-genome", type=int, default=16)
    parser.add_argument("--benchmark-episodes", type=int, default=25)
    parser.add_argument("--baseline-fitness-weight", type=float, default=0.30)
    parser.add_argument("--baseline-fitness-episodes", type=int, default=3)
    parser.add_argument("--hall-of-fame-size", type=int, default=20)
    parser.add_argument("--hall-of-fame-save-freq", type=int, default=5)
    parser.add_argument("--hall-of-fame-opponents", type=int, default=4)
    parser.add_argument("--plot-save-freq", type=int, default=1000)
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
        out_node_activation="tanh",
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
        reset_prob=0.1,
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
    total_tournaments_played = 0
    checkpoint_history = []
    hall_of_fame = []
    checkpoint_dir = out_dir / "checkpoint_genomes"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    if args.plot_save_freq < 1:
        parser.error("--plot-save-freq must be at least 1.")
    if not 0.0 <= args.baseline_fitness_weight <= 1.0:
        parser.error("--baseline-fitness-weight must be between 0.0 and 1.0.")
    if args.baseline_fitness_episodes < 0:
        parser.error("--baseline-fitness-episodes must be at least 0.")
    if args.hall_of_fame_size < 0:
        parser.error("--hall-of-fame-size must be at least 0.")
    if args.hall_of_fame_save_freq < 1:
        parser.error("--hall-of-fame-save-freq must be at least 1.")
    if args.hall_of_fame_opponents < 0:
        parser.error("--hall-of-fame-opponents must be at least 0.")


    for generation in range(total_generations + 1):
        generation_seed_base = args.seed + generation * 10000
        tournaments_remaining = None
        if args.tournaments is not None:
            tournaments_remaining = max(0, args.tournaments - total_tournaments_played)
            if tournaments_remaining == 0:
                break
        tournaments_this_generation = tournaments_per_generation
        if tournaments_remaining is not None:
            tournaments_this_generation = min(tournaments_per_generation, tournaments_remaining)

        selfplay_scores, selfplay_lengths, tournament_baseline_scores = evaluate_selfplay_population(
            population=population,
            opponents_per_genome=args.opponents_per_genome,
            seed_base=generation_seed_base,
            max_tournaments=tournaments_this_generation,
            baseline_seed_base=generation_seed_base + 5000,
            hall_of_fame_genomes=hall_of_fame,
            hall_of_fame_opponents=args.hall_of_fame_opponents,
            baseline_fitness_episodes=args.baseline_fitness_episodes,
            baseline_fitness_weight=args.baseline_fitness_weight,
        )
        selfplay_mean = float(np.mean(selfplay_scores))

        best = population.get_top_genome()
        benchmark_mean, benchmark_std, benchmark_len, benchmark_scores = evaluate_vs_baseline(
            genome=best,
            episodes=args.benchmark_episodes,
            seed_base=generation_seed_base + 7777,
        )
        total_tournaments_played += len(tournament_baseline_scores)
        history_baseline_score.extend(tournament_baseline_scores)
        history_tournament.extend(
            range(
                total_tournaments_played - len(tournament_baseline_scores) + 1,
                total_tournaments_played + 1,
            )
        )
        current_tournament = total_tournaments_played
        topology_score = float(benchmark_scores[0]) if benchmark_scores else benchmark_mean
        genome_history.append((generation, current_tournament, best.copy(), topology_score))

        checkpoints_crossed = current_tournament // args.plot_save_freq
        checkpoints_saved = len(checkpoint_history)
        if checkpoints_crossed > checkpoints_saved:
            checkpoint_genome = best.copy()
            for checkpoint_idx in range(checkpoints_saved + 1, checkpoints_crossed + 1):
                checkpoint_tournament = checkpoint_idx * args.plot_save_freq
                checkpoint_path = checkpoint_dir / f"checkpoint_tournament_{checkpoint_tournament:08d}.pkl"
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_genome.copy(), f)
                checkpoint_history.append((checkpoint_tournament, checkpoint_genome.copy()))

        if benchmark_mean > best_ever_baseline_score:
            best_ever_baseline_score = benchmark_mean
            best_ever = best.copy()
            with open(out_dir / "best_slimevolley_selfplay_genome.pkl", "wb") as f:
                pickle.dump(best_ever, f)

        if args.hall_of_fame_size > 0 and generation % args.hall_of_fame_save_freq == 0:
            hall_of_fame.append(best.copy())
            if len(hall_of_fame) > args.hall_of_fame_size:
                hall_of_fame = hall_of_fame[-args.hall_of_fame_size :]

        print(
            f"Generation {generation:03d} | "
            f"tournament={history_tournament[-1]:06d} | "
            f"best_selfplay={max(selfplay_scores): .3f} | "
            f"mean_selfplay={selfplay_mean: .3f} | "
            f"mean_ep_len={float(np.mean(selfplay_lengths)): .1f} | "
            f"baseline_score={topology_score: .3f} | "
            f"baseline_benchmark={benchmark_mean: .3f} ± {benchmark_std: .3f} | "
            f"species={len(population.species)} | "
            f"hall_of_fame={len(hall_of_fame)}"
        )

        if generation < total_generations:
            population.reproduce()

    if total_tournaments_played > 0 and (
        not checkpoint_history or checkpoint_history[-1][0] != total_tournaments_played
    ):
        final_checkpoint_path = checkpoint_dir / f"checkpoint_tournament_{total_tournaments_played:08d}.pkl"
        with open(final_checkpoint_path, "wb") as f:
            pickle.dump(best.copy(), f)
        checkpoint_history.append((total_tournaments_played, best.copy()))

    plot_tournaments = []
    plot_benchmark_mean = []
    plot_benchmark_std = []
    for checkpoint_tournament, checkpoint_genome in checkpoint_history:
        checkpoint_mean, checkpoint_std, _checkpoint_len, _checkpoint_scores = evaluate_vs_baseline(
            genome=checkpoint_genome,
            episodes=args.benchmark_episodes,
            seed_base=args.seed + 900000 + checkpoint_tournament,
        )
        plot_tournaments.append(checkpoint_tournament / 1000.0)
        plot_benchmark_mean.append(checkpoint_mean)
        plot_benchmark_std.append(checkpoint_std)

    svg_path = save_baseline_benchmark_svg(
        plot_benchmark_mean,
        plot_benchmark_std,
        out_dir,
        x_values=plot_tournaments,
        x_label="Tournament (K)",
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
