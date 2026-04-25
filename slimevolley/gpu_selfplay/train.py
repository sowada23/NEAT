from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from neat import NEATConfig, Population
from slimevolley.gpu_selfplay.evaluation import evaluate_selfplay_population_gpu
from slimevolley.selfplay.output import build_output_dir


def _safe_evaluate_vs_baseline(genome, episodes: int, seed_base: int):
    if episodes <= 0:
        return None
    try:
        from slimevolley.selfplay.evaluation import evaluate_vs_baseline
    except Exception:
        return None
    try:
        return evaluate_vs_baseline(genome=genome, episodes=episodes, seed_base=seed_base)
    except Exception:
        return None


def _safe_save_baseline_benchmark_svg(history_mean, history_std, out_dir):
    try:
        from slimevolley.selfplay.viz.benchmark_plot import save_baseline_benchmark_svg
    except Exception:
        return None
    try:
        return save_baseline_benchmark_svg(history_mean, history_std, out_dir)
    except Exception:
        return None


def _safe_save_champion_gif(genome, out_dir, seed: int, fps: int):
    try:
        from slimevolley.selfplay.viz.gif import save_champion_gif
    except Exception:
        return None, None, None
    try:
        return save_champion_gif(genome, out_dir=out_dir, seed=seed, fps=fps)
    except Exception:
        return None, None, None


def _safe_save_topology_evolution_gif(genome_history, out_dir, fps: int, sample_every: int):
    try:
        from slimevolley.selfplay.viz.topology import save_topology_evolution_gif
    except Exception:
        return None
    try:
        return save_topology_evolution_gif(
            genome_history,
            out_dir=out_dir,
            fps=fps,
            sample_every=sample_every,
        )
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Alternative GPU self-play trainer using a JAX SlimeVolley backend and the existing NEAT implementation."
    )
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--opponents-per-genome", type=int, default=4)
    parser.add_argument("--benchmark-episodes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gif-seed", type=int, default=1234)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--topology-gif-fps", type=int, default=6)
    parser.add_argument("--topology-sample-every", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="selfplay/viz/slimevolley_gpu_selfplay")
    args = parser.parse_args()

    slimevolley_root = Path(__file__).resolve().parent.parent
    out_dir = build_output_dir((slimevolley_root / args.out_dir).resolve())

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
    history_benchmark_mean = []
    history_benchmark_std = []
    genome_history = []

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
        benchmark_result = _safe_evaluate_vs_baseline(
            genome=best,
            episodes=args.benchmark_episodes,
            seed_base=generation_seed_base + 7777,
        )
        if benchmark_result is not None:
            benchmark_mean, benchmark_std, _benchmark_len, _benchmark_scores = benchmark_result
            history_benchmark_mean.append(float(benchmark_mean))
            history_benchmark_std.append(float(benchmark_std))
        else:
            benchmark_mean = float("nan")
            benchmark_std = float("nan")
            history_benchmark_mean.append(float("nan"))
            history_benchmark_std.append(float("nan"))

        if best_score > best_ever_score:
            best_ever_score = best_score
            best_ever = best.copy()
            with open(out_dir / "best_slimevolley_gpu_selfplay_genome.pkl", "wb") as f:
                pickle.dump(best_ever, f)

        genome_history.append((generation, best.copy(), benchmark_mean))
        benchmark_text = (
            f"baseline_benchmark={benchmark_mean: .3f} ± {benchmark_std: .3f} | "
            if benchmark_result is not None
            else "baseline_benchmark= unavailable | "
        )

        print(
            f"Generation {generation:03d} | "
            f"best_selfplay={best_score: .3f} | "
            f"mean_selfplay={float(np.mean(selfplay_scores)): .3f} | "
            f"mean_ep_len={float(np.mean(selfplay_lengths)): .1f} | "
            f"{benchmark_text}"
            f"species={len(population.species)}"
        )

        if generation < args.generations:
            population.reproduce()

    svg_path = None
    if history_benchmark_mean and np.isfinite(np.asarray(history_benchmark_mean)).any():
        svg_path = _safe_save_baseline_benchmark_svg(
            history_benchmark_mean,
            history_benchmark_std,
            out_dir,
        )

    gif_path = None
    gif_score = None
    gif_steps = None
    if best_ever is not None:
        gif_path, gif_score, gif_steps = _safe_save_champion_gif(
            best_ever,
            out_dir=out_dir,
            seed=args.gif_seed,
            fps=args.gif_fps,
        )

    topology_gif_path = None
    if genome_history:
        topology_gif_path = _safe_save_topology_evolution_gif(
            genome_history,
            out_dir=out_dir,
            fps=args.topology_gif_fps,
            sample_every=args.topology_sample_every,
        )

    print("\nGPU self-play training complete.")
    print(f"Best genome: {out_dir / 'best_slimevolley_gpu_selfplay_genome.pkl'}")
    if svg_path is not None:
        print(f"SVG plot: {svg_path}")
    else:
        print("SVG plot: skipped because baseline benchmarking was unavailable.")
    if gif_path is not None:
        print(f"Gameplay GIF: {gif_path} (score={gif_score:.3f}, steps={gif_steps})")
    else:
        print("Gameplay GIF: skipped because baseline rollout or frame capture was unavailable.")
    if topology_gif_path is not None:
        print(f"Topology GIF: {topology_gif_path}")
    else:
        print("Topology GIF: skipped because topology frames were unavailable.")


if __name__ == "__main__":
    main()
