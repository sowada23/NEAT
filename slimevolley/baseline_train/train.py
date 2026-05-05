from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from neat import NEATConfig, Population
from slimevolley.baseline_train.evaluation import evaluate_genome_vs_baseline, evaluate_population_vs_baseline
from slimevolley.baseline_train.output import build_output_dir
from slimevolley.baseline_train.topology_svg import save_topology_svg


def _write_config(path: Path, args, config: NEATConfig) -> None:
    payload = {
        "args": vars(args),
        "neat_config": dict(config.__dict__),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _append_history(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_genome(path: Path, genome) -> None:
    with open(path, "wb") as f:
        pickle.dump(genome, f)


def _summary_from_metrics(metrics):
    fitnesses = np.asarray([m.fitness for m in metrics], dtype=np.float32)
    scores = np.asarray([m.score_mean for m in metrics], dtype=np.float32)
    steps = np.asarray([m.steps_mean for m in metrics], dtype=np.float32)
    return {
        "best_fitness": float(fitnesses.max()),
        "mean_fitness": float(fitnesses.mean()),
        "best_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "mean_steps": float(steps.mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: train NEAT directly against the SlimeVolley baseline.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--episodes-per-genome", type=int, default=8)
    parser.add_argument("--benchmark-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--survival-bonus-weight", type=float, default=0.0)
    parser.add_argument("--complexity-penalty-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--history-print-freq", type=int, default=1)
    parser.add_argument("--history-header-freq", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="slimevolley/baseline_train/output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.generations < 0:
        raise SystemExit("--generations must be >= 0")
    if args.population < 2:
        raise SystemExit("--population must be >= 2")
    if args.episodes_per_genome < 1 or args.benchmark_episodes < 1:
        raise SystemExit("episode counts must be >= 1")
    if args.history_print_freq < 1:
        raise SystemExit("--history-print-freq must be >= 1")
    if args.history_header_freq < 1:
        raise SystemExit("--history-header-freq must be >= 1")

    out_dir = build_output_dir(Path(args.out_dir).resolve())
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.csv"

    config = NEATConfig(
        population_size=args.population,
        genome_shape=(12, 3),
        hid_node_activation="tanh",
        out_node_activation="tanh",
        max_weight=3.0,
        min_weight=-3.0,
        min_bias=-3.0,
        max_bias=3.0,
        add_node_mutation_prob=0.01,
        add_conn_mutation_prob=0.03,
        remove_conn_mutation_prob=0.01,
        remove_node_mutation_prob=0.0,
        num_elites=2,
        selection_share=0.25,
        sigma=0.12,
        perturb_prob=0.90,
        reset_prob=0.05,
        species_threshold=2.0,
        min_species_threshold=0.15,
        max_species_threshold=15.0,
        target_species_number=5,
        adaptive_threshold=0.03,
        c1=1.0,
        c2=1.0,
        c3=0.8,
        save_path=str(out_dir) + "/",
    )
    _write_config(out_dir / "config.json", args, config)

    population = Population(config=config)
    best_ever = None
    best_ever_score = -1e9
    best_ever_metrics = None

    print(f"Output directory: {out_dir}", flush=True)
    print(f"History CSV:      {history_path}", flush=True)
    print("", flush=True)

    history_header = (
        " gen | best_fit | mean_fit | bench_mean | bench_std | best_ever | "
        "mean_score | mean_len | nodes | enabled | species"
    )
    history_rule = "-" * len(history_header)

    def print_history_header() -> None:
        print(history_rule, flush=True)
        print(history_header, flush=True)
        print(history_rule, flush=True)

    print_history_header()

    for generation in range(args.generations + 1):
        generation_seed = args.seed + generation * 100000
        metrics = evaluate_population_vs_baseline(
            population,
            episodes_per_genome=args.episodes_per_genome,
            seed_base=generation_seed,
            threshold=args.threshold,
            max_steps=args.max_steps,
            survival_bonus_weight=args.survival_bonus_weight,
            complexity_penalty_weight=args.complexity_penalty_weight,
        )
        best = population.get_top_genome()
        best_idx = max(range(len(population.members)), key=lambda idx: population.members[idx].fitness)
        best_metrics = metrics[best_idx]

        benchmark_metrics = evaluate_genome_vs_baseline(
            best,
            episodes=args.benchmark_episodes,
            seed_base=generation_seed + 50000,
            threshold=args.threshold,
            max_steps=args.max_steps,
            survival_bonus_weight=args.survival_bonus_weight,
            complexity_penalty_weight=args.complexity_penalty_weight,
        )

        if benchmark_metrics.score_mean > best_ever_score:
            best_ever_score = benchmark_metrics.score_mean
            best_ever = best.copy()
            best_ever_metrics = benchmark_metrics
            _save_genome(out_dir / "best_genome.pkl", best_ever)
            save_topology_svg(
                best_ever,
                out_dir / "best_topology.svg",
                title=f"Best Topology | Generation {generation}",
                baseline_score=best_ever_score,
            )

        summary = _summary_from_metrics(metrics)
        row = {
            "generation": generation,
            "best_fitness": summary["best_fitness"],
            "mean_fitness": summary["mean_fitness"],
            "best_baseline_score": benchmark_metrics.score_mean,
            "best_baseline_std": benchmark_metrics.score_std,
            "mean_baseline_score": summary["mean_score"],
            "mean_episode_length": summary["mean_steps"],
            "best_node_count": best_metrics.node_count,
            "best_connection_count": best_metrics.conn_count,
            "best_enabled_connection_count": best_metrics.enabled_conn_count,
            "species_count": len(population.species),
        }
        _append_history(history_path, row)

        if args.checkpoint_freq > 0 and generation % args.checkpoint_freq == 0:
            _save_genome(checkpoint_dir / f"gen_{generation:06d}.pkl", best.copy())

        if generation % args.history_print_freq == 0:
            if generation > 0 and generation % args.history_header_freq == 0:
                print_history_header()
            print(
                f"{generation:4d} | "
                f"{summary['best_fitness']:8.3f} | "
                f"{summary['mean_fitness']:8.3f} | "
                f"{benchmark_metrics.score_mean:10.3f} | "
                f"{benchmark_metrics.score_std:9.3f} | "
                f"{best_ever_score:9.3f} | "
                f"{summary['mean_score']:10.3f} | "
                f"{summary['mean_steps']:8.1f} | "
                f"{best_metrics.node_count:5d} | "
                f"{best_metrics.enabled_conn_count:7d} | "
                f"{len(population.species):7d}",
                flush=True,
            )

        if generation < args.generations:
            population.reproduce()

    final_best = population.get_top_genome()
    _save_genome(out_dir / "final_genome.pkl", final_best.copy())
    final_metrics = evaluate_genome_vs_baseline(
        final_best,
        episodes=args.benchmark_episodes,
        seed_base=args.seed + 900000,
        threshold=args.threshold,
        max_steps=args.max_steps,
        survival_bonus_weight=args.survival_bonus_weight,
        complexity_penalty_weight=args.complexity_penalty_weight,
    )
    save_topology_svg(
        final_best,
        out_dir / "final_topology.svg",
        title="Final Generation Best Topology",
        baseline_score=final_metrics.score_mean,
    )

    summary_lines = [
        "Stage 2 baseline training complete.",
        f"best_ever_score: {best_ever_score:.6f}",
        f"final_score: {final_metrics.score_mean:.6f} +/- {final_metrics.score_std:.6f}",
        f"best_genome: {out_dir / 'best_genome.pkl'}",
        f"final_genome: {out_dir / 'final_genome.pkl'}",
        f"best_topology: {out_dir / 'best_topology.svg'}",
        f"final_topology: {out_dir / 'final_topology.svg'}",
    ]
    if best_ever_metrics is not None:
        summary_lines.append("best_ever_metrics:")
        summary_lines.append(json.dumps(asdict(best_ever_metrics), indent=2, sort_keys=True))
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
