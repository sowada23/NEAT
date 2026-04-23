
"""
Evaluate how well genomes perform.

This file does not train directly.
Instead, it measures performance by running many episodes and
computing average scores.

It contains two kinds of evaluation:
- self-play evaluation across the current population
- benchmark evaluation against the fixed baseline policy

In short:
- chooses opponents
- runs matches
- averages scores and episode lengths
- assigns fitness values to genomes
"""

import numpy as np

from .episodes import run_selfplay_episode, run_vs_baseline_episode


def _run_two_sided_matchup(genome_right, genome_left, seed):
    score_right, steps_right, _ = run_selfplay_episode(
        genome_right=genome_right,
        genome_left=genome_left,
        seed=seed,
        capture_frames=False,
    )
    score_left_perspective, steps_left, _ = run_selfplay_episode(
        genome_right=genome_left,
        genome_left=genome_right,
        seed=seed + 1,
        capture_frames=False,
    )
    genome_left_score = -score_left_perspective
    return (
        score_right + genome_left_score,
        steps_right + steps_left,
        2,
    )


def evaluate_selfplay_population(
    population,
    opponents_per_genome,
    seed_base,
    max_tournaments=None,
    baseline_seed_base=None,
    hall_of_fame_genomes=None,
    hall_of_fame_opponents=0,
    baseline_fitness_episodes=0,
    baseline_fitness_weight=0.0,
):
    """
    Evaluate each genome only against other genomes in the current population.

    Each sampled pairing plays two matches with swapped sides to reduce side bias.
    """
    n = len(population.members)
    total_scores = [0.0] * n
    total_lengths = [0.0] * n
    games_played = [0] * n

    rng = np.random.default_rng(seed_base)
    hall_of_fame_genomes = hall_of_fame_genomes or []
    tournament_baseline_scores = []
    tournaments_played = 0
    selfplay_fitness_weight = max(0.0, 1.0 - baseline_fitness_weight)

    for i, genome in enumerate(population.members):
        candidate_indices = [j for j in range(n) if j != i]
        if not candidate_indices:
            continue

        sample_size = min(opponents_per_genome, len(candidate_indices))
        opponents = rng.choice(candidate_indices, size=sample_size, replace=False)

        for local_idx, j in enumerate(opponents):
            if max_tournaments is not None and tournaments_played >= max_tournaments:
                break

            opponent = population.members[int(j)]
            match_seed = seed_base + i * 1000 + local_idx * 10

            matchup_score, matchup_steps, matchup_games = _run_two_sided_matchup(
                genome_right=genome,
                genome_left=opponent,
                seed=match_seed,
            )
            total_scores[i] += matchup_score
            total_lengths[i] += matchup_steps
            games_played[i] += matchup_games
            tournaments_played += 1

            if baseline_seed_base is not None:
                baseline_score, _baseline_steps, _baseline_frames = run_vs_baseline_episode(
                    genome=genome,
                    seed=baseline_seed_base + tournaments_played - 1,
                    capture_frames=False,
                )
                tournament_baseline_scores.append(float(baseline_score))

        if max_tournaments is not None and tournaments_played >= max_tournaments:
            break

        if hall_of_fame_genomes and hall_of_fame_opponents > 0:
            sample_size = min(hall_of_fame_opponents, len(hall_of_fame_genomes))
            hall_of_fame_indices = rng.choice(
                len(hall_of_fame_genomes),
                size=sample_size,
                replace=False,
            )
            for local_idx, hall_idx in enumerate(hall_of_fame_indices):
                hall_opponent = hall_of_fame_genomes[int(hall_idx)]
                match_seed = seed_base + 500000 + i * 1000 + local_idx * 10
                matchup_score, matchup_steps, matchup_games = _run_two_sided_matchup(
                    genome_right=genome,
                    genome_left=hall_opponent,
                    seed=match_seed,
                )
                total_scores[i] += matchup_score
                total_lengths[i] += matchup_steps
                games_played[i] += matchup_games

    avg_scores = []
    avg_lengths = []
    for i, genome in enumerate(population.members):
        divisor = max(1, games_played[i])
        avg_score = total_scores[i] / divisor
        avg_length = total_lengths[i] / divisor
        avg_scores.append(avg_score)
        avg_lengths.append(avg_length)
        fitness_score = avg_score

        if baseline_fitness_episodes > 0 and baseline_fitness_weight > 0.0:
            baseline_scores = []
            for episode_idx in range(baseline_fitness_episodes):
                baseline_score, _baseline_steps, _baseline_frames = run_vs_baseline_episode(
                    genome=genome,
                    seed=seed_base + 800000 + i * 100 + episode_idx,
                    capture_frames=False,
                )
                baseline_scores.append(float(baseline_score))

            baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.0
            fitness_score = (
                selfplay_fitness_weight * avg_score
                + baseline_fitness_weight * baseline_mean
            )

        genome.fitness = fitness_score + 6.0

    return avg_scores, avg_lengths, tournament_baseline_scores


def evaluate_vs_baseline(genome, episodes, seed_base):
    """Benchmark a genome against the fixed baseline policy."""
    scores = []
    lengths = []

    for i in range(episodes):
        score, steps, _ = run_vs_baseline_episode(
            genome=genome,
            seed=seed_base + i,
            capture_frames=False,
        )
        scores.append(score)
        lengths.append(steps)

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        float(np.mean(lengths)),
        scores,
    )
