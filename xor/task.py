from __future__ import annotations

import numpy as np

from neat.genetics.genome import Genome
from neat.jax import batched_forward_jax, genome_to_jax

XOR_INPUTS = np.asarray(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)
XOR_TARGETS = np.asarray([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)


def evaluate_genome_xor(genome: Genome) -> dict[str, float | np.ndarray]:
    jax_genome = genome_to_jax(genome)
    outputs = np.asarray(batched_forward_jax(jax_genome, XOR_INPUTS), dtype=np.float32)
    errors = outputs - XOR_TARGETS
    sse = float(np.sum(np.square(errors)))
    mse = float(np.mean(np.square(errors)))
    predictions = (outputs >= 0.5).astype(np.float32)
    accuracy = float(np.mean(predictions == XOR_TARGETS))
    fitness = max(0.0, 4.0 - sse)

    return {
        "fitness": fitness,
        "accuracy": accuracy,
        "mse": mse,
        "sse": sse,
        "outputs": outputs,
        "predictions": predictions,
    }


def evaluate_population_xor(population) -> dict[str, list[float]]:
    best_accuracy = []
    mean_accuracy = []
    best_fitness = []
    mean_fitness = []
    best_mse = []

    for genome in population.members:
        metrics = evaluate_genome_xor(genome)
        genome.fitness = float(metrics["fitness"])
        genome.xor_accuracy = float(metrics["accuracy"])
        genome.xor_mse = float(metrics["mse"])

    accuracies = [float(getattr(genome, "xor_accuracy", 0.0)) for genome in population.members]
    fitnesses = [float(genome.fitness) for genome in population.members]
    mses = [float(getattr(genome, "xor_mse", 0.0)) for genome in population.members]

    best_accuracy.append(max(accuracies) if accuracies else 0.0)
    mean_accuracy.append(float(np.mean(accuracies)) if accuracies else 0.0)
    best_fitness.append(max(fitnesses) if fitnesses else 0.0)
    mean_fitness.append(float(np.mean(fitnesses)) if fitnesses else 0.0)
    best_mse.append(min(mses) if mses else 0.0)

    return {
        "best_accuracy": best_accuracy,
        "mean_accuracy": mean_accuracy,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "best_mse": best_mse,
    }
