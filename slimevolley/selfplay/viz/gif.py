
"""
Create a gameplay GIF of the champion genome.

This file runs the best genome against the baseline policy
and saves the captured frames as an animated GIF.

This is useful for visually checking how the trained agent plays.

In short:
- runs one benchmark episode
- captures frames
- saves them as a GIF
"""


import imageio.v2 as imageio
from ..episodes import run_vs_baseline_episode


def save_champion_gif(genome, out_dir, seed, fps):
    gif_path = out_dir / "champion_vs_baseline.gif"
    score, steps, frames = run_vs_baseline_episode(
        genome,
        seed=seed,
        capture_frames=True,
        frame_skip=2,
        max_frames=900,
    )

    if frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path, score, steps

    return None, score, steps
