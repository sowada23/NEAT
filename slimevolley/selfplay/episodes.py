
"""
Run single game episodes in SlimeVolley.

This file contains the code that actually plays the game.
It can run:
- one genome vs another genome (self-play)
- one genome vs the built-in baseline policy

It can also optionally capture frames for making GIFs.

In short:
- creates the environment
- steps through the game
- collects reward, step count, and frames
"""


import numpy as np

from .bootstrap import gym
from .policy import genome_to_action


def run_selfplay_episode(
    genome_right,
    genome_left,
    seed,
    capture_frames=False,
    frame_skip=2,
    max_frames=900,
):
    """
    Run one self-play match.

    Returns the score from the perspective of the right-side genome.
    """
    env = gym.make("SlimeVolley-v0").unwrapped
    env.seed(seed)

    obs_right = env.reset()
    obs_left = env.game.agent_left.getObservation()
    done = False
    total_reward = 0.0
    steps = 0
    frames = []
    render_env = None
    render_frame = None

    if capture_frames:
        try:
            render_env = gym.make("SlimeVolleyPixel-v0").unwrapped
            render_env.seed(seed)
            render_frame = render_env.reset()
            if render_frame is not None:
                frames.append(np.asarray(render_frame, dtype=np.uint8))
        except Exception:
            capture_frames = False
            render_env = None
            render_frame = None

    while not done:
        action_right = genome_to_action(genome_right, obs_right)
        action_left = genome_to_action(genome_left, obs_left)

        obs_right, reward, done, info = env.step(action_right, action_left)
        obs_left = info["otherObs"]
        total_reward += reward
        steps += 1

        if render_env is not None:
            try:
                render_frame, _render_reward, _render_done, _render_info = render_env.step(
                    action_right,
                    action_left,
                )
            except Exception:
                capture_frames = False
                render_env.close()
                render_env = None
                render_frame = None

        if capture_frames and render_frame is not None and len(frames) < max_frames and (steps % frame_skip == 0):
            frames.append(np.asarray(render_frame, dtype=np.uint8))

    env.close()
    if render_env is not None:
        render_env.close()

    return total_reward, steps, frames


def run_vs_baseline_episode(genome, seed, capture_frames=False, frame_skip=2, max_frames=900):
    """
    Run one evaluation match against the fixed built-in baseline policy.

    Returns the score from the perspective of the genome on the right side.
    """
    env = gym.make("SlimeVolley-v0").unwrapped
    env.seed(seed)

    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    frames = []
    render_env = None
    render_frame = None

    if capture_frames:
        try:
            render_env = gym.make("SlimeVolleyPixel-v0").unwrapped
            render_env.seed(seed)
            render_frame = render_env.reset()
            if render_frame is not None:
                frames.append(np.asarray(render_frame, dtype=np.uint8))
        except Exception:
            capture_frames = False
            render_env = None
            render_frame = None

    while not done:
        action = genome_to_action(genome, obs)
        obs, reward, done, _info = env.step(action)
        total_reward += reward
        steps += 1

        if render_env is not None:
            try:
                render_frame, _render_reward, _render_done, _render_info = render_env.step(action)
            except Exception:
                capture_frames = False
                render_env.close()
                render_env = None
                render_frame = None

        if capture_frames and render_frame is not None and len(frames) < max_frames and (steps % frame_skip == 0):
            frames.append(np.asarray(render_frame, dtype=np.uint8))

    env.close()
    if render_env is not None:
        render_env.close()

    return total_reward, steps, frames
