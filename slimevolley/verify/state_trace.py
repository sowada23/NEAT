from __future__ import annotations

import csv
from pathlib import Path

TRACE_FIELDS = [
    "step",
    "reward",
    "done",
    "right_life",
    "left_life",
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "right_x",
    "right_y",
    "right_vx",
    "right_vy",
    "left_x",
    "left_y",
    "left_vx",
    "left_vy",
]


def gym_trace_row(env, step: int, reward: float, done: bool) -> dict[str, float | int | bool]:
    game = env.game
    return {
        "step": step,
        "reward": float(reward),
        "done": bool(done),
        "right_life": int(game.agent_right.life),
        "left_life": int(game.agent_left.life),
        "ball_x": float(game.ball.x),
        "ball_y": float(game.ball.y),
        "ball_vx": float(game.ball.vx),
        "ball_vy": float(game.ball.vy),
        "right_x": float(game.agent_right.x),
        "right_y": float(game.agent_right.y),
        "right_vx": float(game.agent_right.vx),
        "right_vy": float(game.agent_right.vy),
        "left_x": float(game.agent_left.x),
        "left_y": float(game.agent_left.y),
        "left_vx": float(game.agent_left.vx),
        "left_vy": float(game.agent_left.vy),
    }


def jax_trace_row(state, step: int, reward: float, done: bool, batch_idx: int = 0) -> dict[str, float | int | bool]:
    import numpy as np

    return {
        "step": step,
        "reward": float(reward),
        "done": bool(done),
        "right_life": int(np.asarray(state.agent_right.life)[batch_idx]),
        "left_life": int(np.asarray(state.agent_left.life)[batch_idx]),
        "ball_x": float(np.asarray(state.ball.x)[batch_idx]),
        "ball_y": float(np.asarray(state.ball.y)[batch_idx]),
        "ball_vx": float(np.asarray(state.ball.vx)[batch_idx]),
        "ball_vy": float(np.asarray(state.ball.vy)[batch_idx]),
        "right_x": float(np.asarray(state.agent_right.x)[batch_idx]),
        "right_y": float(np.asarray(state.agent_right.y)[batch_idx]),
        "right_vx": float(np.asarray(state.agent_right.vx)[batch_idx]),
        "right_vy": float(np.asarray(state.agent_right.vy)[batch_idx]),
        "left_x": float(np.asarray(state.agent_left.x)[batch_idx]),
        "left_y": float(np.asarray(state.agent_left.y)[batch_idx]),
        "left_vx": float(np.asarray(state.agent_left.vx)[batch_idx]),
        "left_vy": float(np.asarray(state.agent_left.vy)[batch_idx]),
    }


def write_trace_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

