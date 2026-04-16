from __future__ import annotations

from typing import NamedTuple

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "slimevolley.gpu_selfplay.env requires the optional 'jax' dependency."
    ) from exc


TIMESTEP = 1.0 / 30.0
GRAVITY = -29.4
MAX_BALL_SPEED = 22.5
FRICTION = 1.0
NUDGE = 0.1
REF_W = 48.0
REF_U = 1.5
REF_H = 48.0
REF_WALL_HEIGHT = 3.5
REF_WALL_WIDTH = 1.0
PLAYER_SPEED_X = 17.5
PLAYER_SPEED_Y = 13.5
MAXLIVES = 5
BALL_RADIUS = 0.5
PLAYER_RADIUS = 1.5
INIT_DELAY_FRAMES = 30
OBS_SCALE = 10.0


class BatchedAgentState(NamedTuple):
    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    desired_vx: jax.Array
    desired_vy: jax.Array
    life: jax.Array
    direction: jax.Array


class BatchedBallState(NamedTuple):
    x: jax.Array
    y: jax.Array
    prev_x: jax.Array
    prev_y: jax.Array
    vx: jax.Array
    vy: jax.Array


class BatchedEnvState(NamedTuple):
    ball: BatchedBallState
    agent_left: BatchedAgentState
    agent_right: BatchedAgentState
    delay_life: jax.Array
    steps: jax.Array
    done: jax.Array
    key: jax.Array


def _make_agent(batch_size: int, direction: float, x: float) -> BatchedAgentState:
    return BatchedAgentState(
        x=jnp.full((batch_size,), x, dtype=jnp.float32),
        y=jnp.full((batch_size,), PLAYER_RADIUS, dtype=jnp.float32),
        vx=jnp.zeros((batch_size,), dtype=jnp.float32),
        vy=jnp.zeros((batch_size,), dtype=jnp.float32),
        desired_vx=jnp.zeros((batch_size,), dtype=jnp.float32),
        desired_vy=jnp.zeros((batch_size,), dtype=jnp.float32),
        life=jnp.full((batch_size,), MAXLIVES, dtype=jnp.int32),
        direction=jnp.full((batch_size,), direction, dtype=jnp.float32),
    )


def _sample_ball(keys: jax.Array) -> BatchedBallState:
    vx_keys, vy_keys = jax.vmap(jax.random.split)(keys).transpose((1, 0, 2))
    vx = jax.vmap(lambda k: jax.random.uniform(k, (), minval=-20.0, maxval=20.0))(vx_keys)
    vy = jax.vmap(lambda k: jax.random.uniform(k, (), minval=10.0, maxval=25.0))(vy_keys)
    y0 = jnp.full_like(vx, REF_W / 4.0)
    x0 = jnp.zeros_like(vx)
    return BatchedBallState(x=x0, y=y0, prev_x=x0, prev_y=y0, vx=vx, vy=vy)


def reset_batched_env(keys: jax.Array) -> BatchedEnvState:
    batch_size = keys.shape[0]
    next_keys, ball_keys = jax.vmap(jax.random.split)(keys).transpose((1, 0, 2))
    return BatchedEnvState(
        ball=_sample_ball(ball_keys),
        agent_left=_make_agent(batch_size, -1.0, -REF_W / 4.0),
        agent_right=_make_agent(batch_size, 1.0, REF_W / 4.0),
        delay_life=jnp.full((batch_size,), INIT_DELAY_FRAMES, dtype=jnp.int32),
        steps=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=bool),
        key=next_keys,
    )


def _decode_action(action: jax.Array) -> tuple[jax.Array, jax.Array]:
    action = jnp.asarray(action, dtype=jnp.float32)
    forward = action[:, 0] > 0.0
    backward = action[:, 1] > 0.0
    jump = action[:, 2] > 0.0

    desired_vx = jnp.where(forward & ~backward, -PLAYER_SPEED_X, 0.0)
    desired_vx = jnp.where(backward & ~forward, PLAYER_SPEED_X, desired_vx)
    desired_vy = jnp.where(jump, PLAYER_SPEED_Y, 0.0)
    return desired_vx.astype(jnp.float32), desired_vy.astype(jnp.float32)


def _apply_action(agent: BatchedAgentState, action: jax.Array) -> BatchedAgentState:
    desired_vx, desired_vy = _decode_action(action)
    return BatchedAgentState(
        x=agent.x,
        y=agent.y,
        vx=agent.vx,
        vy=agent.vy,
        desired_vx=desired_vx,
        desired_vy=desired_vy,
        life=agent.life,
        direction=agent.direction,
    )


def _update_agent(agent: BatchedAgentState) -> BatchedAgentState:
    vy = agent.vy + GRAVITY * TIMESTEP
    grounded = agent.y <= (REF_U + NUDGE * TIMESTEP)
    vy = jnp.where(grounded, agent.desired_vy, vy)
    vx = agent.desired_vx * agent.direction

    x = agent.x + vx * TIMESTEP
    y = agent.y + vy * TIMESTEP

    on_floor = y <= REF_U
    y = jnp.where(on_floor, REF_U, y)
    vy = jnp.where(on_floor, 0.0, vy)

    lower = REF_WALL_WIDTH / 2.0 + PLAYER_RADIUS
    upper = REF_W / 2.0 - PLAYER_RADIUS
    signed_x = jnp.clip(x * agent.direction, lower, upper)
    x = signed_x * agent.direction
    hit_side = (signed_x == lower) | (signed_x == upper)
    vx = jnp.where(hit_side, 0.0, vx)

    return BatchedAgentState(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        desired_vx=agent.desired_vx,
        desired_vy=agent.desired_vy,
        life=agent.life,
        direction=agent.direction,
    )


def _ball_apply_gravity(ball: BatchedBallState) -> BatchedBallState:
    vx = ball.vx
    vy = ball.vy + GRAVITY * TIMESTEP
    speed = jnp.sqrt(vx * vx + vy * vy)
    safe_speed = jnp.maximum(speed, 1e-6)
    scale = jnp.minimum(1.0, MAX_BALL_SPEED / safe_speed)
    return BatchedBallState(
        x=ball.x,
        y=ball.y,
        prev_x=ball.prev_x,
        prev_y=ball.prev_y,
        vx=vx * scale,
        vy=vy * scale,
    )


def _ball_move(ball: BatchedBallState) -> BatchedBallState:
    return BatchedBallState(
        x=ball.x + ball.vx * TIMESTEP,
        y=ball.y + ball.vy * TIMESTEP,
        prev_x=ball.x,
        prev_y=ball.y,
        vx=ball.vx,
        vy=ball.vy,
    )


def _resolve_circle_collision(
    ball: BatchedBallState,
    px: jax.Array,
    py: jax.Array,
    pvx: jax.Array,
    pvy: jax.Array,
    radius: float,
) -> BatchedBallState:
    dx = ball.x - px
    dy = ball.y - py
    dist2 = dx * dx + dy * dy
    min_dist = BALL_RADIUS + radius
    colliding = dist2 < (min_dist * min_dist)

    dist = jnp.sqrt(jnp.maximum(dist2, 1e-8))
    nx = dx / dist
    ny = dy / dist
    overlap = min_dist - dist + NUDGE * TIMESTEP

    x = jnp.where(colliding, ball.x + nx * overlap, ball.x)
    y = jnp.where(colliding, ball.y + ny * overlap, ball.y)

    rel_vx = ball.vx - pvx
    rel_vy = ball.vy - pvy
    proj = rel_vx * nx + rel_vy * ny
    vx = jnp.where(colliding, ball.vx - 2.0 * proj * nx, ball.vx)
    vy = jnp.where(colliding, ball.vy - 2.0 * proj * ny, ball.vy)

    return BatchedBallState(
        x=x,
        y=y,
        prev_x=ball.prev_x,
        prev_y=ball.prev_y,
        vx=vx,
        vy=vy,
    )


def _check_ball_edges(ball: BatchedBallState) -> tuple[BatchedBallState, jax.Array]:
    x = ball.x
    y = ball.y
    vx = ball.vx
    vy = ball.vy
    reward = jnp.zeros_like(ball.x, dtype=jnp.int32)

    left_wall = x <= (BALL_RADIUS - REF_W / 2.0)
    x = jnp.where(left_wall, BALL_RADIUS - REF_W / 2.0 + NUDGE * TIMESTEP, x)
    vx = jnp.where(left_wall, -FRICTION * vx, vx)

    right_wall = x >= (REF_W / 2.0 - BALL_RADIUS)
    x = jnp.where(right_wall, REF_W / 2.0 - BALL_RADIUS - NUDGE * TIMESTEP, x)
    vx = jnp.where(right_wall, -FRICTION * vx, vx)

    floor_hit = y <= (BALL_RADIUS + REF_U)
    reward = jnp.where(floor_hit & (x <= 0.0), -1, reward)
    reward = jnp.where(floor_hit & (x > 0.0), 1, reward)
    y = jnp.where(floor_hit, BALL_RADIUS + REF_U + NUDGE * TIMESTEP, y)
    vy = jnp.where(floor_hit, -FRICTION * vy, vy)

    ceiling_hit = y >= (REF_H - BALL_RADIUS)
    y = jnp.where(ceiling_hit, REF_H - BALL_RADIUS - NUDGE * TIMESTEP, y)
    vy = jnp.where(ceiling_hit, -FRICTION * vy, vy)

    fence_right = (
        (x <= (REF_WALL_WIDTH / 2.0 + BALL_RADIUS))
        & (ball.prev_x > (REF_WALL_WIDTH / 2.0 + BALL_RADIUS))
        & (y <= REF_WALL_HEIGHT)
    )
    x = jnp.where(fence_right, REF_WALL_WIDTH / 2.0 + BALL_RADIUS + NUDGE * TIMESTEP, x)
    vx = jnp.where(fence_right, -FRICTION * vx, vx)

    fence_left = (
        (x >= (-REF_WALL_WIDTH / 2.0 - BALL_RADIUS))
        & (ball.prev_x < (-REF_WALL_WIDTH / 2.0 - BALL_RADIUS))
        & (y <= REF_WALL_HEIGHT)
    )
    x = jnp.where(fence_left, -REF_WALL_WIDTH / 2.0 - BALL_RADIUS - NUDGE * TIMESTEP, x)
    vx = jnp.where(fence_left, -FRICTION * vx, vx)

    return BatchedBallState(
        x=x,
        y=y,
        prev_x=ball.prev_x,
        prev_y=ball.prev_y,
        vx=vx,
        vy=vy,
    ), reward


def _sample_new_ball(keys: jax.Array, old_ball: BatchedBallState, scored: jax.Array) -> tuple[BatchedBallState, jax.Array]:
    next_keys, ball_keys = jax.vmap(jax.random.split)(keys).transpose((1, 0, 2))
    new_ball = _sample_ball(ball_keys)
    mask = scored.astype(jnp.float32)
    return BatchedBallState(
        x=jnp.where(scored, new_ball.x, old_ball.x),
        y=jnp.where(scored, new_ball.y, old_ball.y),
        prev_x=jnp.where(scored, new_ball.prev_x, old_ball.prev_x),
        prev_y=jnp.where(scored, new_ball.prev_y, old_ball.prev_y),
        vx=jnp.where(scored, new_ball.vx, old_ball.vx),
        vy=jnp.where(scored, new_ball.vy, old_ball.vy),
    ), next_keys


def batched_observations(state: BatchedEnvState) -> tuple[jax.Array, jax.Array]:
    right = state.agent_right
    left = state.agent_left
    ball = state.ball

    obs_right = jnp.stack(
        [
            right.x * right.direction,
            right.y,
            right.vx * right.direction,
            right.vy,
            ball.x * right.direction,
            ball.y,
            ball.vx * right.direction,
            ball.vy,
            left.x * (-right.direction),
            left.y,
            left.vx * (-right.direction),
            left.vy,
        ],
        axis=-1,
    ) / OBS_SCALE

    obs_left = jnp.stack(
        [
            left.x * left.direction,
            left.y,
            left.vx * left.direction,
            left.vy,
            ball.x * left.direction,
            ball.y,
            ball.vx * left.direction,
            ball.vy,
            right.x * (-left.direction),
            right.y,
            right.vx * (-left.direction),
            right.vy,
        ],
        axis=-1,
    ) / OBS_SCALE
    return obs_right.astype(jnp.float32), obs_left.astype(jnp.float32)


def step_batched_env(
    state: BatchedEnvState,
    action_right: jax.Array,
    action_left: jax.Array,
    max_steps: int,
) -> tuple[BatchedEnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
    active = ~state.done

    left = _apply_action(state.agent_left, action_left)
    right = _apply_action(state.agent_right, action_right)

    left = _update_agent(left)
    right = _update_agent(right)

    delay_open = state.delay_life <= 0
    next_delay = jnp.where(delay_open, 0, state.delay_life - 1)

    ball = _ball_apply_gravity(state.ball)
    ball = BatchedBallState(
        x=jnp.where(delay_open, ball.x, state.ball.x),
        y=jnp.where(delay_open, ball.y, state.ball.y),
        prev_x=jnp.where(delay_open, ball.prev_x, state.ball.prev_x),
        prev_y=jnp.where(delay_open, ball.prev_y, state.ball.prev_y),
        vx=jnp.where(delay_open, ball.vx, state.ball.vx),
        vy=jnp.where(delay_open, ball.vy, state.ball.vy),
    )
    ball = _ball_move(ball)
    ball = BatchedBallState(
        x=jnp.where(delay_open, ball.x, state.ball.x),
        y=jnp.where(delay_open, ball.y, state.ball.y),
        prev_x=jnp.where(delay_open, ball.prev_x, state.ball.prev_x),
        prev_y=jnp.where(delay_open, ball.prev_y, state.ball.prev_y),
        vx=jnp.where(delay_open, ball.vx, state.ball.vx),
        vy=jnp.where(delay_open, ball.vy, state.ball.vy),
    )

    ball = _resolve_circle_collision(ball, left.x, left.y, left.vx, left.vy, PLAYER_RADIUS)
    ball = _resolve_circle_collision(ball, right.x, right.y, right.vx, right.vy, PLAYER_RADIUS)
    ball = _resolve_circle_collision(ball, jnp.zeros_like(ball.x), jnp.full_like(ball.y, REF_WALL_HEIGHT), jnp.zeros_like(ball.x), jnp.zeros_like(ball.y), REF_WALL_WIDTH / 2.0)

    ball, edge_reward = _check_ball_edges(ball)
    reward = (-edge_reward).astype(jnp.float32)
    reward = jnp.where(active, reward, 0.0)

    left_life = left.life - jnp.where(reward > 0, 1, 0)
    right_life = right.life - jnp.where(reward < 0, 1, 0)
    left = BatchedAgentState(left.x, left.y, left.vx, left.vy, left.desired_vx, left.desired_vy, left_life, left.direction)
    right = BatchedAgentState(right.x, right.y, right.vx, right.vy, right.desired_vx, right.desired_vy, right_life, right.direction)

    scored = reward != 0.0
    ball, next_keys = _sample_new_ball(state.key, ball, scored)
    next_delay = jnp.where(scored, INIT_DELAY_FRAMES, next_delay)

    steps = state.steps + active.astype(jnp.int32)
    done = state.done | (steps >= max_steps) | (left_life <= 0) | (right_life <= 0)

    next_state = BatchedEnvState(
        ball=ball,
        agent_left=left,
        agent_right=right,
        delay_life=next_delay,
        steps=steps,
        done=done,
        key=next_keys,
    )
    obs_right, obs_left = batched_observations(next_state)
    return next_state, obs_right, obs_left, reward, done
