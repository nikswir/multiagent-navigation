"""Example-based invariants for the SimpleEnv multi-robot world (stage 1).

Pins the episode contract: reset bookkeeping and valid mutually-separated
spawns and goals (never pre-reached), per-robot step semantics with the right
terminal rewards, the throttle remap the env applies itself, the shaped
step-reward formula, pose integration with heading wrap, the collision /
ray-cast / lidar primitives (including robot-robot sensing and collisions),
out-of-bounds ending as a wall collision, step() returning copies, the
env-owned seeded rng, and the replay buffer's FIFO ring cap.
"""

from __future__ import annotations

import math
import random

import pytest
import numpy as np

from multiagent_navigation.replay_buffer import ReplayBuffer

from multiagent_navigation.environment import (
    Robot,
    World,
    Obstacle,
    SimpleEnv,
)

# Throttle -1 remaps to v = 0: the "hold position" action.
STAY = [-1.0, 0.0]


def _make_env(seed: int, n_robots: int, max_steps: int = 50) -> SimpleEnv:
    """A small env with its own seeded rng: capacity 4."""
    return SimpleEnv(
        robot_radius=0.25,
        max_steps=max_steps,
        n_robots=n_robots,
        max_robots=4,
        seed=seed,
    )


########################################
#            Reset contract            #
########################################


def test_reset_state_and_bookkeeping() -> None:
    env = _make_env(0, 3)

    states = env.reset(n_robots=3)

    assert env.current_step == 0
    assert env.episode_done == [False, False, False]
    assert env.cumulative_reward == [0.0, 0.0, 0.0]
    assert len(states) == 3
    assert all(len(s) == 24 for s in states)


def test_reset_starts_are_valid_and_separated() -> None:
    env = _make_env(1, 4)

    for i in range(env.n_robots):
        robot = env.robots[i]

        # ── Clear of obstacles/walls and inside the shrunk bounds ──
        assert not env.world.check_collision_robot(
            robot.x,
            robot.y,
            robot.radius,
        )
        assert abs(robot.x) < env.world.width / 2 * 0.95
        assert abs(robot.y) < env.world.height / 2 * 0.95

        # ── Mutually separated by at least the body radii ──
        for j in range(i):
            other = env.robots[j]
            dist = math.hypot(robot.x - other.x, robot.y - other.y)
            assert dist >= robot.radius + other.radius


def test_goals_are_never_pre_reached_and_separated() -> None:
    env = _make_env(2, 4)

    for _ in range(20):
        env.reset(n_robots=4)
        for i in range(env.n_robots):
            # ── A goal can never coincide with its robot's start pose ──
            dist = math.hypot(
                env.robots[i].x - env.goals_x[i],
                env.robots[i].y - env.goals_y[i],
            )
            assert dist >= env.goal_reached_dist

            # ── Goals stay mutually separated by the body radii ──
            for j in range(i):
                goal_dist = math.hypot(
                    env.goals_x[i] - env.goals_x[j],
                    env.goals_y[i] - env.goals_y[j],
                )
                assert goal_dist >= (
                    env.robots[i].radius + env.robots[j].radius
                )


def test_env_owns_its_rng() -> None:
    # ── Same seed => same layout; global `random` state untouched ──
    state_before = random.getstate()
    env_a = _make_env(7, 3)
    env_b = _make_env(7, 3)

    assert random.getstate() == state_before
    for i in range(3):
        assert env_a.robots[i].x == env_b.robots[i].x
        assert env_a.robots[i].y == env_b.robots[i].y
        assert env_a.goals_x[i] == env_b.goals_x[i]
        assert env_a.goals_y[i] == env_b.goals_y[i]


########################################
#            Step semantics            #
########################################


def test_step_returns_lists_per_robot_and_increments() -> None:
    env = _make_env(2, 3)

    states, rewards, dones, infos = env.step(np.array([STAY] * 3))

    assert env.current_step == 1
    assert len(states) == 3
    assert len(rewards) == 3
    assert len(dones) == 3
    assert len(infos) == 3


def test_step_remaps_throttle_to_forward_speed() -> None:
    env = _make_env(3, 1)

    # ── Full throttle (+1) moves v * dt = 0.1 along the heading ──
    env.robots[0].x = 0.0
    env.robots[0].y = 0.0
    env.robots[0].theta = 0.0
    env.goals_x[0] = 4.0
    env.goals_y[0] = 0.0
    env.step(np.array([[1.0, 0.0]]))
    assert env.robots[0].x == pytest.approx(0.1)

    # ── Throttle -1 remaps to v = 0: the robot holds position ──
    env.reset(n_robots=1)
    env.robots[0].x = 0.0
    env.robots[0].y = 0.0
    env.goals_x[0] = 4.0
    env.goals_y[0] = 0.0
    env.step(np.array([STAY]))
    assert env.robots[0].x == pytest.approx(0.0)


def test_step_returns_copies_not_internal_state() -> None:
    env = _make_env(4, 1)
    env.robots[0].x = 0.0
    env.robots[0].y = 0.0
    env.goals_x[0] = 4.0
    env.goals_y[0] = 0.0

    _states, _rewards, dones, infos = env.step(np.array([STAY]))

    # ── Mutating the returned lists never touches the env ──
    dones[0] = True
    infos[0] = None
    assert env.episode_done[0] is False
    assert env.info[0] is not None


def test_target_arrival_terminates_with_bonus() -> None:
    env = _make_env(3, 1)

    # ── Park the robot on its goal; the stay action keeps it there ──
    env.robots[0].x = float(env.goals_x[0])
    env.robots[0].y = float(env.goals_y[0])
    _states, rewards, dones, infos = env.step(np.array([STAY]))

    assert dones[0]
    assert rewards[0] == 100.0
    assert infos[0] is not None
    assert infos[0]["reason"] == "target_reached"


def test_collision_terminates_with_penalty() -> None:
    env = _make_env(4, 1)

    # ── Park the robot inside an obstacle, with its goal far away ──
    obstacle = env.world.obstacles[0]
    env.robots[0].x = obstacle.x + obstacle.width / 2
    env.robots[0].y = obstacle.y + obstacle.height / 2
    env.goals_x[0] = -env.robots[0].x
    env.goals_y[0] = -env.robots[0].y
    _states, rewards, dones, infos = env.step(np.array([STAY]))

    assert dones[0]
    assert rewards[0] == -100.0
    assert infos[0] is not None
    assert infos[0]["collision"] is True
    assert infos[0]["reason"] == "collision"


def test_robot_robot_collision_terminates_with_penalty() -> None:
    env = _make_env(5, 2)

    # ── Overlap two robots in open space, goals far away ──
    env.robots[0].x, env.robots[0].y = 0.0, 0.0
    env.robots[1].x, env.robots[1].y = 0.3, 0.0
    env.goals_x[0], env.goals_y[0] = -4.0, -4.0
    env.goals_x[1], env.goals_y[1] = 4.0, 4.0
    _states, rewards, dones, infos = env.step(np.array([STAY, STAY]))

    assert dones == [True, True]
    assert rewards[0] == -100.0
    assert rewards[1] == -100.0
    assert infos[0] is not None
    assert infos[0]["reason"] == "collision"


def test_max_steps_terminates_alive_robots() -> None:
    env = _make_env(5, 1, max_steps=1)

    # ── A free spot with a far goal: only the step cap can end it ──
    env.robots[0].x = 0.0
    env.robots[0].y = 0.0
    env.robots[0].theta = 0.0
    env.goals_x[0] = 4.0
    env.goals_y[0] = 4.0
    _states, _rewards, dones, infos = env.step(np.array([STAY]))

    assert dones[0]
    assert infos[0] is not None
    assert infos[0]["reason"] == "max_steps_reached"


def test_done_robots_are_skipped() -> None:
    env = _make_env(6, 2)

    # ── Mark robot 0 done: it must not move and must earn 0 reward ──
    env.episode_done[0] = True
    pose_before = (env.robots[0].x, env.robots[0].y, env.robots[0].theta)
    _states, rewards, _dones, _infos = env.step(np.ones((2, 2)) * 0.5)

    assert rewards[0] == 0
    assert (env.robots[0].x, env.robots[0].y, env.robots[0].theta) == (
        pose_before
    )


########################################
#           Pose integration           #
########################################


def test_update_pose_wraps_theta() -> None:
    robot = Robot()
    robot.set_velocity(0.0, 1.0)

    # ── 3π/2 wraps to -π/2; the heading stays inside [-π, π) ──
    robot.theta = 3 * math.pi / 2
    robot.update_pose(0.0)
    assert robot.theta == pytest.approx(-math.pi / 2)

    robot.theta = math.pi
    robot.update_pose(0.0)
    assert -math.pi <= robot.theta < math.pi


########################################
#         Collision primitives         #
########################################


def test_world_circle_rect_collision() -> None:
    world = World(10, 10, [[0, 0, 1, 1]])

    assert world.check_collision_robot(0.5, 0.5, 0.1)
    assert world.check_collision_robot(-0.05, 0.5, 0.1)
    assert not world.check_collision_robot(3.0, 3.0, 0.1)


def test_obstacle_raycast_distance() -> None:
    obstacle = Obstacle(1.0, -0.5, 1.0, 1.0)

    # ── A ray straight at the left face hits at distance 1 ──
    hit = obstacle.intersects_segment((0.0, 0.0), (5.0, 0.0))
    assert hit is not None
    assert hit == pytest.approx(1.0)

    # ── A ray pointing away misses ──
    assert obstacle.intersects_segment((0.0, 0.0), (-5.0, 0.0)) is None


def test_robot_raycast_distance() -> None:
    robot = Robot(x=2.0, y=0.0)

    # ── A ray at the disc hits its near rim ──
    hit = robot.intersects_segment((0.0, 0.0), (5.0, 0.0))
    assert hit is not None
    assert hit == pytest.approx(2.0 - robot.radius)

    # ── An orthogonal ray misses; a zero-length one is no ray at all ──
    assert robot.intersects_segment((0.0, 0.0), (0.0, 5.0)) is None
    assert robot.intersects_segment((1.0, 1.0), (1.0, 1.0)) is None


########################################
#           Reward & sensing           #
########################################


def _empty_env(seed: int) -> SimpleEnv:
    """A one-robot env with no interior obstacles (walls only)."""
    return SimpleEnv(
        robot_radius=0.25,
        max_steps=50,
        n_robots=1,
        max_robots=1,
        obstacle_definitions=[],
        seed=seed,
    )


def test_reward_is_shaping_off_a_terminal() -> None:
    """Off a terminal, reward = v/2 - |w|/2 - max(0, 1 - d_min) - 0.05."""
    env = _empty_env(7)

    # ── Centre of the empty field, goal far away: no terminal fires and
    #    every wall is farther than 1, so the proximity term is zero.
    #    Throttle 0.6 remaps to v = 0.8 inside the env ──
    env.robots[0].x = 0.0
    env.robots[0].y = 0.0
    env.robots[0].theta = 0.0
    env.goals_x[0] = 4.0
    env.goals_y[0] = 0.0
    _states, rewards, dones, _infos = env.step(np.array([[0.6, -0.2]]))

    assert not dones[0]
    assert min(env.robots[0].current_lidar_data) > 1
    assert rewards[0] == pytest.approx(0.8 / 2 - 0.2 / 2 - 0.05)


def test_reward_proximity_penalty_below_one() -> None:
    """The proximity term subtracts (1 - d_min) once a reading is < 1."""
    action = np.array([0.6, -0.2])

    far = SimpleEnv.get_reward(False, False, action, min_laser=2.0)
    near = SimpleEnv.get_reward(False, False, action, min_laser=0.4)

    assert far == pytest.approx(0.15)
    assert near == pytest.approx(0.15 - 0.6)


def test_lidar_scan_measures_range_to_obstacle_ahead() -> None:
    """The centre beam reads the body-relative range to a box ahead."""
    robot = Robot(
        x=0.0,
        y=0.0,
        theta=0.0,
        robot_radius=0.25,
        lidar_num_beams=5,
        lidar_max_range=7.0,
    )
    obstacle = Obstacle(2.0, -0.5, 1.0, 1.0)

    scan = robot.lidar_scan([obstacle], [robot])

    # ── Beam 2 (straight ahead) hits the near face at x = 2; beam 4
    #    (+90°) clears it and reads the configured max range ──
    assert scan[2] == pytest.approx(2.0 - robot.radius)
    assert scan[4] == pytest.approx(7.0)


def test_lidar_sees_other_robots() -> None:
    """Another robot occludes the beam exactly like an obstacle."""
    scanner = Robot(
        x=0.0,
        y=0.0,
        theta=0.0,
        robot_radius=0.25,
        lidar_num_beams=5,
        lidar_max_range=7.0,
    )
    other = Robot(x=2.0, y=0.0, robot_radius=0.25)

    scan = scanner.lidar_scan([], [scanner, other])

    # ── The centre beam hits the other robot's near rim ──
    assert scan[2] == pytest.approx(2.0 - other.radius - scanner.radius)


def test_driving_off_the_field_is_a_wall_collision() -> None:
    """Leaving the bounds hits the boundary wall: -100 and 'collision'."""
    env = _empty_env(9)

    # ── Just inside the right wall, heading east at full throttle ──
    env.robots[0].x = 4.7
    env.robots[0].y = 0.0
    env.robots[0].theta = 0.0
    env.goals_x[0] = -4.0
    env.goals_y[0] = 0.0
    _states, rewards, dones, infos = env.step(np.array([[1.0, 0.0]]))

    assert dones[0]
    assert rewards[0] == -100.0
    assert infos[0] is not None
    assert infos[0]["collision"] is True
    assert infos[0]["reason"] == "collision"


########################################
#            Replay buffer             #
########################################


def test_replay_buffer_fifo_cap_and_clear() -> None:
    buffer = ReplayBuffer(buffer_size=3, random_seed=0)
    for k in range(5):
        buffer.add(np.array([k]), np.zeros(2), float(k), 0, np.array([k]))

    # ── Capped at 3, oldest evicted first (ring order is free) ──
    assert buffer.size() == 3
    assert sorted(exp[2] for exp in buffer.buffer) == [2.0, 3.0, 4.0]

    buffer.clear()
    assert buffer.size() == 0
    assert len(buffer.buffer) == 0


def test_replay_buffer_never_touches_global_random() -> None:
    state_before = random.getstate()
    buffer = ReplayBuffer(buffer_size=10, random_seed=42)
    buffer.add(np.zeros(2), np.zeros(2), 0.0, 0, np.zeros(2))
    buffer.sample_batch(1)

    assert random.getstate() == state_before
