"""Animated rollout of a trained TD3 policy in the SimpleEnv world.

Loads the best epoch recorded in the training results log (the same
`{results_dir}/{file_name}` / `{models_dir}/{file_name}_epoch-{epoch}`
contract `lib.train` writes), replays the shared policy for
`cfg.animate.n_robots` robots with matplotlib's FuncAnimation, and dumps every
robot's 24-d state per step to a CSV report (consumed by
`notebooks/develop.ipynb`). The module-level `draw_*` helpers render the same
scene as static vector graphics for the report figure scripts
(`report/scripts/`). Matplotlib is imported at module load, so the training
path (`agent`, `lib`, `run`) never imports this module. Run it as::

    python -m multiagent_navigation.viz                  # animate best epoch
    python -m multiagent_navigation.viz animate.n_robots=4
"""

from __future__ import annotations

import json
import math
import hydra
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import cast
from pathlib import Path
from matplotlib.axes import Axes
from omegaconf import DictConfig
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from collections.abc import Sequence
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow, Circle, Rectangle

from multiagent_navigation.agent import TD3
from multiagent_navigation import config_schema
from multiagent_navigation.config_schema import Config
from multiagent_navigation.environment import Robot, SimpleEnv

# Register the structured-config schema so Hydra type-checks the composed YAML.
config_schema.register()

########################################
#               Palette                #
########################################

# One colour per capacity slot (index = robot id); goals share the colour.
COLORS = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "pink",
    "cyan",
    "brown",
    "magenta",
    "gray",
    "black",
    "yellow",
    "lightblue",
    "lightgreen",
    "lightgray",
    "lightyellow",
]


def report_columns(env: SimpleEnv) -> list[str]:
    """One column per state feature per robot, prefixed by robot colour."""
    columns: list[str] = []
    for i in range(env.n_robots):
        color = COLORS[i]
        columns += [
            f"{color}_lidar_{j}" for j in range(env.robots[0].lidar_num_beams)
        ]
        columns += [
            f"{color}_norm_dist_to_goal",
            f"{color}_norm_relative_angle_to_goal",
            f"{color}_linear_velocity",
            f"{color}_angular_velocity",
        ]
    return columns


########################################
#        Figure drawing helpers        #
########################################

# Flat, muted scene palette shared by the report figures and the demo GIF.
FIELD_FACE = "#f4f6fb"
FIELD_EDGE = "#c2c9d6"
OBSTACLE_FACE = "#9aa7bd"
OBSTACLE_EDGE = "#6b7896"
LIDAR_COLOR = "#f0922f"
INK = "#27324a"


def draw_field(ax: Axes, width: float, height: float) -> None:
    """Paint the bounded world and fix a centred, equal-aspect frame."""
    ax.add_patch(
        Rectangle(
            (-width / 2, -height / 2),
            width,
            height,
            facecolor=FIELD_FACE,
            edgecolor=FIELD_EDGE,
            linewidth=1.5,
            zorder=0,
        ),
    )
    ax.set_xlim(-width / 2 - 0.4, width / 2 + 0.4)
    ax.set_ylim(-height / 2 - 0.4, height / 2 + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_obstacles(ax: Axes, obstacles: list[dict[str, float]]) -> None:
    """Draw each `{x, y, width, height}` rectangle as a soft block."""
    for obs in obstacles:
        ax.add_patch(
            Rectangle(
                (obs["x"], obs["y"]),
                obs["width"],
                obs["height"],
                facecolor=OBSTACLE_FACE,
                edgecolor=OBSTACLE_EDGE,
                linewidth=1.5,
                alpha=0.95,
                zorder=2,
            ),
        )


def draw_goal(
    ax: Axes,
    goal: tuple[float, float],
    threshold: float,
    color: str,
) -> None:
    """A goal dot with its arrival-tolerance halo, in the owner's colour."""
    gx, gy = goal
    ax.add_patch(
        Circle(
            (gx, gy),
            threshold,
            facecolor=color,
            edgecolor=color,
            linestyle=(0, (3, 3)),
            linewidth=1.0,
            alpha=0.18,
            zorder=3,
        ),
    )
    ax.add_patch(
        Circle((gx, gy), 0.07, facecolor=color, edgecolor=INK, zorder=4),
    )


def draw_robot(
    ax: Axes,
    x: float,
    y: float,
    theta: float,
    radius: float,
    color: str,
) -> None:
    """A disc robot body with a short heading tick."""
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            facecolor=color,
            edgecolor=INK,
            linewidth=1.2,
            alpha=0.9,
            zorder=5,
        ),
    )
    ax.plot(
        [x, x + 1.5 * radius * math.cos(theta)],
        [y, y + 1.5 * radius * math.sin(theta)],
        color=INK,
        linewidth=1.4,
        solid_capstyle="round",
        zorder=6,
    )


def draw_lidar(ax: Axes, robot: Robot) -> None:
    """The robot's latest lidar fan, one faint ray per beam."""
    for j, dist in enumerate(robot.current_lidar_data_display):
        # ── Reconstruct the beam angle exactly as the scan cast it ──
        if robot.lidar_num_beams == 1:
            relative = robot.lidar_start_angle_offset + 0.5 * robot.lidar_fov
        else:
            relative = (
                robot.lidar_start_angle_offset
                + (j / (robot.lidar_num_beams - 1)) * robot.lidar_fov
            )
        angle = robot.theta + relative

        ax.plot(
            [robot.x, robot.x + dist * math.cos(angle)],
            [robot.y, robot.y + dist * math.sin(angle)],
            color=LIDAR_COLOR,
            linewidth=0.7,
            alpha=0.45,
            zorder=4,
        )


def draw_trajectory(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    color: str,
    label: str | None = None,
) -> None:
    """A robot's path with a white start marker in the robot's colour."""
    ax.plot(
        list(xs),
        list(ys),
        color=color,
        linewidth=1.8,
        alpha=0.85,
        solid_capstyle="round",
        zorder=6,
        label=label,
    )
    ax.add_patch(
        Circle(
            (xs[0], ys[0]),
            0.09,
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
            zorder=7,
        ),
    )


def render_env(
    env: SimpleEnv,
    ax: Axes | None = None,
    show_lidar: bool = True,
) -> Axes:
    """Draw the current scene: field, obstacles, goals and robots."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 6.4))

    # ── 1. Static scene: field panel and the obstacle course ──
    draw_field(ax, env.world.width, env.world.height)
    draw_obstacles(
        ax,
        [
            {"x": o.x, "y": o.y, "width": o.width, "height": o.height}
            for o in env.world.obstacles
        ],
    )

    # ── 2. Per-robot goal, lidar fan and body, in the slot colour ──
    for i in range(env.n_robots):
        color = COLORS[i]
        draw_goal(
            ax,
            (float(env.goals_x[i]), float(env.goals_y[i])),
            env.goal_reached_dist,
            color,
        )
        if show_lidar:
            draw_lidar(ax, env.robots[i])
        draw_robot(
            ax,
            env.robots[i].x,
            env.robots[i].y,
            env.robots[i].theta,
            env.robots[i].radius,
            color,
        )

    return ax


########################################
#          Checkpoint loading          #
########################################


def load_best_network(cfg: Config, device: torch.device) -> TD3:
    """Rebuild the agent and load the best epoch from the training log."""

    # ── Pick the epoch with the highest average evaluation reward ──
    log_path = Path(cfg.animate.results_dir) / cfg.train.file_name
    with open(log_path) as fh:
        train_report = json.load(fh)
    best = max(train_report, key=lambda e: e["Avg_reward"])
    print(
        f"Loading best epoch: {best['Epoch']}, "
        f"Avg_reward: {best['Avg_reward']:.2f}, "
        f"Avg_arrived: {best['Avg_arrived']:.2f}, "
        f"Avg_collision: {best['Avg_collision']:.2f}",
    )

    # ── Rebuild the network shell and load the epoch checkpoint ──
    state_dim = cfg.env.environment_dim + 4
    network = TD3(
        state_dim,
        cfg.model.action_dim,
        cfg.model.max_action,
        device=device,
        hidden1=cfg.model.hidden1,
        hidden2=cfg.model.hidden2,
    )
    epoch = best["Epoch"]
    network.load(
        f"{cfg.train.file_name}_epoch-{epoch}",
        directory=cfg.animate.models_dir,
    )
    return network


########################################
#            Scene animator            #
########################################


class SceneAnimator:
    """Owns the figure, per-robot artists and the per-step state report."""

    def __init__(
        self,
        env: SimpleEnv,
        network: TD3,
        interval: int,
        figsize: tuple[float, float],
        report_csv: Path,
    ) -> None:
        self.env = env
        self.network = network
        self.interval = interval
        self.report_csv = report_csv
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # ── Per-robot artists, populated by `init_frame` ──
        self.robot_patches: list[Circle] = []
        self.robot_arrows: list[Arrow] = []
        self.goal_patches: list[Circle] = []
        self.lidar_lines: list[Line2D] = []
        self.title_text: Text | None = None
        self.animation: FuncAnimation | None = None

        # ── One row per step, one column per robot state feature ──
        self.state_dim = env.robots[0].lidar_num_beams + 4
        self.report = pd.DataFrame(columns=report_columns(env))

    def init_frame(self) -> list[Artist]:
        """Reset the episode and (re)build every artist from scratch."""
        env = self.env
        env.reset(env.n_robots)

        # ── Fresh axes: bounds, aspect, grid and title ──
        self.ax.clear()
        self.robot_patches = []
        self.robot_arrows = []
        self.goal_patches = []
        self.lidar_lines = []
        self.ax.set_xlim(-env.world.width / 2 - 1, env.world.width / 2 + 1)
        self.ax.set_ylim(-env.world.height / 2 - 1, env.world.height / 2 + 1)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)
        self.title_text = self.ax.set_title("Episode Step: 0, Reward: 0.0")

        # ── Obstacles and world walls ──
        for obs_param in env.world.all_obstacles:
            self.ax.add_patch(
                Rectangle(
                    (obs_param.x, obs_param.y),
                    obs_param.width,
                    obs_param.height,
                    facecolor="gray",
                    edgecolor="black",
                    zorder=2,
                ),
            )

        # ── Robots, heading arrows, goals and lidar-beam lines ──
        for i in range(env.n_robots):
            self.robot_patches.append(
                Circle(
                    (env.robots[i].x, env.robots[i].y),
                    env.robots[i].radius,
                    facecolor=COLORS[i],
                    edgecolor="black",
                    alpha=0.8,
                    zorder=5,
                ),
            )
            self.ax.add_patch(self.robot_patches[-1])

            self.robot_arrows.append(self._heading_arrow(i))
            self.ax.add_patch(self.robot_arrows[-1])

            self.goal_patches.append(
                Circle(
                    (env.goals_x[i], env.goals_y[i]),
                    0.12,
                    facecolor=COLORS[i],
                    edgecolor="black",
                    zorder=4,
                ),
            )
            self.ax.add_patch(self.goal_patches[-1])

            for _ in range(env.robots[i].lidar_num_beams):
                (line,) = self.ax.plot(
                    [],
                    [],
                    "r-",
                    alpha=0.4,
                    linewidth=0.8,
                    zorder=3,
                )
                self.lidar_lines.append(line)

        return self._artists()

    def update(self, frame_num: int) -> list[Artist]:
        """One animation tick: act, step the env, redraw, log the state."""
        env = self.env

        # ── 1. Query the policy per robot (throttle remapped to [0,1]) ──
        actions: list[list[float]] = []
        for i in range(env.n_robots):
            if env.episode_done[i]:
                actions.append([0, 0])
                continue
            current_state = env.current_state[i]
            action = self.network.get_action(np.array(current_state))
            a_in = [(action[0] + 1) / 2, action[1]]
            actions.append(a_in)

        state, rewards, dones, infos = env.step(actions)

        # ── 2. Log every robot's state row into the report ──
        for i, st in enumerate(state):
            cols = self.report.columns[
                i * self.state_dim : (i + 1) * self.state_dim
            ]
            self.report.loc[env.current_step, cols] = st

        # ── 3. Move the robot / goal artists; redraw robot 0's lidar ──
        for i in range(env.n_robots):
            self.robot_patches[i].center = (env.robots[i].x, env.robots[i].y)
            self.goal_patches[i].center = (env.goals_x[i], env.goals_y[i])

            self.robot_arrows[i].remove()
            self.robot_arrows[i] = self._heading_arrow(i)
            self.ax.add_patch(self.robot_arrows[i])

            if i == 0:
                self._draw_lidar(i)

        # ── 4. Title with the per-robot cumulative rewards ──
        rewards_txt = [round(float(item), 3) for item in env.cumulative_reward]
        if self.title_text is not None:
            self.title_text.set_text(
                f"Step: {env.current_step}/{env.max_steps}, "
                f"Rewards: {rewards_txt}",
            )

        # ── 5. Stop when all robots are done; dump the state report ──
        if (
            sum(env.episode_done) == env.n_robots
            and self.animation is not None
            and self.animation.event_source is not None
        ):
            reasons = [
                (info or {}).get("reason", "unknown") for info in env.info
            ]
            total_reward = sum(env.cumulative_reward)
            print(
                f"Episode ended by FuncAnimation. Reason: {reasons}, "
                f"Total Reward: {total_reward}",
            )
            self.report.to_csv(self.report_csv, index=False)
            self.animation.event_source.stop()

        return self._artists()

    ########################################
    #           Drawing helpers            #
    ########################################

    def _heading_arrow(self, i: int) -> Arrow:
        robot = self.env.robots[i]
        return Arrow(
            robot.x,
            robot.y,
            robot.radius * 1.2 * math.cos(robot.theta),
            robot.radius * 1.2 * math.sin(robot.theta),
            width=0.15,
            facecolor="white",
            edgecolor="black",
            zorder=6,
        )

    def _draw_lidar(self, i: int) -> None:
        robot = self.env.robots[i]
        for j, dist in enumerate(robot.current_lidar_data_display):
            # ── Reconstruct the beam angle exactly as the scan cast it ──
            if robot.lidar_num_beams == 1:
                relative_beam_angle = (
                    robot.lidar_start_angle_offset + 0.5 * robot.lidar_fov
                )
            else:
                relative_beam_angle = (
                    robot.lidar_start_angle_offset
                    + (j / (robot.lidar_num_beams - 1)) * robot.lidar_fov
                )
            global_beam_angle = robot.theta + relative_beam_angle

            x_end = robot.x + dist * math.cos(global_beam_angle)
            y_end = robot.y + dist * math.sin(global_beam_angle)
            self.lidar_lines[j].set_data(
                [robot.x, x_end],
                [robot.y, y_end],
            )

    def _artists(self) -> list[Artist]:
        artists: list[Artist] = []
        artists += self.robot_patches
        artists += self.robot_arrows
        artists += self.goal_patches
        if self.title_text is not None:
            artists.append(self.title_text)
        artists += self.lidar_lines
        return artists

    def run(self) -> None:
        """Drive the episode with FuncAnimation until every robot is done."""
        num_frames = self.env.max_steps + 10
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            frames=num_frames,
            init_func=self.init_frame,
            interval=self.interval,
            blit=False,
            repeat=False,
        )
        plt.show()


########################################
#             Entry point              #
########################################


def animate(cfg: Config, *, device: torch.device | None = None) -> None:
    """Build the env and best network from `cfg`, run the live animation."""

    # ── Seed layout randomness for a reproducible scene ──
    random.seed(cfg.animate.seed)

    dev = device if device is not None else torch.device("cpu")
    env = SimpleEnv(
        world_width=cfg.env.world_width,
        world_height=cfg.env.world_height,
        environment_dim=cfg.env.environment_dim,
        robot_radius=cfg.env.robot_radius,
        max_steps=cfg.animate.max_steps,
        n_robots=cfg.animate.n_robots,
        max_robots=cfg.env.max_robots,
        time_delta=cfg.env.time_delta,
        goal_reached_dist=cfg.env.goal_reached_dist,
        lidar_max_range=cfg.env.lidar_max_range,
        obstacle_definitions=[list(o) for o in cfg.env.obstacle_definitions],
    )
    network = load_best_network(cfg, dev)
    animator = SceneAnimator(
        env,
        network,
        interval=cfg.animate.interval,
        figsize=(cfg.animate.fig_width, cfg.animate.fig_height),
        report_csv=Path(cfg.animate.report_csv),
    )
    animator.run()


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # ── Pick the device once; the library stays device-agnostic ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    animate(cast(Config, cfg), device=device)
    print("Animation window closed or animation finished.")


if __name__ == "__main__":
    main()
