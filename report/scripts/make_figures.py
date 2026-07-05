"""Generate every figure used in the report.

Reads the curated training log and rollout artifacts from report/assets/ and
the live environment geometry, then writes vector PDFs (and PNG previews) to
report/figures/.

    uv run python report/scripts/make_figures.py
"""

from __future__ import annotations

import os

# Pin the headless backend before pyplot is imported anywhere below.
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from multiagent_navigation import viz
from multiagent_navigation.config_schema import Config
from multiagent_navigation.environment import SimpleEnv

HERE = Path(__file__).parents[1]
FIG = HERE / "figures"
ASSETS = HERE / "assets"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
        "figure.dpi": 130,
    },
)

INK = "#27324a"
ACCENT = "#2f6df0"
GOOD = "#1f9d6b"
BAD = "#d6453d"
MUTED = "#6b7896"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}")


def build_env(n_robots, seed):
    """A SimpleEnv from the package defaults with a seeded episode."""
    random.seed(seed)
    cfg = Config()
    return SimpleEnv(
        world_width=cfg.env.world_width,
        world_height=cfg.env.world_height,
        environment_dim=cfg.env.environment_dim,
        robot_radius=cfg.env.robot_radius,
        max_steps=cfg.env.max_steps,
        n_robots=n_robots,
        max_robots=cfg.env.max_robots,
        time_delta=cfg.env.time_delta,
        goal_reached_dist=cfg.env.goal_reached_dist,
        lidar_max_range=cfg.env.lidar_max_range,
        obstacle_definitions=[list(o) for o in cfg.env.obstacle_definitions],
    )


########################################
#       1. Environment schematic       #
########################################


def fig_environment():
    env = build_env(4, seed=13)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    viz.render_env(env, ax=ax)

    robot = env.robots[0]
    ax.annotate(
        "robot",
        (robot.x, robot.y),
        (robot.x, robot.y - 1.1),
        color=INK,
        fontsize=10,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
    )
    gx, gy = float(env.goals_x[0]), float(env.goals_y[0])
    ax.annotate(
        "goal",
        (gx, gy),
        (gx + 0.4, gy + 0.8),
        color=GOOD,
        fontsize=10,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": GOOD, "lw": 0.8},
    )
    wide = env.world.obstacles[2]
    ax.text(
        wide.x + wide.width / 2,
        wide.y + wide.height / 2,
        "obstacle",
        color="white",
        fontsize=9,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.set_title(
        "SimpleEnv multi-robot navigation world (one sampled episode)",
    )
    save(fig, "environment")


########################################
#     1b. Random episodes montage      #
########################################


def fig_layouts():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    for ax, seed in zip(axes, (21, 22, 23), strict=False):
        env = build_env(3, seed=seed)
        viz.render_env(env, ax=ax, show_lidar=False)
        ax.set_title(f"sampled episode (seed {seed})", fontsize=10)
    fig.suptitle(
        "Per-episode randomization: start poses, headings and goals vary "
        "over the fixed obstacle course",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "layouts")


########################################
# 2. Actor / twin-critic architecture  #
########################################


def _layer(ax, x, y, w, h, text, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=face,
            edgecolor=INK,
            linewidth=1.3,
        ),
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color=INK,
    )


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            color=INK,
            linewidth=1.2,
        ),
    )


def _row(ax, labels, faces, y, h):
    w, gap = 1.9, 0.45
    total = len(labels) * w + (len(labels) - 1) * gap
    x = (10 - total) / 2
    for i, (lab, face) in enumerate(zip(labels, faces, strict=False)):
        _layer(ax, x, y, w, h, lab, face)
        if i < len(labels) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap


def fig_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # ── Actor: one row of layers, centred vertically ──
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Actor  $\\mu(s)$")
    _row(
        ax,
        ["state\n(24)", "FC 800\nReLU", "FC 600\nReLU", "action (2)\ntanh"],
        ["#eef3ff", "#dbe6ff", "#c6d8ff", "#9bbcff"],
        1.8,
        1.4,
    )

    # ── Twin critic: two stacked Q heads over the same input ──
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Twin critic  $Q_1(s,a)$, $Q_2(s,a)$")
    faces = ["#eafaf2", "#cdeedd", "#b5e6cd", "#8fd9b6"]
    for qlab, y in (("$Q_1$\n(1)", 2.9), ("$Q_2$\n(1)", 0.7)):
        _row(
            ax,
            ["state+action\n(24+2)", "FC 800\nReLU", "FC 600\nReLU", qlab],
            faces,
            y,
            1.4,
        )

    fig.suptitle(
        "TD3 networks: deterministic actor and twin action-value critics",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Polyak-averaged target copies of all three networks; smoothed "
        "target actions;\nactor updated every 2 critic steps against "
        "$\\min(Q_1, Q_2)$ targets.",
        ha="center",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save(fig, "architecture")


########################################
#     3. Learning curve + outcomes     #
########################################


def _moving_avg(a, k=9):
    if len(a) < k:
        return a
    kernel = np.ones(k) / k
    return np.convolve(a, kernel, mode="valid")


def _curriculum_spans(epochs, counts):
    """(start, end, robots) spans where the active-robot count is flat."""
    spans = []
    start = epochs[0]
    for i in range(1, len(epochs)):
        if counts[i] != counts[i - 1]:
            spans.append((start, epochs[i], counts[i - 1]))
            start = epochs[i]
    spans.append((start, epochs[-1], counts[-1]))
    return spans


def _shade_curriculum(ax, spans):
    for k, (a, b, count) in enumerate(spans):
        if k % 2:
            ax.axvspan(a, b, color="#eef1f8", zorder=0)
        label = f"n={count}" if k == 0 else str(count)
        ax.text(
            (a + b) / 2,
            0.965,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
            color=MUTED,
        )


def fig_learning_curve():
    log = json.loads((ASSETS / "TD3_simpleEnv.json").read_text())
    epochs = np.array([e["Epoch"] for e in log])
    reward = np.array([e["Avg_reward"] for e in log])
    arrived = np.array([e["Avg_arrived"] for e in log]) * 100
    collision = np.array([e["Avg_collision"] for e in log]) * 100
    counts = np.rint([e["Avg_N_robots"] for e in log]).astype(int)
    spans = _curriculum_spans(epochs, counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # ── Left: mean evaluation return, curriculum phases shaded ──
    _shade_curriculum(ax1, spans)
    ax1.plot(
        epochs,
        reward,
        color=ACCENT,
        alpha=0.25,
        linewidth=0.8,
        label="eval epoch",
    )
    ma = _moving_avg(reward, 9)
    ax1.plot(
        epochs[4 : 4 + len(ma)],
        ma,
        color=ACCENT,
        linewidth=2.2,
        label="9-epoch moving avg",
    )
    ax1.axhline(0, color=INK, linewidth=0.6, linestyle=":")
    ax1.set_xlabel("evaluation epoch (5000 timesteps each)")
    ax1.set_ylabel("mean per-robot return")
    ax1.set_title("Evaluation return (active robots shaded)")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Right: arrival / collision rates over the same curriculum ──
    _shade_curriculum(ax2, spans)
    ax2.plot(epochs, arrived, color=GOOD, linewidth=1.8, label="arrived")
    ax2.plot(epochs, collision, color=BAD, linewidth=1.8, label="collision")
    ax2.set_xlabel("evaluation epoch (5000 timesteps each)")
    ax2.set_ylabel("per-robot rate (%)")
    ax2.set_title("Terminal outcomes (100 eval episodes)")
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=False, fontsize=9, loc="center right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "learning_curve")


########################################
#  4. Success / failure trajectories   #
########################################


def _draw_rollout(ax, t, title):
    """One recorded episode in its own layout, all from the JSON dump."""
    radius = Config().env.robot_radius
    viz.draw_field(ax, t["width"], t["height"])
    viz.draw_obstacles(ax, t["obstacles"])
    for i, robot in enumerate(t["robots"]):
        color = viz.COLORS[i]
        viz.draw_goal(
            ax,
            (robot["goal"][0], robot["goal"][1]),
            t["goal_threshold"],
            color,
        )
        viz.draw_trajectory(ax, robot["xs"], robot["ys"], color)
        if robot["outcome"] == "collision":
            ax.scatter(
                robot["xs"][-1],
                robot["ys"][-1],
                color=BAD,
                marker="x",
                s=70,
                zorder=8,
            )
        else:
            theta = robot["thetas"][-1] if robot.get("thetas") else 0.0
            viz.draw_robot(
                ax,
                robot["xs"][-1],
                robot["ys"][-1],
                theta,
                radius,
                color,
            )
    ax.set_title(title, fontsize=10)


def fig_trajectory():
    success_path = ASSETS / "trajectory_success.json"
    fail_path = ASSETS / "trajectory.json"
    panels = []
    if success_path.exists():
        t = json.loads(success_path.read_text())
        panels.append((t, "successful episode (every robot arrives)"))
    if fail_path.exists():
        t = json.loads(fail_path.read_text())
        outcomes = [robot["outcome"] for robot in t["robots"]]
        if "collision" in outcomes:
            panels.append((t, "failure episode (collision)"))
    if not panels:
        print("skip trajectory (no rollout assets)")
        return

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.4 * len(panels), 5.2),
    )
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (t, title) in zip(axes, panels, strict=False):
        _draw_rollout(ax, t, title)
    fig.suptitle(
        "Deterministic shared policy on unseen episodes",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "trajectory")


def main():
    fig_environment()
    fig_layouts()
    fig_architecture()
    if (ASSETS / "TD3_simpleEnv.json").exists():
        fig_learning_curve()
    fig_trajectory()


if __name__ == "__main__":
    main()
