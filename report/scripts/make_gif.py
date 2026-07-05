"""Animate trained-policy rollouts into a README GIF.

Loads the report/assets checkpoint, rolls out the deterministic shared policy
on freshly sampled multi-robot episodes until it collects a few where every
robot arrives, then re-renders each recorded episode frame-by-frame with the
same vector helpers (`viz`) that draw the report figures, and writes an
animated GIF preview to docs/assets/demo.gif.

    uv run python report/scripts/make_gif.py
"""

from __future__ import annotations

import os

# Pin the headless backend before pyplot is imported anywhere below.
os.environ.setdefault("MPLBACKEND", "Agg")

import io
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from multiagent_navigation import viz
from multiagent_navigation.agent import TD3
from multiagent_navigation.config_schema import Config
from multiagent_navigation.environment import Robot, SimpleEnv

SEED = 29
N_ROBOTS = 4
N_EPISODES = 6
MAX_STEPS = 200
MAX_TRIES = 200
FRAME_STRIDE = 3
HOLD_LAST = 8
FRAME_MS = 90
DPI = 110

OUT = Path(__file__).parents[2] / "docs" / "assets" / "demo.gif"
ASSETS = Path(__file__).parents[1] / "assets"

INK = "#27324a"
MUTED = "#6b7896"

########################################
#              Checkpoint              #
########################################


def load_agent(cfg: Config, device: torch.device) -> TD3:
    """Rebuild the network shell and load the report checkpoint."""
    name = cfg.train.file_name
    if not (ASSETS / f"{name}_actor.pth").exists():
        raise SystemExit(
            f"missing checkpoint {ASSETS / name}_actor.pth — run "
            "`uv run python report/scripts/run_experiment.py` first",
        )

    state_dim = cfg.env.environment_dim + 4
    agent = TD3(
        state_dim,
        cfg.model.action_dim,
        cfg.model.max_action,
        device=device,
        hidden1=cfg.model.hidden1,
        hidden2=cfg.model.hidden2,
    )
    agent.load(name, ASSETS)
    agent.actor.eval()
    return agent


########################################
#              Recording               #
########################################


def snapshot(env: SimpleEnv) -> list:
    """Per-robot pose + latest display scan for one animation frame."""
    return [
        (
            env.robots[i].x,
            env.robots[i].y,
            env.robots[i].theta,
            env.robots[i].current_lidar_data_display.copy(),
        )
        for i in range(env.n_robots)
    ]


def record_episode(agent: TD3, env: SimpleEnv, cfg: Config):
    """Roll out one deterministic episode and record poses + scans."""
    state = env.reset(n_robots=N_ROBOTS)
    frames = [snapshot(env)]

    # ── Shared policy per robot until every robot settles ──
    while sum(env.episode_done) < env.n_robots:
        action = np.zeros((env.n_robots, cfg.model.action_dim))
        for i, robot_state in enumerate(state):
            action[i] = agent.get_action(np.array(robot_state))
        action[:, 0] = (action[:, 0] + 1) / 2
        state, _rewards, _dones, _infos = env.step(action)
        frames.append(snapshot(env))

    outcomes = [(env.info[i] or {}).get("reason") for i in range(env.n_robots)]
    meta = {
        "goals": [
            [float(env.goals_x[i]), float(env.goals_y[i])]
            for i in range(env.n_robots)
        ],
        "obstacles": [
            {"x": o.x, "y": o.y, "width": o.width, "height": o.height}
            for o in env.world.obstacles
        ],
        "width": env.world.width,
        "height": env.world.height,
        "goal_threshold": env.goal_reached_dist,
    }
    return frames, outcomes, meta


########################################
#              Rendering               #
########################################


def render_frame(fig, ax, cfg, meta, frames, k, episode, total):
    """One frame: the field fills the whole image, HUD text sits on it."""
    w, h = meta["width"], meta["height"]
    ax.clear()

    # ── Full-bleed field with a flat border ──
    ax.set_facecolor(viz.FIELD_FACE)
    ax.set_xlim(-w / 2, w / 2)
    ax.set_ylim(-h / 2, h / 2)
    ax.set_aspect("equal")
    ax.axis("off")

    viz.draw_obstacles(ax, meta["obstacles"])
    for i, goal in enumerate(meta["goals"]):
        viz.draw_goal(
            ax,
            (goal[0], goal[1]),
            meta["goal_threshold"],
            viz.COLORS[i],
        )

    # ── Trails so far, one per robot ──
    for i in range(len(meta["goals"])):
        xs = [frame[i][0] for frame in frames[: k + 1]]
        ys = [frame[i][1] for frame in frames[: k + 1]]
        if len(xs) > 1:
            ax.plot(
                xs,
                ys,
                color=viz.COLORS[i],
                linewidth=1.8,
                alpha=0.7,
                solid_capstyle="round",
                zorder=6,
            )

    # ── Robots and their lidar fans at the current step ──
    probe = Robot(
        robot_radius=cfg.env.robot_radius,
        lidar_num_beams=cfg.env.environment_dim,
        lidar_max_range=cfg.env.lidar_max_range,
    )
    for i, (x, y, theta, scan) in enumerate(frames[k]):
        probe.x, probe.y, probe.theta = x, y, theta
        probe.current_lidar_data_display = scan
        viz.draw_lidar(ax, probe)
        viz.draw_robot(ax, x, y, theta, probe.radius, viz.COLORS[i])

    # ── HUD on the field itself ──
    ax.text(
        -w / 2 + 0.25,
        h / 2 - 0.3,
        "TD3 shared policy on unseen episodes",
        color=INK,
        fontsize=11,
        fontweight="bold",
        va="top",
        zorder=9,
    )
    ax.text(
        w / 2 - 0.25,
        h / 2 - 0.3,
        f"episode {episode:02d}/{total:02d} · step {k:03d}",
        color=MUTED,
        fontsize=9,
        family="monospace",
        ha="right",
        va="top",
        zorder=9,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=viz.FIELD_FACE)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE)


########################################
#             Entry point              #
########################################


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        os.environ.get(
            "MAN_DEVICE",
            "mps" if torch.backends.mps.is_available() else "cpu",
        ),
    )
    cfg = Config()
    agent = load_agent(cfg, device)

    env = SimpleEnv(
        world_width=cfg.env.world_width,
        world_height=cfg.env.world_height,
        environment_dim=cfg.env.environment_dim,
        robot_radius=cfg.env.robot_radius,
        max_steps=MAX_STEPS,
        n_robots=N_ROBOTS,
        max_robots=cfg.env.max_robots,
        time_delta=cfg.env.time_delta,
        goal_reached_dist=cfg.env.goal_reached_dist,
        lidar_max_range=cfg.env.lidar_max_range,
        obstacle_definitions=[list(o) for o in cfg.env.obstacle_definitions],
    )

    # ── Collect episodes where every robot reaches its goal ──
    episodes = []
    for _ in range(MAX_TRIES):
        frames, outcomes, meta = record_episode(agent, env, cfg)
        if all(reason == "target_reached" for reason in outcomes):
            episodes.append((frames, meta))
            print(f"episode {len(episodes)}: arrived in {len(frames)} steps")
        if len(episodes) == N_EPISODES:
            break
    if not episodes:
        raise SystemExit("no all-arrived episodes to animate")

    # ── Re-render every episode into palette frames ──
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    images = []
    for idx, (frames, meta) in enumerate(episodes, start=1):
        picks = list(range(0, len(frames), FRAME_STRIDE))
        if picks[-1] != len(frames) - 1:
            picks.append(len(frames) - 1)
        for k in picks:
            img = render_frame(
                fig,
                ax,
                cfg,
                meta,
                frames,
                k,
                idx,
                len(episodes),
            )
            images.append(img)
        images.extend([images[-1]] * HOLD_LAST)
    plt.close(fig)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        OUT,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({len(images)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
