# Architecture

How a training run flows through the package, from the Hydra entry point to
the checkpoints the animation replays. Each node names the module that owns
it; GitHub renders the diagram below natively (no build step). Keep it in
sync by hand when the pipeline changes — it is a map, not a generated
artifact.

```mermaid
graph TD
    CLI["python -m multiagent_navigation.run"] --> MAIN["run.py · @hydra.main"]

    subgraph Config["Configuration (Hydra)"]
        GROUPS["configs/ groups<br/>env · model · train · animate"]
        SCHEMA["config_schema.py<br/>typed Config (validated)"]
        GROUPS --> SCHEMA
    end

    MAIN --> SCHEMA
    SCHEMA --> TRAIN["lib.py · train(cfg, device, dirs)<br/>the training loop"]

    subgraph Env["environment.py"]
        ENV["SimpleEnv (capacity 16 robots)"]
        RESET["reset(n_robots) · random starts & goals"]
        SCAN["Robot.lidar_scan · 20-beam fan<br/>robots occlude each other"]
        REWARD["get_reward<br/>+100 / -100 / shaped step"]
        ENV --> RESET
        ENV --> SCAN
        ENV --> REWARD
    end

    subgraph Agent["agent.py + replay_buffer.py"]
        ACTOR["Actor net<br/>state 24 → action 2 (tanh)"]
        CRITIC["twin Critic<br/>(state, action) → Q1, Q2"]
        BUFFER["ReplayBuffer<br/>(s, a, r, done, s') FIFO"]
        TARGETS["target nets<br/>delayed soft update τ"]
    end

    TRAIN --> ENV
    TRAIN --> CURR["robot-count curriculum<br/>1 → n_robots over the configured ramp"]
    TRAIN --> ACTOR
    TRAIN --> CRITIC
    TRAIN --> BUFFER
    TRAIN --> TARGETS
    TRAIN --> RESULT["TrainResult<br/>agent + evaluation log"]
    RESULT --> OUT["Hydra per-run output dir<br/>results/ + pytorch_models/"]
    OUT --> VIZ["viz.py · FuncAnimation<br/>replays the best epoch, dumps a state CSV"]
```

## The flow

- **`run.py`** is the CLI entry only — `@hydra.main` composes the config from
  `configs/`, validates it against the typed schema, picks the device via
  `lib.select_device` (`MAN_DEVICE` override, else CUDA → MPS → CPU) and
  routes both output dirs through Hydra's per-run output directory.
  Importing the package never imports it.
- **`config_schema.py`** declares the typed `Config`: the `env` group (world
  geometry, lidar, obstacle course, robot *capacity*), `model` (TD3 network
  sizes and update hyper-parameters), `train` (schedule, exploration decay,
  curriculum cap, checkpoint naming) and `animate` (playback and artifact
  paths).
- **`lib.py`** owns the training loop: it builds `SimpleEnv`, `TD3` and the
  `ReplayBuffer` from the config, ramps the active robot count (curriculum),
  decays exploration noise at runtime, trains after every episode, and every
  `eval_freq` timesteps evaluates the policy and writes the shared checkpoint
  contract — the JSON log `{results_dir}/{file_name}` and weights
  `{models_dir}/{file_name}_epoch-{N}`. The state dimension is *derived*:
  `environment_dim + 4`.
- **`environment.py`** holds `SimpleEnv`: a bounded world with rectangular
  obstacles and wall bounds, up to `max_robots` (capacity 16) disc robots
  with unicycle kinematics. Each robot's 20-beam half-circle lidar sees
  obstacles *and the other robots*; the 24-d state adds normalized goal
  distance/angle and the robot's velocities. Episodes end per robot on goal
  arrival, collision or the step cap; `get_reward` is the fixed pure reward.
- **`agent.py`** holds TD3: the tanh-bounded **Actor**, the **twin Critic**,
  hard-copied target networks with delayed (`policy_freq`) Polyak soft
  updates, clipped smoothing noise on target actions, and device-agnostic
  checkpoint save/load. **`replay_buffer.py`** is the FIFO experience replay
  storing `(s, a, r, done, s')` — done *before* next-state.
- **`viz.py`** replays a trained run: it reads the same results log, picks
  the best epoch by average reward, loads those weights and animates the
  shared policy for `animate.n_robots` robots, dumping every robot's per-step
  state to a CSV for offline analysis.

The public surface (`__all__` in `__init__.py`) is the training entry,
result and config types plus the domain classes, re-exported from their
modules; `test_api_stability.py` pins the contract so it can't drift
silently.
