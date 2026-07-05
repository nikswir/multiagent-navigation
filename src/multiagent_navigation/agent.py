"""TD3 agent: actor / twin-critic networks and the delayed-update trainer.

Twin Delayed DDPG as driven by the training loop in `lib`: a tanh-bounded
Actor, a twin Critic (two Q heads taking state and action), target networks
hard-copied at construction and soft-updated (Polyak `tau`) every
`policy_freq` iterations, and clipped smoothing noise on the replayed target
actions. Tensors are routed through the `device` handed to `TD3` — device
*selection* lives in the entry points (`run`, `viz`), never in here.
"""

from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from multiagent_navigation.replay_buffer import ReplayBuffer

########################################
#               Networks               #
########################################


class Actor(nn.Module):
    """State -> action, tanh-bounded to [-1, 1] per dimension."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 800,
        hidden2: int = 600,
    ) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, hidden1)
        self.layer_2 = nn.Linear(hidden1, hidden2)
        self.layer_3 = nn.Linear(hidden2, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a: torch.Tensor = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    """Twin Q heads; each mixes state and action in its hidden layer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 800,
        hidden2: int = 600,
    ) -> None:
        super().__init__()

        # ── Q1 head ──────────────────────────────
        self.layer_1 = nn.Linear(state_dim, hidden1)
        self.layer_2_s = nn.Linear(hidden1, hidden2)
        self.layer_2_a = nn.Linear(action_dim, hidden2)
        self.layer_3 = nn.Linear(hidden2, 1)

        # ── Q2 head ──────────────────────────────
        self.layer_4 = nn.Linear(state_dim, hidden1)
        self.layer_5_s = nn.Linear(hidden1, hidden2)
        self.layer_5_a = nn.Linear(action_dim, hidden2)
        self.layer_6 = nn.Linear(hidden2, 1)

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ── Q1: state/action mixed via the hidden weights directly ──
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        # ── Q2: same mixing with the second head's weights ──
        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


########################################
#              TD3 agent               #
########################################


class TD3:
    """The agent: online/target networks, action query and the update."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.0005,
        hidden1: int = 800,
        hidden2: int = 600,
    ) -> None:
        self.device = device

        # ── Actor and its hard-copied target ──
        self.actor = Actor(state_dim, action_dim, hidden1, hidden2).to(device)
        self.actor_target = Actor(
            state_dim,
            action_dim,
            hidden1,
            hidden2,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        # ── Twin critic and its hard-copied target ──
        self.critic = Critic(
            state_dim,
            action_dim,
            hidden1,
            hidden2,
        ).to(device)
        self.critic_target = Critic(
            state_dim,
            action_dim,
            hidden1,
            hidden2,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.max_action = max_action
        self.iter_count = 0

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """The deterministic policy action for a single flat state."""
        s = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action: np.ndarray = self.actor(s).cpu().data.numpy().flatten()
        return action

    ########################################
    #           Training update            #
    ########################################

    def train(
        self,
        replay_buffer: ReplayBuffer,
        iterations: int,
        batch_size: int = 256,
        discount: float = 0.9999,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
    ) -> None:
        """Run `iterations` TD3 updates from the replay buffer."""
        av_q = 0.0
        max_q = float("-inf")
        av_critic_loss = 0.0
        av_actor_loss = 0.0

        for it in range(iterations):
            # ── 1. Sample a batch: (s, a, r, t, s2) column order ──
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

            # ── 2. Target action with clipped smoothing noise ──
            next_action = self.actor_target(next_state)
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            )
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(
                -self.max_action,
                self.max_action,
            )

            # ── 3. Clipped double-Q target ──
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            av_q += torch.mean(target_q).item()
            max_q = max(max_q, torch.max(target_q).item())
            target_q = reward + ((1 - done) * discount * target_q).detach()

            # ── 4. Critic update on both heads ──
            current_q1, current_q2 = self.critic(state, action)
            loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2,
                target_q,
            )
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            # ── 5. Delayed actor update + Polyak target updates ──
            if it % policy_freq == 0:
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(
                    self.actor.parameters(),
                    self.actor_target.parameters(),
                    strict=False,
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data,
                    )

                for param, target_param in zip(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    strict=False,
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data,
                    )
                av_actor_loss += actor_grad.item()

            av_critic_loss += loss.item()

        self.iter_count += 1

        # ── One diagnostics line per call ──
        avg_actor_loss = (
            round(av_actor_loss / (iterations // policy_freq), 4)
            if iterations >= policy_freq
            else float("nan")
        )
        print(
            f"Iteration {self.iter_count}  ",
            f"Steps: {iterations}  ",
            f"Average Actor loss: {avg_actor_loss}  ",
            f"Average Critic loss: {round(av_critic_loss / iterations, 4)}  ",
            f"Average Q value: {round(av_q / iterations, 4)}  ",
            f"Max Q value: {round(max_q, 4)}",
        )

    ########################################
    #            Checkpointing             #
    ########################################

    def save(self, filename: str, directory: str | Path) -> None:
        torch.save(
            self.actor.state_dict(),
            Path(directory) / f"{filename}_actor.pth",
        )
        torch.save(
            self.critic.state_dict(),
            Path(directory) / f"{filename}_critic.pth",
        )

    def load(self, filename: str, directory: str | Path) -> None:
        # ── map_location keeps checkpoints loadable on any device ──
        self.actor.load_state_dict(
            torch.load(
                Path(directory) / f"{filename}_actor.pth",
                map_location=self.device,
            ),
        )
        self.critic.load_state_dict(
            torch.load(
                Path(directory) / f"{filename}_critic.pth",
                map_location=self.device,
            ),
        )
