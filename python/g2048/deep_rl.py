"""Chapitre 8 (optionnel) : Deep RL avec PyTorch, via l'API type Gymnasium
(chapitre 3, classe `Env`). DQN avec deux types d'entrée : grille brute
one-hot par exposant, ou vecteur de features (chapitre 5).

Objectif réaliste (chapitre 8 du plan) : apprendre et comparer, pas battre
la recherche expectimax du chapitre 7.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._g2048 import Env, compute_features
from .experience import EXPERIENCE_PATH, load_experience
from .features import DEFAULT_FEATURES as ALL_FEATURES

NUM_ACTIONS = 4
NUM_CELLS = 16
NUM_EXPONENTS = 16
ONEHOT_DIM = NUM_CELLS * NUM_EXPONENTS


def encode_onehot(observation: list[int]) -> torch.Tensor:
    """Grille brute : un vecteur one-hot par case (16 × 16 = 256 dims)."""
    x = torch.zeros(ONEHOT_DIM)
    for cell, exponent in enumerate(observation):
        x[cell * NUM_EXPONENTS + exponent] = 1.0
    return x


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor | None
    legal_next_actions: list[int]


@dataclass
class DqnConfig:
    representation: str = "onehot"  # "onehot" ou "features"
    features: list[str] = field(default_factory=lambda: list(ALL_FEATURES))
    episodes: int = 500
    max_moves: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 20_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 400
    target_update_every: int = 200
    reward_scale: float = 0.001
    eval_every: int = 50
    eval_games: int = 5
    seed: int = 0
    # Précharge le buffer de replay avec les transitions du jeu de
    # données d'expérience persistant (meilleures parties de toutes les
    # expériences déjà lancées, cf. experience.py) avant l'auto-play.
    warm_start_from_experience: bool = False


def _encode_from_exponents(exponents: list[int], config: DqnConfig) -> torch.Tensor:
    if config.representation == "onehot":
        return encode_onehot(exponents)
    if config.representation == "features":
        return torch.tensor(compute_features(exponents, config.features), dtype=torch.float32)
    raise ValueError(f"représentation inconnue: {config.representation}")


def _encode(env: Env, config: DqnConfig) -> torch.Tensor:
    return _encode_from_exponents(env.observation(), config)


def load_experience_into_buffer(
    buffer: deque[Transition], config: DqnConfig, dataset_path: Path = EXPERIENCE_PATH
) -> int:
    """Convertit le jeu de données d'expérience persistant en transitions
    DQN (encodées selon `config.representation`) et les ajoute au buffer.
    Retourne le nombre de transitions ajoutées."""
    df = load_experience(dataset_path)
    if df is None or df.height == 0:
        return 0

    added = 0
    for _keys, group in df.sort("step_index").group_by(["source_experiment", "seed"]):
        rows = group.sort("step_index").to_dicts()
        for i, row in enumerate(rows):
            state = _encode_from_exponents(row["board_before"], config)
            reward = row["gained"] * config.reward_scale
            if not row["terminal"] and i + 1 < len(rows):
                next_row = rows[i + 1]
                next_state = _encode_from_exponents(next_row["board_before"], config)
                legal_next_actions = [d for d, ok in enumerate(next_row["legal"]) if ok]
            else:
                next_state = None
                legal_next_actions = []
            buffer.append(
                Transition(state, row["chosen_direction"], reward, next_state, legal_next_actions)
            )
            added += 1
    return added


def _input_dim(config: DqnConfig) -> int:
    return ONEHOT_DIM if config.representation == "onehot" else len(config.features)


def _epsilon(config: DqnConfig, episode: int) -> float:
    if episode >= config.epsilon_decay_episodes:
        return config.epsilon_end
    frac = episode / max(config.epsilon_decay_episodes, 1)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def _select_action(q_values: torch.Tensor, legal_actions: list[int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice(legal_actions)
    masked = q_values.clone()
    mask = torch.full((NUM_ACTIONS,), float("-inf"))
    mask[legal_actions] = 0.0
    masked = masked + mask
    return int(torch.argmax(masked).item())


def _optimize(net: QNetwork, target_net: QNetwork, optimizer: torch.optim.Optimizer, batch: list[Transition], gamma: float) -> float:
    states = torch.stack([t.state for t in batch])
    actions = torch.tensor([t.action for t in batch], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)

    q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    targets = rewards.clone()
    non_terminal = [i for i, t in enumerate(batch) if t.next_state is not None]
    if non_terminal:
        next_states = torch.stack([batch[i].next_state for i in non_terminal])
        with torch.no_grad():
            next_q = target_net(next_states)
        for row, i in enumerate(non_terminal):
            legal = batch[i].legal_next_actions
            best_next = next_q[row, legal].max().item() if legal else 0.0
            targets[i] += gamma * best_next

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_dqn(config: DqnConfig) -> tuple[QNetwork, list[dict]]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    input_dim = _input_dim(config)
    net = QNetwork(input_dim)
    target_net = QNetwork(input_dim)
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    buffer: deque[Transition] = deque(maxlen=config.buffer_size)

    if config.warm_start_from_experience:
        load_experience_into_buffer(buffer, config)

    history = []
    steps_done = 0

    for episode in range(config.episodes):
        env = Env(config.seed * 1_000_003 + episode)
        env.reset(config.seed * 1_000_003 + episode)
        epsilon = _epsilon(config, episode)

        for _ in range(config.max_moves):
            legal = env.legal_moves()
            if not legal:
                break
            state = _encode(env, config)
            with torch.no_grad():
                q_values = net(state.unsqueeze(0)).squeeze(0)
            action = _select_action(q_values, legal, epsilon)

            _, reward, terminated, _truncated, _info = env.step(action)
            reward *= config.reward_scale

            next_legal = env.legal_moves()
            next_state = None if terminated or not next_legal else _encode(env, config)
            buffer.append(Transition(state, action, reward, next_state, next_legal))

            if len(buffer) >= config.batch_size:
                batch = random.sample(buffer, config.batch_size)
                _optimize(net, target_net, optimizer, batch, config.gamma)

            steps_done += 1
            if steps_done % config.target_update_every == 0:
                target_net.load_state_dict(net.state_dict())

            if terminated:
                break

        if (episode + 1) % config.eval_every == 0 or episode + 1 == config.episodes:
            mean_score = evaluate_dqn(net, config, num_games=config.eval_games)
            history.append({"episode": episode + 1, "mean_score": mean_score})

    return net, history


def evaluate_dqn(net: QNetwork, config: DqnConfig, *, num_games: int, seed_offset: int = 500_000) -> float:
    net.eval()
    total = 0
    with torch.no_grad():
        for i in range(num_games):
            env = Env(seed_offset + i)
            env.reset(seed_offset + i)
            for _ in range(config.max_moves):
                legal = env.legal_moves()
                if not legal:
                    break
                state = _encode(env, config)
                q_values = net(state.unsqueeze(0)).squeeze(0)
                action = _select_action(q_values, legal, epsilon=0.0)
                _, _reward, terminated, _truncated, _info = env.step(action)
                if terminated:
                    break
            total += env.score()
    net.train()
    return total / num_games


def save_weights(net: QNetwork, path: Path) -> None:
    torch.save(net.state_dict(), path)
