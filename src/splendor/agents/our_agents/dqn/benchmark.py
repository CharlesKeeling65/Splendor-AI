"""Paired-deal, paired-seat evaluation with raw games and isolated RNG state."""

import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.minmax import MiniMaxAgent
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.utils import LimitRoundsGameRule
from splendor.template import Agent

from .features import extract_observation
from .network import ACTION_DIM, QNetwork
from .population import HeuristicAgent
from .search import outcome, search_policy

FACTORIES: dict[str, Callable[[int], Agent]] = {
    "random": RandomAgent,
    "minimax": MiniMaxAgent,
    "heuristic": HeuristicAgent,
}


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    """Evaluation cannot change the subsequent training random stream."""
    py_state, np_state, cpu_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)


def wilson(wins: int, games: int) -> list[float]:
    """Descriptive 95% binomial interval; paired games are not independent."""
    if games < 1:
        raise ValueError("need games")
    z = 1.96
    p = wins / games
    center = (p + z * z / (2 * games)) / (1 + z * z / games)
    half = (
        z
        * np.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
        / (1 + z * z / games)
    )
    return [float(center - half), float(center + half)]


@torch.no_grad()
def benchmark(
    net: QNetwork,
    opponent: str,
    seeds: list[int],
    simulations: int = 0,
) -> dict[str, Any]:
    """Each seed is played in BOTH seats; no checkpoint selection occurs here."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("deal seeds must be nonempty and unique")
    device = next(net.parameters()).device
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    was_training = net.training
    net.eval()
    started = time.monotonic()
    try:
        for seed in seeds:
            for seat in (0, 1):
                with isolated_rng(seed):
                    rule = LimitRoundsGameRule(2)
                    rival = FACTORIES[opponent](1 - seat)
                    search_rng = np.random.default_rng(seed + 17)
                    while not rule.gameEnds():
                        state = rule.current_game_state
                        turn = rule.current_agent_index
                        legal = rule.getLegalActions(state, turn)
                        if turn == seat:
                            began = time.monotonic()
                            mapping = create_action_mapping(legal, state, turn)
                            if simulations:
                                pi = search_policy(net, rule, simulations, search_rng)
                                action = int(pi.argmax())
                            else:
                                obs = extract_observation(
                                    state, seat, net.feature_version
                                )
                                mask = np.zeros(ACTION_DIM, np.float32)
                                mask[list(mapping)] = 1
                                action = net.act(
                                    torch.from_numpy(obs).to(device),
                                    torch.from_numpy(mask).to(device),
                                )
                            chosen = mapping[action]
                            latencies.append(time.monotonic() - began)
                        else:
                            chosen = rival.SelectAction(
                                legal, deepcopy(state), deepcopy(rule)
                            )
                        rule.update(chosen)
                    records.append(
                        {
                            "seed": seed,
                            "seat": seat,
                            "outcome": outcome(rule, seat),
                            "score": float(
                                rule.calScore(rule.current_game_state, seat)
                            ),
                            "rival_score": float(
                                rule.calScore(rule.current_game_state, 1 - seat)
                            ),
                            "plies": rule.action_counter,
                        }
                    )
    finally:
        net.train(was_training)
    wins = sum(r["outcome"] > 0 for r in records)
    draws = sum(r["outcome"] == 0 for r in records)
    return {
        "opponent": opponent,
        "simulations": simulations,
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": len(records) - wins - draws,
        "win_rate": wins / len(records),
        "mean_score": float(np.mean([r["score"] for r in records])),
        "win_rate_wilson_descriptive": wilson(wins, len(records)),
        "seat_win_rates": {
            str(s): float(
                np.mean([r["outcome"] > 0 for r in records if r["seat"] == s])
            )
            for s in (0, 1)
        },
        "action_seconds_mean": float(np.mean(latencies)),
        "action_seconds_p95": float(np.quantile(latencies, 0.95)),
        "action_seconds_max": max(latencies),
        "elapsed_seconds": time.monotonic() - started,
        "records": records,
    }
