"""Débit de l'environnement en pas de décision par seconde (validation j5-3).

Ce chiffre fixe le budget d'entraînement : par exemple, 2 millions de pas
à 2 000 pas/s font environ 17 minutes, à 300 pas/s presque 2 heures.

    python scripts/bench_env.py --duration 10 --n-envs 1 2 4 8
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np

import pai_core.envs  # noqa: F401  (enregistre PaiNav-v0)


def bench_single(duration: float) -> float:
    env = gym.make("PaiNav-v0")
    env.reset(seed=0)
    env.action_space.seed(0)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < duration:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        n += 1
        if term or trunc:
            env.reset()
    env.close()
    return n / (time.perf_counter() - t0)


def bench_vector(n_envs: int, duration: float) -> float:
    # processus séparés, comme SubprocVecEnv de Stable-Baselines3 ;
    # la réinitialisation en fin d'épisode est automatique
    envs = gym.vector.AsyncVectorEnv([lambda: gym.make("PaiNav-v0")] * n_envs)
    envs.reset(seed=0)
    envs.action_space.seed(0)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < duration:
        envs.step(envs.action_space.sample())
        n += n_envs
    envs.close()
    return n / (time.perf_counter() - t0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--n-envs", type=int, nargs="+", default=[2, 4])
    a = p.parse_args()

    print(f"Cœurs logiques disponibles : {os.cpu_count()}")
    single = bench_single(a.duration)
    print(f"1 environnement            : {single:8.0f} pas/s  "
          f"({1e3 / single:.2f} ms par décision, soit 50 pas physiques + 36 rayons)")
    for n in a.n_envs:
        r = bench_vector(n, a.duration)
        print(f"{n} environnements parallèles : {r:8.0f} pas/s  (accélération ×{r / single:.1f})")
