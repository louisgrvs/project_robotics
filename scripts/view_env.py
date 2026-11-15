"""Regarder des épisodes dans le visualiseur MuJoCo, en temps réel.

Le robot est piloté par une loi naïve « tourner vers le but, puis avancer »,
qui ignore les obstacles : on doit voir des succès ET des collisions, ce qui
vérifie à l'œil les deux fins d'épisode.

    python scripts/view_env.py --seed 0
"""
import argparse
import time

import mujoco.viewer
import numpy as np

from pai_core.envs.nav_env import NavEnv
from pai_core.perception.features import SLICES

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
seed = p.parse_args().seed

env = NavEnv()
obs, _ = env.reset(seed=seed)
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        cos_b, sin_b = obs[SLICES["goal"]][1:]
        bearing = np.arctan2(sin_b, cos_b)
        action = [max(cos_b, 0.0), np.clip(2.0 * bearing, -1.0, 1.0)]
        obs, r, term, trunc, info = env.step(action)
        viewer.sync()
        if term or trunc:
            fin = "succès" if info["is_success"] else "collision" if info["collision"] else "temps écoulé"
            print(f"{fin:12s}  {info['steps']:3d} pas  {info['n_obstacles']} obstacles")
            time.sleep(0.5)
            obs, _ = env.reset()
        time.sleep(max(0.0, env.cfg.decision_dt - (time.time() - t0)))
