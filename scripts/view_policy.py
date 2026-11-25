"""Regarder la politique PPO entraînée dans le visualiseur MuJoCo, en temps réel.

Même boucle que view_env.py, mais l'action vient du réseau au lieu de la loi naïve.

    python scripts/view_policy.py --model checkpoints/curriculum_s0/best/best_model.zip
    python scripts/view_policy.py --model ... --obstacles 0 0     # sans obstacles, pour comparer

VecNormalize ne normalise que la récompense (norm_obs=False) : la politique
lit les observations brutes, inutile de charger vecnormalize.pkl pour la jouer.
"""
import argparse
import time

import mujoco.viewer

from pai_core.envs.nav_env import NavEnv, NavEnvConfig
from pai_core.rl.ppo_decider import PPODecider

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help=".zip sauvegardé par train_ppo.py")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--obstacles", type=int, nargs=2, default=(4, 8), metavar=("MIN", "MAX"),
               help="nombre d'obstacles ; (4, 8) = dernier niveau du curriculum")
p.add_argument("--stochastic", action="store_true",
               help="échantillonner l'action (avec le bruit d'exploration) au lieu de jouer μ")
a = p.parse_args()

decider = PPODecider(a.model, deterministic=not a.stochastic)
env = NavEnv(NavEnvConfig(n_obstacles=tuple(a.obstacles)))
obs, _ = env.reset(seed=a.seed)
decider.reset()
n_ep, n_succes = 0, 0

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        obs, r, term, trunc, info = env.step(decider.act(obs))
        viewer.sync()
        if term or trunc:
            n_ep += 1
            n_succes += bool(info["is_success"])
            fin = "succès" if info["is_success"] else "collision" if info["collision"] else "temps écoulé"
            print(f"épisode {n_ep:3d} : {fin:12s} {info['steps']:3d} pas  "
                  f"{info['n_obstacles']} obstacles   (succès : {n_succes}/{n_ep})")
            time.sleep(0.5)
            obs, _ = env.reset()
            decider.reset()
        time.sleep(max(0.0, env.cfg.decision_dt - (time.time() - t0)))
