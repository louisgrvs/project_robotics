"""j11-1 : taux de succès du modèle de contrôle sur N épisodes sans obstacle."""
import argparse
import math
from collections import Counter

from pai_core.envs.nav_env import NavEnv, NavEnvConfig
from pai_core.rl.ppo_decider import PPODecider
from pai_core.rl.train_ppo import run_episode


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion (Wilson)."""
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    demi = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - demi, centre + demi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/controle_sans_obstacle/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=200)
    a = p.parse_args()

    env = NavEnv(NavEnvConfig(n_obstacles=(0, 0)))
    decider = PPODecider(a.model)
    fins = Counter()
    for i in range(a.episodes):
        decider.reset()
        # graines 20 000+ : disjointes de l'entraînement (0-3) et de l'évaluation (10 000+)
        _, _, raison = run_episode(env, decider.act, seed=20_000 + i)
        fins[raison] += 1

    k, n = fins["succès"], a.episodes
    lo, hi = wilson(k, n)
    print(dict(fins))
    print(f"succès : {k / n:.1%}  (IC 95 % : {lo:.1%} – {hi:.1%})")


if __name__ == "__main__":
    main()