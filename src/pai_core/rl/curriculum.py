"""Curriculum : densité d'obstacles et distance au but croissantes.

On passe au niveau suivant quand le taux de succès, mesuré sur les derniers
épisodes du niveau courant, dépasse un seuil (80 %).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class Level:
    n_obstacles: tuple[int, int]      # (min, max), tiré uniformément à chaque reset
    goal_dist: tuple[float, float]    # (min, max) en mètres


LEVELS: tuple[Level, ...] = (
    Level(n_obstacles=(0, 0), goal_dist=(1.0, 2.0)),   # 0 : aller au but, court (déjà maîtrisé)
    Level(n_obstacles=(0, 2), goal_dist=(1.5, 3.0)),   # 1 : premiers obstacles, souvent aucun
    Level(n_obstacles=(2, 4), goal_dist=(2.0, 4.0)),   # 2 : contourner devient nécessaire
    Level(n_obstacles=(3, 6), goal_dist=(2.0, 5.0)),   # 3
    Level(n_obstacles=(4, 8), goal_dist=(2.0, 6.0)),   # 4 : la tâche cible
)


def should_advance(successes, threshold: float, min_episodes: int) -> bool:
    n = len(successes)
    return n >= min_episodes and sum(successes) / n >= threshold


class CurriculumCallback(BaseCallback):
    def __init__(self, levels=LEVELS, threshold=0.8, window=200, min_episodes=100, verbose=1):
        super().__init__(verbose)
        assert min_episodes <= window, "la fenêtre doit pouvoir contenir min_episodes résultats"
        self.levels, self.threshold, self.min_episodes = levels, threshold, min_episodes
        self.level = 0
        self.successes = deque(maxlen=window)

    def _apply_level(self, k: int) -> None:
        lvl = self.levels[k]
        self.training_env.env_method("set_task", lvl.n_obstacles, lvl.goal_dist, level=k)
        self.level = k
        self.successes.clear()
        if self.verbose:
            print(f"[curriculum] niveau {k} à {self.num_timesteps} pas : "
                  f"{lvl.n_obstacles} obstacles, but à {lvl.goal_dist} m")

    def _on_training_start(self) -> None:
        self._apply_level(0)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and info.get("level") == self.level:
                self.successes.append(bool(info["is_success"]))
        dernier = self.level == len(self.levels) - 1
        if not dernier and should_advance(self.successes, self.threshold, self.min_episodes):
            self._apply_level(self.level + 1)
        self.logger.record("curriculum/level", self.level)
        if self.successes:
            self.logger.record("curriculum/success_window",
                               sum(self.successes) / len(self.successes))
        return True