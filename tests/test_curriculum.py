"""Tests du curriculum (jour 11) : règle de passage, niveaux, set_task, rappel.

Le rappel est testé avec un faux VecEnv et un faux journal : on vérifie sa
logique sans lancer d'entraînement.
"""
import math

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from pai_core.envs.nav_env import NavEnv
from pai_core.rl.curriculum import LEVELS, CurriculumCallback, should_advance


# ============================================================================
# 1. Règle de passage
# ============================================================================

def test_should_advance_needs_enough_episodes():
    """Fenêtre vide ou trop courte : jamais, même à 100 % de succès."""
    assert not should_advance([], 0.8, 100)
    assert not should_advance([True] * 3, 0.8, 100)
    assert not should_advance([True] * 99, 0.8, 100)


def test_should_advance_threshold():
    """79 % : non ; 80 % : oui (seuil inclus)."""
    assert not should_advance([True] * 79 + [False] * 21, 0.8, 100)
    assert should_advance([True] * 80 + [False] * 20, 0.8, 100)


# ============================================================================
# 2. Niveaux
# ============================================================================

def test_levels_get_harder():
    """Bornes des obstacles et de la distance jamais décroissantes d'un niveau au suivant."""
    for a, b in zip(LEVELS, LEVELS[1:]):
        assert b.n_obstacles[0] >= a.n_obstacles[0] and b.n_obstacles[1] >= a.n_obstacles[1]
        assert b.goal_dist[0] >= a.goal_dist[0] and b.goal_dist[1] >= a.goal_dist[1]


def test_levels_are_valid():
    for lvl in LEVELS:
        lo, hi = lvl.n_obstacles
        assert 0 <= lo <= hi
        assert 0 < lvl.goal_dist[0] <= lvl.goal_dist[1]


def test_first_and_last_level():
    """Le premier niveau est l'entraînement de contrôle, le dernier la tâche cible."""
    assert LEVELS[0].n_obstacles == (0, 0)
    assert LEVELS[-1].n_obstacles == (4, 8)
    assert LEVELS[-1].goal_dist == (2.0, 6.0)


# ============================================================================
# 3. set_task dans l'environnement
# ============================================================================

@pytest.mark.parametrize("k", [0, len(LEVELS) - 1])
def test_set_task_applies_level(k):
    """Après set_task puis reset : nombre d'obstacles, distance au but et niveau conformes."""
    env = NavEnv()
    lvl = LEVELS[k]
    env.set_task(lvl.n_obstacles, lvl.goal_dist, level=k)
    for seed in range(10):
        _, info = env.reset(seed=seed)
        s = env.scene
        d = math.hypot(s.goal[0] - s.start[0], s.goal[1] - s.start[1])
        assert lvl.n_obstacles[0] <= info["n_obstacles"] <= lvl.n_obstacles[1]
        assert lvl.goal_dist[0] <= d <= lvl.goal_dist[1]
        assert info["level"] == k
    env.close()


def test_level_frozen_during_episode():
    """Un changement de niveau en cours d'épisode ne touche que l'épisode suivant."""
    env = NavEnv()
    env.set_task(LEVELS[0].n_obstacles, LEVELS[0].goal_dist, level=0)
    env.reset(seed=0)
    env.set_task(LEVELS[-1].n_obstacles, LEVELS[-1].goal_dist, level=len(LEVELS) - 1)
    info = env.step([0.0, 0.0])[4]
    assert info["level"] == 0
    _, info = env.reset(seed=1)
    assert info["level"] == len(LEVELS) - 1
    env.close()


# ============================================================================
# 4. Le rappel, avec un faux VecEnv
# ============================================================================

class FakeVecEnv:
    """Enregistre les appels à env_method au lieu de les transmettre."""
    def __init__(self):
        self.calls = []

    def env_method(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))


class FakeLogger:
    def __init__(self):
        self.values = {}

    def record(self, key, value):
        self.values[key] = value


class FakeModel:
    """Ce que SB3 donne au rappel au début de learn() (init_callback).

    BaseCallback ne stocke pas training_env ni logger : ce sont des propriétés
    qui les demandent au modèle (self.model.get_env(), self.model.logger).
    On fournit donc un faux modèle qui renvoie nos faux objets.
    """
    def __init__(self):
        self.env = FakeVecEnv()
        self.logger = FakeLogger()

    def get_env(self):
        return self.env


def make_callback(**kwargs):
    cb = CurriculumCallback(verbose=0, **kwargs)
    cb.model = FakeModel()
    return cb


def finish_episodes(cb, n, success, level=None):
    """Simule n fins d'épisode (une par appel de _on_step, un seul environnement)."""
    for _ in range(n):
        cb.locals = {"dones": np.array([True]),
                     "infos": [{"is_success": success,
                                "level": cb.level if level is None else level}]}
        cb._on_step()


def test_callback_starts_at_level_0():
    cb = make_callback()
    cb._on_training_start()
    name, args, kwargs = cb.training_env.calls[-1]
    assert name == "set_task"
    assert args == (LEVELS[0].n_obstacles, LEVELS[0].goal_dist) and kwargs == {"level": 0}


def test_callback_advances_and_clears_window():
    cb = make_callback(min_episodes=100, window=200)
    cb._on_training_start()
    finish_episodes(cb, 99, success=True)
    assert cb.level == 0                         # pas encore assez d'épisodes
    finish_episodes(cb, 1, success=True)
    assert cb.level == 1
    assert len(cb.successes) == 0                # fenêtre vidée au changement
    assert cb.training_env.calls[-1][2] == {"level": 1}


def test_callback_ignores_stale_episodes():
    """Les épisodes commencés à l'ancien niveau ne comptent pas pour le nouveau."""
    cb = make_callback(min_episodes=10, window=20)
    cb._on_training_start()
    cb._apply_level(1)
    finish_episodes(cb, 50, success=True, level=0)
    assert cb.level == 1 and len(cb.successes) == 0


def test_callback_stays_on_last_level():
    cb = make_callback(min_episodes=10, window=20)
    cb._on_training_start()
    for _ in range(len(LEVELS) + 3):
        finish_episodes(cb, 10, success=True)
    assert cb.level == len(LEVELS) - 1
    assert cb.logger.values["curriculum/level"] == len(LEVELS) - 1


def test_callback_does_not_advance_when_failing():
    cb = make_callback(min_episodes=10, window=20)
    cb._on_training_start()
    finish_episodes(cb, 100, success=False)
    assert cb.level == 0
    assert cb.logger.values["curriculum/success_window"] == 0.0
