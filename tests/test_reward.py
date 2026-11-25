"""Tests de la récompense (case j11-2).

Deux parties :
  1. fonctions pures de reward.py : signes, théorème de Ng, termes terminaux
     (aucune simulation, moins d'une seconde) ;
  2. branchement dans NavEnv : collision et troncature dans la vraie physique.
"""
import math

import numpy as np
import pytest

from pai_core.envs.nav_env import NavEnv, NavEnvConfig
from pai_core.envs.reward import (RewardConfig, compute_reward, potential,
                                  proximity_penalty, shaping)
from pai_core.sim.world import Scene

CFG = RewardConfig()
LOIN = 1.0          # marge confortable (m) : pénalité de proximité nulle


# ============================================================================
# 1. Fonctions pures
# ============================================================================

@pytest.mark.parametrize("d_prev,d,signe", [(3.0, 2.5, +1), (2.5, 3.0, -1)])
def test_progress_sign(d_prev, d, signe):
    """Se rapprocher du but rapporte, s'éloigner coûte."""
    assert np.sign(shaping(d_prev, d, terminated=False, cfg=CFG)) == signe


def test_standing_still():
    """Robot immobile à 3 m du but : F = gamma phi (s) - phi (s) = (1 - gamma )·w_p·d > 0.

    Avec gamma  < 1, le potentiel futur est actualisé, donc
    « rester » à un potentiel négatif rapporte un peu. Ça ne change pas la
    politique optimale : sur tout l'épisode, les F se télescopent en
    -phi (s₀), une constante qui ne dépend pas des actions (test suivant).
    En pratique, ce petit gain (0,3 par décision à 3 m) est dominé par le
    terme de progrès dès que le robot avance (≈ 0,5 par décision à 0,5 m/s).
    """
    d = 3.0
    f = shaping(d, d, terminated=False, cfg=CFG)
    assert f == pytest.approx((1 - CFG.gamma) * CFG.w_progress * d)
    assert f > 0


@pytest.mark.parametrize("zero_terminal", [True, False])
def test_telescoping(zero_terminal):
    """Théorème de Ng : Σ gamma ᵗ F_t = gamma ᵀ phi (s_T) - phi (s₀).

    Version stricte (phi  = 0 à la terminaison) : la somme vaut -phi (s₀), quelle que
    soit la trajectoire. C'est exactement ce qui rend le façonnage inoffensif.
    """
    cfg = RewardConfig(zero_terminal_potential=zero_terminal)
    rng = np.random.default_rng(0)
    d = rng.uniform(0.0, 6.0, size=51)           # d_0, …, d_T avec T = 50
    T = len(d) - 1

    total = sum(cfg.gamma**t * shaping(d[t], d[t + 1], terminated=(t == T - 1), cfg=cfg)
                for t in range(T))

    phi_T = 0.0 if zero_terminal else potential(d[T], cfg)
    assert total == pytest.approx(cfg.gamma**T * phi_T - potential(d[0], cfg))


def test_proximity_penalty():
    """Nulle au-delà du seuil, 1 au contact, décroissante entre les deux, bornée."""
    s = CFG.safe_clearance
    assert proximity_penalty(s, CFG) == 0.0
    assert proximity_penalty(2 * s, CFG) == 0.0
    assert proximity_penalty(0.0, CFG) == pytest.approx(1.0)
    assert proximity_penalty(-0.1, CFG) == pytest.approx(1.0)     # chevauchement : borné

    marges = np.linspace(0.0, s, 20)
    valeurs = [proximity_penalty(c, CFG) for c in marges]
    assert all(a >= b for a, b in zip(valeurs, valeurs[1:]))       # décroissante
    assert all(0.0 <= v <= 1.0 for v in valeurs)


@pytest.mark.parametrize("success,collision", [(True, False), (False, True), (False, False)])
def test_terminal_terms(success, collision):
    """Le bonus OU le malus terminal, jamais les deux, et rien sur un pas ordinaire."""
    _, t = compute_reward(1.0, 0.9, LOIN, collision=collision, success=success, cfg=CFG)
    assert t["success"] == (CFG.r_success if success else 0.0)
    assert t["collision"] == (CFG.r_collision if collision else 0.0)


def test_signs_of_task_terms():
    """Le temps coûte toujours ; la proximité coûte (ou est nulle), jamais ne rapporte."""
    _, loin = compute_reward(2.0, 2.0, LOIN, False, False, CFG)
    _, pres = compute_reward(2.0, 2.0, 0.05, False, False, CFG)
    assert loin["time"] == -CFG.w_time
    assert loin["proximity"] == 0.0
    assert pres["proximity"] < 0.0


def test_terms_sum_to_total():
    r, t = compute_reward(2.0, 1.8, 0.1, collision=False, success=False, cfg=CFG)
    assert set(t) == {"progress", "time", "proximity", "success", "collision"}
    assert r == pytest.approx(sum(t.values()))


def test_collision_never_pays():
    """Même départ, même durée : finir par un succès rapporte plus que finir
    par une collision, en retour actualisé (c'est ce que PPO optimise)."""
    def retour(distances, success, collision):
        T = len(distances) - 1
        g = 0.0
        for t in range(T):
            fin = t == T - 1
            r, _ = compute_reward(distances[t], distances[t + 1], LOIN,
                                  collision=collision and fin, success=success and fin, cfg=CFG)
            g += CFG.gamma**t * r
        return g

    vers_le_but = np.linspace(4.0, 0.2, 20)      # arrive au but
    vers_le_mur = np.linspace(4.0, 5.0, 20)      # s'éloigne et heurte un mur
    assert retour(vers_le_but, True, False) > retour(vers_le_mur, False, True)


# ============================================================================
# 2. Dans l'environnement
# ============================================================================

def empty_scene(start, goal):
    d = math.hypot(goal[0] - start[0], goal[1] - start[1])
    return Scene(start=start, goal=goal, obstacles=(), path_length=d)


@pytest.fixture
def env():
    e = NavEnv()
    yield e
    e.close()


def test_collision_terminates_with_penalty(env):
    """Foncer dans le mur x = +3 : terminated, pas truncated, malus présent.

    On ne teste PAS r < 0 : avec phi  = 0 à la terminaison, F = -phi (s) = +w_p·d,
    soit environ +40 à 4 m du but, ce qui dépasse le malus de -10. Voir
    test_collision_never_pays pour la propriété qui compte vraiment.
    """
    env.reset(seed=0, options={"scene": empty_scene((2.0, 0.0, 0.0), (-2.0, 0.0))})
    for _ in range(50):
        _, r, terminated, truncated, info = env.step([1.0, 0.0])
        if terminated or truncated:
            break
    t = info["reward_terms"]
    assert terminated and not truncated
    assert info["collision"] and not info["is_success"]
    assert t["collision"] == CFG.r_collision and t["success"] == 0.0
    assert t["proximity"] < 0.0                  # au contact, la marge est nulle
    assert r == pytest.approx(sum(t.values()))


def test_truncation_is_not_termination(env):
    """Immobile pendant max_steps pas : truncated, pas terminated, aucun terme
    terminal, et phi (s') N'EST PAS annulé dans le façonnage."""
    env.cfg.max_steps = 5
    _, info = env.reset(seed=0, options={"scene": empty_scene((0.0, 0.0, 0.0), (2.0, 0.0))})
    for _ in range(10):
        d_prev = info["goal_distance"]
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    t = info["reward_terms"]
    assert truncated and not terminated
    assert t["success"] == 0.0 and t["collision"] == 0.0
    # Avec phi (s') annulé, F vaudrait -phi (s) = +20 ; ici F ≈ (1 - gamma )·w_p·2 = 0,2
    assert t["progress"] == pytest.approx(shaping(d_prev, info["goal_distance"], False, CFG))
    assert t["progress"] < 1.0