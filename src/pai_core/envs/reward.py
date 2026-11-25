"""Récompense de NavEnv : façonnage par potentiel (Ng et al., 1999) + termes de tâche.

    r = F + r_tâche

    F       = w_p · (gamma ·phi(s') - phi(s)),  phi(s) = -d(s),  phi(s') = 0 si s' terminal
              → ne change PAS les politiques optimales (accélère seulement l'apprentissage)
    r_tâche = -w_t - w_prox·pénalité(c) + r_succès·mathbb{1}[succès] + r_collision·mathbb{1}[collision]
              → DÉFINIT la tâche : aller vite, garder de la marge, ne pas heurter

Fonctions pures : aucun accès à MuJoCo, testables en isolation (tests/test_reward.py).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99            # doit être le gamma  de PPO, sinon le théorème de Ng ne s'applique plus (on peut changer la politique optimale)
    w_progress: float = 10.0       # par mètre
    w_time: float = 0.01           # par décision
    w_proximity: float = 0.1       # pénalité maximale par décision, au contact
    safe_clearance: float = 0.3    # m : en dessous, la pénalité de proximité s'active
    r_success: float = 10.0
    r_collision: float = -10.0
    zero_terminal_potential: bool = True   # phi(s') = 0 si terminal (version stricte de Ng)


def potential(dist: float, cfg: RewardConfig) -> float:
    """phi(s) = -w_p · d(s). Plus on est loin du but, plus le potentiel est bas.
    """
    return -cfg.w_progress * dist


def shaping(d_prev: float, d: float, terminated: bool, cfg: RewardConfig) -> float:
    
    phi_s = potential(d_prev, cfg)                    # Φ(s)  : AVANT le pas
    phi_sp = potential(d, cfg)                        # Φ(s') : APRÈS le pas
    if terminated and cfg.zero_terminal_potential:
        phi_sp = 0.0                                  # c'est Φ(s') qu'on annule
    return cfg.gamma * phi_sp - phi_s


def proximity_penalty(clearance: float, cfg: RewardConfig) -> float:
    """Pénalité dans [0, 1] : 0 si clearance >= safe_clearance, 1 au contact.

    clearance = portée minimale du lidar - rayon du robot (m). Elle peut être
    légèrement négative (le cercle circonscrit chevauche l'obstacle).

    Indice : max(0, 1 - clearance / safe_clearance), puis borner à 1.
    """
    return np.clip(max(0, 1 - clearance / cfg.safe_clearance)**2,0,1)


def compute_reward(d_prev: float, d: float, clearance: float,
                   collision: bool, success: bool,
                   cfg: RewardConfig) -> tuple[float, dict[str, float]]:
    """Récompense totale et détail des termes.

    Renvoie (r, termes), avec termes = {"progress", "time", "proximity",
    "success", "collision"}, et r = somme des termes.
    Le détail sert aux tests et à la journalisation (EpisodeStatsCallback).

      1. terminated = collision or success
      2. calculer chaque terme AVEC SON SIGNE (la pénalité de temps est négative, etc.)
      3. vérifier que r == sum(termes.values())
    """
    terminated = collision or success
    progress = shaping(d_prev,d,terminated,cfg)
    time = -cfg.w_time
    proximity = -cfg.w_proximity * proximity_penalty(clearance,cfg)
    success_r = cfg.r_success if success else 0.0
    terms = {
        "progress": shaping(d_prev, d, terminated, cfg),
        "time": -cfg.w_time,
        "proximity": -cfg.w_proximity * proximity_penalty(clearance, cfg),
        "success": cfg.r_success if success else 0.0,
        "collision": cfg.r_collision if collision else 0.0,
    }
    return float(sum(terms.values())), terms