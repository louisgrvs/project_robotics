"""Aller au but sans obstacle : loi polaire de Lyapunov.

Le robot est décrit relativement au but par trois grandeurs :
    rho   : distance au but
    alpha : angle entre le cap du robot et la direction du but
    beta  : angle qui fixe l'orientation finale (inutile si h = 0)

Avec V = (rho² + alpha² + h·beta²) / 2, la loi choisie donne
    dV/dt = -v·rho·cos(alpha) - k·alpha² <= 0,
d'où la convergence vers le but (Aicardi et al., 1995 ; Siegwart et al., ch. 3).
"""
from dataclasses import dataclass, field

import numpy as np

from pai_core.robot.diff_drive import DiffDriveParams
from pai_core.robot.odometry import wrap_angle


@dataclass(frozen=True)
class GoToGoalParams:
    gamma: float = 1.0               # gain sur la vitesse d'avance (1/s), doit être > 0
    k: float = 3.0                   # gain sur la rotation (1/s), doit être > 0
    h: float = 0.0                   # poids de l'orientation finale ; 0 = cap final libre
    forward_only: bool = True        # interdit la marche arrière
    goal_tolerance: float = 0.1      # rayon de succès (m), même valeur que nav_env
    diff_drive: DiffDriveParams = field(default_factory=DiffDriveParams)  # pour la saturation


def polar_coordinates(pose, goal) -> tuple[float, float, float]:
    """Coordonnées polaires (rho, alpha, beta) du robot par rapport au but.

    pose = (x, y, theta) et goal = (xb, yb), dans le repère monde.
    Sert aux tests, où l'on simule soi-même le modèle unicycle. Dans
    l'environnement, rho et alpha se lisent directement dans l'observation.
    """
    x, y, theta = pose
    xb, yb = goal

    rho = float(np.hypot(xb - x, yb - y))
    # direction du but vue du robot, moins le cap du robot, ramenée dans ]-pi, pi]
    alpha = wrap_angle(np.arctan2(yb - y, xb - x) - theta)
    # convention de Siegwart : beta = -theta - alpha
    beta = wrap_angle(-theta - alpha)
    return rho, alpha, beta


def polar_law(rho, alpha, beta, params) -> tuple[float, float]:
    """Commande (v, w) de la loi de Lyapunov, sans saturation.

    v = gamma·rho·c
    w = k·alpha + (v/rho)·(sin(alpha)/alpha)·(alpha - h·beta)

    où c = cos(alpha), ou max(cos(alpha), 0) en marche avant seule.
    Deux simplifications évitent toute division par zéro :
      - v/rho = gamma·c, valable même si rho = 0 ;
      - sin(alpha)/alpha = np.sinc(alpha/pi), qui vaut 1 en alpha = 0.
    """
    # but derrière le robot (cos < 0) : v = 0, le robot tourne sur place
    c = max(0.0, np.cos(alpha)) if params.forward_only else np.cos(alpha)

    v = params.gamma * rho * c
    # le premier terme ramène alpha vers 0 ; le second annule le terme croisé de dV/dt
    w = params.k * alpha + params.gamma * c * np.sinc(alpha / np.pi) * (alpha - params.h * beta)
    return v, w


def lyapunov(rho, alpha, beta, h) -> float:
    """Fonction de Lyapunov V, qui doit décroître le long de toute trajectoire (tests)."""
    return 0.5 * rho**2 + 0.5 * (alpha**2 + h * beta**2)