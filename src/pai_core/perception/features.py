"""Construction de l'observation : la fonction unique, partagée par Gym et ROS.

Elle ne prend que des grandeurs physiques (distances, pose, vitesses), jamais
l'objet MuJoCo. Le noeud de perception ROS 2 l'appelle avec les messages
/scan et /odom ; l'environnement Gym l'appelle avec les valeurs lues dans
MuJoCo. C'est ce qui rend possible le test d'identité du jour 10.

Disposition du vecteur (43 composantes, toutes de l'ordre de 1) :

    [0:36]   distances lidar / portée max              dans [0, 1]
    [36]     distance au but / distance de référence   dans [0, 1]
    [37:39]  cos, sin du cap du but (repère du robot)  dans [-1, 1]
    [39:41]  v / v_max, omega  / omega _max                      écrêtés à [-2, 2]
    [41:43]  action précédente                         dans [-1, 1]

Le cap est codé par (cos, sin) et non par l'angle : l'angle saute de +π à −π
quand le but passe derrière le robot, alors que le cosinus et le sinus varient
continûment. Un réseau apprend mal une entrée discontinue.

Distances lidar en entrée : en mètres. « Pas de retour » arrive sous deux
formes équivalentes : range_max (NavEnv, sensors.py) ou +inf (messages ROS,
REP 117, publiés ainsi par sim_bridge). preprocess_ranges les ramène toutes
deux à 1 : l'observation est la même des deux côtés.

Convention des faisceaux (REP 103) : faisceau 0 vers l'avant (+x), angles
croissants vers la gauche. Voir lidar_angles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pai_core.geometry.se2 import wrap_angle

N_BEAMS = 36
OBS_DIM = N_BEAMS + 3 + 2 + 2
SLICES = {
    "lidar": slice(0, N_BEAMS),
    "goal": slice(N_BEAMS, N_BEAMS + 3),
    "velocity": slice(N_BEAMS + 3, N_BEAMS + 5),
    "prev_action": slice(N_BEAMS + 5, N_BEAMS + 7),
}
VEL_CLIP = 2.0


@dataclass(frozen=True)
class ObsConfig:
    range_max: float = 4.0        # portée du lidar (m)
    goal_dist_max: float = 8.5    # approx diagonale de l'arène (m)
    v_max: float = 0.5            # mêmes échelles que l'action
    w_max: float = 2.0


def lidar_angles(n_beams: int = N_BEAMS) -> np.ndarray:
    """Angle de chaque faisceau dans le repère du lidar, (n_beams,) : 2pik / n.

    Faisceau 0 vers l'avant, angles croissants vers la gauche.
    """
    return 2.0 * np.pi * np.arange(n_beams) / n_beams


def goal_polar(pose: tuple[float, float, float], goal: tuple[float, float]) -> tuple[float, float]:
    """Distance et cap du but dans le repère du robot (cap > 0 : but à gauche)."""
    x, y, yaw = pose
    dx, dy = goal[0] - x, goal[1] - y
    return math.hypot(dx, dy), float(wrap_angle(math.atan2(dy, dx) - yaw))


def observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.zeros(OBS_DIM, dtype=np.float32)
    high = np.ones(OBS_DIM, dtype=np.float32)
    low[37:39], low[39:41], low[41:43] = -1.0, -VEL_CLIP, -1.0
    high[39:41] = VEL_CLIP
    return low, high


def preprocess_ranges(ranges, cfg: ObsConfig = ObsConfig()) -> np.ndarray:
    """N_BEAMS distances (m, inf = pas de retour) -> N_BEAMS valeurs dans [0, 1], float32.

    C'est le seul endroit où le lidar est normalisé : build_observation et
    /scan_proc l'appellent tous les deux.
    inf et NaN sont remplacés par range_max AVANT la division : np.clip laisse
    passer NaN, et un seul NaN dans l'observation suffit à casser PPO.
    """
    r = np.asarray(ranges, dtype=np.float64)
    if r.shape != (N_BEAMS,):
        raise ValueError(f"{N_BEAMS} distances attendues, reçu la forme {r.shape}")
    r = np.where(np.isfinite(r), r, cfg.range_max)
    return np.clip(r / cfg.range_max, 0.0, 1.0).astype(np.float32)


def build_observation(
    ranges: np.ndarray,                       # N_BEAMS distances (m)
    pose: tuple[float, float, float],         # x, y, lacet (repère map)
    velocity: tuple[float, float],            # v, omega  (repère du robot)
    goal: tuple[float, float],                # x, y du but (repère map)
    prev_action: np.ndarray,                  # action normalisée du pas précédent
    cfg: ObsConfig = ObsConfig(),
) -> np.ndarray:
    dist, bearing = goal_polar(pose, goal)

    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[SLICES["lidar"]] = preprocess_ranges(ranges, cfg)
    obs[SLICES["goal"]] = (min(dist / cfg.goal_dist_max, 1.0), math.cos(bearing), math.sin(bearing))
    obs[SLICES["velocity"]] = np.clip((velocity[0] / cfg.v_max, velocity[1] / cfg.w_max),
                                      -VEL_CLIP, VEL_CLIP)
    obs[SLICES["prev_action"]] = np.clip(prev_action, -1.0, 1.0)
    return obs


def split_obs(obs, cfg: ObsConfig = ObsConfig()) -> tuple:
    """Inverse de build_observation, en unités physiques.

    Renvoie (distances en m (N_BEAMS,), distance au but en m, cap du but en rad,
    v en m/s, omega  en rad/s). L'action précédente n'est pas renvoyée.

    L'inversion perd de l'information : la distance au but est saturée à
    goal_dist_max, les vitesses à pm VEL_CLIP fois leur échelle, les distances
    lidar à range_max (un faisceau à range_max signifie « pas de retour »).
    """
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape != (OBS_DIM,):
        raise ValueError(f"observation de {OBS_DIM} composantes attendue, reçu la forme {obs.shape}")
    distances = obs[SLICES["lidar"]] * cfg.range_max
    d_norm, c, s = obs[SLICES["goal"]]
    v_norm, w_norm = obs[SLICES["velocity"]]
    return (distances, float(d_norm * cfg.goal_dist_max), float(math.atan2(s, c)),
            float(v_norm * cfg.v_max), float(w_norm * cfg.w_max))