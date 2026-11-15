"""Construction de l'observation : LA fonction unique, partagée par Gym et ROS.

Elle ne prend que des grandeurs physiques (distances, pose, vitesses), jamais
l'objet MuJoCo. Le nœud de perception ROS 2 l'appellera avec les messages
/scan et /odom ; l'environnement Gym l'appelle avec les valeurs lues dans
MuJoCo. C'est ce qui rend possible le test d'identité du jour 10.

Disposition du vecteur (43 composantes, toutes de l'ordre de 1) :

    [0:36]   distances lidar / portée max              dans [0, 1]
    [36]     distance au but / distance de référence   dans [0, 1]
    [37:39]  cos, sin du cap du but (repère du robot)  dans [-1, 1]
    [39:41]  v / v_max, ω / ω_max                      écrêtés à [-2, 2]
    [41:43]  action précédente                         dans [-1, 1]

Le cap est codé par (cos, sin) et non par l'angle : l'angle saute de +π à −π
quand le but passe derrière le robot, alors que le cosinus et le sinus varient
continûment. Un réseau apprend mal une entrée discontinue.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
    goal_dist_max: float = 8.5    # ≈ diagonale de l'arène (m)
    v_max: float = 0.5            # mêmes échelles que l'action
    w_max: float = 2.0


def wrap_angle(a: float) -> float:
    """Ramène un angle dans [−π, π[."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def goal_polar(pose: tuple[float, float, float], goal: tuple[float, float]) -> tuple[float, float]:
    """Distance et cap du but dans le repère du robot (cap > 0 : but à gauche)."""
    x, y, yaw = pose
    dx, dy = goal[0] - x, goal[1] - y
    return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - yaw)


def observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.zeros(OBS_DIM, dtype=np.float32)
    high = np.ones(OBS_DIM, dtype=np.float32)
    low[37:39], low[39:41], low[41:43] = -1.0, -VEL_CLIP, -1.0
    high[39:41] = VEL_CLIP
    return low, high


def build_observation(
    ranges: np.ndarray,                       # N_BEAMS distances (m)
    pose: tuple[float, float, float],         # x, y, lacet (repère map)
    velocity: tuple[float, float],            # v, ω (repère du robot)
    goal: tuple[float, float],                # x, y du but (repère map)
    prev_action: np.ndarray,                  # action normalisée du pas précédent
    cfg: ObsConfig = ObsConfig(),
) -> np.ndarray:
    ranges = np.asarray(ranges, dtype=np.float64)
    assert ranges.shape == (N_BEAMS,), ranges.shape
    dist, bearing = goal_polar(pose, goal)

    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[SLICES["lidar"]] = np.clip(ranges / cfg.range_max, 0.0, 1.0)
    obs[SLICES["goal"]] = (min(dist / cfg.goal_dist_max, 1.0), math.cos(bearing), math.sin(bearing))
    obs[SLICES["velocity"]] = np.clip((velocity[0] / cfg.v_max, velocity[1] / cfg.w_max),
                                      -VEL_CLIP, VEL_CLIP)
    obs[SLICES["prev_action"]] = np.clip(prev_action, -1.0, 1.0)
    return obs
