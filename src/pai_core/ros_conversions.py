"""Conversions entre les grandeurs de pai_core et les conventions des messages ROS.

Ces fonctions ne dépendent PAS de ROS : elles ne manipulent que des nombres.
sim_bridge et decision_node les appellent, et le test d'identité
(tests/test_identity.py) aussi. Une seule implémentation, utilisée partout :
si le test passe, c'est bien le code des nœuds qui est validé, pas une copie.

    sim_bridge                         decision_node
    ranges_to_msg ──── /scan ────────> ranges_from_msg
    velocity_from_poses ── /odom ────> (lecture directe du message)
    map_to_odom ─────── /tf ─────────> tf2 (goal_in_frame, dans decision_node)
"""
from __future__ import annotations

import math

import numpy as np

from pai_core.geometry.se2 import compose, inverse, wrap_angle


def ranges_to_msg(ranges, max_range: float) -> list[float]:
    """Distances du lidar (m) -> champ `ranges` d'un LaserScan.

    REP 117 : « aucun retour » s'écrit +inf, pas range_max. sensors.py met
    range_max pour « pas d'impact » et pour les faisceaux perdus : les deux
    deviennent inf. Le message stocke des float32 : on arrondit ici, pour que
    le test d'identité voie exactement ce que voit le réseau ROS.
    """
    r = np.asarray(ranges, dtype=np.float64).copy()
    r[r >= max_range] = np.inf
    return r.astype(np.float32).tolist()


def ranges_from_msg(msg_ranges, range_max: float) -> np.ndarray:
    """Champ `ranges` d'un LaserScan -> distances (m), sans inf ni NaN.

    Faisceau sans impact (inf) ou invalide (NaN) : ramené à la portée
    maximale, comme dans NavEnv. preprocess_ranges le ferait aussi, mais on
    ne laisse pas circuler de NaN dans le nœud.
    """
    r = np.asarray(msg_ranges, dtype=float)
    return np.nan_to_num(r, nan=range_max, posinf=range_max, neginf=0.0)


def velocity_from_poses(p0, p1, dt: float) -> tuple[float, float]:
    """(v, ω) dans le repère du robot, déduits de deux poses successives.

    On inverse le schéma du point milieu de l'odométrie : dθ = θ1 − θ0, et
    Δs = projection du déplacement sur le cap moyen θ0 + dθ/2.
    """
    x0, y0, th0 = (float(c) for c in p0)
    x1, y1, th1 = (float(c) for c in p1)
    dth = float(wrap_angle(th1 - th0))
    th_mid = th0 + dth / 2.0
    ds = (x1 - x0) * math.cos(th_mid) + (y1 - y0) * math.sin(th_mid)
    return ds / dt, dth / dt


def map_to_odom(pose_gt, pose_odom) -> np.ndarray:
    """T_map_odom = T_map_base ∘ (T_odom_base)⁻¹ (localisation parfaite).

    Composée avec odom -> base_link, elle redonne exactement la pose vraie :
    c'est ce qui permet à TF de fournir map -> base_link sans que base_link
    ait deux parents.
    """
    return compose(pose_gt, inverse(pose_odom))
