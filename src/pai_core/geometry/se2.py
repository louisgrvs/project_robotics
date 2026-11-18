"""Poses planes : le groupe SE(2) des rotations et translations du plan.

Convention
    Une pose est un np.ndarray de forme (3,) : (x, y, θ).
    T_a_b désigne la pose du repère b exprimée dans le repère a. Elle convertit
    les coordonnées d'un point : p_a = R(θ)·p_b + t.
        T_map_base   vérité terrain
        T_odom_base  odométrie
        T_map_odom   compose(T_map_base, inverse(T_odom_base))

Les fonctions acceptent aussi des lots de poses de forme (..., 3) : toutes
les opérations se font composante par composante.

Attention : (x, y, θ) n'est qu'un système de coordonnées du groupe, pas un
vecteur. On n'additionne pas deux poses : on les compose.

Aucune dépendance à ROS : ce module sert dans sim_bridge, dans la grille
d'occupation, dans le DWA et plus tard dans le World Model.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Ramène un angle (ou un tableau d'angles) dans [−π, π[.

    Version de référence : odometry.py et features.py doivent l'importer d'ici.
    """
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _as_pose(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.shape[-1] != 3:
        raise ValueError(f"une pose a 3 composantes (x, y, θ), reçu la forme {p.shape}")
    return p


def compose(a, b) -> np.ndarray:
    """a ∘ b : si a = T_x_y et b = T_y_z, renvoie T_x_z.

    Translation : t_a + R(θ_a)·t_b.  Angle : θ_a + θ_b.
    """
    a, b = _as_pose(a), _as_pose(b)
    xa, ya, tha = a[..., 0], a[..., 1], a[..., 2]
    xb, yb, thb = b[..., 0], b[..., 1], b[..., 2]
    c, s = np.cos(tha), np.sin(tha)
    return np.stack([xa + c * xb - s * yb,
                     ya + s * xb + c * yb,
                     wrap_angle(tha + thb)], axis=-1)


def inverse(a) -> np.ndarray:
    """Si a = T_x_y, renvoie T_y_x.

    Translation : −R(θ)ᵀ·t.  Angle : −θ.
    """
    a = _as_pose(a)
    x, y, th = a[..., 0], a[..., 1], a[..., 2]
    c, s = np.cos(th), np.sin(th)
    return np.stack([-c * x - s * y,
                     s * x - c * y,
                     wrap_angle(-th)], axis=-1)


def between(a, b) -> np.ndarray:
    """inverse(a) ∘ b : pose de b vue depuis a (déplacement relatif).

    Si a = T_w_a et b = T_w_b, renvoie T_a_b. Exemple : between(pose_t, pose_t+1)
    donne le déplacement du robot exprimé dans son propre repère.
    """
    return compose(inverse(a), b)


def transform_points(pose, pts) -> np.ndarray:
    """Points de forme (N, 2) exprimés dans le repère enfant → repère parent.

    Si pose = T_a_b, les points sont donnés dans b et renvoyés dans a :
    p_a = R(θ)·p_b + t. Une seule pose, pas un lot.
    """
    x, y, th = _as_pose(pose)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points attendus de forme (N, 2), reçu {pts.shape}")
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


def to_matrix(pose) -> np.ndarray:
    """Matrice homogène 3×3 [[R, t], [0, 0, 1]] d'une pose."""
    x, y, th = _as_pose(pose)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x],
                     [s, c, y],
                     [0.0, 0.0, 1.0]])


def from_matrix(M) -> np.ndarray:
    """Pose (x, y, θ) d'une matrice homogène 3×3 (inverse de to_matrix)."""
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"matrice 3×3 attendue, reçu {M.shape}")
    return np.array([M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0])])