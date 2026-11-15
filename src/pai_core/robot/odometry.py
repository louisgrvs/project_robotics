"""Odométrie par encodeurs de roues (robot à entraînement différentiel).

Origine au point de départ du robot. Le repère est fixe et ne bouge jamais. Le repère qui bouge avec le robot est base_link.

Ce module ne dépend pas de MuJoCo : il reçoit des angles de roues, comme le
ferait le logiciel d'un vrai robot. La pose (x, y, theta ) est celle du milieu de
l'essieu (repère base_link), exprimée dans le repère odom.

Schéma du point milieu, pour des déplacements de roues Delta s_L et Delta s_R :
    Delta s_L                                        distance parcouru par le milieu de la roue gauche 
    Delta s_R                                        distance parcouru par le milieu de la roue droite
    Delta s = (Delta s_R + Delta s_L) / 2            avancée du milieu de l'essieu, la moyenne de la distance des deux roues pusiqu'il est entre elles
    Delta theta  = (Delta s_R - Delta s_L) / L       de combien le robot a tourné
    x += Delta s·cos(theta  + Delta theta /2)        nouvelle position du centre 
    y += Delta s·sin(theta  + Delta theta /2)
    theta  += Delta theta                            nouveau cap 

Si le robot décrit exactement un arc de cercle pendant le pas, le vrai
déplacement est la corde, de longueur Delta s·sinc(Delta theta /2) avec sinc(u) = sin(u)/u,
dans la même direction theta  + Delta theta /2. L'erreur locale du point milieu est donc
Delta s·(1 - sinc(Delta theta /2)) approx Delta s·Delta theta ²/24 : elle est d'ordre 2 en Delta theta  par pas.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Ramène un angle (ou un tableau d'angles) dans [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi

def midpoint_step(pose, ds_l: float, ds_r: float, wheel_base: float) -> np.ndarray:
    """Un pas d'odométrie par le schéma du point milieu. Pour comprendre : aller voir là-haut je trouve c'est bien expliqué"""
    x, y, th = pose
    ds = 0.5 * (ds_r + ds_l)
    dth = (ds_r - ds_l) / wheel_base
    th_mid = th + 0.5 * dth
    return np.array([x + ds * np.cos(th_mid), y + ds * np.sin(th_mid), th + dth])

#TODO : à quoi elle sert cette fonction ?
def arc_step(pose, ds_l: float, ds_r: float, wheel_base: float) -> np.ndarray:
    """Un pas exact si la trajectoire est un arc de cercle (sert de référence)."""
    x, y, th = pose
    ds = 0.5 * (ds_r + ds_l)
    dth = (ds_r - ds_l) / wheel_base
    chord = ds * np.sinc(dth / (2.0 * np.pi))  # np.sinc(x) = sin(pi x)/(pi x)
    th_mid = th + 0.5 * dth
    return np.array([x + chord * np.cos(th_mid), y + chord * np.sin(th_mid), th + dth])


class WheelOdometry:
    """Intègre les angles absolus des encodeurs pour estimer la pose.

    Les paramètres sont ceux que le robot *croit* avoir. Les fausser (rayon,
    entraxe) reproduit les erreurs systématiques de Borenstein et Feng.
    """

    def __init__(self, wheel_radius_left: float, wheel_radius_right: float, wheel_base: float):
        self.r_l = float(wheel_radius_left) 
        self.r_r = float(wheel_radius_right)
        self.wheel_base = float(wheel_base) 
        self.pose = np.zeros(3) #position du robot dans l'axe odometrie
        self._last = None  # derniers angles d'encodeurs (gauche, droite) TODO : pas compris

    def reset(self, pose=(0.0, 0.0, 0.0), wheel_angles=None) -> None:
        """Pose de départ, et angles actuels des encodeurs s'ils sont connus.

        Sans angles, la première mise à jour sert seulement de référence.
        """
        self.pose = np.array(pose, dtype=float)
        self._last = None if wheel_angles is None else np.array(wheel_angles, dtype=float)

    def update(self, wheel_angles) -> np.ndarray:
        """Intègre les incréments depuis le dernier appel ; renvoie (x, y, theta ). En gros on fait un pas d'odometry selon la dynamique expliquer là haut"""
        phi = np.asarray(wheel_angles, dtype=float)
        if self._last is not None:
            dphi = phi - self._last
            self.pose = midpoint_step(self.pose, self.r_l * dphi[0], self.r_r * dphi[1], self.wheel_base)
            self.pose[2] = wrap_angle(self.pose[2])
        self._last = phi.copy()
        return self.pose.copy()
