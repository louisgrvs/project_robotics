"""Capteurs simulés : pose vraie, encodeurs de roues et lidar 2D.

qpos[0:3] : position du chassis dans le monde
qpos[3: 7] : orientation du chassis sous la forme d'un quaternion
qpos[7] : angle roue gauche
qpos[8] angle roue droite

Règle suivie dans tout le module : on lit l'état dans data.qpos (qui décrit la position de tout le modèle à un moment donné, pour nous il a 9 nombres), jamais dans
data.xpos, data.xquat ou data.site_xpos. mj_step calcule d'abord la cinématique
puis intègre : juste après mj_step, xpos décrit encore l'état d'avant le pas,
alors que qpos est à jour. Lire qpos garantit que lidar, encodeurs et pose
vraie décrivent le même instant.

Exception : les obstacles. Le lancer de rayons utilise geom_xpos, calculé par
mj_kinematics. Les obstacles étant statiques, il suffit d'appeler mj_forward
une fois après chaque reset (et après avoir déplacé des obstacles).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


# ----------------------------------------------------------------------------
# Géométrie du robot, lue dans le modèle compilé
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotGeometry:
    """Ce qu'il faut savoir du robot pour lire ses capteurs.

    axle_offset : milieu de l'essieu dans le repère du châssis. En gros dans MujoCo y a un repère naturelle qui est le milieu du châssis 
    et la convention veut que en vrai le repère c'est le milieu de l'essieu. C'est le point
    de référence de la cinématique (base_link, le repère du robot) : en rotation sur place, c'est
    lui qui reste immobile, pas l'origine du châssis.
    """

    wheel_radius: float          # rayon des roues
    wheel_base: float            # écart entre les roues L
    axle_offset: np.ndarray      # (3,), centre du robot 
    chassis_id: int              #
    free_qposadr: int            # position (3) puis quaternion (4) dans qpos
    wheel_qposadr: tuple         # (gauche, droite)

    @classmethod
    def from_model(cls, model: mujoco.MjModel, chassis: str = "chassis",
                   left_wheel: str = "roue_g", right_wheel: str = "roue_d",
                   left_joint: str = "g", right_joint: str = "d") -> "RobotGeometry":
        """
        Fonction pour initialiser un objet de la classe RobotGeometry à partir d'un modèle Mujoco déclaré avant 
        """
        chassis_id = model.body(chassis).id
        jnt = model.body_jntadr[chassis_id]
        if jnt < 0 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"le corps « {chassis} » n'a pas de liaison libre")

        left_id, right_id = model.body(left_wheel).id, model.body(right_wheel).id
        geomg = model.body_geomadr[left_id]
        if model.geom_type[geomg] != mujoco.mjtGeom.mjGEOM_CYLINDER and model.geom_type[geomg]!= mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError("la roue gauche devrait être un cylindre ou une sphère")
        geomd = model.body_geomadr[right_id]
        if model.geom_type[geomd] != mujoco.mjtGeom.mjGEOM_CYLINDER and model.geom_type[geomd]!= mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError("la roue droite devrait être un cylindre ou une sphère")
    
        p_l, p_r = model.body_pos[left_id], model.body_pos[right_id]
        return cls(
            wheel_radius=float(model.geom_size[geomg, 0]),
            wheel_base=float(abs(p_l[1] - p_r[1])),
            axle_offset=0.5 * (p_l + p_r),
            chassis_id=chassis_id,
            free_qposadr=int(model.jnt_qposadr[jnt]),
            wheel_qposadr=(int(model.jnt_qposadr[model.joint(left_joint).id]),
                           int(model.jnt_qposadr[model.joint(right_joint).id])),
        )


def yaw_from_quat(q) -> float:
    """Lacet d'un quaternion (w, x, y, z), convention ZYX, dans [-pi, pi].
      C'est genre l'expression d'un angle de rotation à partir d'un quaternion"""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _chassis_frame(data: mujoco.MjData, geom: RobotGeometry):
    """Position et matrice de rotation du châssis, lues dans qpos."""
    a = geom.free_qposadr
    pos = data.qpos[a:a + 3]
    quat = data.qpos[a + 3:a + 7]
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, quat)
    return pos, quat, rot.reshape(3, 3)


# ----------------------------------------------------------------------------
# Pose vraie
# ----------------------------------------------------------------------------

def true_pose(data: mujoco.MjData, geom: RobotGeometry) -> np.ndarray:
    """Pose vraie (x, y, lacet) du milieu de l'essieu, dans le repère monde. C'est genre (x,y, angle d'inclinaison)"""
    pos, quat, rot = _chassis_frame(data, geom)
    p = pos + rot @ geom.axle_offset
    return np.array([p[0], p[1], yaw_from_quat(quat)])


# ----------------------------------------------------------------------------
# Encodeurs de roues
# ----------------------------------------------------------------------------

class WheelEncoders:
    """Angles des roues (rad), éventuellement quantifiés comme un vrai codeur. 
    Cela nous indique combien la roue à tourner depuis le départ. Super utile pour connaître la position actuelle du robot !

    ticks_per_rev=None donne des encodeurs parfaits.
    Convention : angle positif = la roue fait avancer le robot.
    """

    def __init__(self, geom: RobotGeometry, ticks_per_rev: int | None = None):
        self.adr = geom.wheel_qposadr
        self.resolution = None if ticks_per_rev is None else 2.0 * np.pi / ticks_per_rev

    def read(self, data: mujoco.MjData) -> np.ndarray:
        phi = np.array([data.qpos[self.adr[0]], data.qpos[self.adr[1]]])
        if self.resolution is not None:
            phi = np.floor(phi / self.resolution) * self.resolution
        return phi


# ----------------------------------------------------------------------------
# Lidar 2D
# ----------------------------------------------------------------------------

@dataclass
class LidarConfig:
    n_beams: int = 36
    max_range: float = 4.0         # m
    noise_std: float = 0.0         # écart type du bruit gaussien (m)
    dropout_prob: float = 0.0      # probabilité qu'un faisceau ne revienne pas
    # Position du lidar dans le repère du châssis. À 7 cm au-dessus du centre,
    # soit 12 cm du sol : au-dessus des roues (10 cm). Les obstacles devront
    # donc dépasser 12 cm de haut pour être vus.
    mount: tuple = (0.0, 0.0, 0.07)


class Lidar:
    """Lidar 2D : n_beams faisceaux horizontaux répartis sur 360°.

    Faisceau k : angle 2pik/n dans le repère du robot. Le faisceau 0 pointe
    vers l'avant (+x), les angles croissent vers la gauche (aksip convention : REP 103).

    Les rayons sont tracés dans le plan horizontal, avec le seul lacet du
    robot : c'est un lidar 2D.
    En vrai, un lidar s'incline avec le châssis et toucherait le sol au loin.
    """

    #TODO : comprendre chaque élément qui c'est et la classe en général
    def __init__(self, geom: RobotGeometry, config: LidarConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.geom = geom
        self.cfg = config or LidarConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.angles = 2.0 * np.pi * np.arange(self.cfg.n_beams) / self.cfg.n_beams
        self._mount = np.asarray(self.cfg.mount, dtype=float)
        n = self.cfg.n_beams
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n)

    def _cast(self, model, data, origin, vec) -> np.ndarray:
        """Distances brutes de mj_multiRay (-1 si aucun impact)."""
        n = self.cfg.n_beams
        args = (model, data, origin, vec.ravel(), None, True,
                self.geom.chassis_id, self._geomid, self._dist)
        try:  # MuJoCo récent : argument « normal » ajouté avant nray
            mujoco.mj_multiRay(*args, None, n, self.cfg.max_range)
        except TypeError:  # versions plus anciennes
            mujoco.mj_multiRay(*args, n, self.cfg.max_range)
        return self._dist.copy()

    def scan_raw(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Distances sans bruit ; aucun impact -> portée maximale."""
        pos, quat, rot = _chassis_frame(data, self.geom)
        origin = pos + rot @ self._mount
        a = yaw_from_quat(quat) + self.angles
        vec = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)

        d = self._cast(model, data, origin, vec)
        # Le châssis est exclu des rayons, mais pas les roues : d'où la hauteur
        # du lidar, au-dessus d'elles (voir LidarConfig.mount).
        d[(d < 0.0) | (d > self.cfg.max_range)] = self.cfg.max_range
        return d

    def scan(self, model: mujoco.MjModel, data: mujoco.MjData,
             rng: np.random.Generator | None = None) -> np.ndarray:
        """Distances avec bruit gaussien et pertes de faisceaux."""
        rng = rng if rng is not None else self.rng
        d = self.scan_raw(model, data)
        if self.cfg.noise_std > 0.0:
            d = d + rng.normal(0.0, self.cfg.noise_std, size=d.shape)
        if self.cfg.dropout_prob > 0.0:
            lost = rng.random(d.shape) < self.cfg.dropout_prob
            d[lost] = self.cfg.max_range  # pas de retour, comme hors de portée
        return np.clip(d, 0.0, self.cfg.max_range)