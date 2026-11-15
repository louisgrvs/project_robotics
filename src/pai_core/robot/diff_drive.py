"""Cinématique d'un robot à entraînement différentiel.

Conventions
-----------
* Repère du robot conforme définit comme suit : x vers l'avant, y vers la gauche,
  z vers le haut ; les angles croissent dans le sens trigonométrique.
* ``(v, w)`` : vitesse linéaire (m/s) et vitesse de lacet (rad/s) du point
  milieu de l'essieu des roues motrices, qui sert d'origine à ``base_link``.
* ``(w_l, w_r)`` : vitesses angulaires des roues gauche et droite
  (rad/s), positives quand la roue fait avancer le robot.

Modèle, sous l'hypothèse de roulement sans glissement (ni longitudinal, ni
latéral), avec r le rayon des roues et L l'entraxe :

    v = r (w_R + w_L) / 2            w_R = (v + w L / 2) / r
    w = r (w_R − w_L) / L            w_L = (v − w L / 2) / r

Toutes les fonctions acceptent des scalaires ou des tableaux NumPy de formes
compatibles (règles de diffusion) : le DWA et MPPI pourront ainsi évaluer
des lots entiers de commandes en un seul appel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "DiffDriveParams",
    "forward",
    "inverse",
    "turning_radius",
    "saturate_wheels",
    "saturate",
]

#On rend la classe immuable : 
@dataclass(frozen=True)
class DiffDriveParams:
    """Géométrie du robot et limite des moteurs. 
    C'est une classe python, créée à partir d'un modèle MuJoco qui contient la taille des roues,
      leurs vitesses maximale et la taille d'écart entre les roues
    
        Attributs :

        - wheel_radius : rayon r des roues motrices (m).
        - wheel_base : entraxe L, distance entre les plans médians des roues (m).
        - max_wheel_speed : vitesse angulaire maximale d'une roue (rad/s), égale à
            la borne ``ctrlrange`` des actionneurs ``velocity`` qu'on peut retrouver dans robot.xml dans robot.
    """
    
    wheel_radius: float = 0.05     # r : rayon des cylindres de roue (size="0.05 0.01")
    wheel_base: float = 0.24       # L : distance entre les roues (y = ±0.12)
    max_wheel_speed: float = 10.0  # ctrlrange="-10 10" dans Mujoco (robot.xml)

    #fonction pour vérifier que on a bien des valeurs cohérente pour les attributs
    def __post_init__(self) -> None:
        for nom in ("wheel_radius", "wheel_base", "max_wheel_speed"):
            valeur = getattr(self, nom)
            if not (np.isfinite(valeur) and valeur > 0):
                raise ValueError(f"{nom} doit être un réel strictement positif (là on a donné : {valeur!r})")

    @property
    def max_linear_speed(self) -> float:
        """Vitesse maximale en ligne droite : r · w_max."""
        return self.wheel_radius * self.max_wheel_speed

    @property
    def max_yaw_rate(self) -> float:
        """Vitesse de lacet maximale, atteinte en rotation sur place : 2 r · w_max / L."""
        return 2.0 * self.wheel_radius * self.max_wheel_speed / self.wheel_base

    @classmethod
    def from_mujoco(cls, model, left_joint: str = "g", right_joint: str = "d") -> DiffDriveParams:
        """Lit r, L et w_max dans un ``mujoco.MjModel`` compilé, et nous créé un objet de la classe avec ces paramètres.

        Le MJCF reste ainsi la seule source de vérité pour la géométrie. La
        fonction vérifie au passage les conventions dont dépend ce module :
        deux charnières d'axe +y dans le repère du châssis (une vitesse
        positive fait alors avancer la roue), roue gauche du côté y > 0,
        roues coaxiales et de même rayon, un actionneur borné par roue.
        """
        import mujoco  # import local : le reste du module n'en a pas besoin

        corps, rayons, limites = [], [], []
        for nom in (left_joint, right_joint):
            j = model.joint(nom).id
            b = int(model.jnt_bodyid[j]) #TODO : C'est qui b ?
            if model.jnt_type[j] != int(mujoco.mjtJoint.mjJNT_HINGE): #TODO : Pas compris
                raise ValueError(f"l'articulation {nom!r} n'est pas une charnière")

            # Axe de la charnière exprimé dans le repère du corps parent (le châssis).
            axe = np.zeros(3)
            mujoco.mju_rotVecQuat(axe, model.jnt_axis[j], model.body_quat[b])
            if not np.allclose(axe, (0.0, 1.0, 0.0), atol=1e-9):
                raise ValueError(
                    f"l'axe de {nom!r} vaut {axe} dans le repère du châssis, attendu (0, 1, 0) : "
                    "avec un autre axe, une vitesse positive ne fait plus avancer la roue"
                )

            # Rayon : l'unique géométrie cylindrique ou sphérique de la roue.
            debut = int(model.body_geomadr[b])
            types_roue = (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE))
            geoms = [g for g in range(debut, debut + int(model.body_geomnum[b]))
                     if model.geom_type[g] in types_roue] 
            #Donc geoms ça va être un tableau qui est vide si y a pas de sphere ou de cylindre en rouge et qui à 1 élément sinon
            if len(geoms) != 1:
                raise ValueError(f"la roue {nom!r} doit porter exactement un cylindre ou une sphère")
            rayons.append(float(model.geom_size[geoms[0], 0]))

            # Limite de vitesse : ctrlrange de l'actionneur qui agit sur cette articulation.
            act = np.flatnonzero((model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
                                 & (model.actuator_trnid[:, 0] == j)) #TODO : Pas compris
            if act.size != 1:
                raise ValueError(f"il faut exactement un actionneur sur {nom!r} (trouvé : {act.size})")
            a = int(act[0])
            if not model.actuator_ctrllimited[a]:
                raise ValueError(f"l'actionneur de {nom!r} n'a pas de ctrlrange")
            bas, haut = model.actuator_ctrlrange[a]
            # La commande porte sur gear · \dot{q} : la limite sur \dot{q} est donc ctrl / gear. #TODO : Pas compris
            limites.append(min(-bas, haut) / abs(model.actuator_gear[a, 0]))
            corps.append(b)

        #TODO : Pas compris
        bg, bd = corps
        if model.body_parentid[bg] != model.body_parentid[bd]:
            raise ValueError("les deux roues doivent être rattachées au même corps")
        ecart = model.body_pos[bg] - model.body_pos[bd]
        if ecart[1] <= 0:
            raise ValueError("la roue gauche doit être du côté y > 0 (REP 103, enfin moi je connais pas mais je crois que c'est ça)")
        if not np.allclose(ecart[[0, 2]], 0.0, atol=1e-9):
            raise ValueError(f"les roues sont décalées (écart gauche − droite : {ecart})")
        if not np.isclose(rayons[0], rayons[1]):
            raise ValueError(f"rayons de roue différents : {rayons}")

        return cls(wheel_radius=rayons[0], wheel_base=float(ecart[1]),
                   max_wheel_speed=float(min(limites)))


def forward(omega_l: ArrayLike, omega_r: ArrayLike,
            params: DiffDriveParams) -> tuple[np.ndarray, np.ndarray]:
    """Cinématique directe : vitesses des roues -> (v, w) du robot."""
    omega_l = np.asarray(omega_l, dtype=float)
    omega_r = np.asarray(omega_r, dtype=float)
    r, L = params.wheel_radius, params.wheel_base
    #notre modèle : 
    v = 0.5 * r * (omega_r + omega_l)
    w = r * (omega_r - omega_l) / L
    return v, w


def inverse(v: ArrayLike, w: ArrayLike,
            params: DiffDriveParams) -> tuple[np.ndarray, np.ndarray]:
    """Cinématique inverse : (v, w) voulus -> vitesses des roues.

    Attention : aucune saturation ici, le résultat peut dépasser ``max_wheel_speed``.
    Passer d'abord par :func:`saturate` si la commande doit être exécutable.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    r, L = params.wheel_radius, params.wheel_base
    omega_l = (v - 0.5 * L * w) / r
    omega_r = (v + 0.5 * L * w) / r
    return omega_l, omega_r


def turning_radius(omega_l: ArrayLike, omega_r: ArrayLike,
                   params: DiffDriveParams) -> np.ndarray:
    #TODO : Pas compris, à quoi correspond R ? 
    """Rayon de courbure signé R = (L/2)·(w_R + w_L)/(w_R − w_L) = v/w.

    R > 0 : centre de rotation à gauche du robot. R = /pm inf en ligne droite,
    R = 0 en rotation sur place, NaN à l'arrêt (rayon indéfini).
    """
    omega_l = np.asarray(omega_l, dtype=float)
    omega_r = np.asarray(omega_r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * params.wheel_base * (omega_r + omega_l) / (omega_r - omega_l)

def _facteur_echelle(amplitude: ArrayLike, limite: float) -> np.ndarray:
    """Plus grand s in ]0, 1] tel que s · amplitude <= limite (amplitude >= 0). Sert pour la fonction d'après."""
    if not limite > 0:
        raise ValueError(f"la limite doit être strictement positive (reçu : {limite!r})")
    amplitude = np.asarray(amplitude, dtype=float)
    s = np.ones_like(amplitude)
    np.divide(limite, amplitude, out=s, where=amplitude > limite)
    return s

def saturate_wheels(omega_l: ArrayLike, omega_r: ArrayLike,
                    params: DiffDriveParams) -> tuple[np.ndarray, np.ndarray]:
    """Ramène les deux roues sous w_max en les multipliant par un même facteur.

    Changer chaque roue séparément changerait la quantitée w_R - w_L donc la
    trajectoire (avec w).
    """
    omega_l = np.asarray(omega_l, dtype=float)
    omega_r = np.asarray(omega_r, dtype=float)
    s = _facteur_echelle(np.maximum(np.abs(omega_l), np.abs(omega_r)), params.max_wheel_speed)
    return s * omega_l, s * omega_r


#TODO : pas compris
def saturate(v: ArrayLike, w: ArrayLike, params: DiffDriveParams,
             v_max: float | None = None,
             w_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sature une commande (v, w) en préservant sa courbure k = w/v.

    La commande est multipliée par un unique facteur s ∈ ]0, 1], le plus grand
    qui respecte à la fois la limite des roues et, si elles sont fournies, les
    bornes |v| ≤ v_max et |w| ≤ w_max. Le robot suit donc le même arc de
    cercle, simplement moins vite. Conséquence voulue : une forte demande de
    rotation ralentit aussi l'avance, au lieu de sortir de la trajectoire.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    omega_l, omega_r = inverse(v, w, params)
    s = _facteur_echelle(np.maximum(np.abs(omega_l), np.abs(omega_r)), params.max_wheel_speed)
    if v_max is not None:
        s = np.minimum(s, _facteur_echelle(np.abs(v), v_max))
    if w_max is not None:
        s = np.minimum(s, _facteur_echelle(np.abs(w), w_max))
    return s * v, s * w
