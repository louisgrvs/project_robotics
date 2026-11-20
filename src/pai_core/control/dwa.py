"""Dynamic Window Approach (Fox, Burgard et Thrun, 1997).

À chaque décision (10 Hz) :
  1. les distances lidar deviennent des points (x, y) dans base_link ;
  2. on énumère des candidats (v, w) dans la fenêtre dynamique ;
  3. on simule l'arc de cercle de chaque candidat sur l'horizon ;
  4. on mesure le dégagement de chaque arc par rapport aux points lidar ;
  5. on élimine les candidats qui ne peuvent plus freiner avant l'obstacle ;
  6. on garde le candidat de coût minimal.

Repère : base_link, centré sur le milieu de l'essieu, x vers l'avant,
y vers la gauche (REP 103). Le DWA ne lit que l'observation, jamais l'état
MuJoCo : il voit exactement ce que voit PPO.

Notations des dimensions : M candidats, n instants simulés, N points lidar.
"""
from dataclasses import dataclass, field

import numpy as np
from pai_core.perception.features import ObsConfig, split_obs    
from pai_core.perception.features import split_obs
from pai_core.robot.diff_drive import DiffDriveParams


# ---------------------------------------------------------------- paramètres

@dataclass(frozen=True)
class DWAParams:
    diff_drive: DiffDriveParams = field(default_factory=DiffDriveParams)

    # limites dynamiques : à reprendre des mesures du jour 3, choisies prudentes
    acc_v: float = 1.0           # accélération linéaire maximale (m/s²)
    acc_w: float = 4.0           # accélération angulaire maximale (rad/s²)
    dec_v: float = 1.0           # décélération de freinage, pour l'admissibilité (m/s²)

    # échantillonnage et simulation
    dt: float = 0.1              # période de décision, largeur de la fenêtre (s)
    horizon: float = 1.5         # durée des arcs simulés (s)
    dt_sim: float = 0.1          # pas de temps le long des arcs (s)
    n_v: int = 15                # nombre de valeurs de v dans la fenêtre
    n_w: int = 31                # nombre de valeurs de w (impair : w = 0 en fait partie)

    # robot et capteur
    robot_radius: float = 0.24   # rayon circonscrit depuis le milieu de l'essieu (m)
    lidar_offset_x: float = 0.0  # position du lidar par rapport à l'essieu (m), à vérifier dans le MJCF
    max_range: float = 4.0       # portée du lidar (m)
    n_beams: int = 36

    # coût : poids et saturation du dégagement
    w_heading: float = 1.0
    w_clearance: float = 1.0
    w_velocity: float = 0.5
    d_max: float = 1.0           # au-delà, un dégagement plus grand ne rapporte plus rien (m)
    w_limit: float = 2.0

    # repli quand aucun candidat n'est admissible
    w_fallback: float = 1.0      # rotation sur place (rad/s)

    @property
    def v_max(self) -> float:
        """Sommet du losange sur l'axe v : r·w_roue_max. C'est la vitesse maximale du robot en ligne droite"""
        return self.diff_drive.wheel_radius * self.diff_drive.max_wheel_speed

    @property
    def w_max(self) -> float:
        """Sommet du losange sur l'axe w : 2·r·w_roue_max / L. Rotation maximale du robot I guess"""
        return 2*self.diff_drive.wheel_radius * self.diff_drive.max_wheel_speed /  self.diff_drive.wheel_base


# ---------------------------------------------------------------- perception

def lidar_angles(n_beams: int) -> np.ndarray:
    """Angle de chaque faisceau dans le repère du lidar, (n_beams,).

    Faisceau 0 vers l'avant, angles croissants vers la gauche : la même
    convention que sensors.py , sinon le DWA voit le monde en miroir.
    """
    
    return  2.0 * np.pi * np.arange(n_beams) / n_beams #même truc que dans sensors.py


def lidar_to_points(distances: np.ndarray, angles: np.ndarray,
                    max_range: float, offset_x: float = 0.0,
                    eps: float = 1e-6) -> np.ndarray:
    """Points d'impact (x, y) dans base_link, (N, 2), avec N <= n_beams.

    Passage polaire -> cartésien dans le repère du lidar, puis translation
    de offset_x vers base_link. Les faisceaux à portée maximale n'ont rien
    touché : on les retire, sinon ils créent un mur fantôme à max_range.
    """
    hit = distances < max_range - eps           # masque des faisceaux qui ont touché
    d, a = distances[hit], angles[hit]
    x = offset_x + d * np.cos(a)                # direction du faisceau, mise à l'échelle par la distance mesurée
    y = d * np.sin(a)
    return np.stack([x, y], axis=1)             # (N, 2)


# ---------------------------------------------------------------- candidats

def dynamic_window(v_c: float, w_c: float, p: DWAParams) -> np.ndarray:
    """Candidats (v, w) atteignables en p.dt depuis la vitesse actuelle, (M, 2).

    Fenêtre : rectangle [v_c pm acc_v·dt] times [w_c pm acc_w·dt], discrétisé en
    une grille n_v times n_w, avec v >= 0 (pas de marche arrière). On ne garde
    que les points du losange des roues : |v| / v_max + |w| / w_max <= 1.
    n_w impair : la commande actuelle w_c fait partie des candidats.
    w = 0 (ligne droite exacte) est ajouté quand la fenêtre le contient.
    """
    # bornes du rectangle, rognées par les limites absolues pour ne pas gaspiller de points
    v_lo = max(0.0, v_c - p.acc_v * p.dt)
    v_hi = min(p.v_max, v_c + p.acc_v * p.dt)
    w_lim = min(p.w_max, p.w_limit)
    w_lo = max(-w_lim, w_c - p.acc_w * p.dt)
    w_hi = min(w_lim, w_c + p.acc_w * p.dt)

    vs = np.linspace(v_lo, v_hi, p.n_v)
    ws = np.linspace(w_lo, w_hi, p.n_w)
    if w_lo <= 0.0 <= w_hi:
        ws = np.union1d(ws, [0.0])      # ajoute 0 (sans doublon), résultat trié

    # grille : toutes les combinaisons (v, w)
    V, W = np.meshgrid(vs, ws, indexing="ij")
    cand = np.stack([V.ravel(), W.ravel()], axis=1)            # (n_v·n_w, 2)

    # losange des roues ; la petite tolérance garde les points situés sur le bord
    inside = np.abs(cand[:, 0]) / p.v_max + np.abs(cand[:, 1]) / p.w_max <= 1.0 + 1e-9
    return cand[inside]


def rollout_unicycle(cand: np.ndarray, horizon: float, dt_sim: float) -> np.ndarray:
    """Arcs simulés, (M, n, 3) avec (x, y, theta) aux instants dt_sim, ..., horizon.

    Forme close, depuis l'origine de base_link :
        x = (v/w)·sin(w·t),  y = (v/w)·(1 - cos(w·t)),  theta = w·t.
    Si |w| est presque nul : x = v·t, y = v·w·t²/2 (limite de la formule).
    Réutilisée telle quelle par MPPI au jour 18.
    """
    t = np.arange(1, int(round(horizon / dt_sim)) + 1) * dt_sim             # (n,)
    v, w = cand[:, 0:1], cand[:, 1:2]                               # (M, 1)
    th = w * t                                                      # (M, n)
    droit = np.abs(w) < 1e-6
    w_sur = np.where(droit, 1.0, w)                                 # évite la division par 0
    x = np.where(droit, v * t, v / w_sur * np.sin(th))
    y = np.where(droit, 0.5 * v * w * t**2, v / w_sur * (1 - np.cos(th)))
    return np.stack([x, y, th], axis=-1)                            # (M, n, 3)


# ---------------------------------------------------------------- sécurité

def clearance(traj: np.ndarray, points: np.ndarray, robot_radius: float,
              dt_sim: float) -> tuple[np.ndarray, np.ndarray]:
    """Dégagement de chaque arc.

    Renvoie :
      d_min     (M,) : plus petite distance entre l'emprise du robot et un
                       point lidar, le long de l'arc (négative = contact) ;
      t_contact (M,) : instant du premier contact prédit, np.inf si aucun.

    Distances calculées en une fois par diffusion numpy : (M, n, N).
    Sans point lidar, tout vaut np.inf.
    """
    M, n = traj.shape[:2]
    if len(points) == 0:
        return np.full(M, np.inf), np.full(M, np.inf)
    diff = traj[:, :, None, :2] - points[None, None, :, :]                  # (M, n, N, 2)
    dist = np.linalg.norm(diff, axis=-1).min(axis=2) - robot_radius         # (M, n)
    d_min = dist.min(axis=1)                                                # (M,)
    contact = dist < 0
    # indice k du premier contact ; k·dt_sim est l'instant du dernier point sûr (choix prudent)
    t_contact = np.where(contact.any(axis=1), contact.argmax(axis=1) * dt_sim, np.inf)
    return d_min, t_contact


def admissible(cand: np.ndarray, t_contact: np.ndarray, p: DWAParams) -> np.ndarray:
    """Masque booléen (M,) des candidats qui peuvent encore freiner.

    Distance avant contact s = v·t_contact ; condition v <= sqrt(2·dec_v·s).
    Sans contact prévu, s = inf : on l'écrit explicitement, car v = 0
    donnerait 0·inf = nan, et une comparaison avec nan vaut toujours False.
    """
    v = cand[:, 0]
    s = np.full_like(v, np.inf)                       # pas de contact prévu : distance infinie
    fini = np.isfinite(t_contact)
    s[fini] = v[fini] * t_contact[fini]               # distance avant contact, seulement si contact il y a
    return v <= np.sqrt(2 * p.dec_v * s)


# ---------------------------------------------------------------- choix

def cost(cand: np.ndarray, traj: np.ndarray, d_min: np.ndarray,
         goal_xy: np.ndarray, p: DWAParams) -> np.ndarray:
    """Coût de chaque candidat, (M,), à minimiser. Trois termes dans [0, 1] :

      cap        : |écart| entre le cap en bout d'arc et la direction du but
                   vue depuis ce point, divisé par pi (écart ramené dans [0, pi]) ;
      dégagement : 1 - min(d_min, d_max) / d_max ;
      vitesse    : 1 - v / v_max.

    goal_xy : position du but dans base_link, (2,).
    """
    x, y, th = traj[:, -1, 0], traj[:, -1, 1], traj[:, -1, 2]       # fin de chaque arc
    psi = np.arctan2(goal_xy[1] - y, goal_xy[0] - x)                  # direction du but vue de là
    ecart = np.abs(np.angle(np.exp(1j * (psi - th))))               # écart d'angle dans [0, π]
    c_cap = ecart / np.pi
    c_deg = 1 - np.minimum(d_min, p.d_max) / p.d_max
    c_vit = 1 - cand[:, 0] / p.v_max
    return p.w_heading * c_cap + p.w_clearance * c_deg + p.w_velocity * c_vit 


# ---------------------------------------------------------------- décideur

class DWA:
    """Décideur DWA, même interface que GoToGoal et que la politique PPO."""

    def __init__(self, params: DWAParams | None = None):
        self.p = params or DWAParams()
        self.angles = lidar_angles(self.p.n_beams)   # calculés une seule fois
        self.last: dict = {}                          # détails de la dernière décision

    def reset(self) -> None:
        """Début d'épisode : le DWA n'a pas de mémoire, on vide seulement self.last."""
        self.last = {}

    def act(self, obs: np.ndarray) -> np.ndarray:
        """(docstring inchangée)"""
        p = self.p
        distances, d_but, cap_but, v_c, w_c = split_obs(obs)            # seul lien avec features.py
        points = lidar_to_points(distances, self.angles, p.max_range, p.lidar_offset_x)
        cand = dynamic_window(v_c, w_c, p)                              # candidats (v, w)
        traj = rollout_unicycle(cand, p.horizon, p.dt_sim)              # arc imaginé de chaque candidat
        d_min, t_contact = clearance(traj, points, p.robot_radius, p.dt_sim)
        ok = admissible(cand, t_contact, p)

        if ok.any():
            goal_xy = d_but * np.array([np.cos(cap_but), np.sin(cap_but)])
            J = cost(cand, traj, d_min, goal_xy, p)
            J[~ok] = np.inf                                             # les inadmissibles ne peuvent pas gagner
            choice = int(np.argmin(J))
            v, w = cand[choice]
        else:
            # repli : arrêt et rotation sur place, du côté du but
            J, choice = None, None
            v, w = 0.0, p.w_fallback * (np.sign(cap_but) or 1.0)

        self.last = dict(points=points, cand=cand, traj=traj, d_min=d_min,
                         ok=ok, cost=J, choice=choice)
        return self._to_action(v, w)

    def _to_action(self, v: float, w: float) -> np.ndarray:
        """(v, w) physiques -> action normalisée dans [-1, 1]², comme PPO."""
        cfg = ObsConfig()
        return np.array([v / cfg.v_max, w / cfg.w_max], dtype=np.float32)