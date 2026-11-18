"""Carte d'occupation en log-odds (Probabilistic Robotics, chapitre 9).

Cadre : cartographie AVEC POSES CONNUES. La pose du lidar dans map vient de
TF, qui repose sur la vérité terrain (hypothèse de localisation parfaite).
Hypothèses du modèle : monde statique, cases indépendantes, mesures
indépendantes sachant la carte.

Mise à jour d'une case m par la mesure z_t (dérivation par Bayes, puis rapport
m / non-m, puis logarithme) :

    l_t = l_{t-1} + l_inv(m, z_t, x_t) - l_0,      l = log(p / (1 - p))

Modèle inverse du capteur, pour un faisceau :
    cases traversées avant l'impact   l_free < 0
    case de l'impact                  l_occ  > 0
    cases au-delà                     inchangées (on ne sait rien)
Un faisceau sans retour (inf, REP 117) libère toutes les cases jusqu'à la
portée maximale, mais n'en occupe aucune.

Conventions
    La grille est fixe dans le repère map.
    Tableau interne de forme (nx, ny), indexé [ix, iy] : ix suit x, iy suit y
    (comme indexing="ij" dans world.py).
    origin = (x0, y0) est le COIN bas-gauche de la case (0, 0), pas son centre
    (convention de nav_msgs/OccupancyGrid).
    Un point (x, y) tombe dans la case ix = floor((x − x0) / res), iy = floor((y − y0) / res).

Aucune dépendance à ROS : perception_node lit la TF, appelle update(), puis
recopie to_ros_values() dans le message.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pai_core.geometry.se2 import transform_points


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GridConfig:
    resolution: float = 0.05          # m, comme la grille A*
    size: float = 6.4                 # m : arène de 6 m + 0,2 m de marge de chaque côté
    origin: tuple = (-3.2, -3.2)      # coin bas-gauche de la case (0, 0), dans map
    p_occ: float = 0.7                # modèle inverse : case de l'impact
    p_free: float = 0.35              # modèle inverse : cases traversées
    p_prior: float = 0.5              # a priori : ℓ0 = 0
    l_min: float = -3.5               # saturation : une case doit pouvoir changer d'avis
    l_max: float = 3.5
    occ_threshold: float = 0.65       # au-dessus : occupée (mesures)
    free_threshold: float = 0.35      # en dessous : libre (mesures)
    free_on_no_return: bool = True    # un faisceau sans retour libère-t-il les cases ?

    @classmethod
    def for_arena(cls, arena_size: float, margin: float = 0.2, **kw) -> "GridConfig":
        """Grille centrée sur l'arène, avec une marge.

        La marge est indispensable : la face intérieure des murs est exactement
        en ±arena_size/2. Sans marge, les impacts sur les murs tombent juste
        hors de la grille et les murs n'apparaissent jamais.
        """
        size = arena_size + 2.0 * margin
        return cls(size=size, origin=(-size / 2.0, -size / 2.0), **kw)


# ----------------------------------------------------------------------------
# Fonctions libres
# ----------------------------------------------------------------------------

def logodds(p):
    """log(p / (1 − p))."""
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def prob(l):
    """Inverse de logodds : 1 − 1 / (1 + e^ℓ)."""
    return 1.0 - 1.0 / (1.0 + np.exp(np.asarray(l, dtype=float)))


def bresenham(i0: int, j0: int, i1: int, j1: int) -> np.ndarray:
    """Cases (K, 2) du segment de (i0, j0) à (i1, j1), extrémités comprises.

    Ordre : depuis (i0, j0). Les cases sont 8-connexes : deux cases
    successives diffèrent d'au plus 1 sur chaque indice. Valable dans les
    huit octants (algorithme à erreur entière, sans division).
    """
    i0, j0, i1, j1 = int(i0), int(j0), int(i1), int(j1)
    di, dj = abs(i1 - i0), -abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di + dj
    cells = []
    i, j = i0, j0
    while True:
        cells.append((i, j))
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 >= dj:      # pas selon i
            err += dj
            i += si
        if e2 <= di:      # pas selon j
            err += di
            j += sj
    return np.array(cells, dtype=np.int64)


# ----------------------------------------------------------------------------
# Grille
# ----------------------------------------------------------------------------

class OccupancyGrid:
    def __init__(self, cfg: GridConfig = GridConfig()):
        self.cfg = cfg
        n = cfg.size / cfg.resolution
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"size ({cfg.size}) doit être un multiple de resolution ({cfg.resolution})")
        self.nx = self.ny = int(round(n))
        self.origin = np.asarray(cfg.origin, dtype=float)
        self.l_occ = float(logodds(cfg.p_occ))
        self.l_free = float(logodds(cfg.p_free))
        self.l0 = float(logodds(cfg.p_prior))
        self.log_odds = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.observed = np.zeros((self.nx, self.ny), dtype=bool)
        self.reset()

    def reset(self) -> None:
        self.log_odds[:] = self.l0
        self.observed[:] = False

    # ------------------------------------------------------------ conversions
    def world_to_cell(self, xy) -> np.ndarray:
        """Points (N, 2) dans map → cases (N, 2) entières, SANS bornage.

        Un point hors de la grille donne une case hors de la grille : c'est
        in_bounds qui les écarte. Borner (comme World.to_cell pour A*)
        empilerait de faux impacts sur le bord.
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.floor((xy - self.origin) / self.cfg.resolution).astype(np.int64)

    def in_bounds(self, cells) -> np.ndarray:
        cells = np.atleast_2d(cells)
        return ((cells[:, 0] >= 0) & (cells[:, 0] < self.nx)
                & (cells[:, 1] >= 0) & (cells[:, 1] < self.ny))

    def cell_center(self, cells) -> np.ndarray:
        """Cases (N, 2) → centres (N, 2) dans map."""
        cells = np.atleast_2d(np.asarray(cells, dtype=float))
        return self.origin + (cells + 0.5) * self.cfg.resolution

    def _linear(self, cells) -> np.ndarray:
        """Indice linéaire dans le tableau (nx, ny) rangé en C : ix·ny + iy."""
        return cells[:, 0] * self.ny + cells[:, 1]

    # ------------------------------------------------------------ mise à jour
    def update(self, laser_pose, angles, ranges, max_range: float, range_min: float = 0.0) -> None:
        """Intègre un scan.

        laser_pose : (x, y, θ) du lidar dans map, c'est-à-dire T_map_laser (via TF).
        angles     : (N,) angles des faisceaux dans le repère du lidar (rad).
        ranges     : (N,) distances (m) ; inf ou NaN = pas de retour.
        """
        cfg = self.cfg
        angles = np.asarray(angles, dtype=float)
        ranges = np.asarray(ranges, dtype=float)
        if angles.shape != ranges.shape:
            raise ValueError(f"angles {angles.shape} et ranges {ranges.shape} incompatibles")

        # 1. Tri des faisceaux. Une distance finie sous range_min est ignorée.
        finite = np.isfinite(ranges)
        hit = finite & (ranges >= range_min) & (ranges < max_range)
        no_return = ~finite | (ranges >= max_range)
        used = hit | (no_return & cfg.free_on_no_return)
        if not used.any():
            return

        # 2. Extrémités dans map (portée maximale pour les faisceaux sans retour).
        r = np.where(hit, ranges, max_range)
        local = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        ends = self.world_to_cell(transform_points(laser_pose, local))
        start = self.world_to_cell(np.asarray(laser_pose, dtype=float)[:2])[0]

        # 3. Parcours des rayons.
        free_parts, occ_parts = [], []
        for k in np.flatnonzero(used):
            cells = bresenham(start[0], start[1], ends[k, 0], ends[k, 1])
            if hit[k]:
                free_parts.append(cells[:-1])
                occ_parts.append(cells[-1:])
            else:
                free_parts.append(cells)

        free = np.concatenate(free_parts) if free_parts else np.empty((0, 2), np.int64)
        occ = np.concatenate(occ_parts) if occ_parts else np.empty((0, 2), np.int64)

        # 4. Hors de la grille : on écarte, on ne borne pas.
        free = free[self.in_bounds(free)] if len(free) else free
        occ = occ[self.in_bounds(occ)] if len(occ) else occ

        # 5. Une seule mise à jour par case et par scan ; l'impact l'emporte.
        occ_lin = np.unique(self._linear(occ)) if len(occ) else np.empty(0, np.int64)
        free_lin = np.unique(self._linear(free)) if len(free) else np.empty(0, np.int64)
        free_lin = np.setdiff1d(free_lin, occ_lin, assume_unique=True)

        # 6. Mise à jour log-odds, saturation, cases observées.
        lo = self.log_odds.reshape(-1)          # vue, pas copie (tableau contigu)
        obs = self.observed.reshape(-1)
        lo[free_lin] += self.l_free - self.l0
        lo[occ_lin] += self.l_occ - self.l0
        touched = np.concatenate([free_lin, occ_lin])
        lo[touched] = np.clip(lo[touched], cfg.l_min, cfg.l_max)
        obs[touched] = True

    # ------------------------------------------------------------ lecture
    def probabilities(self) -> np.ndarray:
        """(nx, ny) dans [0, 1]."""
        return prob(self.log_odds)

    def occupied(self) -> np.ndarray:
        """(nx, ny) booléen : p > occ_threshold."""
        return self.probabilities() > self.cfg.occ_threshold

    def free(self) -> np.ndarray:
        """(nx, ny) booléen : observée et p < free_threshold."""
        return self.observed & (self.probabilities() < self.cfg.free_threshold)

    def to_ros_values(self) -> np.ndarray:
        """int8 aplati, prêt pour nav_msgs/OccupancyGrid.data.

        −1 : jamais observée ; sinon round(100·p).
        Ordre ROS : index = ix + iy·width (x varie le plus vite). Le tableau
        interne étant [ix, iy], il faut transposer avant d'aplatir.
        """
        values = np.rint(100.0 * self.probabilities()).astype(np.int8)
        values[~self.observed] = -1
        return values.T.ravel()