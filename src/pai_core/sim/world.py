"""Arène 6 times 6 m, obstacles aléatoires, placement du robot et du but.

Principe : le modèle MuJoCo est compilé UNE fois, avec K obstacles
pré-alloués. Ce sont des corps " mocap " : MuJoCo les traite comme fixes
(la physique ne les déplace jamais) mais on peut changer leur position à
tout moment en écrivant dans data.mocap_pos, sans recompiler. À chaque
reset, on en place n <= K dans l'arène et on gare les autres sous le sol.

Dcp pour le moment les obstacles sont fixes. Au fur et à mesure, peut être que je vais changer ça.
Une scène est rejetée puis retirée si :
  - le départ et le but sont trop proches ;
  - un obstacle ne trouve pas de place loin du départ et du but ;
  - A* ne trouve aucun chemin entre le départ et le but.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from pai_core.planning.astar import astar, path_length
from pai_core.sim.sensors import RobotGeometry, true_pose

# src/pai_core/sim/world.py -> racine du dépôt = parents[3]
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "mjcf"
OBSTACLE_GROUP = 1   # murs et obstacles : sert à reconnaître les collisions
FLOOR_GROUP = 2      # sol et marqueur du but


# --------------------------------------------------------------------------
# Configuration et structures de données
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WorldConfig:
    robot_xml: Path = ASSETS_DIR / "robot.xml"
    arena_size: float = 6.0          # côté intérieur de l'arène (m)
    wall_thickness: float = 0.1
    obstacle_height: float = 0.3     # doit dépasser la hauteur du lidar (0,12 m), sinon le lidar les détectes pas (cf sensors.py)
    n_slots: int = 12                # K obstacles pré-alloués
    obstacle_half_size: tuple[float, float] = (0.15, 0.40)  # demi-côtés ou rayons (m)
    shape_seed: int = 0              # fixe les formes des K obstacles, pas leur position
    robot_radius: float = 0.23       # cercle circonscrit, centré au milieu de l'essieu (0,224 m)
    clearance: float = 0.15          # marge libre autour du départ et du but (m)
    min_start_goal_dist: float = 2.0 #distance minimal entre le start et le goal dcp (m)
    grid_resolution: float = 0.05    # grille A* (m)
    max_tries: int = 200             #nombre d'essais maximal


@dataclass(frozen=True)
class ObstacleShape:
    kind: str     # "box" ou "cylinder"
    hx: float     # demi-côté en x, ou rayon
    hy: float     # demi-côté en y, ou rayon

    @property
    def bounding_radius(self) -> float:
        return math.hypot(self.hx, self.hy) if self.kind == "box" else self.hx


@dataclass(frozen=True)
class PlacedObstacle:
    slot: int
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Scene:
    start: tuple[float, float, float]    # x, y, lacet du milieu de l'essieu (base_link)
    goal: tuple[float, float]
    obstacles: tuple[PlacedObstacle, ...]
    path_length: float                   # plus court chemin A* (m), pour la SPL


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# --------------------------------------------------------------------------
# Monde
# --------------------------------------------------------------------------
class World:
    def __init__(self, cfg: WorldConfig = WorldConfig()):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.shape_seed)
        lo, hi = cfg.obstacle_half_size
        self.shapes: list[ObstacleShape] = []
        for k in range(cfg.n_slots):
            if k % 2 == 0:
                self.shapes.append(ObstacleShape("box", rng.uniform(lo, hi), rng.uniform(lo, hi)))
            else:
                r = rng.uniform(lo, hi)
                self.shapes.append(ObstacleShape("cylinder", r, r))

        self.xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model

        # Robot : géométrie lue dans le modèle (sensors.py), actionneurs
        self.geom = RobotGeometry.from_model(m)
        self.chassis_id = self.geom.chassis_id
        self.qpos_adr = self.geom.free_qposadr
        self.dof_adr = m.jnt_dofadr[m.body_jntadr[self.chassis_id]]
        self.z0 = m.qpos0[self.qpos_adr + 2]
        # indices de ctrl : on les retrouve par l'articulation pilotée, pour ne
        # pas dépendre de l'ordre des actionneurs dans robot.xml
        self.ctrl_left = int(np.flatnonzero(m.actuator_trnid[:, 0] == m.joint("g").id)[0])
        self.ctrl_right = int(np.flatnonzero(m.actuator_trnid[:, 0] == m.joint("d").id)[0])

        # Obstacles et but : indices dans data.mocap_pos
        self.obstacle_mocap = [m.body_mocapid[m.body(f"obs_{k}").id] for k in range(cfg.n_slots)]
        self.goal_mocap = m.body_mocapid[m.body("goal_marker").id]

        # Masques de géométries pour la détection des collisions
        self.robot_geom = m.body_rootid[m.geom_bodyid] == self.chassis_id
        self.obstacle_geom = m.geom_group == OBSTACLE_GROUP
        # _forbidden[g1, g2] vaut True si un contact g1-g2 est une collision
        r, o = self.robot_geom, self.obstacle_geom
        self._forbidden = np.outer(r, o) | np.outer(o, r)

        # Grille A* : centres des cases
        h = cfg.arena_size / 2
        n = int(round(cfg.arena_size / cfg.grid_resolution))
        c = -h + cfg.grid_resolution * (np.arange(n) + 0.5)
        self._gx, self._gy = np.meshgrid(c, c, indexing="ij")

    # ------------------------------------------------------------------ XML
    def _build_xml(self) -> str:
        """Ajoute au robot le sol, les murs, les K obstacles et le marqueur du but."""
        cfg = self.cfg
        h, t, hz = cfg.arena_size / 2, cfg.wall_thickness / 2, cfg.obstacle_height / 2
        root = ET.parse(cfg.robot_xml).getroot()
        root.set("model", "pai_arena")
        ET.SubElement(root, "statistic", center="0 0 0", extent=f"{h + 1}")

        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                      rgb1=".1 .2 .3", rgb2=".2 .3 .4", width="300", height="300",
                      mark="edge", markrgb=".2 .3 .4")
        ET.SubElement(asset, "material", name="grid", texture="grid",
                      texrepeat="6 6", texuniform="false", reflectance=".1")

        wb = root.find("worldbody")
        ET.SubElement(wb, "light", pos="0 0 5", dir="0 0 -1", directional="true")
        ET.SubElement(wb, "camera", name="top", pos="0 0 8.5", xyaxes="1 0 0 0 1 0")
        ET.SubElement(wb, "geom", name="floor", type="plane", size=f"{h} {h} 0.1",
                      material="grid", group=str(FLOOR_GROUP))

        walls = [(0, h + t, h + 2 * t, t), (0, -h - t, h + 2 * t, t),
                 (h + t, 0, t, h), (-h - t, 0, t, h)]
        for i, (x, y, sx, sy) in enumerate(walls):
            ET.SubElement(wb, "geom", name=f"wall_{i}", type="box",
                          size=f"{sx} {sy} {hz}", pos=f"{x} {y} {hz}",
                          group=str(OBSTACLE_GROUP), rgba=".6 .6 .65 1")

        for k, s in enumerate(self.shapes):
            body = ET.SubElement(wb, "body", name=f"obs_{k}", mocap="true",
                                 pos=self._park_pos_str(k))
            size = f"{s.hx} {s.hy} {hz}" if s.kind == "box" else f"{s.hx} {hz}"
            ET.SubElement(body, "geom", type=s.kind, size=size,
                          group=str(OBSTACLE_GROUP), rgba=".85 .45 .2 1")

        goal = ET.SubElement(wb, "body", name="goal_marker", mocap="true", pos="0 0 0.002")
        # contype=conaffinity=0 : visible mais sans collision ; au ras du sol, sous les rayons du lidar
        ET.SubElement(goal, "geom", type="cylinder", size="0.12 0.002", contype="0",
                      conaffinity="0", group=str(FLOOR_GROUP), rgba=".2 .9 .3 .7")
        return ET.tostring(root, encoding="unicode")

    def _park_pos_str(self, k: int) -> str:
        """Position de garage d'un obstacle inutilisé : sous le sol, hors de vue.

        Deux corps mocap, ou un corps mocap et le sol, sont tous deux fixes :
        MuJoCo ne calcule pas de contact entre eux, le garage ne coûte rien.
        """
        return "0 0 -1"

    # ------------------------------------------------------- tirage de scène
    def sample_scene(self, rng: np.random.Generator, n_obstacles: int) -> Scene:
        cfg = self.cfg
        h = cfg.arena_size / 2
        m = cfg.robot_radius + cfg.clearance
        n = min(n_obstacles, cfg.n_slots)

        for _ in range(cfg.max_tries):
            sx, sy = rng.uniform(-h + m, h - m, 2)
            gx, gy = rng.uniform(-h + m, h - m, 2)
            if math.hypot(gx - sx, gy - sy) < cfg.min_start_goal_dist:
                continue
            yaw = rng.uniform(-math.pi, math.pi)

            placed = []
            for k in rng.choice(cfg.n_slots, size=n, replace=False):
                p = self._place_obstacle(rng, int(k), (sx, sy), (gx, gy))
                if p is None:
                    break
                placed.append(p)
            if len(placed) < n:
                continue

            path = astar(self.occupancy_grid(placed), self.to_cell(sx, sy), self.to_cell(gx, gy))
            if path is None:
                continue
            return Scene((sx, sy, yaw), (gx, gy), tuple(placed),
                         path_length(path, cfg.grid_resolution))

        raise RuntimeError(f"Aucune scène valide en {cfg.max_tries} essais "
                           f"({n} obstacles) : arène trop encombrée ?")

    def _place_obstacle(self, rng, k, start, goal, tries=50) -> PlacedObstacle | None:
        cfg = self.cfg
        h = cfg.arena_size / 2
        rb = self.shapes[k].bounding_radius
        keep = rb + cfg.robot_radius + cfg.clearance
        for _ in range(tries):
            x, y = rng.uniform(-h + rb, h - rb, 2)   # entièrement dans l'arène
            if (math.hypot(x - start[0], y - start[1]) >= keep
                    and math.hypot(x - goal[0], y - goal[1]) >= keep):
                return PlacedObstacle(k, float(x), float(y), float(rng.uniform(-math.pi, math.pi)))
        return None

    # ---------------------------------------------------------- grille A*
    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        h, res = self.cfg.arena_size / 2, self.cfg.grid_resolution
        n = self._gx.shape[0]
        return (min(max(int((x + h) / res), 0), n - 1),
                min(max(int((y + h) / res), 0), n - 1))

    def occupancy_grid(self, obstacles) -> np.ndarray:
        """Cases interdites au CENTRE du robot : obstacles et murs gonflés du rayon du robot.

        Gonfler les obstacles permet de traiter le robot comme un point. Pour
        une boîte, la case est interdite si sa distance à la boîte est au plus
        r (somme de Minkowski exacte, coins arrondis).
        """
        h, r = self.cfg.arena_size / 2, self.cfg.robot_radius
        X, Y = self._gx, self._gy
        occ = (np.abs(X) > h - r) | (np.abs(Y) > h - r)
        for ob in obstacles:
            s = self.shapes[ob.slot]
            dx, dy = X - ob.x, Y - ob.y
            if s.kind == "cylinder":
                occ |= np.hypot(dx, dy) <= s.hx + r
            else:
                c, si = math.cos(ob.yaw), math.sin(ob.yaw)
                lx, ly = c * dx + si * dy, -si * dx + c * dy   # repère de la boîte
                qx = np.maximum(np.abs(lx) - s.hx, 0.0)
                qy = np.maximum(np.abs(ly) - s.hy, 0.0)
                occ |= np.hypot(qx, qy) <= r
        return occ

    # ------------------------------------------------- application d'une scène
    def apply(self, scene: Scene) -> None:
        """Remet la simulation à zéro et y installe la scène (sans recompiler)."""
        m, d = self.model, self.data
        hz = self.cfg.obstacle_height / 2
        mujoco.mj_resetData(m, d)   # remet aussi les obstacles à leur position de garage

        for ob in scene.obstacles:
            mid = self.obstacle_mocap[ob.slot]
            d.mocap_pos[mid] = (ob.x, ob.y, hz)
            d.mocap_quat[mid] = yaw_to_quat(ob.yaw)
        d.mocap_pos[self.goal_mocap] = (scene.goal[0], scene.goal[1], 0.002)

        # scene.start est la pose du milieu de l'essieu : on en déduit celle de
        # l'origine du châssis, qui est ce que contient qpos
        x, y, yaw = scene.start
        ox, oy = self.geom.axle_offset[0], self.geom.axle_offset[1]
        c, s = math.cos(yaw), math.sin(yaw)
        a = self.qpos_adr
        d.qpos[a:a + 3] = (x - (c * ox - s * oy), y - (s * ox + c * oy), self.z0)
        d.qpos[a + 3:a + 7] = yaw_to_quat(yaw)
        mujoco.mj_forward(m, d)   # nécessaire : le lidar a besoin de la position des obstacles

    # ------------------------------------------------- état du robot (vérité terrain)
    def robot_pose(self) -> tuple[float, float, float]:
        """Pose vraie (x, y, lacet) du milieu de l'essieu, lue dans qpos."""
        x, y, yaw = true_pose(self.data, self.geom)
        return float(x), float(y), float(yaw)

    def robot_velocity(self) -> tuple[float, float]:
        """(v, ω) : vitesse d'avance et vitesse de lacet.

        Pour une liaison libre, qvel[0:3] est la vitesse linéaire de l'origine
        du châssis dans le repère MONDE, qvel[3:6] la vitesse angulaire dans le
        repère du CORPS. Comme dans sensors.py, la rotation est calculée à
        partir de qpos, à jour après mj_step, et non de xmat, en retard d'un pas.
        La vitesse d'avance est la même à l'origine du châssis et au milieu de
        l'essieu : les deux points sont sur l'axe x du robot.
        """
        qv = self.data.qvel[self.dof_adr:self.dof_adr + 6]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7])
        R = R.reshape(3, 3)
        v = float(R[:, 0] @ qv[0:3])          # projection sur l'axe x du robot
        w = float((R @ qv[3:6])[2])           # lacet dans le repère monde
        return v, w

    def in_collision(self) -> bool:
        """Vrai si une géométrie du robot touche un mur ou un obstacle."""
        g = self.data.contact.geom            # (ncon, 2) : paires de géométries en contact
        return bool(self._forbidden[g[:, 0], g[:, 1]].any())
