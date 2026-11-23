"""Environnement Gymnasium de navigation vers un but.

  Action      : (a_v, a_ω) dans [-1, 1]², converties en v = a_v·v_max et
                ω = a_ω·ω_max, puis en vitesses de roues (cinématique inverse
                et saturation qui conserve la courbure).
  Observation : voir pai_core/perception/features.py (43 composantes).
  Décision    : 10 Hz, soit frame_skip = 50 pas physiques de 2 ms.
  Fin         : terminated si collision ou but atteint ; truncated à max_steps.

La récompense est PROVISOIRE (le jour 11 la remplace par un façonnage par
potentiel justifié) : progrès vers le but, petite pénalité de temps, bonus
de succès, malus de collision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np

from pai_core.perception.features import ObsConfig, build_observation, goal_polar, observation_bounds
from pai_core.robot.diff_drive import DiffDriveParams, inverse, saturate
from pai_core.sim.sensors import Lidar, LidarConfig
from pai_core.sim.world import Scene, World, WorldConfig


@dataclass
class NavEnvConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    robot: DiffDriveParams = field(default_factory=DiffDriveParams)
    decision_dt: float = 0.1            # 10 Hz
    max_steps: int = 300                # 30 s
    goal_tolerance: float = 0.25        # m
    n_obstacles: tuple[int, int] = (4, 8)   # tiré uniformément dans [min, max]
    settle_steps: int = 25              # pas physiques à commande nulle après placement
    lidar_noise_std: float = 0.0
    lidar_dropout: float = 0.0
    # récompense provisoire
    w_progress: float = 10.0            # par mètre gagné vers le but
    w_time: float = 0.01                # par décision
    r_success: float = 10.0
    r_collision: float = -10.0


class NavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: NavEnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config or NavEnvConfig()
        self.render_mode = render_mode
        self.world = World(self.cfg.world)
        self.model, self.data = self.world.model, self.world.data

        dt = self.model.opt.timestep
        self.frame_skip = int(round(self.cfg.decision_dt / dt))
        assert abs(self.frame_skip * dt - self.cfg.decision_dt) < 1e-9, \
            "decision_dt doit être un multiple du pas physique"

        self.lidar = Lidar(self.world.geom, LidarConfig(max_range=self.cfg.obs.range_max,
                                                        noise_std=self.cfg.lidar_noise_std,
                                                        dropout_prob=self.cfg.lidar_dropout))

        low, high = observation_bounds()
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.scene: Scene | None = None
        self._renderer = None

    # ------------------------------------------------------------------ outils
    def _observe(self) -> np.ndarray:
        ranges = self.lidar.scan(self.model, self.data, self.np_random)
        return build_observation(ranges, self.world.robot_pose(), self.world.robot_velocity(),
                                 self.scene.goal, self._prev_action, self.cfg.obs)

    def _goal_distance(self) -> float:
        return goal_polar(self.world.robot_pose(), self.scene.goal)[0]

    def _info(self, success=False, collision=False) -> dict:
        return {"is_success": success, "collision": collision,
                "goal_distance": self._goal_distance(), "steps": self._steps,
                "path_length": self.scene.path_length,
                "n_obstacles": len(self.scene.obstacles)}

    # ------------------------------------------------------------------- API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)          # initialise self.np_random à partir de la graine
        options = options or {}
        if "scene" in options:            # scène imposée : tests, banc d'essai (jour 12)
            self.scene = options["scene"]
        else:                             # densité imposée (curriculum, jour 11) ou tirée
            if "n_obstacles" in options:
                n = int(options["n_obstacles"])
            else:
                lo, hi = self.cfg.n_obstacles
                n = int(self.np_random.integers(lo, hi + 1))
            self.scene = self.world.sample_scene(self.np_random, n)

        self.world.apply(self.scene)
        self.data.ctrl[:] = 0.0
        for _ in range(self.cfg.settle_steps):   # laisse les contacts roues-sol s'établir
            mujoco.mj_step(self.model, self.data)

        self._steps = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_dist = self._goal_distance()
        return self._observe(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v, w = float(a[0]) * self.cfg.obs.v_max, float(a[1]) * self.cfg.obs.w_max
        # saturate travaille sur (v, omega) : il réduit les deux du même facteur si
        # une roue dépasse sa vitesse maximale (la courbure est conservée).
        # inverse donne ensuite les vitesses des roues (gauche, droite).
        v, w = saturate(v, w, self.cfg.robot)
        wl, wr = inverse(v, w, self.cfg.robot)
        self.data.ctrl[self.world.ctrl_left] = float(wl)
        self.data.ctrl[self.world.ctrl_right] = float(wr)

        collision = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.world.in_collision():     # testé à chaque pas : un frôlement bref compte
                collision = True
                break

        self._steps += 1
        self._prev_action = a
        dist = self._goal_distance()
        success = (not collision) and dist < self.cfg.goal_tolerance

        reward = self.cfg.w_progress * (self._prev_dist - dist) - self.cfg.w_time
        if collision:
            reward += self.cfg.r_collision
        elif success:
            reward += self.cfg.r_success
        self._prev_dist = dist

        terminated = collision or success
        truncated = (not terminated) and self._steps >= self.cfg.max_steps
        return self._observe(), float(reward), terminated, truncated, self._info(success, collision)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        self._renderer.update_scene(self.data, camera="top")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
