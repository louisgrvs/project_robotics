"""Validation du jour 5 (cases j5-1 et j5-2) et tests de world.py / A*."""
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env

from pai_core.envs.nav_env import NavEnv, NavEnvConfig
from pai_core.planning.astar import astar, path_length
from pai_core.sim.world import PlacedObstacle, Scene, World

@pytest.fixture
def env():
    e = NavEnv()
    yield e
    e.close()


def empty_scene(start, goal):
    """Scène sans obstacle, pour piloter le robot à la main."""
    d = math.hypot(goal[0] - start[0], goal[1] - start[1])
    return Scene(start=start, goal=goal, obstacles=(), path_length=d)


# ---------------------------------------------------------------- j5-1
def test_check_env(env):
    gym_check_env(env, skip_render_check=True)


def test_check_env_sb3():
    sb3 = pytest.importorskip("stable_baselines3.common.env_checker")
    sb3.check_env(NavEnv(), warn=True)


def test_same_seed_same_scene_and_trajectory():
    def rollout(seed):
        e = NavEnv(NavEnvConfig(lidar_noise_std=0.01))   # le bruit doit aussi être reproductible
        obs, _ = e.reset(seed=seed)
        e.action_space.seed(seed)
        traj = [obs]
        for _ in range(30):
            obs, *_ = e.step(e.action_space.sample())
            traj.append(obs)
        return e.scene, np.array(traj)

    s1, t1 = rollout(123)
    s2, t2 = rollout(123)
    s3, _ = rollout(124)
    assert s1 == s2
    np.testing.assert_array_equal(t1, t2)    # égalité exacte, pas approchée
    assert s1 != s3


# ---------------------------------------------------------------- j5-2
def test_crash_into_wall_terminates_with_collision(env):
    # robot à 1 m du mur x = +3, orienté vers lui ; but derrière
    env.reset(seed=0, options={"scene": empty_scene((2.0, 0.0, 0.0), (-2.0, 0.0))})
    for _ in range(50):
        _, r, terminated, truncated, info = env.step([1.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["collision"] and not info["is_success"]


def test_reaching_goal_terminates_with_success(env):
    env.reset(seed=0, options={"scene": empty_scene((-1.0, 0.0, 0.0), (0.5, 0.0))})
    for _ in range(100):
        _, r, terminated, truncated, info = env.step([1.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["is_success"] and not info["collision"]
    assert r > 0


def test_time_limit_truncates(env):
    env.cfg.max_steps = 5
    env.reset(seed=0, options={"scene": empty_scene((0.0, 0.0, 0.0), (2.0, 2.0))})
    for k in range(5):
        _, _, terminated, truncated, _ = env.step([0.0, 0.0])
        assert not terminated
        assert truncated == (k == 4)


# ---------------------------------------------------------------- monde et A*
def test_obstacles_avoid_start_and_goal(env):
    cfg = env.world.cfg
    for seed in range(20):
        env.reset(seed=seed)
        s = env.scene
        assert math.dist(s.start[:2], s.goal) >= cfg.min_start_goal_dist
        assert s.path_length >= math.dist(s.start[:2], s.goal) - 2 * cfg.grid_resolution
        for ob in s.obstacles:
            keep = env.world.shapes[ob.slot].bounding_radius + cfg.robot_radius
            assert math.dist((ob.x, ob.y), s.start[:2]) >= keep
            assert math.dist((ob.x, ob.y), s.goal) >= keep
        assert not env.world.in_collision()


def test_unused_obstacles_are_parked(env):
    env.reset(seed=0, options={"n_obstacles": 3})
    used = {ob.slot for ob in env.scene.obstacles}
    for k, mid in enumerate(env.world.obstacle_mocap):
        z = env.data.mocap_pos[mid][2]
        assert (z > 0) == (k in used)


def test_astar_wall_with_gap():
    occ = np.zeros((20, 20), dtype=bool)
    occ[10, :] = True
    assert astar(occ, (2, 2), (17, 2)) is None       # mur fermé : pas de chemin
    occ[10, 15] = False
    path = astar(occ, (2, 2), (17, 2))
    assert path is not None and (10, 15) in path
    assert path_length(path, 1.0) >= 15.0


def test_start_pose_is_axle_midpoint(env):
    env.reset(seed=0, options={"scene": empty_scene((1.0, -0.5, 0.8), (-2.0, 0.0))})
    x, y, yaw = env.world.robot_pose()
    assert abs(x - 1.0) < 1e-3 and abs(y + 0.5) < 1e-3 and abs(yaw - 0.8) < 1e-3


def test_lidar_sees_wall_at_known_distance(env):
    # milieu de l'essieu en (2, 0) face au mur intérieur x = 3. Le lidar est
    # au-dessus de l'origine du châssis, 5 cm derrière l'essieu : il voit le mur à 1,05 m.
    env.reset(seed=0, options={"scene": empty_scene((2.0, 0.0, 0.0), (-2.0, 0.0))})
    ranges = env.lidar.scan(env.model, env.data)
    assert abs(ranges[0] - 1.00) < 0.01              # faisceau 0 vers l'avant
    assert abs(ranges[18] - 4.0) < 1e-9              # vers l'arrière : mur à 4,95 m, hors portée
    assert abs(ranges[9] - 3.0) < 0.01               # vers la gauche : mur y = 3

@pytest.mark.parametrize("lo,hi,n", [(1.0, 2.0, 0), (2.0, 6.0, 8)])
def test_goal_distance_range(lo, hi, n):
    world = World()
    rng = np.random.default_rng(0)
    lim = world.cfg.arena_size / 2 - world.cfg.robot_radius - world.cfg.clearance
    for _ in range(50):
        s = world.sample_scene(rng, n, goal_dist=(lo, hi))
        d = math.hypot(s.goal[0] - s.start[0], s.goal[1] - s.start[1])
        assert lo <= d <= hi
        assert abs(s.goal[0]) <= lim and abs(s.goal[1]) <= lim