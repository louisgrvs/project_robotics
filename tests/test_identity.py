"""Test d'identité : un même état MuJoCo donne la même observation par Gym et par ROS.

C'est le test le plus important de la semaine. Il garantit qu'une politique
entraînée dans NavEnv voit, une fois branchée dans ROS, exactement ce qu'elle a
vu à l'entraînement. Sans lui, un mauvais repère, un signe inversé ou une
mauvaise convention pour « pas de retour » passeraient inaperçus : le robot
roulerait mal, et on accuserait la politique.

Les deux voies, pour un même état de la simulation :

    voie Gym : NavEnv._observe()
               lidar (m) ─┐
               pose vraie, but dans map ─┼─> build_observation
               vitesse vraie (qvel) ─┘

    voie ROS : le chemin réel des messages, avec le code des nœuds
               lidar ─> ranges_to_msg (sim_bridge) ─> ranges_from_msg (decision_node)
               pose vraie + odométrie ─> TF map->odom->base_link (sim_bridge)
                   ─> but transformé dans base_link par tf2 (goal_in_frame, decision_node)
               vitesse : message Odometry de /ground_truth
               ─> build_observation(pose = (0, 0, 0) : on est dans base_link)

Trois tests :
    1. test_lidar_identique          sans ROS : conventions inf / range_max / float32
    2. test_observation_identique    avec ROS (tf2) : l'observation complète, écart < 1e-6
    3. test_vitesse_odometrie        sans ROS : vitesse des encodeurs contre vitesse vraie.
                                     Égales en régime établi ; en transitoire, l'écart est
                                     MESURÉ et affiché (pytest -s), c'est un résultat.

Lancer, depuis la racine du dépôt (avec ROS sourcé pour le test 2) :
    source ros2_ws/install/setup.bash
    pytest tests/test_identity.py -v -s
Sans ROS, le test 2 est sauté (skipped) avec la raison, les autres tournent.
"""
from __future__ import annotations

import numpy as np
import pytest

from pai_core.envs.nav_env import NavEnv, NavEnvConfig
from pai_core.geometry.se2 import compose
from pai_core.perception.features import OBS_DIM, SLICES, build_observation, preprocess_ranges
from pai_core.robot.diff_drive import inverse, saturate
from pai_core.robot.odometry import WheelOdometry
from pai_core.ros_conversions import map_to_odom, ranges_from_msg, ranges_to_msg, velocity_from_poses
from pai_core.sim.sensors import WheelEncoders

TOL = 1e-6                 # écart maximal admis entre les deux observations
SEEDS = [0, 1, 2, 3, 4]    # scènes testées
N_STEPS = 40               # décisions par scène
ODOM_RADIUS_ERROR = 0.02   # odométrie volontairement fausse : map -> odom n'est pas l'identité


# ----------------------------------------------------------------------------
# Outils
# ----------------------------------------------------------------------------

def states(seed: int, n_steps: int = N_STEPS):
    """Fait rouler le robot avec des actions aléatoires et rend la main après chaque décision.

    Rend (env, obs_gym, pose_odom) : l'environnement (donc l'état MuJoCo),
    l'observation calculée par NavEnv, et la pose estimée par une odométrie
    tenue comme dans sim_bridge (fausse de 2 % sur la roue droite, exprès).
    """
    env = NavEnv(NavEnvConfig())                  # avec obstacles : le lidar voit des choses
    rng = np.random.default_rng(seed)
    geom = env.world.geom
    encoders = WheelEncoders(geom)
    odom = WheelOdometry(geom.wheel_radius, geom.wheel_radius * (1 + ODOM_RADIUS_ERROR), geom.wheel_base)

    obs, _ = env.reset(seed=seed)
    odom.reset(wheel_angles=encoders.read(env.data))
    for _ in range(n_steps):
        a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        a[0] = abs(a[0])                          # surtout vers l'avant, pour aller loin
        obs, _, terminated, truncated, _ = env.step(a)
        pose_odom = odom.update(encoders.read(env.data))
        yield env, obs, pose_odom
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
            odom.reset(wheel_angles=encoders.read(env.data))


def describe_gap(obs_a: np.ndarray, obs_b: np.ndarray) -> str:
    """Où est le plus grand écart, pour un message d'échec lisible."""
    gap = np.abs(obs_a.astype(np.float64) - obs_b.astype(np.float64))
    i = int(np.argmax(gap))
    part = next(name for name, sl in SLICES.items() if sl.start <= i < sl.stop)
    return f"écart max {gap[i]:.3g} à la composante {i} ({part}) : Gym {obs_a[i]:.7f}, ROS {obs_b[i]:.7f}"


# ----------------------------------------------------------------------------
# 1. Lidar (sans ROS)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_lidar_identique(seed):
    """inf (REP 117) puis retour à range_max, et arrondi float32 du message :
    la partie lidar de l'observation doit être IDENTIQUE, pas seulement proche."""
    for env, _, _ in states(seed, n_steps=15):
        r = env.lidar.scan(env.model, env.data)
        max_range = env.lidar.cfg.max_range
        msg = ranges_to_msg(r, max_range)
        assert all(np.isinf(m) for m, x in zip(msg, r) if x >= max_range), "pas de retour -> inf"
        np.testing.assert_array_equal(preprocess_ranges(r), preprocess_ranges(ranges_from_msg(msg, max_range)))


@pytest.mark.parametrize("seed", SEEDS)
def test_map_odom_redonne_la_pose_vraie(seed):
    """T_map_odom ∘ T_odom_base = T_map_base, même avec une odométrie fausse."""
    for env, _, pose_odom in states(seed, n_steps=15):
        gt = np.array(env.world.robot_pose())
        back = compose(map_to_odom(gt, pose_odom), pose_odom)
        assert np.abs(back - gt).max() < 1e-12


# ----------------------------------------------------------------------------
# 2. Observation complète (avec ROS : tf2 et le code des nœuds)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_observation_identique(seed):
    """LE test du jour 10 : même état MuJoCo -> même vecteur, écart < 1e-6."""
    pytest.importorskip("rclpy", reason="ROS 2 non sourcé")
    tf2_ros = pytest.importorskip("tf2_ros", reason="ROS 2 non sourcé")
    decision = pytest.importorskip("pai_ros.decision_node",
                                   reason="pai_ros non construit ou non sourcé (colcon build, source install/setup.bash)")
    bridge = pytest.importorskip("pai_ros.sim_bridge_node", reason="pai_ros non construit ou non sourcé")
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from rclpy.time import Time

    make_transform = bridge.SimBridge.make_transform     # méthode statique : pas besoin de nœud
    n_checked = 0
    for env, obs_gym, pose_odom in states(seed):
        cfg = env.cfg.obs
        stamp = Time(nanoseconds=round(env.data.time * 1e9)).to_msg()

        # ---- ce que publie sim_bridge -------------------------------------
        msg_ranges = ranges_to_msg(env.lidar.scan(env.model, env.data), env.lidar.cfg.max_range)

        pose_gt = np.array(env.world.robot_pose())
        tf_buffer = tf2_ros.Buffer()
        tf_buffer.set_transform(make_transform(stamp, "map", "odom", *map_to_odom(pose_gt, pose_odom)), "test")
        tf_buffer.set_transform(make_transform(stamp, "odom", "base_link", *pose_odom), "test")

        gt_msg = Odometry()                       # /ground_truth : vitesse vraie, dans base_link
        gt_msg.twist.twist.linear.x, gt_msg.twist.twist.angular.z = env.world.robot_velocity()

        goal = PoseStamped()                      # le but, donné dans map (RViz ou action)
        goal.header.frame_id = "map"
        goal.header.stamp = stamp
        goal.pose.position.x, goal.pose.position.y = (float(c) for c in env.scene.goal)
        goal.pose.orientation.w = 1.0

        # ---- ce que fait decision_node ------------------------------------
        ranges = ranges_from_msg(msg_ranges, env.lidar.cfg.max_range)
        velocity = (float(gt_msg.twist.twist.linear.x), float(gt_msg.twist.twist.angular.z))
        goal_b = decision.goal_in_frame(tf_buffer, goal, "base_link")
        prev_action = np.clip(np.asarray(env._prev_action, dtype=np.float32), -1.0, 1.0)
        obs_ros = build_observation(ranges, pose=(0.0, 0.0, 0.0), velocity=velocity,
                                    goal=goal_b, prev_action=prev_action, cfg=cfg)

        assert obs_ros.shape == obs_gym.shape == (OBS_DIM,)
        gap = float(np.max(np.abs(obs_gym.astype(np.float64) - obs_ros.astype(np.float64))))
        assert gap < TOL, describe_gap(obs_gym, obs_ros)
        n_checked += 1
    assert n_checked == N_STEPS


# ----------------------------------------------------------------------------
# 3. Vitesse : odométrie contre vérité terrain (sans ROS)
# ----------------------------------------------------------------------------

def _drive(env, v_cmd: float, w_cmd: float) -> None:
    """Applique une commande (v, ω) aux roues, comme NavEnv.step et control_node."""
    v, w = saturate(v_cmd, w_cmd, env.cfg.robot)
    wl, wr = inverse(v, w, env.cfg.robot)
    env.data.ctrl[env.world.ctrl_left] = float(wl)
    env.data.ctrl[env.world.ctrl_right] = float(wr)


def _cycle(env, n_sub: int = 10) -> None:
    """Un cycle de sim_bridge : 10 pas physiques de 2 ms = 20 ms."""
    import mujoco
    for _ in range(n_sub):
        mujoco.mj_step(env.model, env.data)


def test_vitesse_odometrie_regime_etabli():
    """Commande constante : après la mise en vitesse, /odom et la vérité concordent.

    Vérifie la convention de velocity_from_poses (signes, repère base_link,
    schéma du point milieu) : si elle était fausse, l'écart resterait grand.
    """
    env = NavEnv(NavEnvConfig(n_obstacles=(0, 0)))
    env.reset(seed=0)
    geom = env.world.geom
    encoders = WheelEncoders(geom)
    odom = WheelOdometry(geom.wheel_radius, geom.wheel_radius, geom.wheel_base)
    cfg, dt = env.cfg.obs, 10 * env.model.opt.timestep

    for v_cmd, w_cmd in [(0.3, 0.0), (0.3, 0.5), (0.2, -1.0), (0.0, 1.5)]:
        odom.reset(wheel_angles=encoders.read(env.data))
        prev = odom.pose.copy()
        _drive(env, v_cmd, w_cmd)
        for k in range(25):                       # 0,5 s
            _cycle(env)
            pose = odom.update(encoders.read(env.data))
            v_o, w_o = velocity_from_poses(prev, pose, dt)
            prev = pose
            if k >= 15:                           # après 0,3 s : régime établi
                v_t, w_t = env.world.robot_velocity()
                assert abs(v_o - v_t) / cfg.v_max < 0.02, (v_cmd, w_cmd, v_o, v_t)
                assert abs(w_o - w_t) / cfg.w_max < 0.02, (v_cmd, w_cmd, w_o, w_t)


def test_vitesse_odometrie_transitoires_mesure():
    """MESURE (affichée avec pytest -s) : écart odométrie / vérité quand la commande change.

    Avec une nouvelle commande toutes les 100 ms, le robot est presque
    toujours en transitoire : le châssis oscille et les roues glissent, et la
    vitesse vraie s'écarte de celle des encodeurs. C'est pourquoi
    decision_node lit /ground_truth par défaut (comme NavEnv), et pourquoi
    passer à velocity_source:=odom est une expérience en soi : c'est
    l'information qu'aurait un vrai robot.
    """
    env = NavEnv(NavEnvConfig(n_obstacles=(0, 0)))
    rng = np.random.default_rng(0)
    geom = env.world.geom
    encoders = WheelEncoders(geom)
    odom = WheelOdometry(geom.wheel_radius, geom.wheel_radius, geom.wheel_base)
    cfg, dt = env.cfg.obs, 10 * env.model.opt.timestep

    gaps = []
    for seed in range(3):
        env.reset(seed=seed)
        odom.reset(wheel_angles=encoders.read(env.data))
        prev = odom.pose.copy()
        for _ in range(40):                       # 40 décisions de 100 ms
            a = rng.uniform(-1.0, 1.0, size=2)
            _drive(env, a[0] * cfg.v_max, a[1] * cfg.w_max)
            for _ in range(5):                    # 5 cycles de 20 ms
                _cycle(env)
                pose = odom.update(encoders.read(env.data))
                v_o, w_o = velocity_from_poses(prev, pose, dt)
                prev = pose
                if env.world.in_collision():      # un contact fausse tout : on l'écarte
                    continue
                v_t, w_t = env.world.robot_velocity()
                gaps.append((abs(v_o - v_t) / cfg.v_max, abs(w_o - w_t) / cfg.w_max))

    gaps = np.asarray(gaps)
    med, p95, mx = np.median(gaps, axis=0), np.percentile(gaps, 95, axis=0), gaps.max(axis=0)
    print("\nécart |odométrie − vérité| (en unités de l'observation, v/v_max et ω/ω_max), "
          f"sur {len(gaps)} cycles :")
    print(f"    v : médiane {med[0]:.3f}, 95 % sous {p95[0]:.3f}, max {mx[0]:.3f}")
    print(f"    ω : médiane {med[1]:.3f}, 95 % sous {p95[1]:.3f}, max {mx[1]:.3f}")
    assert np.all(np.isfinite(gaps))
