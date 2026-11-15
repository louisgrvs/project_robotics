"""Tests du jour 4 : lidar (validations j4-1, j4-2) et pose vraie."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from pai_core.sim.sensors import Lidar, LidarConfig, RobotGeometry, true_pose, yaw_from_quat

ARENA = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "arena.xml"


def scene(obstacles=()):
    """Arène + boîtes fixes ; chaque obstacle est (centre_xy, demi_tailles_xy)."""
    spec = mujoco.MjSpec.from_file(str(ARENA))
    for (cx, cy), (sx, sy) in obstacles:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[sx, sy, 0.5], pos=[cx, cy, 0.5])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # calcule geom_xpos des obstacles
    return model, data, RobotGeometry.from_model(model)


def test_geometrie_du_robot():
    _, _, g = scene()
    assert g.wheel_radius == pytest.approx(0.05)
    assert g.wheel_base == pytest.approx(0.24)
    np.testing.assert_allclose(g.axle_offset, [0.00, 0.0, 0.0])


def test_aucun_impact_donne_la_portee_maximale():
    model, data, g = scene()
    d = Lidar(g).scan_raw(model, data)
    assert d.shape == (36,)
    np.testing.assert_allclose(d, 4.0)


def test_mur_a_distance_connue():
    """j4-1 : face à un mur, chaque faisceau concerné mesure d/cos(angle)."""
    face = 1.0  # face avant du mur à x = 1 m ; le lidar est en x = 0
    model, data, g = scene([((face + 0.05, 0.0), (0.05, 3.0))])
    lidar = Lidar(g)
    d = lidar.scan_raw(model, data)
    a = wrap(lidar.angles)
    vus = np.abs(a) < np.radians(60)
    np.testing.assert_allclose(d[vus], face / np.cos(a[vus]), atol=0.01)
    assert np.all(d[np.abs(a) > np.radians(100)] == 4.0)


def test_faisceau_0_devant_et_angles_vers_la_gauche():
    """j4-2 : un obstacle à gauche (+y) est vu par le faisceau 9 (90°)."""
    model, data, g = scene([((0.0, 1.0), (0.1, 0.1))])
    d = Lidar(g).scan_raw(model, data)
    assert np.argmin(d) == 9
    assert d[27] == 4.0 and d[0] == 4.0


def test_le_lidar_tourne_avec_le_robot():
    """Robot tourné de +90° : l'obstacle à gauche passe devant (faisceau 0)."""
    model, data, g = scene([((0.0, 1.0), (0.1, 0.1))])
    a = g.free_qposadr
    data.qpos[a + 3:a + 7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    mujoco.mj_forward(model, data)
    d = Lidar(g).scan_raw(model, data)
    assert np.argmin(d) == 0


def test_bruit_et_pertes():
    model, data, g = scene([((1.05, 0.0), (0.05, 3.0))])
    cfg = LidarConfig(noise_std=0.02)
    lidar = Lidar(g, cfg, rng=np.random.default_rng(0))
    vrai = lidar.scan_raw(model, data)
    mesures = np.array([lidar.scan(model, data) for _ in range(500)])
    ecarts = (mesures - vrai)[:, vrai < 3.5]  # faisceaux qui voient le mur
    assert np.std(ecarts) == pytest.approx(0.02, rel=0.1)

    cfg = LidarConfig(dropout_prob=0.2)
    lidar = Lidar(g, cfg, rng=np.random.default_rng(0))
    perdus = np.mean([lidar.scan(model, data)[0] == 4.0 for _ in range(2000)])
    assert perdus == pytest.approx(0.2, abs=0.03)


def test_pose_vraie_au_milieu_de_l_essieu():
    model, data, g = scene()
    a = g.free_qposadr
    yaw = 0.7
    data.qpos[a:a + 3] = [1.0, 2.0, 0.05]
    data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    x, y, th = true_pose(data, g)
    assert th == pytest.approx(yaw)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)


def test_lacet_du_quaternion():
    for yaw in np.linspace(-3.1, 3.1, 13):
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        assert yaw_from_quat(q) == pytest.approx(yaw)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
