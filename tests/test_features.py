"""Tests de pai_core.perception.features (sans MuJoCo)."""
import numpy as np
import pytest

from pai_core.geometry.se2 import wrap_angle
from pai_core.perception.features import (N_BEAMS, OBS_DIM, SLICES, ObsConfig, build_observation,
                                          lidar_angles, preprocess_ranges, split_obs)

CFG = ObsConfig()


def test_lidar_angles_convention():
    a = lidar_angles(36)
    assert a[0] == 0.0                                # faisceau 0 vers l'avant
    assert a[9] == pytest.approx(np.pi / 2)           # faisceau 9 vers la gauche


@pytest.mark.parametrize("no_return", [np.inf, np.nan, CFG.range_max, 10.0])
def test_preprocess_no_return_is_one(no_return):
    r = np.full(N_BEAMS, 2.0)
    r[5] = no_return
    out = preprocess_ranges(r, CFG)
    assert out.dtype == np.float32 and out.shape == (N_BEAMS,)
    assert out[5] == 1.0 and out[0] == pytest.approx(0.5)
    assert np.isfinite(out).all()


def test_preprocess_negative_is_zero():
    r = np.full(N_BEAMS, 1.0)
    r[0] = -0.1
    assert preprocess_ranges(r, CFG)[0] == 0.0


def test_preprocess_wrong_shape():
    with pytest.raises(ValueError):
        preprocess_ranges(np.ones(N_BEAMS - 1), CFG)


def test_inf_and_range_max_give_same_observation():
    """Garde-fou du test d'identité du jour 10 : ROS (inf) et Gym (range_max) voient la même chose."""
    rng = np.random.default_rng(0)
    r = rng.uniform(0.2, 3.9, N_BEAMS)
    lost = rng.random(N_BEAMS) < 0.3
    r_gym, r_ros = r.copy(), r.copy()
    r_gym[lost], r_ros[lost] = CFG.range_max, np.inf
    args = ((1.0, -0.5, 0.3), (0.2, -0.4), (2.0, 1.5), np.array([0.1, -0.2]))
    np.testing.assert_array_equal(build_observation(r_gym, *args, CFG),
                                  build_observation(r_ros, *args, CFG))


def test_split_obs_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        ranges = rng.uniform(0.1, 3.9, N_BEAMS)
        pose = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-np.pi, np.pi))
        goal = (rng.uniform(-2, 2), rng.uniform(-2, 2))            # distance < goal_dist_max
        v, w = rng.uniform(-0.5, 0.5), rng.uniform(-2, 2)          # dans les bornes d'écrêtage
        obs = build_observation(ranges, pose, (v, w), goal, np.zeros(2), CFG)
        assert obs.shape == (OBS_DIM,)

        dist, d_goal, bearing, v2, w2 = split_obs(obs, CFG)
        np.testing.assert_allclose(dist, ranges, rtol=1e-6)        # float32 : précision relative ~1e-7
        true_d = np.hypot(goal[0] - pose[0], goal[1] - pose[1])
        true_b = np.arctan2(goal[1] - pose[1], goal[0] - pose[0]) - pose[2]
        assert d_goal == pytest.approx(true_d, rel=1e-5)
        assert wrap_angle(bearing - true_b) == pytest.approx(0.0, abs=1e-5)
        assert (v2, w2) == pytest.approx((v, w), abs=1e-6)


def test_split_obs_saturates():
    """Au-delà des bornes, l'inversion sature : c'est attendu et documenté."""
    obs = build_observation(np.full(N_BEAMS, 1.0), (0, 0, 0), (5.0, 0.0), (100.0, 0.0),
                            np.zeros(2), CFG)
    _, d_goal, _, v, _ = split_obs(obs, CFG)
    assert d_goal == pytest.approx(CFG.goal_dist_max)
    assert v == pytest.approx(2.0 * CFG.v_max)
    assert obs[SLICES["goal"]][0] == 1.0
