"""Tests de pai_core.geometry.se2 : propriétés de groupe et accord avec les matrices."""
import numpy as np
import pytest

from pai_core.geometry.se2 import (between, compose, from_matrix, inverse, to_matrix,
                                   transform_points, wrap_angle)

RNG = np.random.default_rng(0)
N = 200


def random_poses(n=N):
    return np.column_stack([RNG.uniform(-5, 5, n), RNG.uniform(-5, 5, n),
                            RNG.uniform(-np.pi, np.pi, n)])


def assert_pose_close(p, q, tol=1e-9):
    """Compare deux poses ; l'angle via wrap(θ1 − θ2), car π et −π sont le même angle."""
    p, q = np.broadcast_arrays(np.asarray(p, dtype=float), np.asarray(q, dtype=float))
    np.testing.assert_allclose(p[..., :2], q[..., :2], atol=tol)
    np.testing.assert_allclose(wrap_angle(p[..., 2] - q[..., 2]), 0.0, atol=tol)


def test_wrap_angle_range():
    a = wrap_angle(np.linspace(-20, 20, 1001))
    assert np.all(a >= -np.pi) and np.all(a < np.pi)
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)


def test_inverse_is_neutral():
    p = random_poses()
    assert_pose_close(compose(p, inverse(p)), np.zeros(3))
    assert_pose_close(compose(inverse(p), p), np.zeros(3))


def test_inverse_involution():
    p = random_poses()
    assert_pose_close(inverse(inverse(p)), p)


def test_associativity():
    a, b, c = random_poses(), random_poses(), random_poses()
    assert_pose_close(compose(compose(a, b), c), compose(a, compose(b, c)))


def test_matches_matrices():
    for a, b in zip(random_poses(50), random_poses(50)):
        assert_pose_close(compose(a, b), from_matrix(to_matrix(a) @ to_matrix(b)))
        assert_pose_close(inverse(a), from_matrix(np.linalg.inv(to_matrix(a))))
        assert_pose_close(from_matrix(to_matrix(a)), a)


def test_map_odom_chain():
    """Le cas de sim_bridge : (map→odom) ∘ (odom→base) redonne la vérité terrain."""
    gt, od = random_poses(), random_poses()
    map_odom = compose(gt, inverse(od))
    assert_pose_close(compose(map_odom, od), gt)


def test_between():
    a, b = random_poses(), random_poses()
    assert_pose_close(compose(a, between(a, b)), b)


def test_transform_points_example():
    """Robot en (2, 1) tourné de 90° : un point à 1 m devant lui est en (2, 2)."""
    out = transform_points([2.0, 1.0, np.pi / 2], [[1.0, 0.0]])
    np.testing.assert_allclose(out, [[2.0, 2.0]], atol=1e-12)


def test_transform_points_matches_compose():
    pose = random_poses(1)[0]
    pts = RNG.uniform(-3, 3, (20, 2))
    as_poses = np.column_stack([pts, np.zeros(len(pts))])
    np.testing.assert_allclose(transform_points(pose, pts), compose(pose, as_poses)[:, :2], atol=1e-12)
