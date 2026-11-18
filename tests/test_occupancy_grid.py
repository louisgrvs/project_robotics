"""Tests de pai_core.perception.occupancy_grid, sans ROS ni MuJoCo.

Les distances des scans synthétiques visent le MILIEU d'une case (x = 1,025 m
plutôt que 1 m) : un point exactement sur une frontière peut tomber d'un côté
ou de l'autre selon l'arrondi en virgule flottante.
"""
import itertools

import numpy as np
import pytest

from pai_core.perception.occupancy_grid import (GridConfig, OccupancyGrid, bresenham, logodds,
                                                prob)

RES = 0.05


def make_grid(**kw):
    return OccupancyGrid(GridConfig.for_arena(6.0, **kw))


# ---------------------------------------------------------------- fonctions libres

def test_logodds_prob_inverse():
    p = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(prob(logodds(p)), p, atol=1e-12)
    assert logodds(0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("di,dj", [(7, 3), (3, 7), (-7, 3), (-3, 7), (7, -3), (3, -7),
                                   (-7, -3), (-3, -7), (5, 0), (0, 5), (-5, 0), (0, -5),
                                   (4, 4), (-4, 4), (0, 0)])
def test_bresenham_all_octants(di, dj):
    i0, j0 = 10, 20
    cells = bresenham(i0, j0, i0 + di, j0 + dj)
    assert tuple(cells[0]) == (i0, j0)                       # départ inclus
    assert tuple(cells[-1]) == (i0 + di, j0 + dj)            # arrivée incluse
    steps = np.abs(np.diff(cells, axis=0))
    assert np.all(steps.max(axis=1) == 1)                     # cases 8-connexes, sans répétition
    assert len(cells) == max(abs(di), abs(dj)) + 1            # nombre minimal de cases


# ---------------------------------------------------------------- conversions

def test_grid_size_and_margin():
    g = make_grid()
    assert (g.nx, g.ny) == (128, 128)
    np.testing.assert_allclose(g.origin, [-3.2, -3.2])
    # La face intérieure des murs (x = ±3) doit être DANS la grille.
    assert g.in_bounds(g.world_to_cell([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])).all()


def test_origin_is_a_corner():
    g = make_grid()
    assert tuple(g.world_to_cell([-3.2 + 1e-9, -3.2 + 1e-9])[0]) == (0, 0)
    np.testing.assert_allclose(g.cell_center([[0, 0]]), [[-3.2 + RES / 2, -3.2 + RES / 2]])


def test_world_cell_round_trip():
    g = make_grid()
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 128, size=(200, 2))
    np.testing.assert_array_equal(g.world_to_cell(g.cell_center(cells)), cells)


def test_out_of_grid_is_not_clamped():
    g = make_grid()
    c = g.world_to_cell([[10.0, 0.0]])
    assert not g.in_bounds(c).any()


# ---------------------------------------------------------------- mise à jour

def test_single_beam_wall():
    """Lidar en (0,0,0), faisceau vers +x, impact à 1,025 m."""
    g = make_grid()
    g.update((0.0, 0.0, 0.0), [0.0], [1.025], max_range=4.0)

    hit = tuple(g.world_to_cell([1.025, 0.0])[0])
    assert g.log_odds[hit] == pytest.approx(g.l_occ)
    for x in np.arange(0.025, 1.0, RES):                      # cases traversées : libres
        c = tuple(g.world_to_cell([x, 0.0])[0])
        assert g.log_odds[c] == pytest.approx(g.l_free), x
    beyond = tuple(g.world_to_cell([1.525, 0.0])[0])          # au-delà : inconnu
    assert not g.observed[beyond] and g.log_odds[beyond] == 0.0


def test_rotated_lidar_same_cell():
    """Même mur vu par un lidar tourné de +90°, faisceau à −90° : même case."""
    g1, g2 = make_grid(), make_grid()
    g1.update((0.0, 0.0, 0.0), [0.0], [1.025], max_range=4.0)
    g2.update((0.0, 0.0, np.pi / 2), [-np.pi / 2], [1.025], max_range=4.0)
    np.testing.assert_array_equal(g1.occupied(), g2.occupied())
    np.testing.assert_allclose(g1.log_odds, g2.log_odds)


def test_translated_lidar():
    g = make_grid()
    g.update((1.0, -1.0, np.pi), [0.0], [0.525], max_range=4.0)   # regarde vers −x
    hit = tuple(g.world_to_cell([1.0 - 0.525, -1.0])[0])
    assert g.log_odds[hit] == pytest.approx(g.l_occ)


@pytest.mark.parametrize("r", [np.inf, np.nan, 4.0, 7.0])
def test_no_return_occupies_nothing(r):
    g = make_grid()
    g.update((0.0, 0.0, 0.0), [0.0], [r], max_range=4.0)
    assert not (g.log_odds > 0).any()
    assert g.observed[tuple(g.world_to_cell([3.0, 0.0])[0])]        # mais libère le chemin


def test_no_return_ignored_when_disabled():
    g = make_grid(free_on_no_return=False)
    g.update((0.0, 0.0, 0.0), [0.0], [np.inf], max_range=4.0)
    assert not g.observed.any()


def test_below_range_min_ignored():
    g = make_grid()
    g.update((0.0, 0.0, 0.0), [0.0], [0.02], max_range=4.0, range_min=0.05)
    assert not g.observed.any()


def test_one_update_per_cell_per_scan():
    """36 faisceaux partent de la même case : elle ne reçoit qu'UNE mise à jour."""
    g = make_grid()
    angles = 2 * np.pi * np.arange(36) / 36
    g.update((0.025, 0.025, 0.0), angles, np.full(36, 1.0), max_range=4.0)
    origin = tuple(g.world_to_cell([0.025, 0.025])[0])
    assert g.log_odds[origin] == pytest.approx(g.l_free)


def test_hit_wins_over_free_in_same_scan():
    """Un faisceau touche la case C, un autre la traverse : C reste occupée."""
    g = make_grid()
    g.update((0.0, 0.0, 0.0), [0.0, 0.0], [1.025, 2.025], max_range=4.0)
    c = tuple(g.world_to_cell([1.025, 0.0])[0])
    assert g.log_odds[c] == pytest.approx(g.l_occ)


def test_saturation():
    g = make_grid()
    for _ in range(100):
        g.update((0.0, 0.0, 0.0), [0.0], [1.025], max_range=4.0)
    hit = tuple(g.world_to_cell([1.025, 0.0])[0])
    near = tuple(g.world_to_cell([0.525, 0.0])[0])
    assert g.log_odds[hit] == pytest.approx(g.cfg.l_max)
    assert g.log_odds[near] == pytest.approx(g.cfg.l_min)


def test_circle_room():
    """Scan complet dans une « pièce » circulaire de rayon 2 m : le cercle est occupé, l'intérieur libre."""
    g = make_grid()
    angles = 2 * np.pi * np.arange(360) / 360
    for _ in range(3):
        g.update((0.0, 0.0, 0.0), angles, np.full(360, 2.0), max_range=4.0)
    centers = g.cell_center(np.array(list(itertools.product(range(g.nx), range(g.ny)))))
    dist = np.hypot(centers[:, 0], centers[:, 1]).reshape(g.nx, g.ny)
    occ = g.occupied()
    assert occ.any()
    assert np.all(np.abs(dist[occ] - 2.0) < 2 * RES)          # toutes les cases occupées sont sur le cercle
    assert g.free()[dist < 1.8].mean() > 0.95                  # l'intérieur est libre


# ---------------------------------------------------------------- sortie ROS

def test_ros_order():
    """Une case occupée en (ix, iy) = (3, 7) se retrouve à l'indice 3 + 7·width."""
    g = make_grid()
    g.log_odds[3, 7] = g.cfg.l_max
    g.observed[3, 7] = True
    v = g.to_ros_values()
    assert v.dtype == np.int8 and v.shape == (g.nx * g.ny,)
    assert v[3 + 7 * g.nx] == 97
    assert (v == -1).sum() == g.nx * g.ny - 1


def test_ros_values_range():
    g = make_grid()
    g.update((0.0, 0.0, 0.0), [0.0, 1.0], [1.025, np.inf], max_range=4.0)
    v = g.to_ros_values()
    assert v.min() >= -1 and v.max() <= 100
    assert v[v >= 0].size == g.observed.sum()
