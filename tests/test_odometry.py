"""Tests du jour 4 : odométrie par le point milieu (validation j4-3)."""

import numpy as np
import pytest

from pai_core.robot.odometry import WheelOdometry, arc_step, midpoint_step

R, L = 0.05, 0.24


def test_ligne_droite():
    odo = WheelOdometry(R, R, L)
    odo.reset(wheel_angles=[0.0, 0.0])
    x, y, th = odo.update([10.0, 10.0])
    assert (x, y, th) == pytest.approx((R * 10.0, 0.0, 0.0))


def test_rotation_sur_place():
    odo = WheelOdometry(R, R, L)
    odo.reset(wheel_angles=[0.0, 0.0])
    x, y, th = odo.update([-1.0, 1.0])  # gauche en arrière, droite en avant
    assert (x, y) == pytest.approx((0.0, 0.0))
    assert th == pytest.approx(2 * R * 1.0 / L)  # sens trigonométrique


@pytest.mark.parametrize("dth", [0.4, 0.2, 0.1, 0.05])
def test_erreur_locale_du_point_milieu(dth):
    """j4-3 : erreur locale = Δs·(1 − sinc(Δθ/2)) ≈ Δs·Δθ²/24."""
    rayon = 1.0
    ds = rayon * dth                       # longueur d'arc parcourue
    ds_l = (rayon - L / 2) * dth           # roue intérieure (gauche)
    ds_r = (rayon + L / 2) * dth           # roue extérieure
    p0 = np.array([0.3, -0.2, 0.8])
    err = np.linalg.norm(midpoint_step(p0, ds_l, ds_r, L)[:2] - arc_step(p0, ds_l, ds_r, L)[:2])
    exacte = ds * (1 - np.sin(dth / 2) / (dth / 2))
    assert err == pytest.approx(exacte, rel=1e-9)
    assert err == pytest.approx(ds * dth**2 / 24, rel=dth**2 / 30 + 1e-6)


def test_demi_cercle_ordre_2():
    """Sur un demi-tour, l'erreur finale est divisée par 4 quand Δθ est divisé par 2.

    Les erreurs locales sont toutes tangentes au cercle : elles ne
    s'accumulent pas comme une marche aléatoire. Les points de l'odométrie
    tombent exactement sur un cercle de rayon R / sinc(Δθ/2), légèrement
    trop grand, d'où une erreur globale ≈ R·Δθ²/12 après un demi-tour.
    (Sur un tour complet, elles s'annulent et l'odométrie se referme.)
    """
    rayon = 1.0
    erreurs = []
    for n in (50, 100, 200):
        dth = np.pi / n
        odo = WheelOdometry(R, R, L)
        odo.reset(pose=(0.0, 0.0, 0.0), wheel_angles=[0.0, 0.0])
        phi = np.zeros(2)
        for _ in range(n):
            phi += np.array([(rayon - L / 2), (rayon + L / 2)]) * dth / R
            pose = odo.update(phi)
        err = np.linalg.norm(pose[:2] - [0.0, 2 * rayon])  # arrivée exacte : (0, 2R)
        assert err == pytest.approx(rayon * dth**2 / 12, rel=0.01)
        erreurs.append(err)
    assert erreurs[0] / erreurs[1] == pytest.approx(4.0, rel=0.01)
    assert erreurs[1] / erreurs[2] == pytest.approx(4.0, rel=0.01)
