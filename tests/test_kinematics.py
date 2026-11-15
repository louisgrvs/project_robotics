"""Tests de la cinématique différentielle (pai_core.robot.diff_drive)."""

from pathlib import Path

import numpy as np
import pytest

from pai_core.robot.diff_drive import (
    DiffDriveParams,
    forward,
    inverse,
    saturate,
    saturate_wheels,
    turning_radius,
)

# Valeurs de assets/mjcf/robot.xml : roues de 5 cm de rayon, à ±12 cm du plan
# médian, ctrlrange de ±10 rad/s. test_robot_du_projet vérifie la concordance.
P = DiffDriveParams(wheel_radius=0.05, wheel_base=0.24, max_wheel_speed=10.0)
RACINE = Path(__file__).resolve().parents[1]
TOL = 1e-12


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def commandes_aleatoires(rng, n=1000):
    """(v, ω) tirés bien au-delà des limites, pour couvrir aussi les cas saturés."""
    return rng.uniform(-1.0, 1.0, n), rng.uniform(-8.0, 8.0, n)


# ---------------------------------------------------------------- cas connus

def test_ligne_droite():
    v, w = forward(5.0, 5.0, P)
    assert v == pytest.approx(0.25)  # r·ω
    assert w == 0.0


def test_rotation_sur_place_dans_le_sens_trigonometrique():
    v, w = forward(-5.0, 5.0, P)  # gauche en arrière, droite en avant
    assert v == 0.0
    assert w == pytest.approx(2 * 0.05 * 5.0 / 0.24)  # 2rω/L, positif (REP 103)


def test_virage_a_gauche():
    v, w = forward(3.0, 6.0, P)  # la troisième manœuvre de robot_mouvement_test.py
    assert v > 0 and w > 0
    assert turning_radius(3.0, 6.0, P) == pytest.approx(0.36)  # (L/2)·9/3


def test_vitesses_maximales():
    assert P.max_linear_speed == pytest.approx(0.5)
    assert P.max_yaw_rate == pytest.approx(2 * 0.05 * 10.0 / 0.24)


# ------------------------------------------------- directe et inverse

def test_inverse_puis_directe_donne_l_identite(rng):
    v, w = commandes_aleatoires(rng)
    v2, w2 = forward(*inverse(v, w, P), P)
    np.testing.assert_allclose(v2, v, rtol=0, atol=TOL)
    np.testing.assert_allclose(w2, w, rtol=0, atol=TOL)


def test_directe_puis_inverse_donne_l_identite(rng):
    omega_l, omega_r = rng.uniform(-15.0, 15.0, (2, 1000))
    omega_l2, omega_r2 = inverse(*forward(omega_l, omega_r, P), P)
    np.testing.assert_allclose(omega_l2, omega_l, rtol=0, atol=TOL)
    np.testing.assert_allclose(omega_r2, omega_r, rtol=0, atol=TOL)


def test_diffusion_des_formes():
    v = np.linspace(-0.5, 0.5, 7)[:, None]  # (7, 1)
    w = np.linspace(-2.0, 2.0, 5)[None, :]  # (1, 5)
    omega_l, omega_r = inverse(v, w, P)
    assert omega_l.shape == omega_r.shape == (7, 5)
    vs, ws = saturate(v, w, P)
    assert vs.shape == ws.shape == (7, 5)


# ------------------------------------------------------ rayon de courbure

def test_rayon_egal_a_v_sur_omega(rng):
    omega_l, omega_r = rng.uniform(-10.0, 10.0, (2, 1000))
    v, w = forward(omega_l, omega_r, P)
    np.testing.assert_allclose(turning_radius(omega_l, omega_r, P), v / w, rtol=1e-9)


def test_rayon_cas_limites():
    assert turning_radius(5.0, 5.0, P) == np.inf      # ligne droite en avant
    assert turning_radius(-5.0, -5.0, P) == -np.inf   # ligne droite en arrière
    assert turning_radius(-5.0, 5.0, P) == 0.0        # rotation sur place
    assert np.isnan(turning_radius(0.0, 0.0, P))      # arrêt : rayon indéfini


# ------------------------------------------------------------- saturation

def test_saturation_sans_effet_sous_la_limite():
    assert saturate_wheels(4.0, -7.0, P) == (4.0, -7.0)
    assert saturate(0.2, 1.0, P) == (0.2, 1.0)


def test_commande_nulle():
    assert saturate(0.0, 0.0, P) == (0.0, 0.0)


def test_saturation_des_roues_preserve_la_courbure():
    omega_l, omega_r = saturate_wheels(6.0, 12.0, P)
    assert max(abs(omega_l), abs(omega_r)) == pytest.approx(10.0)
    assert turning_radius(omega_l, omega_r, P) == pytest.approx(0.36)
    # Contre-exemple : écrêter chaque roue séparément change le rayon.
    assert turning_radius(6.0, 10.0, P) == pytest.approx(0.48)


def test_rotation_pure_reste_pure():
    vs, ws = saturate(0.0, 10.0, P)
    assert vs == 0.0
    assert ws == pytest.approx(P.max_yaw_rate)


def test_saturation_en_lot(rng):
    v, w = commandes_aleatoires(rng)
    vs, ws = saturate(v, w, P)

    # 1. Les roues respectent la limite.
    pic = np.maximum(*np.abs(inverse(vs, ws, P)))
    assert np.all(pic <= P.max_wheel_speed + TOL)

    # 2. (vs, ws) = s·(v, w) avec 0 < s ≤ 1 : même direction, donc même courbure.
    np.testing.assert_allclose(vs * w - ws * v, 0.0, atol=TOL)
    s = np.where(np.abs(v) > np.abs(w), vs / v, ws / w)
    assert np.all((s > 0) & (s <= 1 + TOL))

    # 3. On ne ralentit pas plus que nécessaire : une commande saturée touche la limite.
    trop_rapide = np.maximum(*np.abs(inverse(v, w, P))) > P.max_wheel_speed
    assert trop_rapide.any() and (~trop_rapide).any()  # les deux cas sont couverts
    np.testing.assert_allclose(pic[trop_rapide], P.max_wheel_speed, rtol=1e-12)
    np.testing.assert_array_equal(vs[~trop_rapide], v[~trop_rapide])


def test_bornes_v_max_et_w_max(rng):
    v, w = commandes_aleatoires(rng)
    vs, ws = saturate(v, w, P, v_max=0.3, w_max=1.5)
    assert np.all(np.abs(vs) <= 0.3 + TOL)
    assert np.all(np.abs(ws) <= 1.5 + TOL)
    np.testing.assert_allclose(vs * w - ws * v, 0.0, atol=TOL)


# ------------------------------------------------------------- paramètres

@pytest.mark.parametrize("champ", ["wheel_radius", "wheel_base", "max_wheel_speed"])
@pytest.mark.parametrize("valeur", [0.0, -1.0, np.nan, np.inf])
def test_parametres_invalides(champ, valeur):
    kwargs = dict(wheel_radius=0.05, wheel_base=0.24, max_wheel_speed=10.0)
    kwargs[champ] = valeur
    with pytest.raises(ValueError):
        DiffDriveParams(**kwargs)


MJCF_TEST = """
<mujoco>
  <worldbody>
    <body name="chassis" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.03"/>
      <body name="roue_g" pos="0 {y} 0">
        <joint name="g" type="hinge" axis="0 {axe} 0"/>
        <geom type="cylinder" size="0.07 0.01" euler="90 0 0"/>
      </body>
      <body name="roue_d" pos="0 -{y} 0">
        <joint name="d" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.07 0.01" euler="90 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="g" kv="1" ctrlrange="-8 8"/>
    <velocity joint="d" kv="1" ctrlrange="-8 8"/>
  </actuator>
</mujoco>
"""


def test_parametres_lus_dans_un_mjcf():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(MJCF_TEST.format(y=0.15, axe=1))
    p = DiffDriveParams.from_mujoco(model)
    assert p.wheel_radius == pytest.approx(0.07)
    assert p.wheel_base == pytest.approx(0.30)
    assert p.max_wheel_speed == pytest.approx(8.0)


def test_axe_de_roue_inverse_detecte():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(MJCF_TEST.format(y=0.15, axe=-1))
    with pytest.raises(ValueError, match="axe"):
        DiffDriveParams.from_mujoco(model)


def test_robot_du_projet():
    mujoco = pytest.importorskip("mujoco")
    chemin = RACINE / "assets" / "mjcf" / "robot.xml"
    if not chemin.exists():
        pytest.skip(f"{chemin} introuvable")
    p = DiffDriveParams.from_mujoco(mujoco.MjModel.from_xml_path(str(chemin)))
    assert p.wheel_radius == pytest.approx(P.wheel_radius)
    assert p.wheel_base == pytest.approx(P.wheel_base)
    assert p.max_wheel_speed == pytest.approx(P.max_wheel_speed)
