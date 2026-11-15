"""E1 : comment dérive l'odométrie ?

Le robot suit une même séquence de commandes (graine fixée), avec des
changements brusques qui provoquent du glissement. On enregistre les angles
des encodeurs à 50 Hz et la pose vraie, pour trois frottements. L'odométrie
est calculée ensuite, à partir des seuls encodeurs, comme sur un vrai robot.
Deux courbes de plus reprennent l'essai μ = 1 avec un entraxe faussé de ±2 %
dans l'odométrie : une erreur systématique, au sens de Borenstein et Feng.

Usage : python scripts/exp_e1_odometry.py [--duree 60] [--graine 0] [--montrer]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from pai_core.robot.odometry import WheelOdometry, wrap_angle
from pai_core.sim.sensors import RobotGeometry, WheelEncoders, true_pose

RACINE = Path(__file__).resolve().parents[1]
ARENA = RACINE / "assets" / "mjcf" / "arena.xml"
ODOM_HZ = 50
STABILISATION = 1.5  # s à commande nulle avant de démarrer l'odométrie

FROTTEMENTS = (0.2, 0.5, 1.0)
CAS = [  # (étiquette, frottement, facteur sur l'entraxe que croit l'odométrie)
    ("μ = 0,2", 0.2, 1.00),
    ("μ = 0,5", 0.5, 1.00),
    ("μ = 1,0", 1.0, 1.00),
    ("μ = 1,0, entraxe +2 %", 1.0, 1.02),
    ("μ = 1,0, entraxe −2 %", 1.0, 0.98),
]


def sequence_de_commandes(duree: float, rng: np.random.Generator):
    """Segments de 1 à 3 s : ligne droite, virage ou rotation sur place.

    Renvoie une liste de (instant de fin, vitesse roue gauche, vitesse roue droite).
    """
    segments, t = [], 0.0
    while t < duree:
        t += rng.uniform(1.0, 3.0)
        v = rng.uniform(3.0, 8.0)
        kind = rng.choice(["droite", "virage", "rotation"], p=[0.4, 0.4, 0.2])
        if kind == "droite":
            w = (v, v)
        elif kind == "virage":
            k = rng.uniform(0.2, 0.7)  # rapport des vitesses des roues
            w = (k * v, v) if rng.random() < 0.5 else (v, k * v)
        else:
            w = (-v, v) if rng.random() < 0.5 else (v, -v)
        segments.append((t, *w))
    return segments


def regler_frottement(model: mujoco.MjModel, mu: float) -> None:
    """Frottement de glissement des roues ET du sol.

    MuJoCo combine les deux géométries en contact en prenant le maximum :
    changer seulement les roues ne changerait rien si le sol reste à 1.
    """
    roues = [model.body_geomadr[model.body(n).id] for n in ("roue_g", "roue_d")]
    plans = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    for g in [*roues, *plans]:
        model.geom_friction[g, 0] = mu


def simuler(mu: float, segments, duree: float) -> dict:
    """Une simulation : pose vraie et angles des encodeurs à ODOM_HZ."""
    model = mujoco.MjModel.from_xml_path(str(ARENA))
    regler_frottement(model, mu)
    data = mujoco.MjData(model)
    geom = RobotGeometry.from_model(model)
    enc = WheelEncoders(geom)

    # Stabilisation : le robot se pose sur le sol avant toute mesure.
    while data.time < STABILISATION:
        mujoco.mj_step(model, data)
    t0 = data.time

    pas_par_mesure = round(1.0 / (ODOM_HZ * model.opt.timestep))
    n = int(duree * ODOM_HZ)
    vrai, phi = np.empty((n + 1, 3)), np.empty((n + 1, 2))
    vrai[0], phi[0] = true_pose(data, geom), enc.read(data)

    i_seg = 0
    for k in range(1, n + 1):
        for _ in range(pas_par_mesure):
            t = data.time - t0
            while i_seg < len(segments) - 1 and t >= segments[i_seg][0]:
                i_seg += 1
            data.ctrl[:] = segments[i_seg][1:]  # ordre des actionneurs : gauche, droite
            mujoco.mj_step(model, data)
        vrai[k], phi[k] = true_pose(data, geom), enc.read(data)

    return {"temps": np.arange(n + 1) / ODOM_HZ, "vrai": vrai, "encodeurs": phi,
            "rayon": geom.wheel_radius, "entraxe": geom.wheel_base}


def odometrie(sim: dict, facteur_entraxe: float = 1.0) -> dict:
    """Odométrie calculée à partir des seuls encodeurs, puis comparée à la vérité."""
    vrai, phi = sim["vrai"], sim["encodeurs"]
    odo = WheelOdometry(sim["rayon"], sim["rayon"], facteur_entraxe * sim["entraxe"])
    odo.reset(pose=vrai[0], wheel_angles=phi[0])
    estime = np.vstack([vrai[0], [odo.update(p) for p in phi[1:]]])
    pas = np.linalg.norm(np.diff(vrai[:, :2], axis=0), axis=1)
    return {
        "temps": sim["temps"],
        "distance": np.concatenate([[0.0], np.cumsum(pas)]),
        "vrai": vrai,
        "estime": estime,
        "err_pos": np.linalg.norm(estime[:, :2] - vrai[:, :2], axis=1),
        "err_cap": wrap_angle(estime[:, 2] - vrai[:, 2]),
    }


def commit_courant() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RACINE,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


def tracer(resultats: dict, chemin: Path, titre: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    deja_traces = []
    for i, (nom, r) in enumerate(resultats.items()):
        c = f"C{i}"
        style = "--" if "entraxe" in nom else "-"
        ax[0].plot(r["distance"], r["err_pos"], style, color=c, label=nom)
        ax[1].plot(r["temps"], np.degrees(r["err_cap"]), style, color=c, label=nom)
        # Les essais à entraxe faussé partagent la simulation μ = 1 : la vraie
        # trajectoire n'est tracée qu'une fois.
        if not any(r["vrai"] is v for v in deja_traces):
            ax[2].plot(r["vrai"][:, 0], r["vrai"][:, 1], "-", color=c, lw=1.4)
            deja_traces.append(r["vrai"])
        ax[2].plot(r["estime"][:, 0], r["estime"][:, 1], ":", color=c, lw=1.4)

    ax[0].set(xlabel="distance parcourue (m)", ylabel="erreur de position (m)",
              title="Erreur de position")
    ax[1].set(xlabel="temps (s)", ylabel="erreur de cap (°)",
              title="Erreur de cap (les rotations sur place n'avancent pas)")
    ax[2].set(xlabel="x (m)", ylabel="y (m)", title="Trajectoires : vraie (trait), odométrie (pointillés)")
    ax[2].set_aspect("equal", adjustable="datalim")
    for a in ax[:2]:
        a.legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(titre)
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--duree", type=float, default=60.0, help="durée de chaque essai (s)")
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--figure", type=Path, default=RACINE / "docs" / "figures" / "e1_odometrie.png")
    p.add_argument("--resultats", type=Path, default=RACINE / "results" / "e1_odometrie.npz")
    p.add_argument("--montrer", action="store_true", help="afficher la figure à l'écran")
    args = p.parse_args()

    segments = sequence_de_commandes(args.duree, np.random.default_rng(args.graine))
    sims = {mu: simuler(mu, segments, args.duree) for mu in FROTTEMENTS}
    resultats = {}
    print(f"{'cas':<24}{'distance':>10}{'err. finale':>13}{'err./dist.':>12}{'cap final':>11}")
    for nom, mu, f in CAS:
        r = odometrie(sims[mu], f)
        resultats[nom] = r
        d, e = r["distance"][-1], r["err_pos"][-1]
        print(f"{nom:<24}{d:>9.2f} m{e:>11.3f} m{100 * e / d:>11.2f} %"
              f"{np.degrees(r['err_cap'][-1]):>9.1f} °")

    commit = commit_courant()
    titre = f"E1 : dérive de l'odométrie (graine {args.graine}, {args.duree:.0f} s, commit {commit})"
    tracer(resultats, args.figure, titre)

    args.resultats.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.resultats, graine=args.graine, duree=args.duree, commit=commit,
             cas=np.array([c[0] for c in CAS]),
             **{f"{k}_{i}": r[k] for i, r in enumerate(resultats.values())
                for k in ("temps", "distance", "vrai", "estime", "err_pos", "err_cap")})
    print(f"figure : {args.figure}\nrésultats : {args.resultats}")
    if args.montrer:
        plt.show()


if __name__ == "__main__":
    main()
