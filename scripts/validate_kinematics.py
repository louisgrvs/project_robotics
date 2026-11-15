"""Validation de la cinématique du robot simulé et étude paramétrique (jour 3).

Trois manœuvres à commande constante sont comparées au modèle cinématique de
``pai_core.robot.diff_drive`` : ligne droite, rotation sur place et cercle.
Chaque écart est décomposé en deux parts :

* l'erreur de l'actionneur : vitesse de roue mesurée contre vitesse demandée ;
* l'erreur cinématique : mouvement du robot contre ce que prédisent les
  vitesses de roue *mesurées* (glissement, écrasement du contact…).

Les grandeurs sont mesurées au milieu de l'essieu (milieu des centres des deux
roues), quel que soit l'endroit où se trouve l'origine du corps ``chassis``.

Usage, depuis la racine du dépôt :

    python scripts/validate_kinematics.py              # tout (≈ 1 min)
    python scripts/validate_kinematics.py --sans-etude # validation seule
    python scripts/validate_kinematics.py --scene autre_arena.xml   # autre modèle

Sorties : tableau dans le terminal, figure docs/figures/glissement_mu.png et
résultats chiffrés dans results/validate_kinematics.json (avec le commit).
Code de retour 1 si une validation échoue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from pai_core.robot.diff_drive import DiffDriveParams, forward, turning_radius

RACINE = Path(__file__).resolve().parents[1]
SCENE = RACINE / "assets" / "mjcf" / "arena.xml"
FIGURES = RACINE / "docs" / "figures"
RESULTATS = RACINE / "results"

# Manœuvres (ω_L, ω_R) en rad/s ; la troisième est celle de robot_mouvement_test.py.
MANOEUVRES = {
    "ligne droite": (5.0, 5.0),
    "rotation sur place": (-5.0, 5.0),
    "cercle": (3.0, 6.0),
}
DUREE = 4.0           # durée de chaque manœuvre (s)
DEBUT_MESURE = 2.0    # on mesure sur [DEBUT_MESURE, DUREE] : régime établi
SEUILS = {"ligne droite": 0.02, "rotation sur place": 0.03, "cercle": 0.03}
SEUIL_PAS_DE_TEMPS = 0.01


@dataclass(frozen=True)
class SimConfig:
    """Paramètres physiques étudiés."""

    mu: float = 1.0                    # frottement de glissement roues–sol
    kv: float = 1.0                    # gain des actionneurs velocity (N·m·s/rad)
    dt: float = 0.002                  # pas de temps (s)
    integrator: str = "implicitfast"   # "euler", "implicitfast", "implicit" ou "rk4"
    scene: str = str(SCENE)            # fichier MJCF de la scène

    def etiquette(self) -> str:
        return f"μ={self.mu:g}, kv={self.kv:g}, Δt={self.dt * 1e3:g} ms, {self.integrator}"


NOMINAL = SimConfig()  # remplacé dans main() si --scene est donné

INTEGRATEURS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
}


# ------------------------------------------------------------------ simulation

class Robot:
    """Modèle MuJoCo configuré, avec les indices utiles à la mesure."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        m = mujoco.MjModel.from_xml_path(cfg.scene)
        m.opt.timestep = cfg.dt
        m.opt.integrator = INTEGRATEURS[cfg.integrator]

        # Actionneur velocity : τ = kv·ctrl − kv·q̇  (gainprm[0] = kv, biasprm[2] = −kv).
        m.actuator_gainprm[:, 0] = cfg.kv
        m.actuator_biasprm[:, 2] = -cfg.kv

        # MuJoCo combine les frottements des deux géométries en prenant le maximum
        # (à priorité égale) : il faut fixer μ sur les roues ET sur le sol.
        # La roulette (priority=1, condim=1) reste sans frottement.
        self.roues = [m.body(n).id for n in ("roue_g", "roue_d")]
        geoms_roues = [g for b in self.roues
                       for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])]
        for g in [*geoms_roues, m.geom("floor").id]:
            m.geom_friction[g, 0] = cfg.mu

        self.model = m
        self.data = mujoco.MjData(m)
        self.params = DiffDriveParams.from_mujoco(m)
        self.chassis = m.body("chassis").id
        self.dofs = [m.jnt_dofadr[m.joint(n).id] for n in ("g", "d")]

    def poser(self, duree_max: float = 3.0) -> None:
        """Remet le robot au départ et attend qu'il soit immobile, commande nulle."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0.0
        while d.time < duree_max:
            mujoco.mj_step(m, d)
            if d.time > 0.2 and np.abs(d.qvel).max() < 1e-5:
                return
        raise RuntimeError(f"le robot ne s'immobilise pas ({self.cfg.etiquette()})")

    def rouler(self, omega_l: float, omega_r: float, duree: float) -> dict[str, np.ndarray]:
        """Applique une commande constante et enregistre la trajectoire à chaque pas."""
        m, d = self.model, self.data
        n = int(round(duree / m.opt.timestep))
        t = np.empty(n)
        p = np.empty((n, 2))       # milieu de l'essieu, dans le plan
        cap = np.empty(n)          # lacet du châssis
        w_roues = np.empty((n, 2))
        t0 = d.time
        d.ctrl[:] = (omega_l, omega_r)
        for k in range(n):
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                raise FloatingPointError(f"simulation instable ({self.cfg.etiquette()})")
            R = d.xmat[self.chassis].reshape(3, 3)
            t[k] = d.time - t0
            p[k] = 0.5 * (d.xpos[self.roues[0], :2] + d.xpos[self.roues[1], :2])
            cap[k] = np.arctan2(R[1, 0], R[0, 0])
            w_roues[k] = d.qvel[self.dofs]
        return {"t": t, "p": p, "cap": np.unwrap(cap), "w_roues": w_roues}


# ------------------------------------------------------------------- mesures

def ajuster_cercle(xy: np.ndarray) -> float:
    """Rayon du cercle des moindres carrés (méthode de Kåsa)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    a, b, c = np.linalg.lstsq(A, x**2 + y**2, rcond=None)[0]
    return float(np.sqrt(c + (a**2 + b**2) / 4))


def mesurer(traj: dict[str, np.ndarray]) -> dict[str, float]:
    """Vitesses en régime établi, sur la fenêtre [DEBUT_MESURE, fin]."""
    t, p, cap = traj["t"], traj["p"], traj["cap"]
    dt = t[1] - t[0]
    fen = t >= DEBUT_MESURE
    # Vitesse du milieu de l'essieu projetée sur le cap (différences centrées).
    vit = np.gradient(p, dt, axis=0)
    v = np.mean(np.einsum("ij,ij->i", vit, np.column_stack([np.cos(cap), np.sin(cap)]))[fen])
    w = np.polyfit(t[fen], cap[fen], 1)[0]
    w_l, w_r = traj["w_roues"][fen].mean(axis=0)
    mes = {"v": float(v), "w": float(w), "omega_l": float(w_l), "omega_r": float(w_r)}
    if abs(w) > 1e-3 and abs(v) > 1e-3:  # trajectoire courbe : rayon mesurable
        mes["R"] = float(np.sign(w) * np.sign(v) * ajuster_cercle(p[fen]))
    return mes


def ecart_relatif(mesure: float, theorie: float) -> float:
    return abs(mesure - theorie) / abs(theorie)


def evaluer(cfg: SimConfig) -> dict[str, dict]:
    """Lance les trois manœuvres et compare chaque mesure à la théorie."""
    robot = Robot(cfg)
    P = robot.params
    sortie = {}
    for nom, (wl, wr) in MANOEUVRES.items():
        robot.poser()
        mes = mesurer(robot.rouler(wl, wr, DUREE))
        v_th, w_th = forward(wl, wr, P)                            # roues parfaites
        v_cin, w_cin = forward(mes["omega_l"], mes["omega_r"], P)  # roues mesurées
        if nom == "ligne droite":
            grandeur, mesure, theorie, cin = "v (m/s)", mes["v"], float(v_th), float(v_cin)
        elif nom == "rotation sur place":
            grandeur, mesure, theorie, cin = "ω (rad/s)", mes["w"], float(w_th), float(w_cin)
        else:
            grandeur, mesure = "R (m)", mes["R"]
            theorie = float(turning_radius(wl, wr, P))
            cin = float(turning_radius(mes["omega_l"], mes["omega_r"], P))
        sortie[nom] = {
            "grandeur": grandeur, "theorie": theorie, "mesure": mesure,
            "ecart": ecart_relatif(mesure, theorie),
            "ecart_actionneur": ecart_relatif(cin, theorie),
            "ecart_cinematique": ecart_relatif(mesure, cin),
            "seuil": SEUILS[nom], **{f"mes_{k}": val for k, val in mes.items()},
        }
    return sortie


# ------------------------------------------------------------------ affichage

def afficher_validation(res: dict[str, dict], titre: str) -> bool:
    print(f"\n{titre}")
    print(f"{'manœuvre':<20}{'grandeur':<11}{'théorie':>10}{'mesure':>10}"
          f"{'écart':>9}{'dont act.':>11}{'dont cin.':>11}{'seuil':>8}")
    tout_ok = True
    for nom, r in res.items():
        ok = r["ecart"] <= r["seuil"]
        tout_ok &= ok
        print(f"{nom:<20}{r['grandeur']:<11}{r['theorie']:>10.4f}{r['mesure']:>10.4f}"
              f"{r['ecart']:>8.2%}{r['ecart_actionneur']:>10.2%}{r['ecart_cinematique']:>10.2%}"
              f"{r['seuil']:>7.0%}  {'OK' if ok else 'ÉCHEC'}")
    if "rotation sur place" in res:
        print(f"  (rotation sur place : vitesse résiduelle du milieu de l'essieu "
              f"{res['rotation sur place']['mes_v']:+.2e} m/s)")
    return tout_ok


def comparer_pas_de_temps(cfg: SimConfig) -> bool:
    """Critère : les mesures varient de moins de 1 % quand Δt est divisé par deux."""
    a, b = evaluer(cfg), evaluer(replace(cfg, dt=cfg.dt / 2))
    print(f"\nSensibilité au pas de temps : Δt = {cfg.dt * 1e3:g} ms contre {cfg.dt * 500:g} ms")
    tout_ok = True
    for nom in MANOEUVRES:
        var = ecart_relatif(b[nom]["mesure"], a[nom]["mesure"])
        ok = var <= SEUIL_PAS_DE_TEMPS
        tout_ok &= ok
        print(f"  {nom:<20}{a[nom]['mesure']:>10.5f}{b[nom]['mesure']:>10.5f}"
              f"{var:>9.3%}  {'OK' if ok else 'ÉCHEC'}")
    return tout_ok


# ------------------------------------------------------- étude paramétrique

def etude_parametrique() -> list[dict]:
    """Un facteur à la fois autour de la configuration nominale."""
    plans = (
        [replace(NOMINAL, mu=mu) for mu in (0.2, 0.5, 1.0)]
        + [replace(NOMINAL, kv=kv) for kv in (0.1, 0.3, 1.0, 3.0, 10.0)]
        + [replace(NOMINAL, dt=dt, integrator=integ)
           for integ in ("euler", "implicitfast") for dt in (0.0005, 0.001, 0.002, 0.004, 0.008)]
    )
    lignes = []
    print("\nÉtude paramétrique (écarts à la théorie ; act. = part due à l'actionneur)")
    print(f"{'configuration':<46}{'droite':>9}{'rotation':>10}{'cercle':>9}{'max act.':>10}")
    for cfg in dict.fromkeys(plans):  # sans doublon, ordre conservé
        try:
            res = evaluer(cfg)
            e = [res[n]["ecart"] for n in MANOEUVRES]
            act = max(res[n]["ecart_actionneur"] for n in MANOEUVRES)
            print(f"{cfg.etiquette():<46}" + "".join(f"{x:>9.2%} " for x in e) + f"{act:>8.2%}")
            lignes.append({"config": asdict(cfg), "resultats": res})
        except (FloatingPointError, RuntimeError) as err:
            print(f"{cfg.etiquette():<46}  {err}")
            lignes.append({"config": asdict(cfg), "erreur": str(err)})
    return lignes


def figure_glissement(chemin: Path, duree: float = 0.6) -> dict[str, float]:
    """Accélération brusque, de l'arrêt à pleine vitesse, pour trois frottements."""
    fig, (ax_v, ax_s) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    pics = {}
    for mu, couleur in zip((0.2, 0.5, 1.0), ("#C0392B", "#E08E0B", "#1F7A8C")):
        robot = Robot(replace(NOMINAL, mu=mu))
        robot.poser()
        traj = robot.rouler(10.0, 10.0, duree)
        t, r = traj["t"], robot.params.wheel_radius
        vit = np.gradient(traj["p"], t[1] - t[0], axis=0)
        v = np.einsum("ij,ij->i", vit, np.column_stack([np.cos(traj["cap"]), np.sin(traj["cap"])]))
        v_roue = r * traj["w_roues"].mean(axis=1)
        # Taux de glissement λ = (rω − v)/rω, défini quand les roues tournent vraiment.
        # Seuil à 0,1 m/s : en dessous, le dénominateur quasi nul rend λ instable.
        lam = np.where(v_roue > 0.1, (v_roue - v) / np.maximum(v_roue, 1e-9), np.nan)
        pics[f"{mu:g}"] = float(np.nanmax(lam))
        ax_v.plot(t, v_roue, color=couleur, ls="--", lw=1)
        ax_v.plot(t, v, color=couleur, label=f"μ = {mu:g}")
        ax_s.plot(t, 100 * lam, color=couleur, label=f"μ = {mu:g} (max {pics[f'{mu:g}']:.0%})")
    ax_v.set_ylabel("vitesse (m/s)")
    ax_v.set_title("Démarrage à pleine commande : robot (trait plein) et r·ω des roues (tirets)")
    ax_v.legend()
    ax_s.set_ylabel("glissement λ (%)")
    ax_s.set_xlabel("temps depuis la commande (s)")
    ax_s.legend()
    for ax in (ax_v, ax_s):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return pics


def charge_roues() -> float:
    """Force normale moyenne sur une roue motrice au repos (N)."""
    robot = Robot(NOMINAL)
    robot.poser()
    m, d = robot.model, robot.data
    roues = {int(g) for b in robot.roues
             for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])}
    total = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in roues or c.geom2 in roues:
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            total += f[0]
    return total / 2


def commit_courant() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RACINE,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


# ----------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sans-etude", action="store_true",
                        help="ne lancer que la validation (pas d'étude paramétrique ni de figure)")
    parser.add_argument("--scene", type=Path, default=SCENE,
                        help="scène MJCF à valider (par défaut assets/mjcf/arena.xml)")
    args = parser.parse_args()
    global NOMINAL
    NOMINAL = replace(NOMINAL, scene=str(args.scene.resolve()))
    print(f"Scène : {args.scene}")

    P = Robot(NOMINAL).params
    print(f"Paramètres lus dans le MJCF : r = {P.wheel_radius} m, L = {P.wheel_base} m, "
          f"ω_max = {P.max_wheel_speed} rad/s")
    ok = afficher_validation(evaluer(NOMINAL), f"Validation, {NOMINAL.etiquette()}")
    ok &= comparer_pas_de_temps(NOMINAL)
    resultats = {"commit": commit_courant(), "nominal": asdict(NOMINAL), "validation_ok": ok}

    if not args.sans_etude:
        resultats["etude"] = etude_parametrique()
        n = charge_roues()
        tau = float(Robot(NOMINAL).model.actuator_forcerange[0, 1])
        print(f"\nCharge par roue au repos : {n:.2f} N ; couple maximal : {tau} N·m")
        print(f"Frottement seuil au repos μ* = τ_max/(r·N) = {tau / (P.wheel_radius * n):.2f}")
        fig = FIGURES / "glissement_mu.png"
        pics = figure_glissement(fig)
        print(f"Glissement maximal au démarrage : {pics}  →  {fig.relative_to(RACINE)}")
        resultats["charge_roue_N"], resultats["glissement_max"] = n, pics

    RESULTATS.mkdir(exist_ok=True)
    with open(RESULTATS / "validate_kinematics.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    print(f"\n{'VALIDATION RÉUSSIE' if ok else 'VALIDATION ÉCHOUÉE'} "
          f"(résultats : results/validate_kinematics.json)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
