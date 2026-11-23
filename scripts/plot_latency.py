"""Histogramme de la latence scan -> commande, à partir du CSV écrit par control_node.

    python3 scripts/plot_latency.py latence.csv
    python3 scripts/plot_latency.py latence.csv --en-mouvement --theme sombre

Le CSV a trois colonnes : t_sim_s, latence_ms, v_cmd. Avec --en-mouvement, on
ne garde que les commandes où le robot avance (v_cmd ≠ 0), c'est-à-dire
pendant une navigation, sans les longues attentes à l'arrêt.

Ce qu'on s'attend à voir : des barres sur les multiples de 20 ms (le temps
simulé n'avance que par cycles de sim_bridge), étalées entre 0 et environ
100 ms : le scan arrive à 10 Hz et la décision tourne à 10 Hz, sans être
synchronisés. En Gym, cette latence est nulle : c'est un écart Gym / ROS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                  # pas de fenêtre : on écrit seulement un fichier
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {   # surface, encre principale, encre secondaire, grille, série
    "clair": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", bar="#2a78d6"),
    "sombre": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#3a3a37", bar="#3987e5"),
}
STEP_MS = 20.0                         # durée d'un cycle de sim_bridge


def load(path: Path, moving_only: bool) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    lat = np.atleast_1d(data["latence_ms"])
    if moving_only:
        lat = lat[np.atleast_1d(data["v_cmd"]) != 0.0]
    if lat.size == 0:
        raise SystemExit(f"aucune mesure dans {path}" + (" avec v_cmd ≠ 0" if moving_only else ""))
    return lat


def plot(lat: np.ndarray, out: Path, theme: str, moving_only: bool) -> None:
    t = THEMES[theme]
    med, p95, mx = np.median(lat), np.percentile(lat, 95), lat.max()

    # Une barre par multiple de 20 ms, centrée dessus
    edges = np.arange(-STEP_MS / 2, mx + STEP_MS, STEP_MS)
    counts, _ = np.histogram(lat, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    share = 100.0 * counts / lat.size

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    ax.bar(centers, share, width=STEP_MS * 0.82, color=t["bar"], zorder=2)
    for x, s in zip(centers, share):                   # étiquettes seulement sur les barres visibles
        if s >= 2.0:
            ax.text(x, s + 0.8, f"{s:.0f} %", ha="center", va="bottom", fontsize=8, color=t["ink2"])

    # Les latences tombent pile sur les centres des barres : des lignes verticales
    # pour la médiane et le 95e centile se superposeraient aux barres. Les
    # statistiques vont donc dans le sous-titre.
    ax.set_ylim(0, share.max() * 1.15)
    ax.set_xlabel("latence scan → commande (ms, temps simulé)", color=t["ink2"])
    ax.set_ylabel("part des commandes (%)", color=t["ink2"])
    scope = "en mouvement" if moving_only else "toutes"
    fig.suptitle("Latence scan → commande", x=0.02, ha="left", color=t["ink"], fontsize=11)
    ax.set_title(f"{lat.size} commandes ({scope}) · médiane {med:.0f} ms · 95 % sous {p95:.0f} ms · "
                 f"max {mx:.0f} ms", loc="left", color=t["ink2"], fontsize=8.5)

    ax.set_xticks(centers)
    ax.tick_params(colors=t["ink2"], labelsize=8, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"])
    print(f"{out} : médiane {med:.1f} ms, 95 % sous {p95:.1f} ms, max {mx:.1f} ms, {lat.size} mesures")


def main() -> None:
    p = argparse.ArgumentParser(description="Histogramme de latence (CSV de control_node)")
    p.add_argument("csv", type=Path, help="fichier écrit par control_node (paramètre latency_log)")
    p.add_argument("-o", "--out", type=Path, default=Path("docs/figures/latence.png"))
    p.add_argument("--en-mouvement", action="store_true", help="seulement les commandes avec v ≠ 0")
    p.add_argument("--theme", choices=sorted(THEMES), default="clair")
    a = p.parse_args()
    plot(load(a.csv, a.en_mouvement), a.out, a.theme, a.en_mouvement)


if __name__ == "__main__":
    main()
