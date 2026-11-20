"""Filtre de sûreté : le dernier rempart entre une décision et les roues.

Pourquoi ce module est dans pai_core et pas dans le noeud ROS :
  - il se teste avec pytest, sans lancer ROS ;
  - le filtre est une variable expérimentale : eval_classic.py et NavEnv
    doivent pouvoir l'activer ou non exactement comme control_node le fait.
    Une seule implémentation pour les deux, sinon les résultats ne sont
    pas comparables.

Règle (volontairement simple, donc facile à défendre) :
  si le robot avance et que la distance libre devant lui est inférieure à
  sa distance d'arrêt plus une marge, on annule v et on garde w.

Repère : base_link (x vers l'avant, y vers la gauche, REP 103).
Hypothèse : pas de marche arrière (v >= 0), comme dans le DWA.
"""
from dataclasses import dataclass

import numpy as np

from pai_core.control import dwa


# ---------------------------------------------------------------- paramètres

@dataclass(frozen=True)
class SafetyParams:
    enabled: bool = True
    robot_radius: float = 0.24     # même valeur que DWAParams.robot_radius (m)
    lidar_offset_x: float = 0.0    # position du lidar par rapport à l'essieu (m)
    corridor_margin: float = 0.05  # marge latérale du couloir (m)
    dec: float = 1.0               # décélération de freinage supposée (m/s²)
    latency: float = 0.15          # temps de réaction : scan + décision + commande (s)
    stop_margin: float = 0.05      # marge longitudinale ajoutée à la distance d'arrêt (m)
    scan_timeout: float = 0.5      # au-delà, le scan est jugé périmé (s)


@dataclass(frozen=True)
class SafetyDecision:
    v: float            # commande linéaire après filtre (m/s)
    w: float            # commande angulaire après filtre (rad/s)
    intervened: bool    # le filtre a-t-il modifié la commande ?
    free_dist: float    # distance libre mesurée (m), utile pour le journal


# ---------------------------------------------------------------- perception

def scan_to_points(ranges: np.ndarray, angle_min: float, angle_increment: float,
                   range_max: float, offset_x: float = 0.0) -> np.ndarray:
    """Points d'impact (x, y) dans base_link, (N, 2).

    Les angles sont reconstruits à partir du nombre de faisceaux : toujours
    exactement un angle par distance, même avec les arrondis flottants.
    Les faisceaux sans impact (>= range_max, inf, nan) sont retirés par
    lidar_to_points : le test « distance < portée » est faux pour inf, et
    toute comparaison avec nan est fausse.
    """
    angles = angle_min + angle_increment * np.arange(len(ranges))
    return dwa.lidar_to_points(ranges, angles, range_max, offset_x)


def free_distance(points: np.ndarray, p: SafetyParams) -> float:
    """Combien de mètres puis-je avancer tout droit avant de toucher quelque chose ?

    Couloir : x > 0 et |y| <= robot_radius + corridor_margin.
    Renvoie min(x) - robot_radius sur les points du couloir, ou +inf s'il
    est vide. Peut être négative : l'obstacle est déjà dans le cercle du robot.

    Pourquoi un couloir plutôt qu'un cône de faisceaux : un cône s'élargit
    avec la distance (il voit des obstacles loin sur le côté qui ne gênent
    pas) et se rétrécit près du robot (il rate un coin juste devant une roue).
    Le couloir a la largeur du robot à toute distance.
    """
    x, y = points[:, 0], points[:, 1]                 # colonnes : tous les x, tous les y
    in_corridor = (x > 0) & (np.abs(y) <= p.robot_radius + p.corridor_margin)  # masque (N,)
    if not in_corridor.any():                         # couloir vide : rien devant
        return float("inf")
    return float(x[in_corridor].min() - p.robot_radius)  # l'avant du robot est à r de l'essieu


def stopping_distance(v: float, p: SafetyParams) -> float:
    """Distance parcourue avant l'arrêt complet : réaction, puis freinage.

    d = v·latency + v² / (2·dec)   (réaction à vitesse constante, puis freinage uniforme)
    """
    return v * p.latency + v**2 / (2 * p.dec)


# ---------------------------------------------------------------- décision

def apply(v: float, w: float, points: np.ndarray, p: SafetyParams) -> SafetyDecision:
    """À cette vitesse, ai-je encore la place de m'arrêter ?

    Commande (v, w) déjà saturée en entrée.
    - p.enabled faux : commande rendue intacte, intervened = False ;
      free_dist est quand même calculée, pour pouvoir la journaliser.
    - v > 0 et free_dist < stopping_distance(v) + stop_margin :
      v = 0, w conservé, intervened = True.
    - sinon : commande intacte, intervened = False.

    À justifier dans le journal :
      1. Pourquoi garder w ? (indice : robot_radius est le rayon du cercle
         circonscrit centré sur l'essieu)
      2. Mettre v à 0 casse la courbure que saturate préservait. Est-ce un
         problème ? (indice : relire l'objectif du jour sur le filtre comme
         variable expérimentale)
      3. Le couloir est droit, mais le robot suit un arc. À forte courbure,
         que regarde-t-on à tort, et que rate-t-on ?
    """
    v, w = float(v), float(w)          # saturate renvoie des tableaux NumPy de dimension 0
    free_dist = free_distance(points, p)

    if not p.enabled:
        return SafetyDecision(v=v, w=w, intervened=False, free_dist=free_dist)
    if v > 0 and free_dist < stopping_distance(v, p) + p.stop_margin:
        return SafetyDecision(v=0.0, w=w, intervened=True, free_dist=free_dist)
    return SafetyDecision(v=v, w=w, intervened=False, free_dist=free_dist)
