"""Recherche A* sur une grille d'occupation, en 8-connexité.

Sert à deux choses : écarter les scènes sans chemin (jour 5) et fournir la
longueur du plus court chemin pour la métrique SPL (jour 12).

Heuristique « octile » : la distance exacte sur une grille 8-connexe sans
obstacle. Elle ne surestime jamais le coût restant (admissible) et respecte
l'inégalité triangulaire (cohérente), donc le premier passage sur le but
donne le chemin optimal sur la grille.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

_SQRT2 = math.sqrt(2.0)
_MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2)]

Cell = tuple[int, int]


def _octile(a: Cell, b: Cell) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)


def astar(occupied: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Plus court chemin de start à goal (indices de cases), ou None.

    occupied[i, j] vaut True si la case est interdite. Un pas diagonal n'est
    permis que si les deux cases qu'il longe sont libres : le robot ne coupe
    pas le coin d'un obstacle.
    """
    occ = occupied.tolist()  # listes Python : l'accès case par case y est bien plus rapide
    nx, ny = occupied.shape
    if occ[start[0]][start[1]] or occ[goal[0]][goal[1]]:
        return None

    g = [[math.inf] * ny for _ in range(nx)]
    closed = [[False] * ny for _ in range(nx)]
    parent: dict[Cell, Cell] = {}
    g[start[0]][start[1]] = 0.0
    heap = [(_octile(start, goal), 0.0, start)]

    while heap:
        _, gc, cell = heapq.heappop(heap)
        i, j = cell
        if closed[i][j]:
            continue
        if cell == goal:
            path = [cell]
            while cell in parent:
                cell = parent[cell]
                path.append(cell)
            return path[::-1]
        closed[i][j] = True
        for di, dj, cost in _MOVES:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or occ[ni][nj] or closed[ni][nj]:
                continue
            if di and dj and (occ[i + di][j] or occ[i][j + dj]):
                continue
            ng = gc + cost
            if ng < g[ni][nj]:
                g[ni][nj] = ng
                parent[(ni, nj)] = cell
                heapq.heappush(heap, (ng + _octile((ni, nj), goal), ng, (ni, nj)))
    return None


def path_length(path: list[Cell], resolution: float) -> float:
    """Longueur du chemin en mètres (pas droit : resolution, diagonal : √2·resolution)."""
    total = 0.0
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        total += _SQRT2 if (i0 != i1 and j0 != j1) else 1.0
    return total * resolution
