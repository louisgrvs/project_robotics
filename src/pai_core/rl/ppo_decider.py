"""Politique PPO entraînée, présentée avec la même interface que le DWA.

    decider.reset()            début d'un nouveau but
    decider.act(obs) -> action normalisée dans [-1, 1]², float32

C'est cette interface commune qui permet à decision_node de passer du DWA à
PPO en changeant un seul paramètre (decision_mode), sans rien savoir de la
méthode utilisée.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class PPODecider:
    """Charge un modèle Stable-Baselines3 sauvegardé par train_ppo.py.

    deterministic=True : on joue la moyenne μ de la gaussienne, sans le bruit
    d'exploration, comme EvalCallback pendant l'entraînement.
    """

    def __init__(self, model_path: str | Path, deterministic: bool = True):
        # Import ici et non en haut du fichier : decision_node n'a pas besoin
        # de PyTorch quand on utilise le DWA.
        from stable_baselines3 import PPO

        path = Path(model_path).expanduser()
        if not path.exists() and not path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"modèle PPO introuvable : {path}")
        self.model = PPO.load(str(path), device="cpu")
        self.deterministic = deterministic

    def reset(self) -> None:
        """Rien à oublier : la politique MLP n'a pas de mémoire entre deux pas.

        (L'action précédente fait partie de l'observation, c'est decision_node
        qui la remet à zéro.)
        """

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32),
                                       deterministic=self.deterministic)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
