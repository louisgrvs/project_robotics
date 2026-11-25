"""Statistiques d'épisodes pour TensorBoard (jour 11).

SB3 journalise déjà rollout/ep_rew_mean, rollout/ep_len_mean et
rollout/success_rate. Il manque ce qui dit COMMENT le robot échoue, et quel
terme de récompense pèse le plus :

    rollout/collision_rate   part des épisodes finis par une collision
    rollout/timeout_rate     part des épisodes tronqués (300 pas sans arriver)
    reward/<terme>           valeur moyenne PAR DÉCISION de chaque terme
                             (progress, time, proximity, success, collision)

Les taux portent sur les 100 derniers épisodes, comme rollout/success_rate
de SB3 : les trois courbes sont comparables, et succès + collision + troncature
fait 1 (à l'arrondi près).
"""
from __future__ import annotations

from collections import defaultdict, deque

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    """Rappel de mesure : il n'agit sur rien, il observe et journalise.

    Cycle d'appel par SB3, pour chaque collecte de n_steps pas :
        _on_rollout_start()   on remet à zéro les sommes de termes
        _on_step()  × n_steps on accumule (chaque appel voit les n_envs environnements)
        _on_rollout_end()     on écrit les moyennes dans le journal
    puis SB3 vide le journal dans TensorBoard et fait la mise à jour PPO.
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        # Fin des 100 derniers épisodes : "success", "collision" ou "timeout".
        # Une deque garde le contenu d'une collecte à l'autre : les taux sont
        # glissants, pas remis à zéro (sinon ils seraient très bruités).
        self.endings = deque(maxlen=window)
        # Sommes des termes de récompense sur la collecte en cours.
        self.term_sums: dict[str, float] = defaultdict(float)
        self.n_transitions = 0

    # ------------------------------------------------------------ collecte
    def _on_rollout_start(self) -> None:
        """Début d'une collecte : les moyennes de termes sont par collecte."""
        self.term_sums = defaultdict(float)
        self.n_transitions = 0

    def _on_step(self) -> bool:
        """Appelée après chaque pas du VecEnv.

        self.locals est le dictionnaire des variables locales de la boucle de
        collecte de SB3, au moment de l'appel :
            self.locals["dones"] : tableau de n_envs booléens, True si
                                   l'épisode de cet environnement vient de finir
                                   (terminé OU tronqué, SB3 ne distingue pas ici)
            self.locals["infos"] : liste de n_envs dictionnaires, ceux que
                                   NavEnv.step a renvoyés (plus quelques clés
                                   ajoutées par SB3)
        """
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            # 1. Termes de récompense, à chaque transition
            terms = info.get("reward_terms")
            if terms is not None:
                for name, value in terms.items():
                    self.term_sums[name] += value
                self.n_transitions += 1

            # 2. Manière dont l'épisode s'est terminé, seulement en fin d'épisode
            if done:
                if info.get("is_success"):
                    self.endings.append("success")
                elif info.get("collision"):
                    self.endings.append("collision")
                elif info.get("TimeLimit.truncated"):
                    # Clé ajoutée par le VecEnv de SB3 : True si l'épisode a été
                    # tronqué sans être terminé. C'est elle qui permet à PPO
                    # d'utiliser V(s') au lieu de 0 à la troncature (GAE).
                    self.endings.append("timeout")
        return True        # renvoyer False arrêterait l'entraînement

    def _on_rollout_end(self) -> None:
        """Fin de collecte : on écrit dans le journal (SB3 l'envoie à TensorBoard juste après)."""
        if self.n_transitions > 0:
            for name, total in self.term_sums.items():
                self.logger.record(f"reward/{name}", total / self.n_transitions)

        n = len(self.endings)
        if n > 0:          # aucun épisode fini depuis le début : rien à écrire
            self.logger.record("rollout/collision_rate", self.endings.count("collision") / n)
            self.logger.record("rollout/timeout_rate", self.endings.count("timeout") / n)