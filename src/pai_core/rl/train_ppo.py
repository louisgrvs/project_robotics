"""Entraînement PPO sur NavEnv : version minimale (entraînement de contrôle).

But : vérifier que la chaîne observation -> récompense -> apprentissage
fonctionne, sur des scènes sans obstacle. Si la récompense moyenne ne monte pas
ici, le problème vient de la récompense, des échelles ou du câblage, pas de la
difficulté de la tâche.

    ┌──────────────┐   make_env(rang)   ┌───────────────────────┐
    │ TrainConfig  │ ─────────────────> │ NavEnv + Monitor (×N) │ ── VecEnv
    └──────────────┘                    └───────────────────────┘      │
           │                                                           ▼
           │  hyperparamètres                       ┌──────────────────────────┐
           └──────────────────────────────────────> │ PPO("MlpPolicy", VecEnv) │
                                                     └──────────────────────────┘
                                                        │ learn()       │ rappels
                                                        ▼               ▼
                                          runs/<nom>/ (TensorBoard)   checkpoints/<nom>/

Utilisation, depuis la racine du dépôt :
    python -m pai_core.rl.train_ppo --name controle_sans_obstacle --steps 2_000_000
    tensorboard --logdir runs

Ce qu'on doit voir dans TensorBoard :
    rollout/ep_rew_mean     monte
    rollout/success_rate    passe nettement au-dessus de 0
    rollout/ep_len_mean     baisse (le robot arrive plus vite)
    eval/mean_reward        part d'environ -1 et s'approche de 40 (score de « vers le but »)
    train/approx_kl, train/clip_fraction : ni nuls ni énormes (voir le jour 12)
"""
from __future__ import annotations

# Bibliothèque standard
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Bibliothèques externes
import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Projet
from pai_core.envs.nav_env import NavEnv, NavEnvConfig
from pai_core.perception.features import SLICES


# ============================================================================
# 1. Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Tout ce qui définit une expérience. Elle est sauvegardée en JSON à côté
    des résultats : on doit pouvoir dire avec quels réglages une courbe a été obtenue.
    """
    name: str = "controle_sans_obstacle"
    seed: int = 0
    total_timesteps: int = 2_000_000    # budget total, tous environnements confondus
    n_envs: int = 4                     # CPU modeste : 4, à baisser si ROS tourne en même temps

    # Environnement
    n_obstacles: tuple[int, int] = (0, 0)   # entraînement de contrôle : aucun obstacle

    # PPO : valeurs par défaut de Stable-Baselines3, un bon point de départ.
    learning_rate: float = 3e-4   # pas de l'optimiseur Adam : de combien on bouge les poids à chaque pas de gradient
    n_steps: int = 1024           # pas par environnement entre deux mises à jour (4 × 1024 = 4096 transitions)
    batch_size: int = 256         # taille d'un mini-lot : 4096 / 256 = 16 mini-lots par passe
    n_epochs: int = 10            # passes sur les mêmes 4096 transitions à chaque mise à jour
    gamma: float = 0.99           # actualisation : 0 = agent myope, proche de 1 = agent prévoyant ;
                                  #   horizon effectif 1/(1-γ) = 100 décisions = 10 s (épisode max 30 s)
    gae_lambda: float = 0.95      # avantage GAE, compromis biais/variance : 0 = un pas (biaisé), 1 = Monte-Carlo (bruité)
    clip_range: float = 0.2       # ε : le rapport π_nouveau/π_ancien est écrêté dans [0,8 ; 1,2]
    ent_coef: float = 0.0         # bonus d'entropie (exploration) ; à augmenter si l'écart-type s'effondre

    # Suivi (en pas d'environnement cumulés ; build_callbacks fait la conversion)
    eval_every: int = 50_000      # évaluation déterministe (sans bruit) sur un environnement à part
    n_eval_episodes: int = 20     # épisodes par évaluation
    checkpoint_every: int = 100_000   # sauvegarde du modèle

    @property
    def run_dir(self) -> Path:
        return Path("runs") / self.name

    @property
    def ckpt_dir(self) -> Path:
        return Path("checkpoints") / self.name


# ============================================================================
# 2. Environnements
# ============================================================================

def make_env(cfg: TrainConfig, rank: int, eval_env: bool = False):
    """Renvoie une FONCTION qui fabrique un environnement (et non l'environnement).

    SubprocVecEnv crée chaque environnement dans son propre processus : il lui
    faut la recette, pas l'objet.

    Graines : une par rang, sinon les N environnements tireraient les mêmes
    scènes dans le même ordre (N copies des mêmes données). Les graines
    d'évaluation sont dans une plage à part.
    """
    def _init():
        env = NavEnv(NavEnvConfig(n_obstacles=cfg.n_obstacles))
        env = Monitor(env, info_keywords=("is_success",))   # journalise récompense, longueur et succès
        offset = 10_000 if eval_env else 0
        env.reset(seed=cfg.seed + offset + rank)
        return env
    return _init


def build_vec_env(cfg: TrainConfig, eval_env: bool = False):
    """N environnements en parallèle pour l'entraînement, 1 seul pour l'évaluation.

    SubprocVecEnv : un processus par environnement, utile car la physique
    (50 pas MuJoCo + lidar par décision) coûte cher. DummyVecEnv : tout dans
    un processus, plus simple à déboguer.
    Pas de VecNormalize : features.py normalise déjà les observations.
    """
    if eval_env:
        return DummyVecEnv([make_env(cfg, 0, eval_env=True)])
    return SubprocVecEnv([make_env(cfg, i) for i in range(cfg.n_envs)])


# ============================================================================
# 3. Modèle
# ============================================================================

def build_model(cfg: TrainConfig, env) -> PPO:
    """PPO avec une politique MLP, journaux TensorBoard dans cfg.run_dir.

    « MlpPolicy » : deux réseaux séparés, activation tanh
        acteur   : 43 entrées → 64 → 64 → 2 sorties  (moyenne μ de a_v et a_ω ;
                   l'écart-type σ est un paramètre appris à part)
        critique : 43 entrées → 64 → 64 → 1 sortie   (V(s))
    """
    assert (cfg.n_steps * cfg.n_envs) % cfg.batch_size == 0, \
        "n_steps × n_envs doit se couper en mini-lots entiers"
    return PPO("MlpPolicy", env,
               learning_rate=cfg.learning_rate, n_steps=cfg.n_steps, batch_size=cfg.batch_size,
               n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
               clip_range=cfg.clip_range, ent_coef=cfg.ent_coef,
               tensorboard_log=str(cfg.run_dir), seed=cfg.seed, verbose=1, device="cpu")


# ============================================================================
# 4. Rappels (callbacks)
# ============================================================================

def build_callbacks(cfg: TrainConfig, eval_env) -> list:
    """Sauvegardes régulières et évaluation déterministe sur un environnement à part.

    save_freq et eval_freq se comptent en pas du VecEnv : un pas du VecEnv vaut
    n_envs pas d'environnement, d'où la division par cfg.n_envs.
    Évaluation déterministe : on joue μ sans bruit, c'est la politique qu'on
    déploiera dans ROS.
    """
    ckpt = CheckpointCallback(save_freq=max(cfg.checkpoint_every // cfg.n_envs, 1),
                              save_path=str(cfg.ckpt_dir), name_prefix="ppo")
    evaluation = EvalCallback(eval_env,
                              n_eval_episodes=cfg.n_eval_episodes,
                              eval_freq=max(cfg.eval_every // cfg.n_envs, 1),
                              deterministic=True,
                              best_model_save_path=str(cfg.ckpt_dir / "best"),
                              log_path=str(cfg.run_dir / "eval"))
    return [ckpt, evaluation]


# ============================================================================
# 5. Vérifications avant de lancer des heures de calcul
# ============================================================================

def run_episode(env, policy, seed: int) -> tuple[float, int, str]:
    """Joue un épisode et renvoie (récompense totale, nombre de pas, raison de fin)."""
    obs, info = env.reset(seed=seed)
    total, n = 0.0, 0
    while True:
        obs, r, terminated, truncated, info = env.step(policy(obs))
        total += r
        n += 1
        if terminated or truncated:      # terminé (succès, collision) OU tronqué (300 pas)
            if info["is_success"]:
                raison = "succès"
            elif info["collision"]:
                raison = "collision"
            else:
                raison = "troncature"
            return total, n, raison


def vers_le_but(obs: np.ndarray) -> np.ndarray:
    """Agent simple de référence : avancer si le but est devant, tourner vers lui."""
    _, cos_b, sin_b = obs[SLICES["goal"]]     # (distance, cos et sin du cap du but vu du robot)
    a_v = max(float(cos_b), 0.0)   # pleine vitesse si le but est droit devant, 0 s'il est derrière
    a_w = float(sin_b)             # tourne vers le but, d'autant plus fort qu'il est sur le côté
    return np.array([a_v, a_w], dtype=np.float32)


def sanity_checks(cfg: TrainConfig) -> None:
    """Quelques secondes de vérification qui évitent un week-end perdu.

    Au hasard : le plancher (environ -1). Vers le but : l'objectif (environ +40).
    """
    env = NavEnv(NavEnvConfig(n_obstacles=cfg.n_obstacles))
    check_env(env)

    env.action_space.seed(cfg.seed)           # actions aléatoires reproductibles
    au_hasard = lambda obs: env.action_space.sample()
    print("au hasard   :", run_episode(env, au_hasard, seed=cfg.seed))
    print("vers le but :", run_episode(env, vers_le_but, seed=cfg.seed))
    env.close()


# ============================================================================
# 6. Orchestration
# ============================================================================

def parse_args() -> tuple[TrainConfig, bool]:
    p = argparse.ArgumentParser(description="Entraînement PPO sur NavEnv")
    p.add_argument("--name", default=TrainConfig.name)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--steps", type=int, default=TrainConfig.total_timesteps)
    p.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    p.add_argument("--skip-checks", action="store_true", help="sauter les vérifications préalables")
    # Plus tard : --resume <chemin.zip> pour reprendre depuis une sauvegarde (utile si WSL s'arrête)
    a = p.parse_args()
    cfg = TrainConfig(name=a.name, seed=a.seed, total_timesteps=a.steps, n_envs=a.n_envs)
    return cfg, a.skip_checks


def save_config(cfg: TrainConfig) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def main() -> None:
    cfg, skip_checks = parse_args()
    set_random_seed(cfg.seed)

    if not skip_checks:
        sanity_checks(cfg)

    save_config(cfg)
    env = build_vec_env(cfg)
    eval_env = build_vec_env(cfg, eval_env=True)
    model = build_model(cfg, env)
    callbacks = build_callbacks(cfg, eval_env)

    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks,
                    tb_log_name="ppo", progress_bar=False)
    except KeyboardInterrupt:
        print("interrompu : sauvegarde du modèle courant")
    finally:
        # Toujours sauvegarder, même après Ctrl+C : quelques heures de calcul
        # valent plus que la propreté d'un arrêt.
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save(cfg.ckpt_dir / "final")
        env.close()
        eval_env.close()


# Indispensable avec SubprocVecEnv : chaque processus fils réimporte ce module,
# et sans cette garde il relancerait lui-même un entraînement.
if __name__ == "__main__":
    main()