"""Noeud de décision : (scan, vitesse, but) -> /cmd_vel, à 10 Hz.

Rôle : traduire le monde ROS (messages asynchrones, repères, temps simulé)
en exactement le même vecteur d'observation que NavEnv, appeler le décideur,
puis retraduire son action en /cmd_vel. En gros : c'est l'interprète entre ROS et le décideur.

    /scan ──────┐
    /odom ──────┼─> build_observation ─> decider.act(obs) ─>action ∈ [-1, 1]^2
    but ────────┘    (features.py)        (DWA aujourd'hui,        │
    (/goal_pose ou action)                 PPO et MPPI plus tard)  ▼
      + TF2 map→base_link                   (v, omega) = action · échelles
                                                                   ▼
                                                          /cmd_vel TwistStamped

Le décideur ne connaît pas ROS, et ce noeud ne sait pas quel décideur il
appelle : la seule interface est act(obs) -> action normalisée, la même que
dans NavEnv. C'est ce qui rend la comparaison DWA / PPO / MPPI équitable, et
ce que vérifiera le test d'identité.

Entrées  : /scan       sensor_msgs/LaserScan        36 distances
           /odom       nav_msgs/Odometry            vitesses (v, omega) dans base_link
           /goal_pose  geometry_msgs/PoseStamped    but cliqué dans RViz (repère map)
           TF2         map -> base_link             publié par sim_bridge
           /collision  std_msgs/Bool                vrai pendant un contact (publié par sim_bridge)
Sorties  : /cmd_vel    geometry_msgs/TwistStamped   commande demandée, vers control_node
Action   : /navigate_to_goal  pai_interfaces/action/NavigateToGoal   (jour 10)

Deux façons de donner un but, qui passent par la même méthode _start_goal :
    - clic dans RViz sur /goal_pose : pas de retour, pas de résultat ;
    - action /navigate_to_goal : retour (distance restante), résultat
      (succès, durée, collisions) et annulation possible.

Fils d'exécution (jour 10)
    Le rappel d'exécution de l'action dure toute la navigation. Avec un seul
    fil (rclpy.spin), il bloquerait tick et la réception de /tf. On utilise
    donc un MultiThreadedExecutor, et l'action est dans un groupe
    « réentrant » : elle tourne dans son propre fil, pendant que tick continue
    dans le sien. Les deux communiquent par quelques attributs et un
    threading.Event (un drapeau qu'un fil lève et qu'un autre attend).

Paramètres du jour 10
    decision_mode    "dwa" ou "ppo" (MPPI plus tard) : choisit le décideur
    model_path       modèle PPO (.zip de train_ppo.py), si decision_mode = "ppo"
    velocity_source  "ground_truth" ou "odom" : d'où vient (v, ω) dans l'observation.
                     NavEnv utilise la VRAIE vitesse (qvel de MuJoCo) : pour que
                     la politique voie en ROS exactement ce qu'elle a vu à
                     l'entraînement, on lit /ground_truth par défaut. Avec
                     "odom", on mesure l'effet d'une vitesse estimée par les
                     encodeurs, comme sur un vrai robot.

Latence (jour 10)
    Chaque /cmd_vel porte l'horodatage du SCAN qui l'a produit, et non
    l'instant de publication. control_node fait « maintenant − horodatage »
    à la réception : c'est la latence scan -> commande.

Rappel : dans une fonction de rappel, `msg` est LE message reçu (avec ses
valeurs) ; LaserScan, Odometry... sont seulement les types (les moules).
"""
import math
import threading
import time

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (enregistre PoseStamped auprès de tf2 : sans lui, transform échoue)
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from pai_interfaces.action import NavigateToGoal

from pai_core.control.dwa import DWA, DWAParams
from pai_core.perception.features import N_BEAMS, ObsConfig, build_observation
from pai_core.ros_conversions import ranges_from_msg

DECISION_MODES = ("dwa", "ppo")       # "mppi" s'ajoutera ici
VELOCITY_SOURCES = {"ground_truth": "/ground_truth", "odom": "/odom"}


def make_decider(mode: str, dwa_params: DWAParams, model_path: str = ""):
    """Fabrique le décideur demandé. Tous ont la même interface : reset() et act(obs)."""
    if mode == "dwa":
        return DWA(dwa_params)
    if mode == "ppo":
        if not model_path:
            raise ValueError("decision_mode = ppo : donner model_path (le .zip sauvegardé par train_ppo.py)")
        from pai_core.rl.ppo_decider import PPODecider   # import tardif : PyTorch seulement si besoin
        return PPODecider(model_path)
    raise ValueError(f"decision_mode inconnu : {mode!r} (choix : {', '.join(DECISION_MODES)})")


def goal_in_frame(tf_buffer: Buffer, goal: PoseStamped, frame: str) -> tuple[float, float]:
    """Position (x, y) du but dans `frame`, avec la transformation la plus récente.

    Piège important : RViz horodate le but à l'instant du clic. Si on le
    transforme tel quel, TF2 donne le but dans base_link à cet instant-là,
    c'est-à-dire là où était le robot au moment du clic. Le but est fixe dans
    map : on transforme donc une copie dont l'horodatage vaut zéro
    (Time().to_msg()), ce qui veut dire « la dernière disponible ».

    Fonction hors de la classe : le test d'identité l'appelle avec son propre
    Buffer, sans lancer de nœud. Lève TransformException si TF n'est pas prête.
    """
    g = PoseStamped()
    g.header.frame_id = goal.header.frame_id
    g.header.stamp = Time().to_msg()                  # zéro : « la plus récente »
    g.pose = goal.pose
    out = tf_buffer.transform(g, frame)
    return float(out.pose.position.x), float(out.pose.position.y)


class DecisionNode(Node):

    def __init__(self):
        super().__init__("decision_node")

        # ---------------------------------------------------------- paramètres
        self.rate = float(self.declare_parameter("rate", 10.0).value)                 # Hz, comme NavEnv
        self.goal_tolerance = float(self.declare_parameter("goal_tolerance", 0.25).value)  # m, comme NavEnv
        self.base_frame = str(self.declare_parameter("base_frame", "base_link").value)
        self.max_duration = float(self.declare_parameter("max_duration", 30.0).value)  # s, comme max_steps de NavEnv
        self.decision_mode = str(self.declare_parameter("decision_mode", "dwa").value)
        model_path = str(self.declare_parameter("model_path", "").value)
        self.velocity_source = str(self.declare_parameter("velocity_source", "ground_truth").value)
        if self.velocity_source not in VELOCITY_SOURCES:
            raise ValueError(f"velocity_source inconnue : {self.velocity_source!r} "
                             f"(choix : {', '.join(VELOCITY_SOURCES)})")

        # Échelles de l'action : les mêmes que NavEnv (ObsConfig), sinon une
        # action de 1 ne voudrait pas dire la même vitesse en ROS et en Gym.
        self.obs_cfg = ObsConfig()

        d = DWAParams()
        dwa_params = DWAParams(
            w_heading=float(self.declare_parameter("dwa.w_heading", d.w_heading).value),
            w_clearance=float(self.declare_parameter("dwa.w_clearance", d.w_clearance).value),
            w_velocity=float(self.declare_parameter("dwa.w_velocity", d.w_velocity).value),
            horizon=float(self.declare_parameter("dwa.horizon", d.horizon).value),
        )
        self.decider = make_decider(self.decision_mode, dwa_params, model_path)

        # ---------------------------------------------------------- état
        self.ranges: np.ndarray | None = None              # 36 distances du dernier scan (m)
        self.velocity: tuple[float, float] | None = None   # (v, omega) de la dernière odométrie
        self.goal: PoseStamped | None = None               # but courant, dans le repère où il a été donné
        self.prev_action = np.zeros(2, dtype=np.float32)   # action normalisée du cycle précédent
        self.t_decision_max = 0.0                          # pire temps de calcul d'une décision (s)
        self.scan_stamp = None                             # horodatage du dernier scan (pour la latence)

        # État partagé entre tick (fil du minuteur) et l'action (son propre fil)
        self.goal_dist: float | None = None                # dernière distance au but calculée par tick
        self.goal_reached = threading.Event()              # levé par tick quand le but est atteint
        self.in_contact = False                            # dernier état de /collision
        self.n_collisions = 0                              # nombre de contacts DISTINCTS depuis le début du but
        self.action_active = False                         # une action est en cours

        # ---------------------------------------------------------- TF2
        # Le Buffer garde les dernières transformations reçues sur /tf ;
        # le TransformListener le remplit en arrière-plan.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------------------------------------- ROS
        self.pub_cmd = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        # Vitesse : /ground_truth (comme NavEnv) ou /odom (comme un vrai robot).
        # Les deux sont des Odometry, avec la vitesse exprimée dans base_link.
        self.create_subscription(Odometry, VELOCITY_SOURCES[self.velocity_source], self.on_odom, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.create_subscription(Bool, "/collision", self.on_collision, 10)

        # Serveur d'action, dans un groupe réentrant : son rappel d'exécution
        # tourne en parallèle de tick (voir l'en-tête).
        self.action_server = ActionServer(
            self, NavigateToGoal, "navigate_to_goal",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        # Avec use_sim_time, ce minuteur suit le temps simulé : 10 Hz de simulation.
        self.timer = self.create_timer(1.0 / self.rate, self.tick)

        self.get_logger().info(f"décideur : {self.decision_mode}, vitesse : {self.velocity_source} ; "
                               "en attente d'un but (2D Goal Pose de RViz ou action /navigate_to_goal)")

    # -------------------------------------------------------------- callbacks

    def on_scan(self, msg: LaserScan) -> None:
        """Range les distances du scan dans self.ranges, en tableau NumPy.

        Un scan de mauvaise taille est signalé puis IGNORÉ (return) : on garde
        le précédent plutôt que de faire planter build_observation dans tick.
        """
        ranges = np.asarray(msg.ranges, dtype=float)
        if ranges.shape != (N_BEAMS,):
            self.get_logger().warn(f"scan de {ranges.size} distances, {N_BEAMS} attendues : ignoré",
                                   throttle_duration_sec=2.0)
            return
        # inf (pas de retour, REP 117) ou NaN : ramenés à la portée maximale,
        # avec la fonction que le test d'identité vérifie.
        self.ranges = ranges_from_msg(msg.ranges, msg.range_max)
        self.scan_stamp = msg.header.stamp                # repris dans /cmd_vel, pour la latence

    def on_odom(self, msg: Odometry) -> None:
        """Range (v, omega) dans self.velocity (depuis /ground_truth ou /odom).

        Les vitesses d'un message Odometry sont exprimées dans child_frame_id,
        ici base_link. Odometry contient un TwistWithCovariance, d'où twist.twist.
        """
        self.velocity = (float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z))

    def on_goal(self, msg: PoseStamped) -> None:
        """But cliqué dans RViz. Ignoré si une action est en cours."""
        if self.action_active:
            self.get_logger().warn("action en cours : but RViz ignoré")
            return
        self._start_goal(msg)

    def _start_goal(self, goal: PoseStamped) -> None:
        """Nouveau but : on le garde et on démarre un nouvel « épisode ».

        Comme NavEnv.reset : rien de l'ancien but ne doit influencer le nouveau.
        Utilisée par le clic RViz ET par l'action.
        """
        self.goal = goal                                  # mémoriser
        self.decider.reset()
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.goal_dist = None
        self.goal_reached.clear()                         # baisser le drapeau
        self.n_collisions = 0
        p = goal.pose.position
        self.get_logger().info(                           # journaliser : écrire dans le journal du noeud
            f"nouveau but : ({p.x:.2f}, {p.y:.2f}) dans le repère {goal.header.frame_id}")

    def on_collision(self, msg: Bool) -> None:
        """Compte les contacts distincts : un contact qui dure 1 s ne compte qu'une fois.

        On compte un « front montant » : passage de « pas de contact » à
        « contact ». self.in_contact garde l'état ACTUEL, vrai ou faux.
        """
        if msg.data and not self.in_contact:
            self.n_collisions += 1
        self.in_contact = msg.data

    # -------------------------------------------------------------- action

    def goal_callback(self, goal_request) -> GoalResponse:
        """Appelé quand un client envoie un but : on l'accepte ou on le refuse.

        Choix simple : un seul but à la fois. Un deuxième but pendant une
        navigation est refusé (le client voit « Goal was rejected »).
        """
        if self.action_active:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        """Appelé quand le client demande l'annulation. On accepte toujours ;
        l'arrêt effectif se fait dans la boucle de execute_callback."""
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Toute la navigation vers un but, dans son propre fil.

        Ce rappel ne commande pas le robot : c'est toujours tick qui décide.
        Lui se contente de surveiller, environ 10 fois par seconde :
            annulé ?  -> arrêter, goal_handle.canceled(), renvoyer le résultat
            atteint ? -> goal_handle.succeed(), renvoyer le résultat
            trop long ? -> arrêter, goal_handle.abort(), renvoyer le résultat
            sinon     -> publier le retour (distance restante) et attendre

        Durée : en temps SIMULÉ (self.get_clock(), avec use_sim_time), pour
        être comparable à NavEnv.
        """
        self.action_active = True
        self._start_goal(goal_handle.request.goal)
        t_start = self.get_clock().now()
        feedback = NavigateToGoal.Feedback()

        try:
            while rclpy.ok():
                duration = (self.get_clock().now() - t_start).nanoseconds * 1e-9

                # 1. annulation demandée
                if goal_handle.is_cancel_requested:
                    self.goal = None                  # tick n'a plus de but : il publie (0, 0)
                    goal_handle.canceled()
                    return self._make_result(success=False, duration=duration)

                # 2. but atteint (drapeau levé par tick)
                if self.goal_reached.is_set():
                    goal_handle.succeed()
                    return self._make_result(success=True, duration=duration)

                # 3. trop long : même limite que NavEnv (30 s)
                if duration > self.max_duration:
                    self.get_logger().warn(f"but non atteint en {self.max_duration:.0f} s : abandon")
                    self.goal = None
                    goal_handle.abort()
                    return self._make_result(success=False, duration=duration)

                # 4. sinon : retour au client, puis attendre un peu
                if self.goal_dist is not None:
                    feedback.distance_remaining = float(self.goal_dist)
                    goal_handle.publish_feedback(feedback)
                time.sleep(0.1)       # temps réel : on ne fait que surveiller, pas besoin de précision
        finally:
            self.action_active = False   # exécuté quoi qu'il arrive, même après un return

    def _make_result(self, success: bool, duration: float):
        """Construit le résultat de l'action.

        Les classes générées vérifient les types : duration doit être un float
        (champ float32), collisions un int (champ uint32), d'où float() et int().
        """
        result = NavigateToGoal.Result()
        result.success = bool(success)
        result.duration = float(duration)
        result.collisions = int(self.n_collisions)
        return result

    def tick(self) -> None:
        """Une décision, à chaque période du minuteur.

        Pourquoi publier (0, 0) au lieu de ne rien publier : control_node et le
        chien de garde de sim_bridge reçoivent un ordre explicite d'arrêt, et
        rqt_graph / ros2 topic hz montrent que le noeud est vivant.
        """
        # Copie locale : l'action (autre fil) peut remettre self.goal à None
        # pendant ce tick. Sans cette copie, self.goal.header pourrait planter
        # entre le test ci-dessous et son utilisation (situation de compétition).
        goal = self.goal

        # 1. il manque une donnée : on reste à l'arrêt
        if goal is None or self.ranges is None or self.velocity is None:
            self._publish_cmd(0.0, 0.0)
            return

        # 2. le but par rapport au robot, maintenant
        goal_b = self._goal_in_base_link(goal)
        if goal_b is None:                                # TF pas encore prête
            self._publish_cmd(0.0, 0.0)
            return

        # 3. but atteint : même critère que NavEnv
        dist = math.hypot(goal_b[0], goal_b[1])
        self.goal_dist = dist                             # lu par execute_callback pour le retour
        if dist < self.goal_tolerance:
            self._publish_cmd(0.0, 0.0)
            self.get_logger().info(f"but atteint (à {dist:.2f} m)")
            self.goal = None
            self.goal_reached.set()                       # lever le drapeau pour l'action
            return

        # 4. observation : dans base_link, le robot est à l'origine avec un cap nul,
        #    donc la même fonction que NavEnv donne la même observation
        obs = build_observation(self.ranges, pose=(0.0, 0.0, 0.0),
                                velocity=self.velocity, goal=goal_b,
                                prev_action=self.prev_action, cfg=self.obs_cfg)

        # 5. décision, chronométrée
        t0 = time.perf_counter()
        action = self.decider.act(obs)
        t_dec = time.perf_counter() - t0
        self.t_decision_max = max(self.t_decision_max, t_dec)
        if t_dec > 1.0 / self.rate:
            self.get_logger().warn(f"décision trop lente : {1e3 * t_dec:.0f} ms "
                                   f"pour une période de {1e3 / self.rate:.0f} ms",
                                   throttle_duration_sec=2.0)

        # 6. mémoriser l'action pour la prochaine observation
        self.prev_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # 7. action normalisée -> (v, omega) -> /cmd_vel
        v, w = self._action_to_cmd(self.prev_action)
        self._publish_cmd(v, w)

    # -------------------------------------------------------------- outils

    def _goal_in_base_link(self, goal_in: PoseStamped) -> tuple[float, float] | None:
        """Position (x, y) du but dans base_link, ou None si TF2 n'est pas prête.

        Pas de délai d'attente (timeout) : tick ne doit jamais bloquer. Si TF
        n'est pas prête, on renvoie None et on réessaie au cycle suivant.
        goal_in est passé en argument (et non lu dans self.goal) : voir la
        copie locale au début de tick.
        """
        try:
            return goal_in_frame(self.tf_buffer, goal_in, self.base_frame)
        except TransformException as e:
            self.get_logger().warn(f"TF {goal_in.header.frame_id} -> {self.base_frame} indisponible : {e}",
                                   throttle_duration_sec=2.0)
            return None

    def _action_to_cmd(self, action: np.ndarray) -> tuple[float, float]:
        """Action normalisée -> (v, omega) physiques, exactement comme NavEnv.step.

        v = a_v · obs_cfg.v_max,  omega = a_omega · obs_cfg.w_max, après écrêtage à [-1, 1].
        """
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return float(a[0] * self.obs_cfg.v_max), float(a[1] * self.obs_cfg.w_max)

    def _publish_cmd(self, v: float, w: float) -> None:
        """Publie (v, omega) sur /cmd_vel.

        Horodatage : celui du scan sur lequel la décision s'appuie (et l'instant
        présent seulement tant qu'aucun scan n'est arrivé). control_node en
        déduit la latence scan -> commande.
        """
        msg = TwistStamped()
        msg.header.stamp = self.scan_stamp if self.scan_stamp is not None else self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def destroy_node(self):
        self.get_logger().info(f"pire temps de décision : {1e3 * self.t_decision_max:.1f} ms")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    # Plusieurs fils : l'action (groupe réentrant) tourne pendant que tick et
    # les abonnements (groupe par défaut, un rappel à la fois) continuent.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()