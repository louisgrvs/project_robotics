"""Noeud de décision : (scan, vitesse, but) -> /cmd_vel, à 10 Hz.

Rôle : traduire le monde ROS (messages asynchrones, repères, temps simulé)
en exactement le même vecteur d'observation que NavEnv, appeler le décideur,
puis retraduire son action en /cmd_vel. En gros : c'est l'interprète entre ROS et le décideur.

    /scan ──────┐
    /odom ──────┼─> build_observation ─> decider.act(obs) ─>action ∈ [-1, 1]^2
    /goal_pose ─┘    (features.py)        (DWA aujourd'hui,        │
      + TF2 map→base_link                  PPO et MPPI plus tard)  ▼
                                                    (v, omega) = action · échelles
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
Sorties  : /cmd_vel    geometry_msgs/TwistStamped   commande demandée, vers control_node

Rappel : dans une fonction de rappel, `msg` est LE message reçu (avec ses
valeurs) ; LaserScan, Odometry... sont seulement les types (les moules).
"""
import math
import time

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (enregistre PoseStamped auprès de tf2 : sans lui, transform échoue)
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from pai_core.control.dwa import DWA, DWAParams
from pai_core.perception.features import N_BEAMS, ObsConfig, build_observation


class DecisionNode(Node):

    def __init__(self):
        super().__init__("decision_node")

        # ---------------------------------------------------------- paramètres
        self.rate = float(self.declare_parameter("rate", 10.0).value)                 # Hz, comme NavEnv
        self.goal_tolerance = float(self.declare_parameter("goal_tolerance", 0.25).value)  # m, comme NavEnv
        self.base_frame = str(self.declare_parameter("base_frame", "base_link").value)

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
        # Jour 10 : un paramètre decision_mode choisira ici entre DWA, PPO et MPPI.
        self.decider = DWA(dwa_params)

        # ---------------------------------------------------------- état
        self.ranges: np.ndarray | None = None              # 36 distances du dernier scan (m)
        self.velocity: tuple[float, float] | None = None   # (v, omega) de la dernière odométrie
        self.goal: PoseStamped | None = None               # but courant, dans le repère où RViz l'a donné
        self.prev_action = np.zeros(2, dtype=np.float32)   # action normalisée du cycle précédent
        self.t_decision_max = 0.0                          # pire temps de calcul d'une décision (s)

        # ---------------------------------------------------------- TF2
        # Le Buffer garde les dernières transformations reçues sur /tf ;
        # le TransformListener le remplit en arrière-plan.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------------------------------------- ROS
        self.pub_cmd = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)

        # Avec use_sim_time, ce minuteur suit le temps simulé : 10 Hz de simulation.
        self.timer = self.create_timer(1.0 / self.rate, self.tick)

        self.get_logger().info("décideur : DWA, en attente d'un but (outil 2D Goal Pose de RViz)")

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
        # Faisceau sans impact codé inf ou nan par un autre capteur : on le
        # ramène à la portée maximale, comme le fait sim_bridge. Sinon un nan
        # traverserait build_observation et empoisonnerait l'observation.
        self.ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max)

    def on_odom(self, msg: Odometry) -> None:
        """Range (v, omega) dans self.velocity.

        Les vitesses d'un message Odometry sont exprimées dans child_frame_id,
        ici base_link. Odometry contient un TwistWithCovariance, d'où twist.twist.
        """
        self.velocity = (float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z))

    def on_goal(self, msg: PoseStamped) -> None:
        """Nouveau but : on le garde et on démarre un nouvel « épisode ».

        Comme NavEnv.reset : rien de l'ancien but ne doit influencer le nouveau.
        """
        self.goal = msg                                   # mémoriser
        self.decider.reset()
        self.prev_action = np.zeros(2, dtype=np.float32)
        p = msg.pose.position
        self.get_logger().info(                           # journaliser : écrire dans le journal du noeud
            f"nouveau but : ({p.x:.2f}, {p.y:.2f}) dans le repère {msg.header.frame_id}")

    def tick(self) -> None:
        """Une décision, à chaque période du minuteur.

        Pourquoi publier (0, 0) au lieu de ne rien publier : control_node et le
        chien de garde de sim_bridge reçoivent un ordre explicite d'arrêt, et
        rqt_graph / ros2 topic hz montrent que le noeud est vivant.
        """
        # 1. il manque une donnée : on reste à l'arrêt
        if self.goal is None or self.ranges is None or self.velocity is None:
            self._publish_cmd(0.0, 0.0)
            return

        # 2. le but par rapport au robot, maintenant
        goal_b = self._goal_in_base_link()
        if goal_b is None:                                # TF pas encore prête
            self._publish_cmd(0.0, 0.0)
            return

        # 3. but atteint : même critère que NavEnv
        dist = math.hypot(goal_b[0], goal_b[1])
        if dist < self.goal_tolerance:
            self._publish_cmd(0.0, 0.0)
            self.get_logger().info(f"but atteint (à {dist:.2f} m)")
            self.goal = None
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

    def _goal_in_base_link(self) -> tuple[float, float] | None:
        """Position (x, y) du but dans base_link, ou None si TF2 échoue.

        Piège important : RViz horodate le but à l'instant du clic. Si on
        le transforme tel quel, TF2 donne le but dans base_link à cet instant-là,
        c'est-à-dire là où était le robot au moment du clic.
        Le but est fixe dans map : on veut la transformation la plus
        récente. On transforme donc une copie dont l'horodatage vaut zéro
        (Time().to_msg()), ce qui veut dire « la dernière disponible ».

        Pas de délai d'attente (timeout) : tick tourne dans l'unique fil
        d'exécution du noeud, celui-là même qui reçoit /tf. Attendre ici
        bloquerait la réception des transformations qu'on attend. Si TF
        n'est pas prête, on renvoie None et on réessaie au cycle suivant.
        """
        goal = PoseStamped()
        goal.header.frame_id = self.goal.header.frame_id
        goal.header.stamp = Time().to_msg()               # zéro : « la plus récente »
        goal.pose = self.goal.pose
        try:
            out = self.tf_buffer.transform(goal, self.base_frame)
        except TransformException as e:
            self.get_logger().warn(f"TF {goal.header.frame_id} -> {self.base_frame} indisponible : {e}",
                                   throttle_duration_sec=2.0)
            return None
        return (float(out.pose.position.x), float(out.pose.position.y))

    def _action_to_cmd(self, action: np.ndarray) -> tuple[float, float]:
        """Action normalisée -> (v, omega) physiques, exactement comme NavEnv.step.

        v = a_v · obs_cfg.v_max,  omega = a_omega · obs_cfg.w_max, après écrêtage à [-1, 1].
        """
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return float(a[0] * self.obs_cfg.v_max), float(a[1] * self.obs_cfg.w_max)

    def _publish_cmd(self, v: float, w: float) -> None:
        """Publie (v, omega) sur /cmd_vel."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()