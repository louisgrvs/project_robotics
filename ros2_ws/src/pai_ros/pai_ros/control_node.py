"""Noeud de commande : /cmd_vel (v, omega) -> /wheel_cmd (omega_gauche, omega_droite).

Chaîne appliquée à chaque /cmd_vel reçu, dans cet ordre :
  1. saturate          : la commande devient exécutable, courbure préservée
  2. filtre de sûreté  : peut annuler v si un obstacle est trop proche
  3. inverse           : (v, omega) -> vitesses des roues
  4. publication       : /wheel_cmd, plus /cmd_vel_safe pour le débogage

Entrées  : /cmd_vel   geometry_msgs/TwistStamped  (decision_node)
           /scan      sensor_msgs/LaserScan       (sim_bridge, scan brut)
Sorties  : /wheel_cmd             std_msgs/Float64MultiArray  [omega_g, omega_d] (rad/s)
           /cmd_vel_safe          geometry_msgs/TwistStamped  commande réellement envoyée
           /safety/interventions  std_msgs/UInt32             compteur cumulé

Le noeud ne contient que la plomberie ROS. Toute la logique est dans
pai_core (diff_drive, control/safety) et se teste sans ROS.

Le filtre lit /scan et non /scan_proc : la sûreté ne doit dépendre ni de la
latence ni d'un bogue éventuel de la perception.

Rappel : une fonction de rappel (on_scan, on_cmd_vel) ne renvoie rien, rclpy
jetterait la valeur. Elle communique en rangeant dans self ou en publiant.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, UInt32

from pai_core.control import safety
from pai_core.robot.diff_drive import DiffDriveParams, inverse, saturate


class ControlNode(Node):

    def __init__(self):
        super().__init__("control_node")

        # ---------------------------------------------------------- paramètres
        # Géométrie : les valeurs par défaut de DiffDriveParams, les mêmes que
        # le DWA. À vérifier une fois contre DiffDriveParams.from_mujoco.
        self.dd = DiffDriveParams()

        # Bornes optionnelles en plus de la limite des roues (0 = pas de borne)
        v_max = float(self.declare_parameter("v_max", 0.0).value)
        w_max = float(self.declare_parameter("w_max", 0.0).value)
        self.v_max = v_max or None
        self.w_max = w_max or None

        d = safety.SafetyParams()        # valeurs par défaut, écrasées par params.yaml
        self.sp = safety.SafetyParams(
            enabled=bool(self.declare_parameter("safety.enabled", d.enabled).value),
            robot_radius=float(self.declare_parameter("safety.robot_radius", d.robot_radius).value),
            lidar_offset_x=float(self.declare_parameter("safety.lidar_offset_x", d.lidar_offset_x).value),
            corridor_margin=float(self.declare_parameter("safety.corridor_margin", d.corridor_margin).value),
            dec=float(self.declare_parameter("safety.dec", d.dec).value),
            latency=float(self.declare_parameter("safety.latency", d.latency).value),
            stop_margin=float(self.declare_parameter("safety.stop_margin", d.stop_margin).value),
            scan_timeout=float(self.declare_parameter("safety.scan_timeout", d.scan_timeout).value),
        )

        # ---------------------------------------------------------- état
        self.points: np.ndarray | None = None   # dernier scan, en points base_link (N, 2)
        self.scan_stamp: Time | None = None     # horodatage de ce scan (temps simulé)
        self.was_intervening = False            # pour détecter le début d'une intervention
        self.n_interventions = 0                # nombre d'interventions (fronts montants)
        self.n_cycles_filtered = 0              # nombre de commandes modifiées par le filtre
        self.n_stale = 0                        # commandes bloquées faute de scan récent

        # ---------------------------------------------------------- ROS
        self.pub_wheels = self.create_publisher(Float64MultiArray, "/wheel_cmd", 10)
        self.pub_safe = self.create_publisher(TwistStamped, "/cmd_vel_safe", 10)
        self.pub_count = self.create_publisher(UInt32, "/safety/interventions", 10)

        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, "/cmd_vel", self.on_cmd_vel, 10)

        self.get_logger().info(f"filtre de sûreté : {'actif' if self.sp.enabled else 'inactif'}")

    # -------------------------------------------------------------- callbacks

    def on_scan(self, msg: LaserScan) -> None:
        """Convertit le scan en points base_link et mémorise son horodatage.

        Aucun return : on range dans self, on_cmd_vel relira ces valeurs.
        """
        self.points = safety.scan_to_points(
            np.asarray(msg.ranges, dtype=float),
            msg.angle_min, msg.angle_increment, msg.range_max,
            self.sp.lidar_offset_x,
        )
        self.scan_stamp = Time.from_msg(msg.header.stamp)

    def on_cmd_vel(self, msg: TwistStamped) -> None:
        """Toute la chaîne de commande, appelée à chaque /cmd_vel (10 Hz).

        Chaque étape récupère le résultat de la précédente par une affectation.
        """
        # 1. lecture de la commande demandée
        v = msg.twist.linear.x
        w = msg.twist.angular.z

        # 2. saturation : même arc, parcouru moins vite si nécessaire
        v, w = saturate(v, w, self.dd, self.v_max, self.w_max)
        v, w = float(v), float(w)          # tableaux NumPy de dimension 0 -> nombres

        # 3. filtre de sûreté
        if self.sp.enabled and not self._scan_is_fresh():
            # On ne roule pas à l'aveugle. Ce n'est une intervention que si
            # la commande change réellement (v déjà nul : rien à bloquer).
            blocked = v != 0.0
            if blocked:
                self.n_stale += 1
                self.get_logger().warn("scan absent ou périmé : avance bloquée",
                                       throttle_duration_sec=2.0)
            decision = safety.SafetyDecision(v=0.0, w=w, intervened=blocked,
                                             free_dist=float("nan"))
        else:
            # Filtre inactif sans scan reçu : tableau vide -> distance libre infinie
            points = self.points if self.points is not None else np.empty((0, 2))
            decision = safety.apply(v, w, points, self.sp)

        # 4. comptabilité
        self._count(decision.intervened)

        # 5. cinématique inverse, sur la commande APRÈS filtre (decision.v, decision.w)
        # Pas besoin de re-saturer : avec v = 0, chaque roue tourne à L·|w|/(2r),
        # qui est inférieur à max(|v ± L·w/2|)/r. Si la commande saturée était
        # exécutable, la commande filtrée l'est aussi.
        omega_l, omega_r = inverse(decision.v, decision.w, self.dd)

        # 6. publications
        self.pub_wheels.publish(Float64MultiArray(data=[float(omega_l), float(omega_r)]))

        safe = TwistStamped()
        safe.header = msg.header               # même horodatage et même repère que la demande
        safe.twist.linear.x = decision.v
        safe.twist.angular.z = decision.w
        self.pub_safe.publish(safe)

    # -------------------------------------------------------------- outils

    def _scan_is_fresh(self) -> bool:
        """Vrai si un scan existe et date de moins de scan_timeout.

        âge = maintenant - instant du scan : c'est une durée, qu'on compare
        à la durée scan_timeout. now() donne le temps simulé seulement si
        use_sim_time vaut true pour ce noeud.
        """
        if self.scan_stamp is None:
            return False
        age = (self.get_clock().now() - self.scan_stamp).nanoseconds * 1e-9   # s
        return age < self.sp.scan_timeout

    def _count(self, intervened: bool) -> None:
        """Met à jour les compteurs et publie /safety/interventions.

        Une intervention = un front montant (False -> True). Un robot bloqué
        devant un mur pendant 3 s compte 1 intervention et 30 cycles filtrés.
        """
        if intervened:
            self.n_cycles_filtered += 1
            if not self.was_intervening:       # début d'une nouvelle intervention
                self.n_interventions += 1
                self.get_logger().info(f"intervention n° {self.n_interventions} du filtre de sûreté")
        self.was_intervening = intervened
        self.pub_count.publish(UInt32(data=self.n_interventions))

    def destroy_node(self):
        self.get_logger().info(
            f"bilan : {self.n_interventions} interventions, "
            f"{self.n_cycles_filtered} cycles filtrés, {self.n_stale} commandes sans scan"
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
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
