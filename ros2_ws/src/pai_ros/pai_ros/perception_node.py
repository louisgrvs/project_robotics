"""Nœud de perception (jour 8) : /scan → /scan_proc et /local_map.

Comme sim_bridge, ce nœud ne contient aucun algorithme : il appelle pai_core.
    /scan_proc  : preprocess_ranges (features.py), la MÊME normalisation que NavEnv.
    /local_map  : OccupancyGrid (occupancy_grid.py), mise à jour en log-odds.

Pose du lidar
    La carte est construite dans map_frame (map par défaut) : il faut la pose
    du lidar dans ce repère À L'INSTANT DU SCAN, lue dans TF. Avec map, cette
    pose repose sur la vérité terrain (localisation parfaite). Avec
    map_frame:=odom, elle repose sur l'odométrie seule : avec une odométrie
    faussée, les obstacles se dédoublent, ce qui montre pourquoi map -> odom existe.

Attente de la TF
    La TF d'un cycle et le scan du même cycle sont publiés ensemble, mais rien
    ne garantit que la TF arrive avant le scan. On ne bloque JAMAIS dans un
    rappel (l'exécuteur mono-fil ne pourrait plus recevoir la TF attendue) :
    les scans attendent dans une file, retentée à chaque scan et par un
    minuteur, et sont abandonnés au-delà de tf_wait secondes.

Temps : à lancer avec use_sim_time:=true (les messages portent le temps simulé).

Topics
    reçu    /scan        sensor_msgs/LaserScan       (best effort)
    publié  /scan_proc   std_msgs/Float32MultiArray  36 distances normalisées dans [0, 1]
    publié  /local_map   nav_msgs/OccupancyGrid      (reliable, transient local)
"""
import math
import time
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from pai_core.envs.nav_env import NavEnvConfig
from pai_core.perception.features import N_BEAMS, preprocess_ranges
from pai_core.perception.occupancy_grid import GridConfig, OccupancyGrid


def yaw_from_quaternion(q) -> float:
    """Lacet d'un geometry_msgs/Quaternion (convention ZYX)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception")

        # ------------------------------------------------------------ paramètres
        self.map_frame = str(self.declare_parameter("map_frame", "map").value)
        map_rate = float(self.declare_parameter("map_rate", 2.0).value)           # Hz
        self.tf_wait = float(self.declare_parameter("tf_wait", 0.5).value)        # s (temps simulé)
        resolution = float(self.declare_parameter("resolution", 0.05).value)
        margin = float(self.declare_parameter("margin", 0.2).value)
        free_on_no_return = bool(self.declare_parameter("free_on_no_return", True).value)

        if not self.get_parameter("use_sim_time").value:
            self.get_logger().warn("use_sim_time est faux : les horodatages du temps simulé "
                                   "seront comparés à l'horloge système (lance avec use_sim_time:=true)")

        # Même configuration que NavEnv et sim_bridge : même portée, même arène.
        cfg = NavEnvConfig()
        self.obs_cfg = cfg.obs
        self.grid = OccupancyGrid(GridConfig.for_arena(cfg.world.arena_size, margin=margin,
                                                       resolution=resolution,
                                                       free_on_no_return=free_on_no_return))

        # ------------------------------------------------------------------ TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pending: deque = deque(maxlen=20)       # scans qui attendent leur TF

        # ------------------------------------------------------------------ ROS
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.pub_proc = self.create_publisher(Float32MultiArray, "/scan_proc", 10)
        # Carte : « reliable » et « transient local », comme une carte ROS classique.
        # Un abonné qui arrive plus tard (RViz relancé) reçoit la dernière carte.
        # Attention : l'affichage Map de RViz doit avoir les mêmes réglages.
        map_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(OccupancyGridMsg, "/local_map", map_qos)

        self.create_timer(0.05, self.process_pending)             # retente la file à 20 Hz
        self.create_timer(1.0 / map_rate, self.publish_map)
        self.create_timer(10.0, self.report)

        self.last_stamp = None
        self.n_integrated = self.n_dropped = 0
        self.update_time_max = 0.0
        self.warned_range = False

        g = self.grid
        self.get_logger().info(f"carte {g.nx} × {g.ny} cases de {g.cfg.resolution * 100:.0f} cm "
                               f"dans « {self.map_frame} », origine ({g.origin[0]:.2f}, {g.origin[1]:.2f})")

    # ------------------------------------------------------------------ scans
    def on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        if ranges.shape != (N_BEAMS,):
            self.get_logger().warn(f"/scan : {N_BEAMS} faisceaux attendus, reçu {ranges.size}",
                                   throttle_duration_sec=5.0)
            return
        if not self.warned_range and abs(msg.range_max - self.obs_cfg.range_max) > 1e-6:
            self.get_logger().warn(f"portée du scan ({msg.range_max} m) différente de celle de "
                                   f"l'observation ({self.obs_cfg.range_max} m)")
            self.warned_range = True

        # 1. /scan_proc : aucune TF nécessaire, on publie tout de suite.
        out = Float32MultiArray()
        out.data = preprocess_ranges(ranges, self.obs_cfg).tolist()
        self.pub_proc.publish(out)

        # 2. Carte : il faut la pose du lidar au moment du scan.
        self.pending.append(msg)
        self.process_pending()

    def process_pending(self):
        """Intègre les scans dont la TF est arrivée, dans l'ordre ; abandonne les trop vieux."""
        now = self.get_clock().now()
        while self.pending:
            msg = self.pending[0]
            stamp = Time.from_msg(msg.header.stamp)
            if self.tf_buffer.can_transform(self.map_frame, msg.header.frame_id, stamp):
                self.pending.popleft()
                self.integrate(msg, stamp)
            elif (now - stamp) > Duration(seconds=self.tf_wait):
                self.pending.popleft()                      # TF jamais arrivée : on abandonne
                self.n_dropped += 1
            else:
                break                                       # on attend : la TF devrait arriver

    def integrate(self, msg: LaserScan, stamp: Time):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, stamp)
        except TransformException as e:
            self.n_dropped += 1
            self.get_logger().warn(f"TF {self.map_frame} -> {msg.header.frame_id} : {e}",
                                   throttle_duration_sec=5.0)
            return
        tr = t.transform.translation
        laser_pose = (tr.x, tr.y, yaw_from_quaternion(t.transform.rotation))

        n = len(msg.ranges)
        angles = msg.angle_min + msg.angle_increment * np.arange(n)   # le message fait foi
        t0 = time.perf_counter()
        self.grid.update(laser_pose, angles, np.asarray(msg.ranges, dtype=np.float64),
                         max_range=msg.range_max, range_min=msg.range_min)
        self.update_time_max = max(self.update_time_max, time.perf_counter() - t0)
        self.n_integrated += 1
        self.last_stamp = msg.header.stamp

    # ------------------------------------------------------------------ carte
    def publish_map(self):
        if self.last_stamp is None:
            return
        g = self.grid
        msg = OccupancyGridMsg()
        msg.header.stamp = self.last_stamp               # instant du dernier scan intégré
        msg.header.frame_id = self.map_frame
        msg.info.map_load_time = self.last_stamp
        msg.info.resolution = float(g.cfg.resolution)
        msg.info.width, msg.info.height = g.nx, g.ny     # width : nombre de cases selon x
        msg.info.origin.position.x = float(g.origin[0])  # coin bas-gauche de la case (0, 0)
        msg.info.origin.position.y = float(g.origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = g.to_ros_values().tolist()            # index = ix + iy·width
        self.pub_map.publish(msg)

    def report(self):
        self.get_logger().info(f"scans intégrés {self.n_integrated}, abandonnés {self.n_dropped}, "
                               f"en attente {len(self.pending)} | mise à jour la plus longue "
                               f"{self.update_time_max * 1e3:.1f} ms | cases observées "
                               f"{self.grid.observed.mean():.0%}")
        self.update_time_max = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
