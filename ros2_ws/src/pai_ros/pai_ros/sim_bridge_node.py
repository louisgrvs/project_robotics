"""Pont entre la simulation MuJoCo et ROS 2 (jour 7).

Ce nœud ne contient aucune physique ni aucun capteur : il appelle pai_core,
exactement comme NavEnv. Le monde (World), le lidar (Lidar), les encodeurs
(WheelEncoders) et l'odométrie (WheelOdometry) sont les mêmes objets que
dans l'environnement Gymnasium, ce qui prépare le test d'identité du jour 10.

Horloge
    Ce nœud POSSÈDE l'horloge. Un minuteur cadencé sur le temps réel (temps
    « mur ») avance la physique, puis publie le temps simulé sur /clock.
    Les autres nœuds tourneront avec use_sim_time:=true ; celui-ci jamais,
    sinon son minuteur attendrait un /clock que lui seul publie.

Cadences, avec le pas physique de 2 ms
    1 cycle du minuteur = 10 pas physiques = 20 ms  ->  /clock, /odom, /ground_truth à 50 Hz
    1 scan tous les 5 cycles = 50 pas = 100 ms      ->  /scan à 10 Hz
    50 pas physiques par scan, c'est une décision de NavEnv (frame_skip).

Topics publiés
    /clock         rosgraph_msgs/Clock         temps simulé
    /odom          nav_msgs/Odometry           odométrie des encodeurs  (odom -> base_link)
    /ground_truth  nav_msgs/Odometry           pose vraie               (map  -> base_link)
    /scan          sensor_msgs/LaserScan       36 faisceaux             (laser_link)
Topic reçu
    /wheel_cmd     std_msgs/Float64MultiArray  [ω_gauche, ω_droite] en rad/s

Sûreté : sans commande pendant cmd_timeout secondes de temps simulé, les
roues reçoivent une consigne nulle.
"""
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray

from pai_core.envs.nav_env import NavEnvConfig
from pai_core.robot.odometry import WheelOdometry, wrap_angle
from pai_core.sim.sensors import Lidar, LidarConfig, WheelEncoders
from pai_core.sim.world import World


def set_yaw(q_msg, yaw):
    """Remplit un geometry_msgs/Quaternion pour une rotation de yaw autour de z."""
    q_msg.x, q_msg.y = 0.0, 0.0
    q_msg.z, q_msg.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class SimBridge(Node):
    def __init__(self):
        super().__init__("sim_bridge")

        # ------------------------------------------------------------ paramètres
        seed = int(self.declare_parameter("seed", 0).value)
        n_obstacles = int(self.declare_parameter("n_obstacles", -1).value)  # -1 : tiré comme dans NavEnv
        self.rate = float(self.declare_parameter("rate", 50.0).value)         # Hz
        self.scan_every = int(self.declare_parameter("scan_every", 5).value)   # cycles par scan
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout", 0.5).value)
        lidar_noise = float(self.declare_parameter("lidar_noise_std", 0.0).value)
        lidar_dropout = float(self.declare_parameter("lidar_dropout", 0.0).value)
        use_viewer = bool(self.declare_parameter("viewer", True).value)

        if self.get_parameter("use_sim_time").value:
            raise RuntimeError("sim_bridge publie /clock : lance-le avec use_sim_time:=false")

        # ------------------------------------------------- monde : comme NavEnv.reset
        # Même configuration que l'environnement Gymnasium : mêmes formes d'obstacles,
        # même portée du lidar, même nombre de pas de stabilisation.
        cfg = NavEnvConfig()
        self.world = World(cfg.world)
        self.model, self.data = self.world.model, self.world.data
        m, d = self.model, self.data

        rng = np.random.default_rng(seed)
        if n_obstacles < 0:
            lo, hi = cfg.n_obstacles
            n_obstacles = int(rng.integers(lo, hi + 1))
        self.scene = self.world.sample_scene(rng, n_obstacles)
        self.world.apply(self.scene)
        d.ctrl[:] = 0.0
        for _ in range(cfg.settle_steps):          # laisse les contacts roues-sol s'établir
            mujoco.mj_step(m, d)

        # ------------------------------------------------------------ cadences
        dt = m.opt.timestep
        self.n_sub = round(1.0 / (self.rate * dt))     # pas physiques par cycle (10)
        self.cycle_dt = self.n_sub * dt                # durée simulée d'un cycle (20 ms)
        if abs(self.cycle_dt * self.rate - 1.0) > 1e-9:
            self.get_logger().warn(f"1/rate n'est pas un multiple du pas physique : cycle de {self.cycle_dt} s")

        # ------------------------------------------------------------ capteurs
        geom = self.world.geom
        self.lidar = Lidar(geom, LidarConfig(max_range=cfg.obs.range_max,
                                             noise_std=lidar_noise, dropout_prob=lidar_dropout),
                           rng=np.random.default_rng(seed + 1))
        self.encoders = WheelEncoders(geom)
        self.odom = WheelOdometry(geom.wheel_radius, geom.wheel_radius, geom.wheel_base)
        self.odom.reset(wheel_angles=self.encoders.read(d))   # odom part de (0, 0, 0)
        self.prev_odom_pose = self.odom.pose.copy()

        # ------------------------------------------------------------------ ROS
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_gt = self.create_publisher(Odometry, "/ground_truth", 10)
        # Capteur : « best effort », on préfère perdre un scan que le recevoir en retard.
        # Attention : un abonné « reliable » (réglage par défaut de RViz) ne reçoit rien.
        self.pub_scan = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, "/wheel_cmd", self.on_wheel_cmd, 10)

        # Exécuteur mono-fil (rclpy.spin) : on_wheel_cmd et step ne s'exécutent
        # jamais en même temps, donc aucun verrou n'est nécessaire sur self.data.
        self.timer = self.create_timer(1.0 / self.rate, self.step)

        self.viewer = mujoco.viewer.launch_passive(m, d) if use_viewer else None

        self.cycle = 0
        self.last_cmd_time = -math.inf
        self.rtf_wall0, self.rtf_sim0 = time.perf_counter(), d.time
        self.cb_max = 0.0

        self.log_scene(seed)

    # ------------------------------------------------------------------ commande
    def on_wheel_cmd(self, msg):
        if len(msg.data) != 2:
            self.get_logger().warn("/wheel_cmd attend [ω_gauche, ω_droite]", throttle_duration_sec=2.0)
            return
        # MuJoCo borne lui-même ctrl à ctrlrange (ctrllimited)
        self.data.ctrl[self.world.ctrl_left] = float(msg.data[0])
        self.data.ctrl[self.world.ctrl_right] = float(msg.data[1])
        self.last_cmd_time = self.data.time

    # ------------------------------------------------------------------ boucle
    def step(self):
        t_start = time.perf_counter()
        d = self.data

        if d.time - self.last_cmd_time > self.cmd_timeout:   # chien de garde
            d.ctrl[:] = 0.0

        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, d)
        self.cycle += 1

        if self.world.in_collision():
            self.get_logger().warn("collision", throttle_duration_sec=1.0)

        # Tous les messages d'un cycle portent le même horodatage : le temps simulé.
        stamp = Time(nanoseconds=round(d.time * 1e9)).to_msg()
        clock = Clock()
        clock.clock = stamp
        self.pub_clock.publish(clock)

        self.publish_odom(stamp)
        self.publish_ground_truth(stamp)
        if self.cycle % self.scan_every == 0:
            self.publish_scan(stamp)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        self.cb_max = max(self.cb_max, time.perf_counter() - t_start)
        self.report_real_time_factor()

    # ------------------------------------------------------------------ messages
    def publish_odom(self, stamp):
        x, y, th = self.odom.update(self.encoders.read(self.data))

        # Vitesses déduites de deux poses successives, en inversant le schéma du
        # point milieu : dθ = θ1 − θ0 et Δs = projection du déplacement sur le cap moyen.
        x0, y0, th0 = self.prev_odom_pose
        dth = float(wrap_angle(th - th0))
        th_mid = th0 + dth / 2.0
        ds = (x - x0) * math.cos(th_mid) + (y - y0) * math.sin(th_mid)
        self.prev_odom_pose = np.array([x, y, th])

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        set_yaw(msg.pose.pose.orientation, float(th))
        msg.twist.twist.linear.x = ds / self.cycle_dt     # vitesses exprimées dans base_link
        msg.twist.twist.angular.z = dth / self.cycle_dt
        self.pub_odom.publish(msg)

    def publish_ground_truth(self, stamp):
        x, y, yaw = self.world.robot_pose()       # milieu de l'essieu, repère monde
        v, w = self.world.robot_velocity()

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        set_yaw(msg.pose.pose.orientation, yaw)
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = w
        self.pub_gt.publish(msg)

    def publish_scan(self, stamp):
        ranges = self.lidar.scan(self.model, self.data)
        n = len(ranges)
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = "laser_link"
        msg.angle_min = 0.0                               # faisceau 0 vers l'avant
        msg.angle_increment = 2.0 * math.pi / n           # angles croissants vers la gauche
        msg.angle_max = (n - 1) * msg.angle_increment
        msg.time_increment = 0.0                          # tous les faisceaux au même instant
        msg.scan_time = self.scan_every * self.cycle_dt
        msg.range_min = 0.0
        msg.range_max = float(self.lidar.cfg.max_range)
        msg.ranges = ranges.astype(np.float32).tolist()
        self.pub_scan.publish(msg)

    # ------------------------------------------------------------------ mesures
    def report_real_time_factor(self, every=5.0):
        now = time.perf_counter()
        if now - self.rtf_wall0 < every:
            return
        rtf = (self.data.time - self.rtf_sim0) / (now - self.rtf_wall0)
        self.get_logger().info(
            f"facteur temps réel {rtf:.3f} | cycle le plus long {self.cb_max * 1e3:.1f} ms "
            f"(budget {1e3 / self.rate:.0f} ms) | t_sim = {self.data.time:.1f} s")
        self.rtf_wall0, self.rtf_sim0, self.cb_max = now, self.data.time, 0.0

    def log_scene(self, seed):
        """Affiche la scène, pour vérifier /scan à la main."""
        s = self.scene
        lines = [f"scène (graine {seed}) : départ x={s.start[0]:.2f} y={s.start[1]:.2f} "
                 f"cap={math.degrees(s.start[2]):.0f}°, but ({s.goal[0]:.2f}, {s.goal[1]:.2f})"]
        for ob in s.obstacles:
            shape = self.world.shapes[ob.slot]
            lines.append(f"  obstacle {ob.slot} : {shape.kind} en ({ob.x:.2f}, {ob.y:.2f}), "
                         f"demi-tailles {shape.hx:.2f} × {shape.hy:.2f}, lacet {math.degrees(ob.yaw):.0f}°")
        lines.append(f"  pas physique {self.model.opt.timestep * 1e3:.0f} ms, {self.n_sub} pas par cycle")
        self.get_logger().info("\n".join(lines))

    def destroy_node(self):
        if self.viewer is not None:
            self.viewer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimBridge()
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
