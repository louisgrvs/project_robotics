"""Pont entre la simulation MuJoCo et ROS 2.

Ce nœud ne contient aucune physique ni aucun capteur : il appelle pai_core,
exactement comme NavEnv. Le monde (World), le lidar (Lidar), les encodeurs
(WheelEncoders) et l'odométrie (WheelOdometry) sont les mêmes objets que
dans l'environnement Gymnasium, ce qui prépare le test d'identité du jour 10.

Horloge
    Ce noeud POSSÈDE l'horloge. Un minuteur cadencé sur le temps réel (temps
    « mur ») avance la physique, puis publie le temps simulé sur /clock.
    Les autres nœuds tourneront avec use_sim_time:=true ; celui-ci jamais,
    sinon son minuteur attendrait un /clock que lui seul publie.

Cadences, avec le pas physique de 2 ms
    1 cycle du minuteur = 10 pas physiques = 20 ms  ->  /clock, /odom, /ground_truth, /tf à 50 Hz
    1 scan tous les 5 cycles = 50 pas = 100 ms      ->  /scan à 10 Hz
    50 pas physiques par scan, c'est une décision de NavEnv (frame_skip).

Arbre des repères (REP 105)
    map ──(50 Hz, vérité terrain)──> odom ──(50 Hz, odométrie)──> base_link ──(statique)──> laser_link

    Un repère n'a qu'UN parent : base_link a déjà odom pour parent, donc on ne
    publie jamais map -> base_link sur /tf. La vérité terrain passe par
        T_map_odom = T_map_base ∘ (T_odom_base)⁻¹
    ce qui suppose une localisation parfaite. map -> odom est la correction de
    la dérive de l'odométrie : à t = 0 elle vaut la pose de départ, puis elle
    s'en éloigne à mesure que l'odométrie dérive.

Topics publiés
    /clock         rosgraph_msgs/Clock         temps simulé
    /odom          nav_msgs/Odometry           odométrie des encodeurs  (odom -> base_link)
    /ground_truth  nav_msgs/Odometry           pose vraie               (map  -> base_link), message seulement, pas une TF
    /scan          sensor_msgs/LaserScan       36 faisceaux             (laser_link)
    /tf            map -> odom, odom -> base_link
    /tf_static     base_link -> laser_link
    /collision     std_msgs/Bool               vrai pendant un contact robot-obstacle (jour 10)
Topic reçu
    /wheel_cmd     std_msgs/Float64MultiArray  [ω_gauche, ω_droite] en rad/s
Service (jour 10)
    /reset_world   pai_interfaces/srv/ResetWorld   nouvelle scène, sans relancer les nœuds

Remise à zéro et temps simulé
    mj_resetData remet data.time à 0. Si on le laissait faire, /clock
    reculerait, et TF comme les minuteurs des autres nœuds verraient le temps
    revenir en arrière (TF jette alors ses données). reset_scene conserve donc
    le temps : il continue d'avancer à travers la remise à zéro.

Les conversions vers les conventions ROS (inf pour « pas de retour »,
map -> odom, vitesses de l'odométrie) sont dans pai_core/ros_conversions.py :
le test d'identité appelle exactement les mêmes fonctions.

Sûreté : sans commande pendant cmd_timeout secondes de temps simulé, les
roues reçoivent une consigne nulle.
"""
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from pai_interfaces.srv import ResetWorld

from pai_core.envs.nav_env import NavEnvConfig
from pai_core.geometry.se2 import between
from pai_core.ros_conversions import map_to_odom, ranges_to_msg, velocity_from_poses
from pai_core.robot.odometry import WheelOdometry
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
        # Erreurs d'étalonnage de l'odométrie (jour 8). Elles ne touchent que ce
        # que le robot CROIT de lui-même, jamais la simulation : le vrai robot
        # reste parfait. Erreurs systématiques au sens de Borenstein et Feng.
        radius_err = float(self.declare_parameter("odom_radius_error", 0.0).value)  # relative, roue droite
        base_err = float(self.declare_parameter("odom_base_error", 0.0).value)      # relative, entraxe
        encoder_ticks = int(self.declare_parameter("encoder_ticks", 0).value)       # 0 : encodeurs parfaits

        if self.get_parameter("use_sim_time").value:
            raise RuntimeError("sim_bridge publie /clock : lance-le avec use_sim_time:=false")

        # ------------------------------------------------- monde
        # Même configuration que l'environnement Gymnasium : mêmes formes d'obstacles,
        # même portée du lidar, même nombre de pas de stabilisation.
        self.cfg = NavEnvConfig()
        cfg = self.cfg
        self.world = World(cfg.world)
        self.model, self.data = self.world.model, self.world.data
        m, d = self.model, self.data
        self.seed = seed
        self.n_obstacles_param = n_obstacles        # -1 : tiré comme dans NavEnv
        self.rng = np.random.default_rng(seed)     # tirage des scènes successives

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
        self.encoders = WheelEncoders(geom, ticks_per_rev=None if encoder_ticks <= 0 else encoder_ticks)
        self.odom = WheelOdometry(geom.wheel_radius,
                                  geom.wheel_radius * (1.0 + radius_err),
                                  geom.wheel_base * (1.0 + base_err))

        # Première scène (même code que /reset_world)
        self.map_odom0 = None      # première map -> odom (= pose de départ), pour mesurer la dérive
        self.map_odom = None       # dernière map -> odom publiée
        self.last_cmd_time = -math.inf
        self.reset_scene(self.n_obstacles_param)

        # ------------------------------------------------------------------ ROS
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_gt = self.create_publisher(Odometry, "/ground_truth", 10)
        # Capteur : « best effort », on préfère perdre un scan que le recevoir en retard.
        # Attention : un abonné « reliable » (réglage par défaut de RViz) ne reçoit rien.
        self.pub_scan = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.pub_collision = self.create_publisher(Bool, "/collision", 10)
        self.create_subscription(Float64MultiArray, "/wheel_cmd", self.on_wheel_cmd, 10)
        self.create_service(ResetWorld, "/reset_world", self.on_reset_world)

        # Repères : /tf pour ce qui bouge, /tf_static (publié une fois, conservé
        # pour les abonnés qui arrivent plus tard) pour ce qui est fixé au robot.
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf(self.sim_stamp())

        # Exécuteur mono-fil (rclpy.spin) : on_wheel_cmd, on_reset_world et step ne
        # s'exécutent jamais en même temps, donc aucun verrou n'est nécessaire sur
        # self.data (une remise à zéro ne peut pas tomber au milieu d'un pas).
        self.timer = self.create_timer(1.0 / self.rate, self.step)

        self.viewer = mujoco.viewer.launch_passive(m, d) if use_viewer else None

        self.cycle = 0
        self.rtf_wall0, self.rtf_sim0 = time.perf_counter(), d.time
        self.cb_max = 0.0

        self.log_scene(seed)
        if radius_err or base_err or encoder_ticks > 0:
            self.get_logger().info(f"odométrie faussée : rayon droit {radius_err:+.2%}, "
                                   f"entraxe {base_err:+.2%}, encodeurs "
                                   f"{encoder_ticks if encoder_ticks > 0 else 'parfaits'}")

    # ------------------------------------------------------------------ scène
    def reset_scene(self, n_obstacles: int) -> None:
        """Tire et installe une nouvelle scène, comme NavEnv.reset.

        Appelée au démarrage et par /reset_world. Le temps simulé est conservé
        (voir l'en-tête) ; l'odométrie repart de (0, 0, 0), donc map -> odom
        repart de la pose de départ.
        """
        cfg, m, d = self.cfg, self.model, self.data
        if n_obstacles < 0:
            lo, hi = cfg.n_obstacles
            n_obstacles = int(self.rng.integers(lo, hi + 1))

        t_before = d.time
        self.scene = self.world.sample_scene(self.rng, n_obstacles)
        self.world.apply(self.scene)               # mj_resetData : data.time repasse à 0
        d.ctrl[:] = 0.0
        for _ in range(cfg.settle_steps):          # laisse les contacts roues-sol s'établir
            mujoco.mj_step(m, d)
        d.time = t_before + cfg.settle_steps * m.opt.timestep   # le temps ne recule jamais

        self.odom.reset(wheel_angles=self.encoders.read(d))   # odom part de (0, 0, 0)
        self.prev_odom_pose = self.odom.pose.copy()
        self.map_odom0 = None                      # la dérive se mesure depuis ce départ
        self.last_cmd_time = -math.inf             # pas de commande héritée de l'ancienne scène

    def on_reset_world(self, request, response):
        """Service /reset_world : nouvelle scène, sans relancer les nœuds.

        seed >= 0 : la scène est celle de cette graine (reproductible).
        seed < 0  : on continue le tirage en cours (scène suivante).
        """
        if request.seed >= 0:
            self.rng = np.random.default_rng(request.seed)
            self.seed = int(request.seed)
        n = request.n_obstacles if request.n_obstacles >= 0 else self.n_obstacles_param
        self.reset_scene(int(n))
        self.log_scene(self.seed)

        s = self.scene
        response.success = True
        response.message = f"{len(s.obstacles)} obstacles, but en ({s.goal[0]:.2f}, {s.goal[1]:.2f})"
        response.start_x, response.start_y, response.start_yaw = (float(c) for c in s.start)
        response.goal_x, response.goal_y = float(s.goal[0]), float(s.goal[1])
        return response

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

        contact = self.world.in_collision()
        self.pub_collision.publish(Bool(data=contact))
        if contact:
            self.get_logger().warn("collision", throttle_duration_sec=1.0)

        # Tous les messages d'un cycle portent le même horodatage : le temps simulé.
        stamp = self.sim_stamp()
        clock = Clock()
        clock.clock = stamp
        self.pub_clock.publish(clock)

        # Chaque pose est calculée UNE fois par cycle, puis sert au message ET à la TF.
        # C'est indispensable pour l'odométrie : update() intègre depuis le dernier appel.
        pose_odom = self.publish_odom(stamp)
        pose_gt = self.publish_ground_truth(stamp)
        self.publish_tf(stamp, pose_gt, pose_odom)       # avant le scan, même horodatage
        if self.cycle % self.scan_every == 0:
            self.publish_scan(stamp)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        self.cb_max = max(self.cb_max, time.perf_counter() - t_start)
        self.report_real_time_factor()

    def sim_stamp(self):
        return Time(nanoseconds=round(self.data.time * 1e9)).to_msg()

    # ------------------------------------------------------------------ messages
    def publish_odom(self, stamp) -> np.ndarray:
        """Publie /odom et renvoie la pose d'odométrie T_odom_base."""
        x, y, th = self.odom.update(self.encoders.read(self.data))

        # Vitesses déduites de deux poses successives (schéma du point milieu inversé)
        v, w = velocity_from_poses(self.prev_odom_pose, (x, y, th), self.cycle_dt)
        self.prev_odom_pose = np.array([x, y, th])

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        set_yaw(msg.pose.pose.orientation, float(th))
        msg.twist.twist.linear.x = v                      # vitesses exprimées dans base_link
        msg.twist.twist.angular.z = w
        self.pub_odom.publish(msg)
        return np.array([x, y, th])

    def publish_ground_truth(self, stamp) -> np.ndarray:
        """Publie /ground_truth et renvoie la pose vraie T_map_base."""
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
        return np.array([x, y, yaw])

    def publish_scan(self, stamp):
        max_range = float(self.lidar.cfg.max_range)
        # REP 117 : « aucun retour » s'écrit +inf, pas range_max. Sinon RViz et la
        # grille d'occupation prendraient ces faisceaux pour des impacts à 4 m
        # (cercle fantôme). Pour features.py, rien ne change : inf redevient 1.
        ranges = ranges_to_msg(self.lidar.scan(self.model, self.data), max_range)

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
        msg.range_max = max_range
        msg.ranges = ranges
        self.pub_scan.publish(msg)

    # ------------------------------------------------------------------ repères
    @staticmethod
    def make_transform(stamp, parent, child, x, y, yaw, z=0.0) -> TransformStamped:
        """TransformStamped plan : pose (x, y, yaw) de child dans parent.

        Attention aux noms : transform.translation ici, pose.pose.position dans Odometry.
        """
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        set_yaw(t.transform.rotation, float(yaw))
        return t

    def publish_static_tf(self, stamp):
        """base_link -> laser_link, publié une seule fois.

        Calculé à partir de la configuration (et non écrit en dur) : position du
        lidar dans le châssis moins position du milieu de l'essieu dans le châssis.
        Pas de rotation : le faisceau 0 pointe déjà vers +x de base_link.
        """
        offset = np.asarray(self.lidar.cfg.mount, dtype=float) - self.world.geom.axle_offset
        t = self.make_transform(stamp, "base_link", "laser_link", offset[0], offset[1], 0.0, z=offset[2])
        self.tf_static_broadcaster.sendTransform(t)
        self.get_logger().info(f"TF statique base_link -> laser_link : "
                               f"({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}) m")

    def publish_tf(self, stamp, pose_gt, pose_odom):
        """odom -> base_link (odométrie) et map -> odom (correction de la dérive)."""
        map_odom = map_to_odom(pose_gt, pose_odom)           # T_map_odom = T_map_base ∘ T_odom_base⁻¹
        if self.map_odom0 is None:
            self.map_odom0 = map_odom.copy()                  # doit valoir la pose de départ
            self.get_logger().info(f"map -> odom initiale : x={map_odom[0]:.3f} y={map_odom[1]:.3f} "
                                   f"cap={math.degrees(map_odom[2]):.1f}°")
        self.map_odom = map_odom

        t_odom_base = self.make_transform(stamp, "odom", "base_link", *pose_odom)
        t_map_odom = self.make_transform(stamp, "map", "odom", *map_odom)
        self.tf_broadcaster.sendTransform([t_map_odom, t_odom_base])

    # ------------------------------------------------------------------ mesures
    def report_real_time_factor(self, every=5.0):
        now = time.perf_counter()
        if now - self.rtf_wall0 < every:
            return
        rtf = (self.data.time - self.rtf_sim0) / (now - self.rtf_wall0)
        # Dérive de l'odométrie : de combien le repère odom a bougé dans map depuis le départ.
        drift = ""
        if self.map_odom0 is not None:
            dx, dy, dth = between(self.map_odom0, self.map_odom)
            drift = f" | dérive odom {100 * math.hypot(dx, dy):.1f} cm, {math.degrees(dth):+.1f}°"
        self.get_logger().info(
            f"facteur temps réel {rtf:.3f} | cycle le plus long {self.cb_max * 1e3:.1f} ms "
            f"(budget {1e3 / self.rate:.0f} ms) | t_sim = {self.data.time:.1f} s{drift}")
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