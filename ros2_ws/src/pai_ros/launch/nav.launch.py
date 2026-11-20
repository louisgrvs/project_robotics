"""Lance tout le système de navigation et RViz en une seule commande.

    ros2 launch pai_ros nav.launch.py
    ros2 launch pai_ros nav.launch.py safety:=false seed:=3 rviz:=false

Graphe lancé :
    sim_bridge ─► /scan, /odom, /tf, /clock
    perception_node ─► /scan_proc, /local_map
    decision_node ─► /cmd_vel           (DWA)
    control_node ─► /wheel_cmd          (saturation, filtre de sûreté, cinématique inverse)
    rviz2                                (but avec l'outil 2D Goal Pose)

Tous les paramètres viennent de config/params.yaml. Les arguments de
lancement ne servent qu'aux réglages qu'on change d'une expérience à
l'autre (filtre, graine, affichage), sans toucher au fichier.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Fichiers installés dans install/pai_ros/share/pai_ros/ (voir data_files dans setup.py)
    pkg = FindPackageShare("pai_ros")
    params_file = PathJoinSubstitution([pkg, "config", "params.yaml"])
    rviz_file = PathJoinSubstitution([pkg, "rviz", "nav.rviz"])

    # ------------------------------------------------------------ arguments
    args = [
        DeclareLaunchArgument("safety", default_value="true",
                              description="filtre de sûreté actif (variable expérimentale)"),
        DeclareLaunchArgument("seed", default_value="0",
                              description="graine de la scène MuJoCo"),
        DeclareLaunchArgument("viewer", default_value="true",
                              description="fenêtre MuJoCo"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="lancer RViz"),
        DeclareLaunchArgument("perception", default_value="true",
                              description="lancer perception_node (mettre false s'il n'est pas prêt)"),
    ]

    # Un argument de lancement est toujours une CHAÎNE : "false" est une chaîne
    # non vide, donc bool("false") vaut True. ParameterValue convertit au bon
    # type avant de transmettre le paramètre au noeud.
    safety = ParameterValue(LaunchConfiguration("safety"), value_type=bool)
    seed = ParameterValue(LaunchConfiguration("seed"), value_type=int)
    viewer = ParameterValue(LaunchConfiguration("viewer"), value_type=bool)

    # Tous les noeuds suivent le temps simulé, SAUF sim_bridge : c'est lui qui
    # publie /clock. S'il attendait /clock pour faire tourner son propre
    # minuteur, il attendrait un message qu'il est le seul à pouvoir envoyer.
    sim_time = {"use_sim_time": True}

    # ------------------------------------------------------------ noeuds
    sim_bridge = Node(
        package="pai_ros", executable="sim_bridge", name="sim_bridge",
        parameters=[params_file, {"use_sim_time": False, "seed": seed, "viewer": viewer}],
        output="screen",
    )

    perception = Node(
        package="pai_ros", executable="perception_node", name="perception_node",
        parameters=[params_file, sim_time],
        output="screen",
        condition=IfCondition(LaunchConfiguration("perception")),
    )

    decision = Node(
        package="pai_ros", executable="decision_node", name="decision_node",
        parameters=[params_file, sim_time],
        output="screen",
    )

    control = Node(
        package="pai_ros", executable="control_node", name="control_node",
        parameters=[params_file, sim_time, {"safety.enabled": safety}],
        output="screen",
    )

    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", rviz_file],
        parameters=[sim_time],
        output="log",                     # RViz est bavard : son journal va dans ~/.ros/log
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(args + [sim_bridge, perception, decision, control, rviz])
