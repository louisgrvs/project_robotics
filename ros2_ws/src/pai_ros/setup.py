import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'pai_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Sans ces trois lignes, ros2 launch ne trouve ni le fichier de
        # lancement, ni params.yaml, ni la configuration RViz : ils doivent
        # être copiés dans install/pai_ros/share/pai_ros/ par colcon build.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='louisgrvs',
    maintainer_email='louis.gervais@ens-lyon.fr',
    description='Couche ROS 2 du projet Physical AI : pont MuJoCo, perception, décision, commande',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim_bridge = pai_ros.sim_bridge_node:main',
            'perception_node = pai_ros.perception_node:main',
            'decision_node = pai_ros.decision_node:main',
            'control_node = pai_ros.control_node:main',
        ],
    },
)
