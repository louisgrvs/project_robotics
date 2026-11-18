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
        # Configurations RViz : installées dans share/pai_ros/rviz/
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='louisgrvs',
    maintainer_email='louis.gervais@ens-lyon.fr',
    description='Nœuds ROS 2 du projet Physical AI : pont MuJoCo, perception, décision',
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
        ],
    },
)
