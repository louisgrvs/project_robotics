cmake_minimum_required(VERSION 3.8)
project(pai_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

# Chaque fichier listé ici devient une classe Python (et C++) au colcon build.
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/NavigateToGoal.action"
  "srv/ResetWorld.srv"
  DEPENDENCIES geometry_msgs
)

ament_package()
