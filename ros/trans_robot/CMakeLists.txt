cmake_minimum_required(VERSION 2.8.3)
project(trans_robot)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  moveit_core
  moveit_ros_planning
  roscpp
  moveit_ros_move_group
  moveit_ros_planning_interface
)
find_package( OpenCV REQUIRED )

catkin_package(CATKIN_DEPENDS
  moveit_core moveit_ros_planning
   moveit_ros_planning roscpp)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(shadow_forward_kinematics src/shadow_forward_kinematics.cpp)
add_dependencies(shadow_forward_kinematics ${catkin_EXPORTED_TARGETS})
target_link_libraries(shadow_forward_kinematics ${catkin_LIBRARIES} ${OpenCV_LIBS})

