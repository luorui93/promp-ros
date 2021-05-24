cmake_minimum_required(VERSION 3.0.2)
project(promp_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS 
    system
)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES promp_ros
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
    ${catkin_INCLUDE_DIRS}
)

add_executable(message_filter_node src/message_filter.cpp)
target_link_libraries(message_filter_node ${catkin_LIBRARIES})
add_dependencies(message_filter_node 
	${${PROJECT_NAME}_EXPORTED_TARGETS} 
	${catkin_EXPORTED_TARGETS}
)
