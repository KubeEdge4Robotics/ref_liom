cmake_minimum_required(VERSION 3.0.2)
project(ref_provider)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI) # jsoncpp need it , or it will return undefined reference
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)	# 增加对libigl.cmake的引用

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  # pcl library
  pcl_conversions
  roscpp
  rospy
  rosbag
  std_msgs
  message_filters
	ref_liom
)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(jsoncpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include 
  DEPENDS PCL CERES ref_liom
  CATKIN_DEPENDS 
	roscpp 
	rospy 
	std_msgs
	nav_msgs
  geometry_msgs
  sensor_msgs
  ref_liom
  message_runtime 
  message_generation
)
###########
## Build ##
###########


include(libigl)
igl_include(glfw)
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
	include
)
include_directories(${JSONCPP_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(${PROJECT_NAME} src/ref_provider.cpp)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${ref_liom_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} PUBLIC jsoncpp_lib igl::glfw ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})     


