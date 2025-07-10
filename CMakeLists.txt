cmake_minimum_required(VERSION 4.0)
project(machine_learning VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(FetchContent)

message(STATUS "Fetching Eigen…")
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_MPL2_ONLY ON)

FetchContent_MakeAvailable(Eigen)

add_executable(main src/main.cpp)

get_target_property(EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

add_library(EigenSystem INTERFACE)
target_include_directories(EigenSystem SYSTEM INTERFACE ${EIGEN_INCLUDE_DIRS})

target_link_libraries(main PRIVATE EigenSystem)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/src/machine_learning)