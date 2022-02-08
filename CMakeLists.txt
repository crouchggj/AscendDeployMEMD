cmake_minimum_required(VERSION 3.10)
project(AscendDeployMEMD)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DENABLE_DVPP_INTERFACE)
# Specify target generation path
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O3 -g -Wall -ffast-math -Wno-sign-compare -Wno-terminate")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O3 -Wall -ffast-math -Wno-sign-compare -Wno-terminate")
set(DDK_PATH /opt/ascend-toolkit/latest)
set(INC_PATH ${DDK_PATH})
message(STATUS "env INC_PATH: ${INC_PATH}")

set(NPU_HOST_LIB ${DDK_PATH}/acllib/lib64/stub)
set(LIB_PATH ${NPU_HOST_LIB})
message(STATUS "env LIB_PATH: ${LIB_PATH}")

set(OpenCV_DIR /opt/ascend_ddk/lib/cmake/opencv4)
find_package(OpenCV)

# Header path
include_directories(
        /opt/ascend_ddk/include
        ${INC_PATH}/acllib/include/
        ${OpenCV_INCLUDE_DIRS}
)

# add host lib path
link_directories(
        /opt/ascend_ddk/lib
        /opt/ascend_ddk/usr/lib/
        ${LIB_PATH}
)

add_executable(AscendDeployMEMD memd_deploy.cpp
        acl_context.cpp acl_dataset.cpp decode_box.cpp)
target_link_libraries(AscendDeployMEMD
        ascendcl acl_cblas acl_dvpp stdc++ pthread dl
        opencv_core opencv_imgcodecs opencv_imgproc)