cmake_minimum_required (VERSION 3.1.3)
project (LoadImagesInLMDB)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set (CMAKE_CXX_STANDARD 11)

# ---[ Protobuf
find_package( Protobuf REQUIRED )
include_directories(${PROTOBUF_INCLUDE_DIR})
list(APPEND LIBRARIES ${PROTOBUF_LIBRARIES})

# ---[ Boost
find_package(Boost 1.55 REQUIRED COMPONENTS system thread filesystem)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND LIBRARIES ${Boost_LIBRARIES})

# ---[ Google-gflags
find_package(GFlags REQUIRED)

# ---[ Google-GLog
find_package(GLOG REQUIRED)

# ---[ LMDB
find_package(LMDB REQUIRED)

# ---[ OpenCB
find_package(OpenCV REQUIRED)

# ---[ CUDA
find_package(CUDA REQUIRED)

# ---[ Caffe
list(APPEND LIBRARIES "${CMAKE_SOURCE_DIR}/../caffe/.build_release/lib/libcaffe.a")
include_directories("${CMAKE_SOURCE_DIR}/../caffe/include/")
include_directories("${CMAKE_SOURCE_DIR}/../caffe/.build_release/src/")
add_definitions(${Caffe_DEFINITIONS})


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")

add_subdirectory (src)

enable_testing ()
add_subdirectory (test)

