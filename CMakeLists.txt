cmake_minimum_required (VERSION 2.8.8)
project (LoadImagesInLMDB)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# ---[ Protobuf
find_package( Protobuf REQUIRED )
include_directories(${PROTOBUF_INCLUDE_DIR})
list(APPEND LIBRARIES ${PROTOBUF_LIBRARIES})

# ---[ Boost
find_package(Boost 1.55 REQUIRED COMPONENTS system thread filesystem)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND LIBRARIES ${Boost_LIBRARIES})

# ---[ Google-gflags
find_package(gflags REQUIRED)

# ---[ LMDB
find_package(lmdb REQUIRED)

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

add_executable(load_images_in_lmdb ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/lmdb.cpp)
target_link_libraries(load_images_in_lmdb gflags glog lmdb)
target_link_libraries(load_images_in_lmdb ${LIBRARIES})
