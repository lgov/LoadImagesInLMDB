add_executable(load_images_in_lmdb ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/lmdb.cpp)
target_link_libraries(load_images_in_lmdb gflags glog lmdb)
target_link_libraries(load_images_in_lmdb ${LIBRARIES})
target_link_libraries(load_images_in_lmdb ${OpenCV_LIBS} )
