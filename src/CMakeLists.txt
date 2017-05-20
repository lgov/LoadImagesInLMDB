add_executable(load_images_in_lmdb ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/lmdb.cpp)
target_link_libraries(load_images_in_lmdb ${BOOST_LIBRARIES}
                                          ${Boost_FILESYSTEM_LIBRARY}
                                          ${Boost_SYSTEM_LIBRARY})
target_link_libraries(load_images_in_lmdb ${LIBRARIES} ${LMDB_LIBRARIES} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARY})
target_link_libraries(load_images_in_lmdb ${OpenCV_LIBS} )
target_link_libraries(load_images_in_lmdb ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
