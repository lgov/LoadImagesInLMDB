find_package (Boost COMPONENTS system filesystem  unit_test_framework REQUIRED)
include_directories (${LoadImagesInLMDB_SOURCE_DIR}/src
                     ${Boost_INCLUDE_DIRS}
                     )

add_definitions (-DBOOST_TEST_DYN_LINK)
add_executable (test_load_images_in_lmdb test_image_loading.cpp
                                         test_lmdb_database.cpp
                                         test_main.cpp
                                         ../src/lmdb.cpp)
target_link_libraries(test_load_images_in_lmdb ${BOOST_LIBRARIES}
                                          ${Boost_FILESYSTEM_LIBRARY}
                                          ${Boost_SYSTEM_LIBRARY}
                                          ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
target_link_libraries(test_load_images_in_lmdb ${LIBRARIES} ${LMDB_LIBRARIES} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARY})
target_link_libraries(test_load_images_in_lmdb ${OpenCV_LIBS} )
target_link_libraries(test_load_images_in_lmdb ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})

