find_package (Boost COMPONENTS system filesystem  unit_test_framework REQUIRED)
include_directories (${LoadImagesInLMDB_SOURCE_DIR}/src
                     ${Boost_INCLUDE_DIRS}
                     )

add_definitions (-DBOOST_TEST_DYN_LINK)
add_executable (test_load_images_in_lmdb test_image_loading.cpp
                                         test_lmdb_database.cpp
                                         test_main.cpp
                                         ../src/lmdb.cpp)

target_link_libraries (test_load_images_in_lmdb
                       gflags glog lmdb
                       ${LIBRARIES}
                       ${OpenCV_LIBS}
                       ${Boost_FILESYSTEM_LIBRARY}
                       ${Boost_SYSTEM_LIBRARY}
                       ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      )
