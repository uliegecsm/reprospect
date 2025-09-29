add_library(reprospect INTERFACE)
target_include_directories(reprospect INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cpp>)

add_subdirectory(errors)

#---- Install.
install(TARGETS reprospect EXPORT ReprospectConfig FILE_SET headers DESTINATION include/reprospect COMPONENT cpp)
install(EXPORT ReprospectConfig DESTINATION lib/cmake/Reprospect NAMESPACE reprospect:: COMPONENT cpp)
