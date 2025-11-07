add_executable(occa occa.cpp)
add_executable(${namespace}occa ALIAS occa)

target_link_libraries(occa libocca)
target_include_directories(occa
                           PRIVATE $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

install(
  TARGETS occa
  EXPORT occaExport
  DESTINATION bin)
