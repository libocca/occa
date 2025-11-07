target_sources(libocca PRIVATE forLoop.cpp iteration.cpp typelessForLoop.cpp)

target_include_directories(
  libocca
  PRIVATE $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>
          $<BUILD_INTERFACE:${OCCA_BUILD_DIR}/include>)
