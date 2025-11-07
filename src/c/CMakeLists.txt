add_library(c_api SHARED)
add_library(${namespace}c_api ALIAS c_api)
target_sources(c_api
  PRIVATE
  base.cpp
  device.cpp
  dtype.cpp
  io.cpp
  json.cpp
  kernel.cpp
  memory.cpp
  memoryPool.cpp
  scope.cpp
  stream.cpp
)
add_subdirectory(experimental)

target_link_libraries(c_api
  PRIVATE
  libocca
)
