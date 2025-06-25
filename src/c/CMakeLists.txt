target_sources(libocca PRIVATE base.cpp device.cpp dtype.cpp
  experimental io.cpp json.cpp kernel.cpp kernel_fortran_interface.cpp
  memory.cpp memoryPool.cpp scope.cpp stream.cpp)
add_subdirectory(experimental)
