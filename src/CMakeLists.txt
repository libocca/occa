file(GLOB_RECURSE sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
list(REMOVE_ITEM sources core/kernelOperators.cpp tools/runFunction.cpp)

if (WITH_METAL)
  file(GLOB_RECURSE sources_metal RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.mm")
  set(sources ${sources} ${sources_metal})
endif()

# dontCompile = $(OCCA_DIR)/src/core/kernelOperators.cpp $(OCCA_DIR)/src/tools/runFunction.cpp

add_library(libocca SHARED ${sources})

# without this, CMake will create liblibocca.so
set_target_properties(libocca PROPERTIES OUTPUT_NAME occa)

set(LIBOCCA_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

if (WITH_CUDA)
  set(LIBOCCA_LIBRARIES cuda ${LIBOCCA_LIBRARIES})
endif()

if (WITH_OPENCL)
  set(LIBOCCA_LIBRARIES ${LIBOCCA_LIBRARIES} ${OpenCL_LIBRARIES} ${LIBOCCA_LIBRARIES} )
endif()

if (WITH_MPI)
  set(LIBOCCA_LIBRARIES ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${LIBOCCA_LIBRARIES})
endif()

if (WITH_METAL)
  set(LIBOCCA_LIBRARIES ${METAL_LIBRARY} ${CORE_SERVICES} ${APP_KIT} ${LIBOCCA_LIBRARIES})
endif()

target_link_libraries(libocca ${LIBOCCA_LIBRARIES} )

install(TARGETS libocca DESTINATION lib )
