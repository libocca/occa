file(GLOB_RECURSE tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
list(REMOVE_ITEM sources core/kernelOperators.cpp tools/runFunction.cpp)

# dontCompile = $(OCCA_DIR)/src/core/kernelOperators.cpp $(OCCA_DIR)/src/tools/runFunction.cpp

add_library(libocca ${sources})
