add_executable(occa occa.cpp)
target_link_libraries(occa libocca)
install(TARGETS occa DESTINATION bin)
