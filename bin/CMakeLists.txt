add_executable(occa occa.cpp)
target_link_libraries(occa libocca)
install(TARGETS occa DESTINATION bin)

add_custom_command(TARGET occa POST_BUILD COMMAND occa info)