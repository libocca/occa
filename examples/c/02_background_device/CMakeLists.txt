compile_c_example(background_device main.c)

add_custom_target(c_example_background_devices_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_c_background_device c_example_background_devices_okl)
