compile_cpp_example_with_modes(oklt_v3_moving_avg main.cpp)

add_custom_target(cpp_example_oklt_v3_moving_avg_cpy ALL 
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/constants.h constants.h
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/movingAverage.okl movingAverage.okl)
add_dependencies(examples_cpp_oklt_v3_moving_avg cpp_example_oklt_v3_moving_avg_cpy)
target_sources(examples_cpp_oklt_v3_moving_avg 
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/movingAverage.okl
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/constants.h
	)
