####################################################################################################
# This function recursivelly update git submodules.
# Params: project base directory
# Example:
# init_submodules(${PROJECT_SOURCE_DIR})
####################################################################################################

function(init_submodules
	 project_dir)

 find_package(Git QUIET)
 if(GIT_FOUND AND EXISTS "${project_dir}/.git")
   # Update submodules as needed
   message(STATUS "Submodule update")
   execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${project_dir}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
   if(NOT GIT_SUBMOD_RESULT EQUAL "0")
     message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
   endif()
 else()
   message(STATUS "Submodule update has not been run")
 endif()

endfunction()

