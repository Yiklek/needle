
# add_pybind11_module <target> <output_path> [SRCS srcs...] [DEPENDS targets...]
function(add_pybind11_module target_name)
  cmake_parse_arguments(_ARG "" "" "SRCS;DEPENDS" ${ARGN})

  pybind11_add_module(${target_name} ${_ARG_SRCS})
  target_compile_definitions(${target_name} PRIVATE VERSION_INFO=${PROJECT_VERSION}
                                                    PYBIND11_CURRENT_MODULE_NAME=${target_name})
  target_link_libraries(${target_name} PRIVATE ${_ARG_DEPENDS})
  # add_dependencies(pymodule ${target_name})
  # set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/${output_path}")
endfunction(add_pybind11_module)
