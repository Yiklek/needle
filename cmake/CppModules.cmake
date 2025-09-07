
function(add_cpp_module_library target_name target_type)
  cmake_parse_arguments(_ARG "" "" "SRCS" ${ARGN})
  message(STATUS "============= ${target_name} ${target_type} ${_ARG_SRCS}")
  add_library(${target_name} ${target_type} ${_ARG_SRCS})
  target_sources(${target_name}
        PUBLIC
        FILE_SET cxx_modules_${target_name}
        TYPE CXX_MODULES
        FILES ${_ARG_SRCS}
  )
  set_target_properties(${target_name} PROPERTIES CXX_MODULE_STD on)
  target_compile_options(${target_name} PRIVATE -fvisibility=default)
endfunction(add_cpp_module_library)
