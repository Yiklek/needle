# FindCXXStdModules.cmake
#
# CMake's built-in detection runs clang++ --print-file-name=<stdlib>.modules.json
# which returns only a bare filename when the file lives outside the compiler's
# default library search path (e.g. Homebrew LLVM puts libc++.modules.json under
# lib/c++/ but clang only searches lib/clang/<ver>/).
#
# Include this after project() and call find_cxx_stdlib_modules_json() to fix
# up CMAKE_CXX_STDLIB_MODULES_JSON automatically on any platform.

function(find_cxx_stdlib_modules_json)
  # Already a valid absolute path?
  if(IS_ABSOLUTE "${CMAKE_CXX_STDLIB_MODULES_JSON}")
    if(EXISTS "${CMAKE_CXX_STDLIB_MODULES_JSON}")
      return()
    endif()
  endif()

  # Determine which .modules.json file we need
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    if(CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libstdc++")
      set(_stdlib_name "libstdc++")
    else()
      set(_stdlib_name "libc++")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(_stdlib_name "libstdc++")
  else()
    return()
  endif()

  set(_modules_json_filename "${_stdlib_name}.modules.json")

  # Strategy 1: ask the compiler via --print-file-name
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_ID_ARG1}
            "--print-file-name=${_modules_json_filename}"
    OUTPUT_VARIABLE _print_output
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _print_result
  )
  if(_print_result EQUAL 0 AND IS_ABSOLUTE "${_print_output}" AND EXISTS "${_print_output}")
    set(CMAKE_CXX_STDLIB_MODULES_JSON "${_print_output}" PARENT_SCOPE)
    message(STATUS "Found ${_modules_json_filename} via --print-file-name: ${_print_output}")
    return()
  endif()

  # Strategy 2: search around the compiler's install prefix.
  # Resolve symlinks to get the real compiler path, then walk up to <prefix>.
  get_filename_component(_compiler_real "${CMAKE_CXX_COMPILER}" REALPATH)
  get_filename_component(_compiler_dir "${_compiler_real}" DIRECTORY)
  get_filename_component(_prefix_dir "${_compiler_dir}" DIRECTORY)

  # For Clang we additionally probe the resource directory as an anchor.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_COMPILER_ID_ARG1}
              -print-resource-dir
      OUTPUT_VARIABLE _resource_dir
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_resource_dir)
      # resource dir: <prefix>/lib/clang/<ver>
      # lib/c++/<file> is at the same level as lib/clang/
      get_filename_component(_lib_dir "${_resource_dir}" DIRECTORY)
      get_filename_component(_lib_dir "${_lib_dir}" DIRECTORY)

      set(_candidates
        "${_lib_dir}/c++/${_modules_json_filename}"               # Homebrew LLVM
        "${_prefix_dir}/share/libc++/v1/${_modules_json_filename}" # Linux / standard
        "${_prefix_dir}/share/libstdc++/${_modules_json_filename}"
      )
      foreach(_cand IN LISTS _candidates)
        if(EXISTS "${_cand}")
          set(CMAKE_CXX_STDLIB_MODULES_JSON "${_cand}" PARENT_SCOPE)
          message(STATUS "Found ${_modules_json_filename} near resource dir: ${_cand}")
          return()
        endif()
      endforeach()
    endif()
  endif()

  # Strategy 3: recursive search under the compiler prefix (up to 2 levels up).
  get_filename_component(_search_root "${_prefix_dir}" ABSOLUTE)
  if(NOT EXISTS "${_search_root}")
    get_filename_component(_search_root "${_compiler_dir}/.." ABSOLUTE)
  endif()

  file(GLOB_RECURSE _found_json
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${_search_root}/${_modules_json_filename}"
  )
  # Prefer lib/c++/ over share/ (closer to the actual library layout).
  foreach(_f IN LISTS _found_json)
    string(FIND "${_f}" "/lib/c++/" _pos)
    if(_pos GREATER -1)
      set(CMAKE_CXX_STDLIB_MODULES_JSON "${_f}" PARENT_SCOPE)
      message(STATUS "Found ${_modules_json_filename} via recursive search: ${_f}")
      return()
    endif()
  endforeach()
  if(_found_json)
    list(GET _found_json 0 _first)
    set(CMAKE_CXX_STDLIB_MODULES_JSON "${_first}" PARENT_SCOPE)
    message(STATUS "Found ${_modules_json_filename} via recursive search: ${_first}")
    return()
  endif()

  message(WARNING
    "Could not locate ${_modules_json_filename}. "
    "C++ standard library module (import std;) will not be available."
  )
endfunction()
