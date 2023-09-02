
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# gcc must use the p1689 forked version now. AppleClang does not support modules
# see: https://www.kitware.com/import-cmake-c20-modules/
#      https://discourse.cmake.org/t/compile-c-20-modules-with-clang-16-and-cmake-v3-26-3/8096/2
#      https://github.com/Kitware/CMake/blob/v3.27.2/Help/dev/experimental.rst
#      https://zhuanlan.zhihu.com/p/350136757
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API aa1f7df0-828a-4fcd-9afc-2dc80491aca7)
# set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
set(CMake_TEST_CXXModules_UUID "a246741c-d067-4019-a8fb-3d16b0c9d1d3")


# set(CMAKE_CXX_COMPILE_OBJECT "${CMAKE_CXX_COMPILE_OBJECT} @<OBJECT>.modmap")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
                "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E -x c++ <SOURCE>"
                " -MT <DYNDEP_FILE> -MD -MF <DEP_FILE>"
                " -fmodules-ts"
                "-fdep-file=<DYNDEP_FILE> -fdep-output=<OBJECT> -fdep-format=trtbd"
                " -o <PREPROCESSED_SOURCE>"
  )
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "gcc")
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "-fmodules-ts -fmodule-mapper=<MODULE_MAP_FILE> -fdep-format=trtbd -x c++")

  # set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY};-checks=*,-llvmlibc-*,-fuchsia-*,-cppcoreguidelines-init-variables")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
  #   "${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}"
  #   " -format=p1689"
  #   " --"
  #   " <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS>"
  #   " -x c++ <SOURCE> -c -o <OBJECT>"
  #   " -MT <DYNDEP_FILE>"
  #   " -MD -MF <DEP_FILE>"
  #   " > <DYNDEP_FILE>")
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "clang")
  # set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "@<MODULE_MAP_FILE>")

  # set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "-fmodule-mapper=<MODULE_MAP_FILE>")
  # set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "msvc")
endif()
