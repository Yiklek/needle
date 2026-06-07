# fetch_stdexec — header-only Senders/Receivers
# Downloads sources without building (avoids stdexec's own project()/CMakeLists.txt conflict)
function(fetch_stdexec)
  if(NOT TARGET stdexec)
    include(FetchContent)
    if(POLICY CMP0169)
      cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Declare(
      stdexec
      GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
      GIT_TAG nvhpc-26.05
      GIT_SHALLOW ON
      SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/nvidia-stdexec-src"
    )
    FetchContent_Populate(stdexec)
    if(stdexec_POPULATED)
      add_library(stdexec INTERFACE)
      target_include_directories(stdexec SYSTEM INTERFACE
        "${stdexec_SOURCE_DIR}/include")
    endif()
  endif()
endfunction(fetch_stdexec)
