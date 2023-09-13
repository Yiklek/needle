# require CPM
include(CPM)

# fetch_catch2
function(fetch_catch2)
  if(NOT TARGET Catch2::Catch2WithMain)
    # CPMAddPackage(NAME "gh:catchorg/Catch2@3.4.0" OPTIONS "BUILD_TESTING OFF")
    CPMAddPackage(
      NAME catch2
      GITHUB_REPOSITORY "catchorg/Catch2"
      GIT_TAG v3.4.0 OPTIONS "BUILD_TESTING OFF"
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_catch2)

function(add_cc_test targe_name)
    cmake_parse_arguments(_ARG "" "" "SRCS;DEPENDS;DEFINITIONS" ${ARGN})
    add_executable(
        ${targe_name}
        ${_ARG_SRCS}
    )
    target_link_libraries(
        ${targe_name}
        Catch2::Catch2WithMain
        ${_ARG_DEPENDS}
    )
    target_compile_options(${targe_name} PRIVATE -fno-access-control)
    target_compile_definitions(${targe_name} PRIVATE ${_ARG_DEFINITIONS})
    catch_discover_tests(${targe_name})
endfunction(add_cc_test)
