# require CPM
include(CPM)

# fetch_gtest
function(fetch_gtest)
  if(NOT TARGET GTest::gtest_main)
    CPMAddPackage(
      NAME gtest
      GITHUB_REPOSITORY "google/googletest"
      GIT_TAG v1.17.0 OPTIONS "BUILD_TESTING OFF"
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
    include(GoogleTest)
  endif()
endfunction(fetch_gtest)

function(add_cc_test target_name)
  cmake_parse_arguments(_ARG "" "" "SRCS;DEPENDS;DEFINITIONS" ${ARGN})
  add_executable(
        ${target_name}
        ${_ARG_SRCS}
    )
  target_link_libraries(
        ${target_name}
        GTest::gtest_main
        GTest::gmock
        ${_ARG_DEPENDS}
    )
  target_compile_options(${target_name} PRIVATE -fno-access-control)
  target_compile_definitions(${target_name} PRIVATE ${_ARG_DEFINITIONS})
  gtest_discover_tests(${target_name})
endfunction(add_cc_test)
