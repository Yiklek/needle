# require CPM
include(CPM)

# fetch_result
function(fetch_expected)
  if(NOT TARGET tl::expected)
    CPMAddPackage(
      NAME expected
      GITHUB_REPOSITORY TartanLlama/expected
      GIT_TAG v1.1.0 OPTIONS "EXPECTED_BUILD_TESTS OFF"
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_expected)
