# require CPM
include(CPM)

# fetch_result
function(fetch_expected)
  if(NOT TARGET tl::expected)
    cpmaddpackage(
      NAME
      expected
      GITHUB_REPOSITORY
      TartanLlama/expected
      GIT_TAG
      v1.1.0
      GIT_SHALLOW
      ON
      EXCLUDE_FROM_ALL
      ON)
  endif()
endfunction(fetch_expected)
