# require CPM
include(CPM)

# fetch_spdlog
function(fetch_boost_preprocessor)
  if(NOT TARGET Boost::preprocessor)
    CPMAddPackage(
      NAME boost_preprocessor
      GITHUB_REPOSITORY boostorg/preprocessor
      GIT_TAG boost-1.83.0
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
  endif()
endfunction(fetch_boost_preprocessor)
