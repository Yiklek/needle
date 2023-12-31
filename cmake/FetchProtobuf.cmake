# require CPM
include(CPM)

# fetch_result
function(fetch_protobuf)
  set(ABSL_PROPAGATE_CXX_STD ON)
  if(NOT TARGET protobuf::protoc)
    CPMAddPackage(
      NAME protobuf
      GITHUB_REPOSITORY protocolbuffers/protobuf
      GIT_TAG v24.1 OPTIONS "protobuf_BUILD_TESTS OFF"
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
    include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)
  endif()
endfunction(fetch_protobuf)
