# require CPM
include(CPM)

# fetch_protobuf
function(fetch_protobuf)
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
  if(NOT TARGET protobuf::protoc)
    CPMAddPackage(
      NAME protobuf
      GITHUB_REPOSITORY protocolbuffers/protobuf
      GIT_TAG v35.0
      OPTIONS
        "protobuf_BUILD_TESTS OFF"
        "protobuf_BUILD_SHARED_LIBS OFF"
        # "protobuf_ABSL_PROVIDER module"  # 使用 protobuf 内置的 absl
        # "absl_ENABLE_INSTALL OFF"
        # 禁用可能需要 atomic 的 absl 组件
        # "absl_FIND_CMAKE_PACKAGE OFF"
      GIT_SHALLOW ON
      EXCLUDE_FROM_ALL ON)
    include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)
  endif()
  string(REPLACE "-Wno-error=deprecated-declarations" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wno-deprecated-declarations" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endfunction(fetch_protobuf)
