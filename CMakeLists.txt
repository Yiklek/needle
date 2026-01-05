cmake_minimum_required(VERSION 4.2)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CppModules)
include(CppModulesClang)
# include(CppModulesGCC)

project(needle C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(FINEFLOW_BUILD_TEST "build test" ON)
find_package(
  Python3
  COMPONENTS Development Interpreter
  REQUIRED)
message(STATUS "Python: " ${Python3_EXECUTABLE})

# find pybind
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS ${__pybind_path})
find_package(pybind11 PATHS ${__pybind_path})
include(Pybind11)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")



set(PYTHON_MODULE_PATH needle/backend_ndarray)

add_custom_target(backend COMMENT "backend")
# ##############################################################################
# CPU BACKEND ###
# ##############################################################################
pybind11_add_module(ndarray_backend_cpu src/ndarray_backend_cpu.cc)
add_dependencies(backend ndarray_backend_cpu)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined
                                                   dynamic_lookup)
endif()

install(TARGETS ndarray_backend_cpu LIBRARY DESTINATION ${PYTHON_MODULE_PATH}
                                            COMPONENT backend)

include(FetchFmt)
include(FetchProtobuf)
include(FetchExpected)
include(FetchSpdlog)
include(FetchBoostPreprocessor)

if (CMAKE_CXX_STANDARD LESS 23)
  fetch_fmt()
  fetch_expected()
endif()
fetch_protobuf()
fetch_spdlog()


set(NEEDLE_THIRD_COMMON_LIBS "")
if (CMAKE_CXX_STANDARD LESS 23)
  list(APPEND NEEDLE_THIRD_COMMON_LIBS tl::expected)
  list(APPEND NEEDLE_THIRD_COMMON_LIBS fmt::fmt)
endif()
list(APPEND NEEDLE_THIRD_COMMON_LIBS spdlog::spdlog_header_only)

add_library(ThirdCommon INTERFACE)
target_link_libraries(ThirdCommon INTERFACE ${NEEDLE_THIRD_COMMON_LIBS})

file(GLOB FINEFLOW_CORE_PROTO_SRCS  src/fineflow/core/common/*.proto)
add_library(proto-objects
    OBJECT
    ${FINEFLOW_CORE_PROTO_SRCS}
)

target_link_libraries(proto-objects PUBLIC protobuf::libprotobuf)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_generated")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(proto-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
    TARGET proto-objects
    IMPORT_DIRS "src"
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}"
)

file(GLOB FINEFLOW_COMMON_MODULE_SRCS  src/fineflow/core/common/*.cppm)
add_cpp_module_library(FineflowCommon STATIC
  MODULE_SRCS ${FINEFLOW_COMMON_MODULE_SRCS}
)

target_link_libraries(FineflowCommon PUBLIC proto-objects ThirdCommon)

file(GLOB FINEFLOW_CORE_MODULE_SRCS  src/fineflow/core/*.cppm)
file(GLOB FINEFLOW_CORE_FUNCTOR_MODULE_SRCS  src/fineflow/core/functional/*_functor.cppm)
file(GLOB FINEFLOW_CORE_CPU_KERNEL_MODULE_SRCS  src/fineflow/core/kernels/cpu/*_kernel.cppm)
add_cpp_module_library(FineflowCore STATIC
  MODULE_SRCS
  ${FINEFLOW_CORE_MODULE_SRCS}
  ${FINEFLOW_CORE_FUNCTOR_MODULE_SRCS}
  ${FINEFLOW_CORE_CPU_KERNEL_MODULE_SRCS}
)
target_link_libraries(FineflowCore PUBLIC FineflowCommon)


file(GLOB FINEFLOW_API_PYTHON_MODULE_SRCS  src/fineflow/api/python/*.cppm)
add_cpp_module_library(FineflowPyApiObj OBJECT
  MODULE_SRCS ${FINEFLOW_API_PYTHON_MODULE_SRCS}
)
target_link_libraries(FineflowPyApiObj PUBLIC
    $<LINK_LIBRARY:WHOLE_ARCHIVE,FineflowCore>
)

add_pybind11_module(FineflowPyApi
    SRCS src/fineflow/api/python/fineflow.cpp
    DEPENDS
    FineflowPyApiObj
)


add_compile_definitions(FINEFLOW_SOURCE_DIR=${PROJECT_SOURCE_DIR} FINEFLOW_BINARY_DIR=${PROJECT_BINARY_DIR})
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  add_compile_definitions(DEBUG)
endif()

# ##############################################################################
# CUDA BACKEND ###
# ##############################################################################
# find_package(CUDA)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  message(STATUS "Found cuda, building cuda backend")
  message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
  message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDAToolkit_VERSION_MAJOR: ${CUDAToolkit_VERSION_MAJOR}")
  message(STATUS "CUDAToolkit_VERSION_MINOR: ${CUDAToolkit_VERSION_MINOR}")
  message(STATUS "CUDAToolkit_VERSION_PATCH: ${CUDAToolkit_VERSION_PATCH}")
  message(STATUS "CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
  message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
  message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
  message(STATUS "CUDAToolkit_LIBRARY_ROOT: ${CUDAToolkit_LIBRARY_ROOT}")
  message(STATUS "CUDAToolkit_TARGET_DIR: ${CUDAToolkit_TARGET_DIR}")
  message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")

  enable_language(CUDA)
  set(CMAKE_CUDA_VISIBILITY_PRESET "hidden")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/.clangd.in
                 ${CMAKE_CURRENT_SOURCE_DIR}/.clangd)
  set(CMAKE_CUDA_STANDARD 17)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")

  set(CUDA_VERSION
      ${CUDAToolkit_VERSION}
      CACHE STRING "")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Tesla P40/P4, Quadro Pxxx/Pxxxx, GeForce GTX 10xx, TITAN X/Xp
    list(APPEND CMAKE_CUDA_ARCHITECTURES 60-real)

    # V100, TITAN V
    list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)

    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      # T4, Quadro RTX xxxx, Txxxx, Geforce RTX 20xx, TITAN RTX
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      # A100
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
      # GeForce RTX 30xx
      list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
      # GeForce RTX 40xx
      list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
      # NVIDIA H100
      list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
    endif()
  endif()

  pybind11_add_module(ndarray_backend_cuda src/ndarray_backend_cuda.cu)
  add_dependencies(backend ndarray_backend_cuda)

  install(TARGETS ndarray_backend_cuda LIBRARY DESTINATION ${PYTHON_MODULE_PATH}
                                               COMPONENT backend)
endif()
if(FINEFLOW_BUILD_TEST)
  include(FetchCatch2)
  fetch_catch2()
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
  enable_testing()
  include(CTest)
  include(Catch)

  add_cc_test(test_tensor SRCS tests/cpp/test_tensor.cpp DEPENDS FineflowCore)

endif(FINEFLOW_BUILD_TEST)
