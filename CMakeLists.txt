cmake_minimum_required(VERSION 3.25)
project(needle C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "")

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_CUDA_FLAGS "-std=c++17 ${CMAKE_CUDA_FLAGS}")

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
