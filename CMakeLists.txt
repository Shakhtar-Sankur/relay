cmake_minimum_required(VERSION 3.24)
project(relay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(RELAY_CUDA "Build the CUDA backend" OFF)
option(RELAY_NATIVE "Optimize for this machine's CPU" ON)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_library(relay_engine
  engine/src/json.cpp
  engine/src/safetensors.cpp
  engine/src/config.cpp
  engine/src/weights.cpp
  engine/src/backend.cpp
  engine/src/kv_cache.cpp
  engine/src/cpu_backend.cpp
  engine/src/sampler.cpp
  engine/src/engine.cpp
  engine/src/transport.cpp
  engine/src/kv_transfer.cpp
  engine/src/worker.cpp
  engine/src/c_api.cpp
)
target_include_directories(relay_engine PUBLIC engine/include)
# Position-independent, so the C interface can be a shared library (librelay_c) for Python.
set_target_properties(relay_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(relay_engine PUBLIC OpenMP::OpenMP_CXX Threads::Threads)
target_compile_options(relay_engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers>)
if(RELAY_NATIVE)
  target_compile_options(relay_engine PRIVATE -march=native)
endif()

if(RELAY_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
  endif()
  target_sources(relay_engine PRIVATE engine/src/cuda_backend.cu)
  target_compile_definitions(relay_engine PUBLIC RELAY_CUDA=1)
  target_link_libraries(relay_engine PUBLIC CUDA::cudart CUDA::cublas)
endif()

# The C interface (relay/c_api.h) as a shared library, loaded by Python through ctypes.
add_library(relay_c SHARED engine/src/c_api.cpp)
target_link_libraries(relay_c PRIVATE relay_engine)

add_executable(relay-generate tools/relay_generate.cpp)
target_link_libraries(relay-generate PRIVATE relay_engine)
add_executable(relay-worker tools/relay_worker.cpp)
target_link_libraries(relay-worker PRIVATE relay_engine)
add_executable(relay-transfer-bench tools/relay_transfer_bench.cpp)
target_link_libraries(relay-transfer-bench PRIVATE relay_engine)
add_executable(relay-attention-bench tools/relay_attention_bench.cpp)
target_link_libraries(relay-attention-bench PRIVATE relay_engine)

enable_testing()
foreach(t safetensors kv_cache reference batching sampler transfer worker faults attention c_api)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE relay_engine)
  target_compile_definitions(test_${t} PRIVATE RELAY_FIXTURES="${CMAKE_SOURCE_DIR}/tests/fixtures")
  add_test(NAME ${t} COMMAND test_${t})
endforeach()
