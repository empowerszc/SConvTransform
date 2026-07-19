cmake_minimum_required(VERSION 3.12)
project(SConvNative CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENBLAS "Link OpenBLAS for microkernel" ON)

# Find OpenBLAS
if(USE_OPENBLAS)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h
        PATHS /opt/homebrew/opt/openblas/include
              /usr/include/openblas
              /usr/include/x86_64-linux-gnu
              /usr/local/include
    )
    find_library(OPENBLAS_LIB openblas
        PATHS /opt/homebrew/opt/openblas/lib
              /usr/lib
              /usr/lib/x86_64-linux-gnu
              /usr/local/lib
    )
    if(OPENBLAS_LIB)
        message(STATUS "OpenBLAS found: ${OPENBLAS_LIB}")
        add_definitions(-DUSE_OPENBLAS)
        include_directories(${OPENBLAS_INCLUDE_DIR})
    else()
        message(WARNING "OpenBLAS not found, using naive GEMM fallback")
    endif()
endif()

add_executable(sconv_bench src/sconv.cpp)

if(OPENBLAS_LIB)
    target_link_libraries(sconv_bench ${OPENBLAS_LIB})
endif()

# OpenMP for parallelism (optional, for future multi-core support)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sconv_bench OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
endif()

# Architecture flags for SVE/SME (ARM only)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_compile_options(-march=armv9-a+sve+sme -O3)
    message(STATUS "aarch64: enabling SVE+SME flags")
else()
    add_compile_options(-O3)
endif()

install(TARGETS sconv_bench RUNTIME DESTINATION bin)
