cmake_minimum_required(VERSION 3.12)
project(SConvNative CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENBLAS "Link OpenBLAS for microkernel" ON)
option(USE_KBLAST "Link HPCKit kblas for microkernel" OFF)

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

# Find HPCKit kblas (CBLAS-compatible, SME/SVE optimized)
if(USE_KBLAST)
    if(DEFINED ENV{KMLROOT})
        set(KMLROOT "$ENV{KMLROOT}")
    else()
        set(KMLROOT "/home/envs/HPCKit_26.1.RC1/HPCKit/latest/kml/bisheng")
    endif()
    set(KBLAS_INCLUDE_DIR "${KMLROOT}/include")
    # 0x48-0xd22 (Kunpeng SME) uses matrix/kblas; 0x48-0xd02/d03/d06 use sve/kblas
    if(EXISTS "${KMLROOT}/lib/matrix/kblas/multi")
        set(KBLAS_LIB_DIR "${KMLROOT}/lib/matrix/kblas/multi")
    elseif(EXISTS "${KMLROOT}/lib/sve/kblas/multi")
        set(KBLAS_LIB_DIR "${KMLROOT}/lib/sve/kblas/multi")
    else()
        set(KBLAS_LIB_DIR "${KMLROOT}/lib")
    endif()
    if(EXISTS "${KBLAS_INCLUDE_DIR}/kblas.h" AND EXISTS "${KBLAS_LIB_DIR}/libkblas.so")
        message(STATUS "kblas found: ${KBLAS_LIB_DIR}")
        add_definitions(-DUSE_KBLAST)
        include_directories(${KBLAS_INCLUDE_DIR})
        link_directories(${KBLAS_LIB_DIR})
        set(OPENBLAS_LIB kblas)
        # libomp for kblas runtime; prefer HPCKit Bisheng OpenMP
        find_library(KBLAST_OMP_LIB omp
            PATHS /home/envs/HPCKit_26.1.RC1/HPCKit/latest/compiler/bisheng/lib
                  ${KMLROOT}/lib)
        if(KBLAST_OMP_LIB)
            message(STATUS "kblas libomp: ${KBLAST_OMP_LIB}")
        endif()
        # Runtime paths: kblas lib + omp dir + KMLROOT/lib (libkml_rt.so)
        set(KBLAST_RUNTIME_DIRS "${KBLAS_LIB_DIR}:${KMLROOT}/lib")
        if(KBLAST_OMP_LIB)
            get_filename_component(KBLAST_OMP_DIR "${KBLAST_OMP_LIB}" DIRECTORY)
            set(KBLAST_RUNTIME_DIRS "${KBLAST_RUNTIME_DIRS}:${KBLAST_OMP_DIR}")
        endif()
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${KBLAST_RUNTIME_DIRS}")
    else()
        message(WARNING "kblas not found at ${KMLROOT}, using naive GEMM fallback")
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
