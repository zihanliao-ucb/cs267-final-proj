cmake_minimum_required(VERSION 3.18)
project(spgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

# Path to NVSHMEM
set(NVSHMEM_DIR "/global/common/software/nersc9/nvshmem/3.2.5-1-25.03")
include_directories(${NVSHMEM_DIR}/include)
link_directories(${NVSHMEM_DIR}/lib)

# Mark the target to use CUDA linker

add_executable(spgemm data_loader.cpp main.cu)

# *** This line triggers nvcc for the link step ***
set_target_properties(spgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# still OK to link the static library:
target_link_libraries(spgemm
    PRIVATE
        ${NVSHMEM_DIR}/lib/libnvshmem.a   # or lib64, whichever exists
        cuda  cudart  # MPI/libs as needed
)


# add_executable(spgemm data_loader.cpp spgemm.cu)
# target_link_libraries(spgemm PRIVATE NVSHMEM::nvshmem)

# # CUDA arch flags (A100 on Perlmutter)
# target_compile_options(spgemm PRIVATE
#         $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=sm_80>
# )