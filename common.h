#pragma once

#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cusparse.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <utility>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

int get_pe(int i, int j);
int get_i(int pe);
int get_j(int pe);
int get_row_bias(int N, int pe);
int get_rows(int N, int pe);
int get_col_bias(int N, int pe);
int get_cols(int N, int pe);

