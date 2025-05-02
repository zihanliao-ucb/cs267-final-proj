#include "common.h"
#include "data_loader.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

struct SpMatrixCUDA {
    int rows;
    int cols;
    int nnz;
    int *row_ptr; // csr row pointer to device memory
    int *col_idx; // csr column index to device memory
    float *values; // csr values to device memory
    bool from_global; // whether the matrix is from global memory
    SpMatrixCUDA() {rows = cols = nnz = 0; row_ptr = col_idx = nullptr; values = nullptr; from_global = false;}
    SpMatrixCUDA(const SpMatrixCUDA &mat) {
        rows = mat.rows;
        cols = mat.cols;
        nnz = mat.nnz;
        from_global = false;
        CHECK_CUDA(cudaMalloc(&row_ptr, sizeof(int) * (rows + 1)));
        CHECK_CUDA(cudaMalloc(&col_idx, sizeof(int) * nnz));
        CHECK_CUDA(cudaMalloc(&values, sizeof(float) * nnz));
        CHECK_CUDA(cudaMemcpy(row_ptr, mat.row_ptr, sizeof(int) * (rows + 1), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(col_idx, mat.col_idx, sizeof(int) * nnz, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(values, mat.values, sizeof(float) * nnz, cudaMemcpyDeviceToDevice));
    }
    SpMatrixCUDA(int rows, int cols, int nnz) {
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;
        from_global = false;
    }
    void set_ptrs(int *row_ptr, int *col_idx, float *values) {
        this->row_ptr = row_ptr;
        this->col_idx = col_idx;
        this->values = values;
    }
    void free() {
        if (from_global) return;
        if (row_ptr) CHECK_CUDA(cudaFree(row_ptr)); row_ptr = nullptr;
        if (col_idx) CHECK_CUDA(cudaFree(col_idx)); col_idx = nullptr;
        if (values) CHECK_CUDA(cudaFree(values)); values = nullptr;
    }
};

struct GlobalSpMatrix {
    int *rows;
    int *cols;
    int *nnz;
    int *row_ptr;
    int *col_idx;
    float *values;
    int local_rows() const {
        int local_rows;
        CHECK_CUDA(cudaMemcpy(&local_rows, this->rows, sizeof(int), cudaMemcpyDeviceToHost));
        return local_rows;
    }
    int local_cols() const {
        int local_cols;
        CHECK_CUDA(cudaMemcpy(&local_cols, this->cols, sizeof(int), cudaMemcpyDeviceToHost));
        return local_cols;
    }
    int local_nnz() const {
        int local_nnz;
        CHECK_CUDA(cudaMemcpy(&local_nnz, this->nnz, sizeof(int), cudaMemcpyDeviceToHost));
        return local_nnz;
    }
    void free() {
        if (rows) nvshmem_free(rows); rows = nullptr;
        if (cols) nvshmem_free(cols); cols = nullptr;
        if (nnz) nvshmem_free(nnz); nnz = nullptr;
        if (row_ptr) nvshmem_free(row_ptr); row_ptr = nullptr;
        if (col_idx) nvshmem_free(col_idx); col_idx = nullptr;
        if (values) nvshmem_free(values); values = nullptr;
    }
};

SpMatrixCUDA load(const CSRMatrix &mat);
GlobalSpMatrix make_global(GlobalSpMatrix &mat_buffer, const SpMatrixCUDA &mat);
SpMatrixCUDA to_local(const GlobalSpMatrix &global);
GlobalSpMatrix make_matrix_buffer(int max_rows, int max_nnz);
void put_matrix(GlobalSpMatrix &mat_buffer, const GlobalSpMatrix &mat, int pe);

SpMatrixCUDA sub_matrix_col(const SpMatrixCUDA &mat, int col_bias, int cols);
SpMatrixCUDA sub_matrix_row(const SpMatrixCUDA &mat, int row_start, int rows);
SpMatrixCUDA sub_matrix(const SpMatrixCUDA &mat, int row_bias, int rows, int col_bias, int cols);
SpMatrixCUDA spgemm(const SpMatrixCUDA &A, const SpMatrixCUDA &B);
SpMatrixCUDA spadd(const SpMatrixCUDA &A, const SpMatrixCUDA &B);

// void merge(std::vector<SpMatrixCUDA> &submats, const SpMatrixCUDA &B);
// SpMatrixCUDA concat(std::vector<SpMatrixCUDA> &submats);
