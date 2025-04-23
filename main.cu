#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#include "data_loader.hpp"

// Helper function that convert csr to a dense matrix
std::vector<std::vector<float>> csr_to_dense(const CSRMatrix &A)
{
    std::vector<std::vector<float>> dense(A.rows, std::vector<float>(A.cols, 0.0f));
    for (int i = 0; i < A.rows; ++i)
    {
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx)
        {
            int j = A.col_idx[idx];
            float val = A.values[idx];
            dense[i][j] = val;
        }
    }
    return dense;
}

std::vector<std::vector<float>> naive_matmul(
    const std::vector<std::vector<float>> &A,
    const std::vector<std::vector<float>> &B)
{
    int n = A.size();
    int m = B[0].size();
    int k = B.size();

    std::vector<std::vector<float>> C(n, std::vector<float>(m, 0.0f));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int x = 0; x < k; ++x)
                C[i][j] += A[i][x] * B[x][j];
    return C;
}

__global__ void hello_from_gpu()
{
    int pe = nvshmem_my_pe();
    printf("Hello from PE %d\n", pe);
}

int main(int argc, char **argv)
{
    std::string path = "../data/Spectro_10NN.mtx";
    CSRMatrix A = load_matrix_market_to_csr(path);

    printf("Matrix loaded from %s\n", path.c_str());
    printf("Dimensions: %d x %d\n", A.rows, A.cols);
    printf("Non-zeros: %d\n\n", A.nnz);

    auto denseA = csr_to_dense(A);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    auto C = naive_matmul(denseA, denseA);
    auto end = std::chrono::high_resolution_clock::now();

    // Duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Naive dense matrix multiplication took %.3f ms\n", duration.count());

    // nvshmem_init();
    // int mype = nvshmem_my_pe();
    // int npes = nvshmem_n_pes();

    // printf("Running on PE %d of %d\n", mype, npes);

    // hello_from_gpu<<<1, 1>>>();
    // cudaDeviceSynchronize();

    // nvshmem_finalize();
    return 0;
}
