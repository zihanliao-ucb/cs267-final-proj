#include "common.h"
#include "SpMatrixCUDA.h"
#include "data_loader.h"

// Distributed SpGEMM using 2D SUMMA to calculate mat * mat
int main(int argc, char** argv) {
    // Initialize NVSHMEM
    int mype_node;
    nvshmem_init();
    int pe_per_side = sqrt(nvshmem_n_pes());
    if (pe_per_side * pe_per_side != nvshmem_n_pes()) {
        throw std::runtime_error("Number of PEs must be a perfect square");
    }
    if (argc < 2) {
        throw std::runtime_error("Usage: " + std::string(argv[0]) + " <path_to_matrix.mtx>");
    }
    std::string matrix_path = argv[1];
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    
    // Load matrix
    CSRMatrix mat_host = load_matrix_market_to_csr(matrix_path);
    if (mat_host.rows != mat_host.cols) {
        throw std::runtime_error("Matrix is not square");
    }
    SpMatrixCUDA mat_device = load(mat_host);
    mat_host.free();
    int N = mat_device.rows; // Size of the square matrix
    if (nvshmem_my_pe() == 0) {
        std::cout << "PE " << nvshmem_my_pe() << ": mat_device.rows = " << mat_device.rows << std::endl;
        std::cout << "PE " << nvshmem_my_pe() << ": mat_device.cols = " << mat_device.cols << std::endl;
        std::cout << "PE " << nvshmem_my_pe() << ": mat_device.nnz = " << mat_device.nnz << std::endl;
    }
    nvshmem_barrier_all();

    // Subdivide matrix
    int local_row_start = get_row_bias(N, nvshmem_my_pe());
    int local_rows = get_rows(N, nvshmem_my_pe());
    int local_col_start = get_col_bias(N, nvshmem_my_pe());
    int local_cols = get_cols(N, nvshmem_my_pe());
    SpMatrixCUDA submat = sub_matrix(mat_device, local_row_start, local_rows, local_col_start, local_cols);
    mat_device.free();
    nvshmem_barrier_all();

    // Broadcast sizes of submat to all PEs
    int *nnz_buffer = (int*) nvshmem_malloc(sizeof(int) * nvshmem_n_pes());
    int *rows_buffer = (int*) nvshmem_malloc(sizeof(int) * nvshmem_n_pes());
    CHECK_CUDA(cudaMemcpy(nnz_buffer + nvshmem_my_pe(), &submat.nnz, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rows_buffer + nvshmem_my_pe(), &submat.rows, sizeof(int), cudaMemcpyHostToDevice));
    nvshmem_barrier_all();
    for (int i = 0; i < nvshmem_n_pes(); i++) {
        if (i == nvshmem_my_pe()) continue;
        nvshmem_int_put(nnz_buffer + nvshmem_my_pe(), nnz_buffer + nvshmem_my_pe(), 1, i);
        nvshmem_int_put(rows_buffer + nvshmem_my_pe(), rows_buffer + nvshmem_my_pe(), 1, i);
    }
    nvshmem_barrier_all();

    // Find the maximum nnz and rows to make the matrix buffer large enough for all PEs
    int max_nnz = 0, max_rows = 0;
    int nnzs[nvshmem_n_pes()];
    int rows[nvshmem_n_pes()];
    CHECK_CUDA(cudaMemcpy(nnzs, nnz_buffer, sizeof(int) * nvshmem_n_pes(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(rows, rows_buffer, sizeof(int) * nvshmem_n_pes(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nvshmem_n_pes(); i++) {
        max_nnz = std::max(max_nnz, nnzs[i]);
        max_rows = std::max(max_rows, rows[i]);
    }
    nvshmem_free(nnz_buffer);
    nvshmem_free(rows_buffer);
    if (nvshmem_my_pe() == 0) {
        std::cout << "PE " << nvshmem_my_pe() << ": max_nnz = " << max_nnz << ", max_rows = " << max_rows << std::endl;
    }
    nvshmem_barrier_all();

    // Make global matrix buffer
    GlobalSpMatrix submat_global = make_matrix_buffer(max_rows, max_nnz);
    submat_global = make_global(submat_global, submat);
    submat.free();
    GlobalSpMatrix matrix_buffer_A = make_matrix_buffer(max_rows, max_nnz);
    GlobalSpMatrix matrix_buffer_B = make_matrix_buffer(max_rows, max_nnz);
    nvshmem_barrier_all();

    // 2D SUMMA
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    SpMatrixCUDA subC;
    for (int k = 0; k < pe_per_side; k++) {
        nvshmem_barrier_all();
        // Broadcast A_{ik} to all PEs in row i
        for (int i = 0; i < pe_per_side; i++) {
            int pe = get_pe(i, k);
            if (pe == nvshmem_my_pe()) {
                for (int kk = 0; kk < pe_per_side; kk++) {
                    int pe_kk = get_pe(i, kk);
                    put_matrix(matrix_buffer_A, submat_global, pe_kk);
                }
            }
            nvshmem_barrier_all();
        }
        // Broadcast A_{kj} to all PEs in col j
        for (int j = 0; j < pe_per_side; j++) {
            int pe = get_pe(k, j);
            if (pe == nvshmem_my_pe()) {
                for (int kk = 0; kk < pe_per_side; kk++) {
                    int pe_kk = get_pe(kk, j);
                    put_matrix(matrix_buffer_B, submat_global, pe_kk);
                }
            }
            nvshmem_barrier_all();
        }
        // Local SpGEMM
        SpMatrixCUDA A, B;
        if (get_j(nvshmem_my_pe()) == k) {
            A = to_local(submat_global);
        }
        else {
            A = to_local(matrix_buffer_A);
        }
        if (get_i(nvshmem_my_pe()) == k) {
            B = to_local(submat_global);
        }
        else {
            B = to_local(matrix_buffer_B);
        }
        SpMatrixCUDA C = spgemm(A, B);
        SpMatrixCUDA temp_subC = spadd(subC, C);
        subC.free();
        subC = temp_subC;
        A.free();
        B.free();
        C.free();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    std::cout << "PE " << nvshmem_my_pe() << ": SpGEMM time: " << time << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    nvshmem_barrier_all();

    std::cout << "Final Message: PE " << nvshmem_my_pe() << ": subC.rows = " << subC.rows << ", subC.cols = " << subC.cols << ", subC.nnz = " << subC.nnz << std::endl;
    nvshmem_barrier_all();

    // Free local cuda memory
    subC.free();
    // Free nvshmem memory
    submat_global.free();
    matrix_buffer_A.free();
    matrix_buffer_B.free();
    nvshmem_finalize();
    return 0;
}