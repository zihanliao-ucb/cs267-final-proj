#include "SpMatrixCUDA.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

__global__ void count_submatrix_nnz_per_row(
    const int *row_ptr, const int *col_idx,
    int rows, int col_bias, int cols,
    int *sub_row_counts)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int count = 0;
    for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        int col = col_idx[idx];
        if (col >= col_bias && col < col_bias + cols) {
            count++;
        }
    }
    sub_row_counts[row] = count;
}

__global__ void fill_submatrix(
    const int *row_ptr, const int *col_idx, const float *values,
    int *sub_row_ptr, int *sub_col_idx, float *sub_values,
    int rows, int col_bias, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int write_idx = sub_row_ptr[row];

    for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        int col = col_idx[idx];
        if (col >= col_bias && col < col_bias + cols) {
            sub_col_idx[write_idx] = col - col_bias; // Shift
            sub_values[write_idx] = values[idx];
            write_idx++;
        }
    }
}

SpMatrixCUDA load(const CSRMatrix &mat)
{
    SpMatrixCUDA d_mat;
    d_mat.rows = mat.rows;
    d_mat.cols = mat.cols;
    d_mat.nnz = mat.nnz;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_mat.row_ptr, sizeof(int) * (mat.rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_mat.col_idx, sizeof(int) * mat.nnz));
    CHECK_CUDA(cudaMalloc(&d_mat.values, sizeof(float) * mat.nnz));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_mat.row_ptr, mat.row_ptr.data(), sizeof(int) * (mat.rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat.col_idx, mat.col_idx.data(), sizeof(int) * mat.nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat.values, mat.values.data(), sizeof(float) * mat.nnz, cudaMemcpyHostToDevice));

    return d_mat;
}

GlobalSpMatrix make_global(GlobalSpMatrix &mat_buffer, const SpMatrixCUDA &mat)
{
    CHECK_CUDA(cudaMemcpy(mat_buffer.rows, &mat.rows, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mat_buffer.cols, &mat.cols, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mat_buffer.nnz, &mat.nnz, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mat_buffer.row_ptr, mat.row_ptr, sizeof(int) * (mat.rows + 1), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(mat_buffer.col_idx, mat.col_idx, sizeof(int) * mat.nnz, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(mat_buffer.values, mat.values, sizeof(float) * mat.nnz, cudaMemcpyDeviceToDevice));
    return mat_buffer;
}

SpMatrixCUDA to_local(const GlobalSpMatrix &global)
{
    SpMatrixCUDA mat = SpMatrixCUDA(global.local_rows(), global.local_cols(), global.local_nnz());
    mat.from_global = true;
    mat.set_ptrs(global.row_ptr, global.col_idx, global.values);
    return mat;
}

GlobalSpMatrix make_matrix_buffer(int max_rows, int max_nnz)
{
    GlobalSpMatrix mat;
    mat.rows = (int*) nvshmem_malloc(sizeof(int));
    mat.cols = (int*) nvshmem_malloc(sizeof(int));
    mat.nnz = (int*) nvshmem_malloc(sizeof(int));
    mat.row_ptr = (int*) nvshmem_malloc(sizeof(int) * (max_rows + 1));
    mat.col_idx = (int*) nvshmem_malloc(sizeof(int) * max_nnz);
    mat.values = (float*) nvshmem_malloc(sizeof(float) * max_nnz);
    return mat;
}

void put_matrix(GlobalSpMatrix &mat_buffer, const GlobalSpMatrix &mat, int pe)
{
    if (pe == nvshmem_my_pe()) return;
    nvshmem_int_put(mat_buffer.rows, mat.rows, 1, pe);
    nvshmem_int_put(mat_buffer.cols, mat.cols, 1, pe);
    nvshmem_int_put(mat_buffer.nnz, mat.nnz, 1, pe);
    int rows = mat.local_rows();
    int cols = mat.local_cols();
    int nnz = mat.local_nnz();
    nvshmem_int_put(mat_buffer.row_ptr, mat.row_ptr, rows + 1, pe);
    nvshmem_int_put(mat_buffer.col_idx, mat.col_idx, nnz, pe);
    nvshmem_float_put(mat_buffer.values, mat.values, nnz, pe);
}

SpMatrixCUDA sub_matrix_col(const SpMatrixCUDA &mat, int col_bias, int cols)
{
    SpMatrixCUDA sub;
    sub.rows = mat.rows;
    sub.cols = cols;

    // Step 1: Count nonzeros per row
    int *d_row_counts;
    CHECK_CUDA(cudaMalloc(&d_row_counts, sizeof(int) * mat.rows));

    int blockSize = 256;
    int gridSize = (mat.rows + blockSize - 1) / blockSize;
    count_submatrix_nnz_per_row<<<gridSize, blockSize>>>(
        mat.row_ptr, mat.col_idx, mat.rows, col_bias, cols, d_row_counts);

    // Step 2: Exclusive scan to compute submatrix row_ptr
    CHECK_CUDA(cudaMalloc(&sub.row_ptr, sizeof(int) * (mat.rows + 1)));
    CHECK_CUDA(cudaMemset(sub.row_ptr, 0, sizeof(int)));
    thrust::device_ptr<int> row_counts_ptr(d_row_counts);
    thrust::device_ptr<int> row_ptr_ptr(sub.row_ptr);
    thrust::inclusive_scan(row_counts_ptr, row_counts_ptr + mat.rows, row_ptr_ptr + 1);

    // Step 3: Find total number of nonzeros
    int sub_nnz;
    CHECK_CUDA(cudaMemcpy(&sub_nnz, sub.row_ptr + mat.rows, sizeof(int), cudaMemcpyDeviceToHost));
    sub.nnz = sub_nnz;

    // Step 4: Allocate col_idx and values for submatrix
    CHECK_CUDA(cudaMalloc(&sub.col_idx, sizeof(int) * sub.nnz));
    CHECK_CUDA(cudaMalloc(&sub.values, sizeof(float) * sub.nnz));

    // Step 5: Fill col_idx and values
    fill_submatrix<<<gridSize, blockSize>>>(
        mat.row_ptr, mat.col_idx, mat.values,
        sub.row_ptr, sub.col_idx, sub.values,
        mat.rows, col_bias, cols);

    // Free temporary
    CHECK_CUDA(cudaFree(d_row_counts));

    return sub;
}

SpMatrixCUDA sub_matrix_row(const SpMatrixCUDA &mat, int row_start, int rows)
{
    SpMatrixCUDA sub;
    sub.rows = rows;
    sub.cols = mat.cols;
    // nnz not yet known, we'll compute it below

    // Step 1: Find the nnz range
    int start_nnz, end_nnz;
    CHECK_CUDA(cudaMemcpy(&start_nnz, mat.row_ptr + row_start, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&end_nnz, mat.row_ptr + row_start + rows, sizeof(int), cudaMemcpyDeviceToHost));
    sub.nnz = end_nnz - start_nnz;

    // Step 2: Allocate memory
    CHECK_CUDA(cudaMalloc(&sub.row_ptr, sizeof(int) * (rows + 1)));
    CHECK_CUDA(cudaMalloc(&sub.col_idx, sizeof(int) * sub.nnz));
    CHECK_CUDA(cudaMalloc(&sub.values, sizeof(float) * sub.nnz));

    // Step 3: Copy col_idx and values
    CHECK_CUDA(cudaMemcpy(sub.col_idx, mat.col_idx + start_nnz, sizeof(int) * sub.nnz, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(sub.values, mat.values + start_nnz, sizeof(float) * sub.nnz, cudaMemcpyDeviceToDevice));

    // Step 4: Copy row_ptr and shift it
    thrust::device_ptr<int> src_row_ptr(mat.row_ptr + row_start);
    thrust::device_ptr<int> dst_row_ptr(sub.row_ptr);

    // First, copy original offsets
    thrust::copy(src_row_ptr, src_row_ptr + (rows + 1), dst_row_ptr);

    // Then, shift them all by -(start_nnz)
    thrust::transform(
        dst_row_ptr, dst_row_ptr + (rows + 1),
        thrust::make_constant_iterator(-start_nnz),
        dst_row_ptr,
        thrust::plus<int>());
    return sub;
}

SpMatrixCUDA sub_matrix(const SpMatrixCUDA &mat, int row_bias, int rows, int col_bias, int cols)
{
    SpMatrixCUDA sub_row = sub_matrix_row(mat, row_bias, rows);
    SpMatrixCUDA sub = sub_matrix_col(sub_row, col_bias, cols);
    sub_row.free();
    return sub;
}

SpMatrixCUDA spgemm(const SpMatrixCUDA &A, const SpMatrixCUDA &B)
{
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // === Step 1: Create matA, matB, matC (original C) descriptors ===
    cusparseSpMatDescr_t matA, matB, matC;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                      (void*)A.row_ptr, (void*)A.col_idx, (void*)A.values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B.rows, B.cols, B.nnz,
                      (void*)B.row_ptr, (void*)B.col_idx, (void*)B.values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A.rows, B.cols, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // === Step 2: Create a fresh output result ===
    SpMatrixCUDA result;
    result.rows = A.rows;
    result.cols = B.cols;

    float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    // === Step 3: Work estimation ===
    size_t bufferSize1 = 0, bufferSize2 = 0;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL));
    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1));

    // === Step 4: Compute ===
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL));
    CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));

    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2));

    // === Step 5: Get new size and allocate output ===
    int64_t new_num_rows, new_num_cols, new_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &new_num_rows, &new_num_cols, &new_nnz));

    result.nnz = static_cast<int>(new_nnz);
    CHECK_CUDA(cudaMalloc(&result.row_ptr, sizeof(int) * (new_num_rows + 1)));
    CHECK_CUDA(cudaMalloc(&result.col_idx, sizeof(int) * result.nnz));
    CHECK_CUDA(cudaMalloc(&result.values, sizeof(float) * result.nnz));

    // Update pointers for matC
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, result.row_ptr, result.col_idx, result.values));

    // === Step 6: Final copy ===
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT,
                        spgemmDesc));

    // === Step 7: Clean up ===
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(dBuffer1));
    CHECK_CUDA(cudaFree(dBuffer2));

    return result;
}

SpMatrixCUDA spadd(const SpMatrixCUDA &A, const SpMatrixCUDA &B)
{
    if (A.values == nullptr) return SpMatrixCUDA(B);
    if (B.values == nullptr) return SpMatrixCUDA(A);
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::runtime_error("Matrices must have the same dimensions");
    }
    float alpha = 1.0f, beta = 1.0f;
    SpMatrixCUDA result;
    result.rows = A.rows;
    result.cols = A.cols;
    
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    cusparseMatDescr_t descrA, descrB, descrC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));

    int baseC, nnzC;
    /* alpha, nnzTotalDevHostPtr points to host memory */
    size_t bufferSizeInBytes;
    char *buffer = NULL;
    int *nnzTotalDevHostPtr = &nnzC;
    CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
    CHECK_CUDA(cudaMalloc((void**)&result.row_ptr, sizeof(int)*(A.rows+1)));
    /* prepare buffer */
    CHECK_CUSPARSE(cusparseScsrgeam2_bufferSizeExt(handle, A.rows, A.cols,
        &alpha,
        descrA, A.nnz,
        A.values, A.row_ptr, A.col_idx,
        &beta,
        descrB, B.nnz,
        B.values, B.row_ptr, B.col_idx,
        descrC,
        result.values, result.row_ptr, result.col_idx,
        &bufferSizeInBytes
        ));
    CHECK_CUDA(cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes));
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, A.rows, A.cols,
            descrA, A.nnz, A.row_ptr, A.col_idx,
            descrB, B.nnz, B.row_ptr, B.col_idx,
            descrC, result.row_ptr, nnzTotalDevHostPtr,
            buffer));
    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        CHECK_CUDA(cudaMemcpy(&nnzC, result.row_ptr+A.rows, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&baseC, result.row_ptr, sizeof(int), cudaMemcpyDeviceToHost));
        nnzC -= baseC;
    }
    result.nnz = nnzC;
    CHECK_CUDA(cudaMalloc((void**)&result.col_idx, sizeof(int)*nnzC));
    CHECK_CUDA(cudaMalloc((void**)&result.values, sizeof(float)*nnzC));
    CHECK_CUSPARSE(cusparseScsrgeam2(handle, A.rows, A.cols,
            &alpha,
            descrA, A.nnz,
            A.values, A.row_ptr, A.col_idx,
            &beta,
            descrB, B.nnz,
            B.values, B.row_ptr, B.col_idx,
            descrC,
            result.values, result.row_ptr, result.col_idx,
            buffer));
    
    return result;
}







// __host__
// void merge(std::vector<SpMatrixCUDA> &submats, const SpMatrixCUDA &B)
// {
//     if (submats.empty()) {
//         submats.push_back(B);
//         return;
//     }

//     const SpMatrixCUDA &A = submats.back();

//     if (A.row_bias > B.row_bias) {
//         throw std::runtime_error("A must be above B");
//     }
//     if (A.row_bias + A.rows < B.row_bias) {
//         throw std::runtime_error("A and B are not consecutive");
//     }
//     if (A.row_bias + A.rows > B.row_bias + 1) {
//         throw std::runtime_error("A and B have more than one overlapping row");
//     }

//     if (A.row_bias + A.rows == B.row_bias)
//     {
//         // === Case 1: No overlap — just concatenate ===
//         submats.push_back(B);
//     }
//     else
//     {
//         // === Case 2: One row overlap — need merge ===

//         // Pop A off the list
//         submats.pop_back();

//         SpMatrixCUDA A_top = sub_matrix_row(A, 0, A.rows - 1);      // A without last row
//         SpMatrixCUDA A_last_row = sub_matrix_row(A, A.rows - 1, 1); // A's last row
//         SpMatrixCUDA B_first_row = sub_matrix_row(B, 0, 1);         // B's first row
//         SpMatrixCUDA B_bottom = sub_matrix_row(B, 1, B.rows - 1);   // B without first row

//         // Merge the overlapping row
//         SpMatrixCUDA ab_row = spadd(A_last_row, B_first_row);

//         // Only push non-empty parts
//         if (A_top.rows > 0) {
//             submats.push_back(A_top);
//         }
//         submats.push_back(ab_row);
//         if (B_bottom.rows > 0) {
//             submats.push_back(B_bottom);
//         }

//         // Free temporary slices (we don't own input B)
//         free(A_last_row);
//         free(B_first_row);
//         free(A); // free the old full A
//         free(B); // free the old full B
//     }
// }

// __host__
// SpMatrixCUDA concat(std::vector<SpMatrixCUDA> &submats)
// {
//     if (submats.empty()) {
//         throw std::runtime_error("submats is empty");
//     }

//     SpMatrixCUDA merged;

//     // Setup merged matrix meta
//     merged.row_bias = submats.front().row_bias;
//     merged.cols = submats.front().cols;
//     merged.rows = 0;
//     merged.nnz = 0;

//     // Compute total rows and nnz
//     for (const auto& mat : submats) {
//         merged.rows += mat.rows;
//         merged.nnz += mat.nnz;
//     }

//     // Allocate memory
//     cudaMalloc(&merged.row_ptr, sizeof(int) * (merged.rows + 1));
//     cudaMalloc(&merged.col_idx, sizeof(int) * merged.nnz);
//     cudaMalloc(&merged.values, sizeof(float) * merged.nnz);

//     // Step 1: Fill row_ptr
//     int *d_row_ptr = merged.row_ptr;
//     int *d_col_idx = merged.col_idx;
//     float *d_values = merged.values;

//     int row_offset = 0;
//     int nnz_offset = 0;

//     for (const auto& mat : submats)
//     {
//         // 1. Copy row_ptr

//         thrust::device_ptr<int> src_row_ptr(mat.row_ptr);
//         thrust::device_ptr<int> dst_row_ptr(d_row_ptr + row_offset);

//         // Copy row_ptr[0..mat.rows] and adjust nnz offset
//         if (row_offset == 0) {
//             // First block, copy directly
//             thrust::copy(src_row_ptr, src_row_ptr + (mat.rows + 1), dst_row_ptr);
//         } else {
//             // Subsequent blocks, need to offset the nnz
//             thrust::transform(
//                 src_row_ptr, src_row_ptr + (mat.rows + 1),
//                 thrust::make_constant_iterator(nnz_offset),
//                 dst_row_ptr,
//                 thrust::plus<int>());
//         }

//         // 2. Copy col_idx
//         cudaMemcpy(d_col_idx + nnz_offset, mat.col_idx, sizeof(int) * mat.nnz, cudaMemcpyDeviceToDevice);

//         // 3. Copy values
//         cudaMemcpy(d_values + nnz_offset, mat.values, sizeof(float) * mat.nnz, cudaMemcpyDeviceToDevice);

//         // Advance offsets
//         row_offset += mat.rows;
//         nnz_offset += mat.nnz;
//     }

//     return merged;
// }
