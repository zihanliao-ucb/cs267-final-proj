#pragma once
#include <vector>
#include <string>
#include <tuple>

struct COOMatrix
{
    int rows, cols, nnz;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<float> values;
    ~COOMatrix() {
        free();
    }
    void free() {
        if (row_idx.size() > 0) row_idx.clear();
        if (col_idx.size() > 0) col_idx.clear();
        if (values.size() > 0) values.clear();
    }
};

struct CSRMatrix
{
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    ~CSRMatrix() {
        free();
    }
    void free() {
        if (row_ptr.size() > 0) row_ptr.clear();
        if (col_idx.size() > 0) col_idx.clear();
        if (values.size() > 0) values.clear();
    }
};

COOMatrix load_matrix_market_to_coo(const std::string &filepath);
CSRMatrix coo_to_csr(const COOMatrix &coo);
CSRMatrix load_matrix_market_to_csr(const std::string &filepath);