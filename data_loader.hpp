#pragma once
#include <vector>
#include <string>

struct CSRMatrix
{
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

CSRMatrix load_matrix_market_to_csr(const std::string &filepath);
