#include "data_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

CSRMatrix load_matrix_market_to_csr(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::string line;
    // Skip header and comments
    while (std::getline(file, line))
    {
        if (line[0] != '%')
            break;
    }

    std::istringstream dim_line(line);
    int rows, cols, nnz;
    dim_line >> rows >> cols >> nnz;

    std::vector<int> row_indices(nnz), col_indices(nnz);
    std::vector<float> values(nnz);

    for (int i = 0; i < nnz; ++i)
    {
        int r, c;
        float v;
        file >> r >> c >> v;
        row_indices[i] = r - 1;
        col_indices[i] = c - 1;
        values[i] = v;
    }

    // Convert COO to CSR
    CSRMatrix mat{rows, cols, nnz};
    mat.row_ptr.resize(rows + 1, 0);
    mat.col_idx = col_indices;
    mat.values = values;

    for (int i = 0; i < nnz; ++i)
    {
        mat.row_ptr[row_indices[i] + 1]++;
    }

    for (int i = 0; i < rows; ++i)
    {
        mat.row_ptr[i + 1] += mat.row_ptr[i];
    }

    return mat;
}
