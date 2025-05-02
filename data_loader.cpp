#include "data_loader.h"
#include <fstream>
#include <limits>
#include <sstream>
#include <iostream>
#include <algorithm>

COOMatrix load_matrix_market_to_coo(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    bool is_symmetric = false;

    // Read header
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) {
                is_symmetric = true;
            }
            continue;
        } else {
            break; // Found the size line
        }
    }

    // Read matrix dimensions and nnz
    std::istringstream dim_stream(line);
    int rows, cols, nnz;
    dim_stream >> rows >> cols >> nnz;

    COOMatrix coo;
    coo.rows = rows;
    coo.cols = cols;

    int row, col;
    float val;
    while (file >> row >> col >> val) {
        row--; col--; // Convert to 0-based indexing
        coo.row_idx.push_back(row);
        coo.col_idx.push_back(col);
        coo.values.push_back(val);
        if (is_symmetric && row != col) {
            coo.row_idx.push_back(col);
            coo.col_idx.push_back(row);
            coo.values.push_back(val);
        }
    }

    coo.nnz = static_cast<int>(coo.values.size());
    return coo;
}

CSRMatrix coo_to_csr(const COOMatrix &coo) {
    CSRMatrix csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;

    // Step 1: Create a vector of tuples and sort by (row, col)
    std::vector<std::tuple<int, int, float>> entries;
    entries.reserve(coo.nnz);
    for (int i = 0; i < coo.nnz; ++i) {
        entries.emplace_back(coo.row_idx[i], coo.col_idx[i], coo.values[i]);
    }
    std::sort(entries.begin(), entries.end());

    // Step 2: Initialize row_ptr to size rows + 1 and fill with 0
    csr.row_ptr.assign(csr.rows + 1, 0);
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    // Step 3: Count number of entries in each row to build row_ptr
    for (const auto &entry : entries) {
        int row = std::get<0>(entry);
        csr.row_ptr[row + 1]++;
    }

    // Step 4: Convert counts to cumulative sum
    for (int i = 0; i < csr.rows; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    // Step 5: Fill in col_idx and values using insertion indices
    std::vector<int> row_offsets = csr.row_ptr;  // copy for tracking insertion positions
    for (const auto &entry : entries) {
        int row = std::get<0>(entry);
        int col = std::get<1>(entry);
        float val = std::get<2>(entry);

        int dst = row_offsets[row]++;
        csr.col_idx[dst] = col;
        csr.values[dst] = val;
    }

    return csr;
}

CSRMatrix load_matrix_market_to_csr(const std::string &filepath)
{
    COOMatrix coo = load_matrix_market_to_coo(filepath);
    return coo_to_csr(coo);
}
