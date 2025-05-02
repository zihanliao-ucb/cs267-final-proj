#include "common.h"

int get_pe(int i, int j)
{
    int pe_per_side = std::sqrt(nvshmem_n_pes());
    return i * pe_per_side + j;
}

int get_i(int pe)
{
    int pe_per_side = std::sqrt(nvshmem_n_pes());
    return pe / pe_per_side;
}

int get_j(int pe)
{
    int pe_per_side = std::sqrt(nvshmem_n_pes());
    return pe % pe_per_side;
}

int get_row_bias(int N, int pe)
{
    int pe_per_side = std::sqrt(nvshmem_n_pes());
    int rows_per_pe = (N + pe_per_side - 1) / pe_per_side;
    int i = get_i(pe);
    return i * rows_per_pe;
}

int get_rows(int N, int pe)
{
    int pe_per_side = std::sqrt(nvshmem_n_pes());
    int row_bias = get_row_bias(N, pe);
    int rows_per_pe = (N + pe_per_side - 1) / pe_per_side;
    return std::min(rows_per_pe, N - row_bias);
}

int get_col_bias(int N, int pe)
{
    int pe_per_side = std::sqrt(nvshmem_n_pes());
    int cols_per_pe = (N + pe_per_side - 1) / pe_per_side;
    int j = get_j(pe);
    return j * cols_per_pe;
}

int get_cols(int N, int pe)
{
    int pe_per_side = std::sqrt(nvshmem_n_pes());
    int col_bias = get_col_bias(N, pe);
    int cols_per_pe = (N + pe_per_side - 1) / pe_per_side;
    return std::min(cols_per_pe, N - col_bias);
}