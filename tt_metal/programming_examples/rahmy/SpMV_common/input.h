#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>

#include "mmio.h"
#include "config.h"

// Define the coo_matrix struct
struct coo_matrix {
    std::vector<uint32_t> rows;
    std::vector<uint32_t> cols;
    std::vector<bfloat16> vals;
    int M;    // Number of rows
    int N;    // Number of columns
    int nnz;  // Number of non-zero elements
};

// Function to sort the COO matrix
static void sort_coo(coo_matrix& coo) {
    auto cmp = [&](size_t i, size_t j) {
        if (coo.rows[i] < coo.rows[j]) {
            return true;
        }
        if (coo.rows[i] > coo.rows[j]) {
            return false;
        }
        return coo.cols[i] < coo.cols[j];
    };

    std::vector<size_t> indices(coo.nnz);
    for (size_t i = 0; i < coo.nnz; ++i) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), cmp);

    std::vector<uint32_t> sorted_rows(coo.nnz);
    std::vector<uint32_t> sorted_cols(coo.nnz);
    std::vector<bfloat16> sorted_vals(coo.nnz);

    for (size_t i = 0; i < coo.nnz; ++i) {
        sorted_rows[i] = coo.rows[indices[i]];
        sorted_cols[i] = coo.cols[indices[i]];
        sorted_vals[i] = coo.vals[indices[i]];
    }

    coo.rows = std::move(sorted_rows);
    coo.cols = std::move(sorted_cols);
    coo.vals = std::move(sorted_vals);
}

// Function to read a COO matrix from a Matrix Market file
void read_coo_matrix(coo_matrix& coo, const char* mm_filename) {
    FILE* fid;
    MM_typecode matcode;

    fid = fopen(mm_filename, "r");
    if (fid == nullptr) {
        std::cerr << "Unable to open file " << mm_filename << std::endl;
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0) {
        std::cerr << "Could not process Matrix Market banner." << std::endl;
        exit(1);
    }

    if (!mm_is_valid(matcode)) {
        std::cerr << "Invalid Matrix Market file." << std::endl;
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) &&
          mm_is_sparse(matcode))) {
        std::cerr << "Unsupported Matrix Market type: [" << mm_typecode_to_str(matcode) << "]" << std::endl;
        std::cerr << "Only sparse real-valued or pattern coordinate matrices are supported." << std::endl;
        exit(1);
    }

    int num_rows, num_cols, num_nonzeros;
    if (mm_read_mtx_crd_size(fid, &num_rows, &num_cols, &num_nonzeros) != 0) {
        exit(1);
    }

    coo.M = num_rows;
    coo.N = num_cols;
    coo.nnz = num_nonzeros;

    coo.rows.resize(coo.nnz);
    coo.cols.resize(coo.nnz);
    coo.vals.resize(coo.nnz);

    std::cout << "Reading sparse matrix from file (" << mm_filename << "):" << std::flush;

    if (mm_is_pattern(matcode)) {
        for (uint32_t i = 0; i < coo.nnz; ++i) {
            uint32_t I, J;
            assert(fscanf(fid, " %u %u \n", &I, &J) == 2);
            coo.rows[i] = I - 1;  // Adjust from 1-based to 0-based indexing
            coo.cols[i] = J - 1;
            coo.vals[i] = bfloat16(1.0f);  // Use value 1.0 for all nonzero entries
        }
    } else if (mm_is_real(matcode) || mm_is_integer(matcode)) {
        for (uint32_t i = 0; i < coo.nnz; ++i) {
            uint32_t I, J;
            float V;
            assert(fscanf(fid, " %u %u %f \n", &I, &J, &V) == 3);
            coo.rows[i] = I - 1;  // Adjust from 1-based to 0-based indexing
            coo.cols[i] = J - 1;
            coo.vals[i] = bfloat16(V);
        }
    } else {
        std::cerr << "Unrecognized data type." << std::endl;
        exit(1);
    }

    fclose(fid);
    std::cout << " done" << std::endl;

    if (mm_is_symmetric(matcode)) {  // Duplicate off-diagonal entries
        uint32_t off_diagonals = 0;
        for (uint32_t i = 0; i < coo.nnz; ++i) {
            if (coo.rows[i] != coo.cols[i]) {
                off_diagonals++;
            }
        }

        uint32_t true_nonzeros = 2 * off_diagonals + (coo.nnz - off_diagonals);

        std::vector<uint32_t> new_rows(true_nonzeros);
        std::vector<uint32_t> new_cols(true_nonzeros);
        std::vector<bfloat16> new_vals(true_nonzeros);

        uint32_t ptr = 0;
        for (uint32_t i = 0; i < coo.nnz; ++i) {
            if (coo.rows[i] != coo.cols[i]) {
                new_rows[ptr] = coo.rows[i];
                new_cols[ptr] = coo.cols[i];
                new_vals[ptr] = coo.vals[i];
                ptr++;
                new_rows[ptr] = coo.cols[i];
                new_cols[ptr] = coo.rows[i];
                new_vals[ptr] = coo.vals[i];
                ptr++;
            } else {
                new_rows[ptr] = coo.rows[i];
                new_cols[ptr] = coo.cols[i];
                new_vals[ptr] = coo.vals[i];
                ptr++;
            }
        }

        coo.rows = std::move(new_rows);
        coo.cols = std::move(new_cols);
        coo.vals = std::move(new_vals);
        coo.nnz = true_nonzeros;
    }

    // Sort the COO matrix
    sort_coo(coo);
}
