#ifndef BSR_MATRIX_HPP
#define BSR_MATRIX_HPP

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <random>
#include <cmath>
#include <chrono>
#include <../matmul_common/bmm_op.hpp>
#include "tt-metalium/constants.hpp"

#define RAND true
#define NO_RAND false
#define FILL_ROW 1
#define FILL_COL 2
#define FILL_DIAG 3 // will require the size to be perfect
#define TILE_SIZE 32

// TODO: if we wanted, we could put this in a namespace,
//        then we could define gemm() and spmm() to be
//        non-member functions in the namespace.

template <typename T>
class dense_matrix {
public:
    std::vector<T> data;
    size_t H;
    size_t W;

    dense_matrix() : H(0), W(0) {}

    dense_matrix(int rows, int cols, bool random) : H(rows), W(cols) {
        data.resize(rows * cols);
        if (random)
            std::generate(data.begin(), data.end(), []() { return static_cast<T>(rand()) / static_cast<T>(RAND_MAX); });
        else {
            std::fill(data.begin(), data.end(), 1);
        }
    }

    dense_matrix(int rows, int cols) : H(rows), W(cols) {
        data.resize(rows * cols);
        std::fill(data.begin(), data.end(), 0);
    }

    dense_matrix(int rows, int cols, T val) : H(rows), W(cols) {
        data.resize(rows * cols);
        std::fill(data.begin(), data.end(), val);
    }

    // Constructor from a flat vector
    dense_matrix(const std::vector<T> &data, int rows, int cols) : data(data), H(rows), W(cols) {
        assert(data.size() == rows * cols);
    }

    // Constructor from a flat vector with no size check
    dense_matrix(const dense_matrix<T> &other) : data(other.data), H(other.H), W(other.W) {}

    // Assignment operator
    dense_matrix<T>& operator=(const dense_matrix<T> &other) {
        if (this != &other) {
            data = other.data;
            H = other.H;
            W = other.W;
        }
        return *this;
    }

    // Move constructor
    dense_matrix(dense_matrix<T> &&other) noexcept : data(std::move(other.data)), H(other.H), W(other.W) {
        other.H = 0;
        other.W = 0;
    }

    // Move to bfloat16
    dense_matrix<bfloat16> bfloat16_cast() {
        std::vector<bfloat16> bfloat16_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            bfloat16_data[i] = bfloat16(data[i]);
        }
        return dense_matrix<bfloat16>(bfloat16_data, H, W);
    }

    void print() {
        std::cout << "Dense Matrix:" << std::endl;
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                std::cout << data[i * W + j] << " ";
            }
            std::cout << std::endl;
        }

    }

    dense_matrix<T> gemm(const dense_matrix<T> &other) {
        assert(W == other.H);
        dense_matrix<T> result(H, other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < other.W; ++j) {
                T sum = 0;
                for (size_t k = 0; k < W; ++k) {
                    sum += data[i * W + k] * other.data[k * other.W + j];
                }
                result.data[i * other.W + j] = sum;
            }
        }
        return result;
    }

    dense_matrix<bfloat16> gemm_bfloat16(const dense_matrix<bfloat16> &other) {
        assert(W == other.H);
        dense_matrix<bfloat16> result(H, other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < other.W; ++j) {
                float sum = 0;
                for (size_t k = 0; k < W; ++k) {
                    sum += data[i * W + k].to_float() * other.data[k * other.W + j].to_float();
                }
                result.data[i * other.W + j] = bfloat16(sum);
            }
        }
        return result;
    }
    bool all_close(const dense_matrix<T> &other, T tol = 1e-3) {
        assert(H == other.H && W == other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                if (std::abs(data[i * W + j] - other.data[i * W + j]) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    bool all_close_bfloat16(const dense_matrix<bfloat16> &other, float tol = 1) {
        assert(H == other.H && W == other.W);
        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {

                if (std::abs(data[i * W + j].to_float() - other.data[i * W + j].to_float()) > tol) {
                    return false;
                }
            }
        }
        return true;
    }
};

template <typename T>
class bsr_matrix {
public: // everything is public for now
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<T> data;

    size_t H;
    size_t W;
    size_t nblocks;
    size_t R;
    size_t C;

public:
    bsr_matrix() : H(0), W(0), nblocks(0), R(0), C(0) {}
    bsr_matrix(size_t rows, size_t cols, size_t block_rows, size_t block_cols, size_t num_blocks, int fill_type = FILL_ROW, bool random = RAND) :
        H(rows), W(cols), R(block_rows), C(block_cols), nblocks(num_blocks) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);

        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);
        std::fill(indptr.begin(), indptr.end(), 0);
        // if Fill_row, fill the matrix with blocks in a row-wise manner
        // if Fill_col, fill the matrix with blocks in a column-wise manner
        if (fill_type == FILL_ROW) {
            for (size_t i = 0; i < blocked_matrix_height; i++) {
                for (size_t j = 0; j < blocked_matrix_width; j++) {
                    if (i * blocked_matrix_width + j < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        for (size_t k = 0; k < R * C; k++) {
                            if (random) {
                                data.push_back(static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
                            } else {
                                data.push_back(k);
                            }
                        }
                    }
                }
            }
        } else if (fill_type == FILL_COL) {
            for (size_t j = 0; j < blocked_matrix_width; j++) {
                for (size_t i = 0; i < blocked_matrix_height; i++) {
                    if (i + (j * blocked_matrix_height) < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        for (size_t k = 0; k < R * C; k++) {
                            if (random) {
                                data.push_back(static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
                            } else {
                                data.push_back(k);
                            }
                        }
                    }
                }
            }
        } else if (fill_type == FILL_DIAG){
            assert(nblocks <= std::min(blocked_matrix_height, blocked_matrix_width));
            for (size_t i = 0; i < nblocks; i++) {
                indptr[i + 1]++;
                indices.push_back(i);
                for (size_t k = 0; k < R * C; k++) {
                    if (random) {
                        data.push_back(static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
                    } else {
                        data.push_back(k);
                    }
                }
            }

        } else {
            throw std::invalid_argument("Invalid fill type");
        }
        for (size_t i = 1; i < indptr.size(); i++) {
            indptr[i] += indptr[i - 1];
        }
        indptr.resize(blocked_matrix_height + 1);
    }

    bsr_matrix(size_t rows, size_t cols, size_t block_rows, size_t block_cols, size_t num_blocks, int fill_type = FILL_ROW) :
        H(rows), W(cols), R(block_rows), C(block_cols), nblocks(num_blocks) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);

        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);
        std::fill(indptr.begin(), indptr.end(), 0);

        // if Fill_row, fill the matrix with blocks in a row-wise manner
        // if Fill_col, fill the matrix with blocks in a column-wise manner
        if (fill_type == FILL_ROW) {
            for (size_t i = 0; i < blocked_matrix_height; i++) {
                for (size_t j = 0; j < blocked_matrix_width; j++) {
                    if (i * blocked_matrix_width + j < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        for (size_t k = 0; k < R * C; k++) {
                            data.push_back(static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
                        }
                    }
                }
            }
        } else if (fill_type == FILL_COL) {
            for (size_t j = 0; j < blocked_matrix_width; j++) {
                for (size_t i = 0; i < blocked_matrix_height; i++) {
                    if (i * blocked_matrix_width + j < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(i);
                        for (size_t k = 0; k < R * C; k++) {
                            data.push_back(static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
                        }
                    }
                }
            }
        } else {
            throw std::invalid_argument("Invalid fill type");
        }
    }


    // construct a copy of the dense matrix in this bsr matrix
    // this is the first function that caters to TT hardware by
    // picking block size a multiple of tile size
    // REQUIRES: other has been tilized by the TT utils
    bsr_matrix(dense_matrix<T>& other, uint32_t right_outer_dim = 0) :
        H(other.H),
        W(other.W) 
        {
        // how do we choose block sizes?
        // largest square multiple of 32 that tilizes the dense matrix?
        // 32x32 for everything?
        // 128x128 for everything, falling back to smaller sizes if that doesn't fit? 
        // 64x64 for everything, falling back to smaller sizes. That's it because in0_block_w = 2.
        // TODO: change this to choose block sizes equal to whatever TT would pick for its dense block sizes, 
        //          using the sparse fit-in-SRAM function.
        //          maybe we maintain our own version of bmm_op.hpp after all...
        // TODO: the way we are sizing right now is a bit of a cheat, but it's good for now.
        // TODO: continue the cheat by passing number of cores to constructor


        uint32_t Nt = right_outer_dim > 0 ? right_outer_dim / TILE_WIDTH : std::max(H, W) / TILE_WIDTH;
        auto matmul_params = bmm_op_utils::get_large_matmul_params(H / TILE_HEIGHT, Nt, 8, 8, 2);
        uint32_t per_core_M = std::get<0>(matmul_params);
        uint32_t per_core_N = std::get<1>(matmul_params);
        uint32_t out_subblock_h = std::get<2>(matmul_params);
        uint32_t out_subblock_w = std::get<3>(matmul_params);

        
        // try to let Ct = 2, but accept it otherwise
        R = per_core_M * TILE_HEIGHT;
        C = std::min(per_core_N * TILE_WIDTH, 2 * TILE_WIDTH);
        // if (W % 64 == 0)
        //     C = 64;
        // if (H % 64 == 0)
        //     R = 64; /
        // R = 32;
        // C = 32;

        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;
        nblocks = blocked_matrix_height * blocked_matrix_width;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.resize(nblocks * R * C);

        size_t count = 0;
        for (int i = 0 ; i < nblocks; i++){
            size_t row = i / blocked_matrix_width;
            size_t col = i % blocked_matrix_width;
            indptr[row + 1]++;
            indices.push_back(col);
            for (int j = 0; j < R * C; j++){
                data.push_back(other.data[count++]);
            }
        }

        for (size_t i = 1; i < indptr.size(); i++) {
            indptr[i] += indptr[i - 1];
        }
        indptr.resize(blocked_matrix_height + 1);

        std::vector<T> tilized_input;
        tilized_input.reserve(other.data.size());

        const auto write_face = [](std::vector<T>& tilized_input,
                                const std::vector<T>& input,
                                uint32_t face_height,
                                uint32_t face_width,
                                uint32_t face_idx,
                                uint32_t n) -> void {
            for (uint32_t i = 0; i < face_height; i++) {
                for (uint32_t j = 0; j < face_width; j++) {
                    tilized_input.push_back(input[face_idx + j]);
                }
                face_idx += n;
            }
        };

        uint32_t TILE_HEIGHT = R;
        uint32_t TILE_WIDTH = C;
        uint32_t FACE_HEIGHT = R;
        uint32_t FACE_WIDTH = C;
        uint32_t row_tiles = H / TILE_HEIGHT;
        uint32_t col_tiles = W / TILE_WIDTH;
        uint32_t row_of_tiles_num_elements = TILE_HEIGHT * W;
        uint32_t tile_start = 0;
        for (uint32_t row_tile = 0; row_tile < row_tiles; row_tile++) {
            uint32_t row_tile_start = tile_start;
            for (uint32_t col_tile = 0; col_tile < col_tiles; col_tile++) {
                uint32_t face0_id = row_tile_start;
                uint32_t face1_id = face0_id + FACE_WIDTH;
                uint32_t face2_id = face0_id + W * FACE_HEIGHT;
                uint32_t face3_id = face2_id + FACE_WIDTH;

                write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face0_id, W);
                // write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face1_id, W);
                // write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face2_id, W);
                // write_face(tilized_input, other.data, FACE_HEIGHT, FACE_WIDTH, face3_id, W);
                row_tile_start += TILE_WIDTH;
            }
            tile_start += row_of_tiles_num_elements;
        }


        data = std::move(tilized_input);
    }

    // construct from custom data members. Most flexible constructor. 
    bsr_matrix(
        std::vector<T> data,
        std::vector<int> indptr,
        std::vector<int> indices,
        size_t rows,
        size_t cols,
        size_t block_rows,
        size_t block_cols,
        size_t num_blocks) :
        data(std::move(data)),
        indptr(std::move(indptr)),
        indices(std::move(indices)),
        H(rows),
        W(cols),
        R(block_rows),
        C(block_cols),
        nblocks(num_blocks) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);
    }

    bsr_matrix(int rows, int cols, size_t block_rows, size_t block_cols, size_t num_blocks, bool random = false) {
        H = rows;
        W = cols;
        R = block_rows;
        C = block_cols;
        nblocks = num_blocks;

        assert(H*W >= nblocks * R * C);
        assert(R > 0);
        assert(C > 0);
        assert(H % R == 0);
        assert(W % C == 0);
        assert(H >= R);
        assert(W >= C);

        size_t blocked_matrix_height = H / R;
        size_t blocked_matrix_width = W / C;

        indptr.resize(blocked_matrix_height + 1);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);
        std::fill(indptr.begin(), indptr.end(), 0);

        std::vector<int> block_indices(blocked_matrix_height * blocked_matrix_width);
        std::fill(block_indices.begin(), block_indices.begin() + nblocks, 1);
        std::fill(block_indices.begin() + nblocks, block_indices.end(), 0);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(block_indices.begin(), block_indices.end(), std::default_random_engine(seed));

        for (size_t i = 0; i < block_indices.size(); i++) {
            if (block_indices[i] == 1) {
                size_t row = i / blocked_matrix_width;
                size_t col = i % blocked_matrix_width;
                indptr[row + 1]++;
                indices.push_back(col);
                for (size_t j = 0; j < R * C; j++) {
                    if (random) {
                        data.push_back(static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
                    } else {
                        data.push_back(1);
                    }
                }
            }
        }
        for (size_t i = 1; i < indptr.size(); i++) {
            indptr[i] += indptr[i - 1];
        }
        indptr.resize(blocked_matrix_height + 1);
    }

    // Moves inptr, indices, and data to the new bsr_matrix
    bsr_matrix<bfloat16> bfloat16_cast() {
        std::vector<bfloat16> bfloat16_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            bfloat16_data[i] = bfloat16(data[i]);
        }
        return bsr_matrix<bfloat16>(bfloat16_data, indptr, indices, H, W, R, C, nblocks);
    }

        dense_matrix<bfloat16> spmm_bfloat16(dense_matrix<bfloat16> &B) {
            assert(W == B.H);
            dense_matrix<bfloat16> output(H, B.W);

            // FORALL parallelism on the block rows
            for (size_t i = 0; i < indptr.size() - 1; i++) {                        // runtime args
                for (size_t r = 0; r < R; r += TILE_SIZE) {                         // comptime args
                    for (size_t p = 0; p < B.W; p += TILE_SIZE) {                   // comptime args
                        std::vector<float> output_tile(TILE_SIZE * TILE_SIZE, 0);   // DST register
                        for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {  // reading raw data from a CB
                            size_t j = indices[idx];                                // reading raw data from a CB
                            auto iter_start = data.begin() + idx * R * C;           // src0_addr, determined from args
                            auto iter_B_start = B.data.begin() + j * C * B.W;       // src1_addr, determined from args
                            for (size_t c = 0; c < C; c += TILE_SIZE) {             // comptime args
                                // begin matmul_tiles API call
                                for (size_t rr = r; rr < std::min(r + TILE_SIZE, R); rr++) {
                                    for (size_t pp = p; pp < std::min(p + TILE_SIZE, B.W); pp++) {
                                        float sum = 0;
                                        for (size_t cc = c; cc < std::min(C, c + TILE_SIZE); cc++) {
                                            bfloat16 a_val = *(iter_start + rr * C + cc);
                                            bfloat16 b_val = *(iter_B_start + cc * B.W + pp);
                                            sum += a_val.to_float() * b_val.to_float();
                                        }
                                        output_tile[(rr - r) * TILE_SIZE + pp - p] += sum;
                                    }
                                }
                                // end matmul_tiles API call
                            }
                        }
                        // write tile to DRAM starting at tile i*R + r, p (output is dense, no more blocks)
                        // On TT:
                        // 1. pack DST reg to output CB
                        // 2. writer kernel pops from output CB
                        // 3. writer kernel NoC's to DRAM
                        for (size_t rr = r; rr < std::min(R, r + TILE_SIZE); rr++) {
                            for (size_t pp = p; pp < std::min(B.W, p + TILE_SIZE); pp++) {
                                *(output.data.begin() + (i * R + rr) * output.W + pp) =
                                    bfloat16(output_tile[(rr - r) * TILE_SIZE + pp - p]);
                            }
                        }
                    }
                }
            }
        return output;
    }

    dense_matrix<T> spmm(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                size_t j = indices[idx];
                auto iter_start = data.begin() + idx * R * C;
                auto iter_B_start = B.data.begin() + j * C * B.W;
                for (size_t r = 0; r < R; r++) {
                    for (size_t p = 0; p < B.W; p++) {
                        T sum = 0;
                        for (size_t c = 0; c < C; c++) {
                            T a_val = *(iter_start + r * C + c);
                            T b_val = *(iter_B_start + c * B.W + p);
                            sum += a_val * b_val;
                        }
                        *(output.data.begin() + (i * R + r) * output.W + p) += sum;
                    }
                }
            }
        }
        return output;
    }

    dense_matrix<T> omp_spmm(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);
        #pragma omp parallel for
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                size_t j = indices[idx];
                auto iter_start = data.begin() + idx * R * C;
                auto iter_B_start = B.data.begin() + j * C * B.W;
                for (size_t r = 0; r < R; r++) {
                    for (size_t p = 0; p < B.W; p++) {
                        T sum = 0;
                        #pragma omp reduction(+:sum)
                        for (size_t c = 0; c < C; c++) {
                            T a_val = *(iter_start + r * C + c);
                            T b_val = *(iter_B_start + c * B.W + p);
                            sum += a_val * b_val;
                        }
                        *(output.data.begin() + (i * R + r) * output.W + p) += sum;
                    }
                }
            }
        }
        return output;
    }

    dense_matrix<T> tiled_spmm_CPU(dense_matrix<T> &B) {
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);
        // FORALL parallelism on the block rows
        for (size_t i = 0; i < indptr.size() - 1; i++) { // host
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) { // host
                size_t j = indices[idx]; // host

                // now. read the code below as though it were the reader kernel on TT,
                // reader_bmm_8bank_output_tiles_partitioned.cpp
                // R --> M, Mt <-- R / TILE_SIZE
                // C --> K, Kt <-- C / TILE_SIZE
                // QUIRK: Kt = Ct, BUT KtNt = KtNt. Kt is used to index into A (of shape RxC)
                //     while KtNt is used to index into B (of shape CxN, or KxN)
                // B.W --> N, Nt <-- B.W / TILE_SIZE
                auto iter_start = data.begin() + idx * R * C;     // src0_addr
                auto iter_B_start = B.data.begin() + j * C * B.W; // src1_addr
                for (size_t r = 0; r < R; r += TILE_SIZE) {
                    for (size_t p = 0; p < B.W; p += TILE_SIZE) {
                        for (size_t rr = r; rr < std::min(R, r + TILE_SIZE); rr++) {
                            for (size_t pp = p; pp < std::min(B.W, p + TILE_SIZE); pp++) {
                                T sum = 0;
                                for (size_t c = 0; c < C; c += TILE_SIZE) {
                                    for (size_t cc = c; cc < std::min(C, c + TILE_SIZE); cc++) {
                                        T a_val = *(iter_start + rr * C + cc);
                                        T b_val = *(iter_B_start + cc * B.W + pp);
                                        sum += a_val * b_val;
                                    }
                                }
                                *(output.data.begin() + (i * R + rr) * output.W + pp) += sum;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }


    // DEAR READER, this is a tile-wise parallel implementation of SpMM on BSR matrices.
    // It is not optimized for CPU and in fact has several egregious inefficiencies from that perspective.
    // It is written to demonstrate the tile-wise parallel implementation on TT hardware.
    // (In fact, this version is not as numerically stable on CPU as the others).
    dense_matrix<T> tiled_spmm(dense_matrix<T> &B){
        /*
        Some Tenstorrent terminology:
        - tile: a contiguous block of 2048 bytes, usually 32x32 elements of a bfloat16 matrix.
                The APIs into the FPU and SFPU operate on tiles. The NoC architecture is optmized
                to send pages of sizes between 1-4MB, so the tile size is strategically set
                to benefit from both the compute engines and the dataflow architecture.
        - comptime args: the JIT compile time args to compute and dataflow kernels.
                         These can be set at the host's compile time or the host's runtime.

        */
        assert(W == B.H);
        dense_matrix<T> output(H, B.W);

        // FORALL parallelism on the block rows
        for (size_t i = 0; i < indptr.size() - 1; i++) { // runtime args
            for (size_t r = 0; r < R; r += TILE_SIZE) { // comptime args
                for (size_t p = 0; p < B.W; p += TILE_SIZE) { // comptime args
                    std::vector<T> output_tile(TILE_SIZE * TILE_SIZE, 0); // DST register
                    for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) { // reading raw data from a CB
                        size_t j = indices[idx];                               // reading raw data from a CB
                        auto iter_start = data.begin() + idx * R * C;     // src0_addr, determined from args
                        auto iter_B_start = B.data.begin() + j * C * B.W; // src1_addr, determined from args
                        for (size_t c = 0; c < C; c += TILE_SIZE) { // comptime args
                            // begin matmul_tiles API call
                            for (size_t rr = r; rr < std::min(r + TILE_SIZE, R); rr++) {
                                for (size_t pp = p; pp < std::min(p + TILE_SIZE, B.W); pp++) {
                                    T sum = 0;
                                    for (size_t cc = c; cc < std::min(C, c + TILE_SIZE); cc++) {
                                        T a_val = *(iter_start + rr * C + cc);
                                        T b_val = *(iter_B_start + cc * B.W + pp);
                                        sum += a_val * b_val;
                                    }
                                    output_tile[(rr - r) * TILE_SIZE + pp - p] += sum;
                                }
                            }
                            // end matmul_tiles API call
                        }
                    }
                    // write tile to DRAM starting at tile i*R + r, p (output is dense, no more blocks)
                    // On TT:
                    // 1. pack DST reg to output CB
                    // 2. writer kernel pops from output CB
                    // 3. writer kernel NoC's to DRAM
                    for (size_t rr = r; rr < std::min(R, r + TILE_SIZE); rr++) {
                        for (size_t pp = p; pp < std::min(B.W, p + TILE_SIZE); pp++) {
                            *(output.data.begin() + (i * R + rr) * output.W + pp) = output_tile[(rr - r) * TILE_SIZE + pp - p];
                        }
                    }
                }
            }
        }
        return output;
    }

    dense_matrix<T> to_dense() {
        dense_matrix<T> output(H, W);
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                size_t j = indices[idx];
                auto iter_start = data.begin() + idx * R * C;
                for (size_t r = 0; r < R; r++) {
                    for (size_t c = 0; c < C; c++) {
                        output.data[(i * R + r) * W + (j * C + c)] = *(iter_start + r * C + c);
                    }
                }
            }
        }
        return output;
    }

    void print() {
        std::cout << "BSR Matrix:" << std::endl;
        std::cout << "Size: " << H << " x " << W << std::endl;
        std::cout << "Block Size: " << R << " x " << C << std::endl;
        std::cout << "Number of blocks: " << nblocks << std::endl;
        std::cout << "Indptr:" << std::endl;
        for (size_t i = 0; i < indptr.size(); ++i) {
            std::cout << indptr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Indices:" << std::endl;
        for (size_t i = 0; i < indices.size(); ++i) {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Data:" << std::endl;
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }

    void pretty_print() {
                std::cout << "BSR Matrix:" << std::endl;
        std::cout << "Size: " << H << " x " << W << std::endl;
        std::cout << "Block Size: " << R << " x " << C << std::endl;
        std::cout << "Number of blocks: " << nblocks << std::endl;
        std::cout << "Indptr:" << std::endl;
        for (size_t i = 0; i < indptr.size(); ++i) {
            std::cout << indptr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Indices:" << std::endl;
        for (size_t i = 0; i < indices.size(); ++i) {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Data:" << std::endl;
        for (size_t i = 0; i < H / R; i++) {
            for (size_t j = 0; j < W / C; j++) {
                char nz = '_';
                for (size_t idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                    if (indices[idx] == j) {
                        nz = '*';
                        break;
                    }
                }
                std::cout << nz << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};

#endif  // BSR_MATRIX_HPP
