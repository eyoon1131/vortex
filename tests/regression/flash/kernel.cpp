#include <vx_spawn.h>
#include <vx_tensor.h>
#include <VX_config.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "common.h"

// SIMT path constants (need to know at compile time for unrolling)
static constexpr uint32_t HEAD_DIM = 8;
static constexpr uint32_t BLOCK_SIZE_C = 4;

// TCU tile shape (8x8x8 fp16 inputs, fp32 accumulate)
namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<8, vt::fp16, vt::fp32>;
static constexpr uint32_t TCU_K = tcu_ctx::tileK;

static inline uint16_t f2h(float x) {
    __fp16 h = (__fp16)x;
    uint16_t out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

// head_dim=8, block_size_r=8, block_size_c=8 
static void flashattention_tcu(kernel_arg_t* arg) {
    if (arg->head_dim != 8 || arg->block_size_r != 8 || arg->block_size_c != 8) {
        // Fallback if dimensions don't match tile; SIMT path will handle.
        return;
    }

    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;

    // local memory: Q, K, V tiles in fp16 + scratch for scores in fp32
    auto local_ptr = __local_mem(
        3 * TCU_K * TCU_K * sizeof(uint16_t) +
        TCU_K * TCU_K * sizeof(float)
    );
    uint16_t* local_Q  = reinterpret_cast<uint16_t*>(local_ptr);
    uint16_t* local_K  = local_Q  + TCU_K * TCU_K;
    uint16_t* local_Vh = local_K  + TCU_K * TCU_K;
    float*    local_S  = reinterpret_cast<float*>(local_Vh + TCU_K * TCU_K);

    uint32_t g_row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t l_row = threadIdx.x;

    // load Q row (fp32 -> fp16, pad to 8)
    for (uint32_t c = 0; c < 8; ++c)
        local_Q[l_row * TCU_K + c] = f2h(Q_ptr[g_row * 8 + c]);
    for (uint32_t c = 8; c < TCU_K; ++c)
        local_Q[l_row * TCU_K + c] = 0;

    float O_buf[8] = {0};
    float m = -INFINITY, l = 0;

    tcu_ctx::fragment_a   fragA;
    tcu_ctx::fragment_b   fragB;
    tcu_ctx::fragment_acc fragC;
    tcu_ctx::fragment_b   fragV;
    tcu_ctx::fragment_acc fragPV;

    for (uint32_t j = 0; j < seq_len; j += 8) {
        // load K block (transpose on the fly into col-major)
        for (uint32_t r = 0; r < 8; ++r) {
            for (uint32_t c = 0; c < 8; ++c)
                local_K[c * TCU_K + r] = f2h(K_ptr[(j + r) * 8 + c]);
            for (uint32_t c = 8; c < TCU_K; ++c)
                local_K[c * TCU_K + r] = 0;
        }
        for (uint32_t r = 8; r < TCU_K; ++r)
            for (uint32_t c = 0; c < TCU_K; ++c)
                local_K[c * TCU_K + r] = 0;

        // load V block transposed
        for (uint32_t r = 0; r < 8; ++r) {
            for (uint32_t c = 0; c < 8; ++c)
                local_Vh[c * TCU_K + r] = f2h(V_ptr[(j + r) * 8 + c]);
            for (uint32_t c = 8; c < TCU_K; ++c)
                local_Vh[c * TCU_K + r] = 0;
        }
        for (uint32_t r = 8; r < TCU_K; ++r)
            for (uint32_t c = 0; c < TCU_K; ++c)
                local_Vh[c * TCU_K + r] = 0;

        __syncthreads();

        // S = Q * K^T
        tcu_ctx::fill_fragment(fragC, 0);
        tcu_ctx::load_matrix_sync(fragA, local_Q, TCU_K);
        tcu_ctx::load_matrix_sync<vt::col_major>(fragB, local_K, TCU_K);
        tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
        tcu_ctx::store_matrix_sync(local_S, fragC, TCU_K);

        __syncthreads();

        float sp[8];
        for (uint32_t c = 0; c < 8; ++c)
            sp[c] = local_S[l_row * TCU_K + c];

        float rowmax = sp[0];
        for (uint32_t c = 1; c < 8; ++c)
            if (sp[c] > rowmax) rowmax = sp[c];

        for (uint32_t c = 0; c < 8; ++c)
            sp[c] = expf(sp[c] - rowmax);

        float rowsum = 0;
        for (uint32_t c = 0; c < 8; ++c)
            rowsum += sp[c];

        float new_m = fmaxf(m, rowmax);
        float old_w = expf(m - new_m);
        float new_w = expf(rowmax - new_m);
        float new_l = old_w * l + new_w * rowsum;

        //  P = softmax block, pad to 8x8 
        uint16_t* local_P = local_K; // reuse K storage
        for (uint32_t c = 0; c < TCU_K; ++c)
            local_P[l_row * TCU_K + c] = (c < 8 ? f2h(sp[c]) : 0);

        __syncthreads();

        // PV 
        tcu_ctx::fill_fragment(fragPV, 0);
        tcu_ctx::load_matrix_sync(fragA, local_P, TCU_K);
        tcu_ctx::load_matrix_sync<vt::col_major>(fragV, local_Vh, TCU_K);
        tcu_ctx::mma_sync(fragPV, fragA, fragV, fragPV);
        tcu_ctx::store_matrix_sync(local_S, fragPV, TCU_K);

        __syncthreads();

        for (uint32_t c = 0; c < 8; ++c) {
            float dot = local_S[l_row * TCU_K + c];
            O_buf[c] = old_w * O_buf[c] + new_w * dot;
        }

        m = new_m;
        l = new_l;
    }

    float inv_l = 1.f / l;
    for (uint32_t c = 0; c < 8; ++c)
        O_ptr[g_row * 8 + c] = O_buf[c] * inv_l;
}

void kernel_body(kernel_arg_t *arg) {
    // Setup buffer arguments
    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto block_size_r = arg->block_size_r;

    // Allocate local memory
    // Must store tile of Q (b_r x d) +  K, V (b_c x d)
    auto local_ptr = __local_mem((block_size_r + 2 * BLOCK_SIZE_C) * HEAD_DIM * sizeof(float));
    auto local_Q = (float*)local_ptr;
    auto local_K = (float*)local_Q + block_size_r * HEAD_DIM;
    auto local_V = (float*)local_K + BLOCK_SIZE_C * HEAD_DIM;

    // Determine global/local row index
    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto l_row = threadIdx.x;

    auto g_row_offset = g_row * HEAD_DIM;
    auto l_row_offset = l_row * HEAD_DIM;

    // Load Q_i from HBM
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
        local_Q[l_row_offset + col] = Q_ptr[g_row_offset + col];

    // Initialize O_i in registers
    float O_buf[HEAD_DIM];
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
      O_buf[col] = 0.0f;

    // Create buffer to store row of S and P
    float sp_buf[BLOCK_SIZE_C];

    // Initialize m_i, l_i
    float m = -INFINITY;
    float l = 0.0f;

    // Thread's row of Q block
    float* Q_row = local_Q + l_row * HEAD_DIM;

    // Loop over blocks of K and V
    for (uint32_t j = 0; j < seq_len; j += BLOCK_SIZE_C) {
        uint32_t block_offset = j * HEAD_DIM;

        // Load K_j and V_j^T
        // BLOCK_SIZE_C % block_size_r = 0
        for (uint32_t k = 0; k < BLOCK_SIZE_C / block_size_r; ++k) {
          auto row = k * block_size_r + l_row;
          auto row_offset = row * HEAD_DIM;
          for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            auto offset = row_offset + col;
            local_K[offset] = K_ptr[block_offset + offset];
            // Store transpose of V_j
            local_V[col * BLOCK_SIZE_C + row] = V_ptr[block_offset + offset];
          }
        }

        __syncthreads();

        // Thread's row of S_ij = Q_i · K_j^T
        // Compute dot product of thread's Q row and each row of K_j
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k) {
          // sp_buf[k] = 0;
          #pragma clang loop unroll(full)
          for (uint32_t elem = 0; elem < HEAD_DIM; ++elem)
            sp_buf[k] += Q_row[elem] * local_K[k * HEAD_DIM + elem];
        }

        // Row max
        float rowmax = sp_buf[0];
        #pragma clang loop unroll(full)
        for (uint32_t k = 1; k < BLOCK_SIZE_C; ++k)
          if (sp_buf[k] > rowmax) rowmax = sp_buf[k];

        // Threads row of P_ij
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = expf(sp_buf[k] - rowmax);

        // Row sum
        float rowsum = 0;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          rowsum += sp_buf[k];

        // Compute new m and l
        float new_m = (m > rowmax ? m : rowmax);
        float new_l = expf(m - new_m) * l + expf(rowmax - new_m) * rowsum;

        // Weights of old and new O
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);

        // Update O
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < HEAD_DIM; ++k) {
          // Compute dot product of thread's P_ij row and each col of V_j
          float dot = 0;
          #pragma clang loop unroll(full)
          for (uint32_t elem = 0; elem < BLOCK_SIZE_C; ++elem)
            dot += sp_buf[elem] * local_V[k * BLOCK_SIZE_C + elem];
          O_buf[k] = old_weight * O_buf[k] + new_weight * dot;
        }

        // Update m and l for next block
        m = new_m;
        l = new_l;
    }

    // Normalize O and write back to HBM
    float inv_l = 1.0f / l;
    #pragma clang loop unroll(full)
    for (uint32_t k = 0; k < HEAD_DIM; ++k)
      O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

// TCU when kernel_type == 1 and dims match, else SIMT.
void kernel_body_dispatch(kernel_arg_t* arg) {
    if (arg->kernel_type == 1 && arg->head_dim == 8 && arg->block_size_r == 8 && arg->block_size_c == 8) {
        flashattention_tcu(arg);
    } else {
        kernel_body(arg);
    }
}

int main() {
    auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(
        1,
        arg->grid_dim,
        arg->block_dim,
        (vx_kernel_func_cb)kernel_body_dispatch,
        arg
    );
}
