#include <vx_spawn.h>
#include <vx_tensor.h>
#include <VX_config.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstring>
#include "common.h"

// Upper bounds to allow runtime-configurable head_dim/block_size_c.
static constexpr uint32_t HEAD_DIM_MAX = 64;
static constexpr uint32_t BLOCK_SIZE_C_MAX = 16;

// Tensor core path follows configured TCU tile (fp16 inputs / fp32 accumulate).
namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_THREADS, vt::fp16, vt::fp32>;
static constexpr uint32_t TCU_M = tcu_ctx::tileM;
static constexpr uint32_t TCU_N = tcu_ctx::tileN;
static constexpr uint32_t TCU_K = tcu_ctx::tileK;
static_assert(TCU_N <= BLOCK_SIZE_C_MAX, "tensor tileN exceeds BLOCK_SIZE_C_MAX");
static_assert(TCU_M <= HEAD_DIM_MAX, "tensor tileM exceeds HEAD_DIM_MAX");

static inline uint16_t float_to_fp16(float value) {
    __fp16 h = (__fp16)value;
    uint16_t out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

template <uint32_t BLOCK_SIZE_R, uint32_t BLOCK_SIZE_C>
static void flashattention_simt_fixed(kernel_arg_t *arg) {
    // Compile-time specialization for head_dim == 8 to enable full unrolling.
    static_assert(BLOCK_SIZE_C <= BLOCK_SIZE_C_MAX, "block_size_c too large");
    constexpr uint32_t HEAD_DIM = 8;

    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto scale = arg->scale;
    bool causal = arg->causal != 0;

    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto l_row = threadIdx.x;
    auto g_row_offset = g_row * HEAD_DIM;
    auto l_row_offset = l_row * HEAD_DIM;

    auto local_ptr = __local_mem((BLOCK_SIZE_R + 2 * BLOCK_SIZE_C) * HEAD_DIM * sizeof(float));
    auto local_Q = (float*)local_ptr;
    auto local_K = (float*)local_Q + BLOCK_SIZE_R * HEAD_DIM;
    auto local_V = (float*)local_K + BLOCK_SIZE_C * HEAD_DIM;

    auto local_ptr = __local_mem((BLOCK_SIZE_R + 2 * BLOCK_SIZE_C) * HEAD_DIM * sizeof(float));
    auto local_Q = (float*)local_ptr;
    auto local_K = (float*)local_Q + BLOCK_SIZE_R * HEAD_DIM;
    auto local_V = (float*)local_K + BLOCK_SIZE_C * HEAD_DIM;

    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
        local_Q[l_row_offset + col] = Q_ptr[g_row_offset + col];

    // Initialize O_i in registers
    float O_buf[HEAD_DIM];
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
      O_buf[col] = 0.0f;

    // Create buffer to store row of S and P
    float sp_buf[BLOCK_SIZE_C_MAX];
    float sp_buf[BLOCK_SIZE_C_MAX];

    // Initialize m_i, l_i
    float m = -INFINITY;
    float l = 0.0f;

    float* Q_row = local_Q + l_row * HEAD_DIM;

    for (uint32_t j = 0; j < seq_len; j += BLOCK_SIZE_C) {
        uint32_t block_offset = j * HEAD_DIM;

        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C / BLOCK_SIZE_R; ++k) {
          auto row = k * BLOCK_SIZE_R + l_row;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C / BLOCK_SIZE_R; ++k) {
          auto row = k * BLOCK_SIZE_R + l_row;
          auto row_offset = row * HEAD_DIM;
          #pragma clang loop unroll(full)
          #pragma clang loop unroll(full)
          for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            auto offset = row_offset + col;
            local_K[offset] = K_ptr[block_offset + offset];
            local_V[col * BLOCK_SIZE_C + row] = V_ptr[block_offset + offset];
          }
        }

        __syncthreads();

        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k) {
          #pragma clang loop unroll(full)
          for (uint32_t elem = 0; elem < HEAD_DIM; ++elem)
            sp_buf[k] += Q_row[elem] * local_K[k * HEAD_DIM + elem];
          auto key_idx = j + k;
          sp_buf[k] = (causal && key_idx > g_row) ? -INFINITY : sp_buf[k] * scale;
        }

        float rowmax = sp_buf[0];
        #pragma clang loop unroll(full)
        for (uint32_t k = 1; k < BLOCK_SIZE_C; ++k)
          if (sp_buf[k] > rowmax) rowmax = sp_buf[k];

        // Compute exp even if rowmax is -INFINITY to avoid deadlock
        // (All threads must follow the same control flow)
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = expf(sp_buf[k] - rowmax);

        float rowsum = 0;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          rowsum += sp_buf[k];

        float new_m = (m > rowmax ? m : rowmax);
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);
        float new_l = old_weight * l + new_weight * rowsum;
        float new_l = old_weight * l + new_weight * rowsum;

        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < HEAD_DIM; ++k) {
          float dot = 0;
          #pragma clang loop unroll(full)
          for (uint32_t elem = 0; elem < BLOCK_SIZE_C; ++elem)
            dot += sp_buf[elem] * local_V[k * BLOCK_SIZE_C + elem];
          float old_contrib = old_weight * O_buf[k];
          float new_contrib = new_weight * dot;
          O_buf[k] = old_contrib + new_contrib;
        }

        m = new_m;
        l = new_l;

        __syncthreads();
    }

    float inv_l = (l > 0) ? (1.0f / l) : 0.0f;
    #pragma clang loop unroll(full)
    for (uint32_t k = 0; k < HEAD_DIM; ++k)
      O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

static void flashattention_simt_generic(kernel_arg_t *arg) {
    // Setup buffer arguments
    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto head_dim = arg->head_dim;
    auto block_size_r = arg->block_size_r;
    auto block_size_c = arg->block_size_c;
    auto scale = arg->scale;
    bool causal = arg->causal != 0;

    if (head_dim > HEAD_DIM_MAX || block_size_c > BLOCK_SIZE_C_MAX)
        return;

    // Allocate local memory
    // Must store tile of Q (b_r x d) + K, V (b_c x d)
    auto local_ptr = __local_mem((block_size_r + 2 * block_size_c) * head_dim * sizeof(float));
    auto local_Q = (float*)local_ptr;
    auto local_K = (float*)local_Q + block_size_r * head_dim;
    auto local_V = (float*)local_K + block_size_c * head_dim;

    // Determine global/local row index
    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto l_row = threadIdx.x;

    auto g_row_offset = g_row * head_dim;
    auto l_row_offset = l_row * head_dim;

    // Load Q_i from HBM
    for (uint32_t col = 0; col < head_dim; ++col)
        local_Q[l_row_offset + col] = Q_ptr[g_row_offset + col];

    // Initialize O_i in registers (accumulates l_i * O_i)
    float O_buf[HEAD_DIM_MAX];
    for (uint32_t col = 0; col < head_dim; ++col)
      O_buf[col] = 0.0f;

    // Create buffer to store row of S and P
    float sp_buf[BLOCK_SIZE_C_MAX];

    // Initialize m_i, l_i
    float m = -INFINITY;
    float l = 0.0f;

    // Thread's row of Q block
    float* Q_row = local_Q + l_row * head_dim;

    // Loop over blocks of K and V
    for (uint32_t j = 0; j < seq_len; j += block_size_c) {
        uint32_t block_offset = j * head_dim;

        // Load K_j and V_j^T
        // block_size_c % block_size_r = 0
        for (uint32_t k = 0; k < block_size_c / block_size_r; ++k) {
          auto row = k * block_size_r + l_row;
          auto row_offset = row * head_dim;
          for (uint32_t col = 0; col < head_dim; ++col) {
            auto offset = row_offset + col;
            local_K[offset] = K_ptr[block_offset + offset];
            // Store transpose of V_j
            local_V[col * block_size_c + row] = V_ptr[block_offset + offset];
          }
        }

        __syncthreads();

        // Thread's row of S_ij = Q_i · K_j^T
        // Compute dot product of thread's Q row and each row of K_j
        for (uint32_t k = 0; k < block_size_c; ++k)
          sp_buf[k] = 0.0f;
        for (uint32_t k = 0; k < block_size_c; ++k) {
          for (uint32_t elem = 0; elem < head_dim; ++elem)
            sp_buf[k] += Q_row[elem] * local_K[k * head_dim + elem];
          auto key_idx = j + k;
          sp_buf[k] = (causal && key_idx > g_row) ? -INFINITY : sp_buf[k] * scale;
        }

        // Row max
        float rowmax = sp_buf[0];
        for (uint32_t k = 1; k < block_size_c; ++k)
          if (sp_buf[k] > rowmax) rowmax = sp_buf[k];

        // Compute exp even if rowmax is -INFINITY to avoid deadlock
        // (All threads must follow the same control flow)
        // Threads row of P_ij
        for (uint32_t k = 0; k < block_size_c; ++k)
          sp_buf[k] = expf(sp_buf[k] - rowmax);

        // Row sum
        float rowsum = 0;
        for (uint32_t k = 0; k < block_size_c; ++k)
          rowsum += sp_buf[k];

        // Compute new m and l using streaming softmax update
        float new_m = (m > rowmax ? m : rowmax);
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);
        float new_l = old_weight * l + new_weight * rowsum;

        // Update O with normalized probabilities
        for (uint32_t k = 0; k < head_dim; ++k) {
          float dot = 0;
          for (uint32_t elem = 0; elem < block_size_c; ++elem)
            dot += sp_buf[elem] * local_V[k * block_size_c + elem];
          float old_contrib = old_weight * O_buf[k];
          float new_contrib = new_weight * dot;
          O_buf[k] = old_contrib + new_contrib;
        }

        // Update m and l for next block
        m = new_m;
        l = new_l;

        __syncthreads();
    }

    // Normalize O and write back to HBM
    float inv_l = (l > 0) ? (1.0f / l) : 0.0f;
    for (uint32_t k = 0; k < head_dim; ++k)
      O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

static void flashattention_tcu(kernel_arg_t *arg) {
    // Setup buffer arguments
    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto head_dim = arg->head_dim;
    auto block_size_r = arg->block_size_r; // expected 8 for TCU path
    auto block_size_c = arg->block_size_c; // TCU tileN
    auto scale = arg->scale;
    bool causal = arg->causal != 0;
    uint32_t padded_head = ((head_dim + TCU_K - 1) / TCU_K) * TCU_K;

    // Fallback to SIMT if the tile sizes don't match the tensor-core config.
    if (block_size_r != TCU_M || block_size_c != TCU_N || head_dim > HEAD_DIM_MAX) {
        // Fall back to SIMT. Use fixed 8x8 path when possible for unrolling.
        if (head_dim == 8 && block_size_r == 8 && block_size_c == 8)
            flashattention_simt_fixed<8, 8>(arg);
        else
            flashattention_simt_generic(arg);
        return;
    }

    // Allocate local memory: Q_fp16, K_fp16, scores (fp32), V (fp16)
    size_t qk_fp16_elems = (block_size_r + block_size_c) * padded_head;
    size_t p_fp16_elems  = block_size_r * block_size_c;
    size_t v_fp16_elems  = block_size_c * padded_head;
    size_t scores_elems  = block_size_r * padded_head;
    auto local_ptr = __local_mem((qk_fp16_elems + p_fp16_elems + v_fp16_elems) * sizeof(uint16_t)
                                 + scores_elems * sizeof(float));
    auto local_Q  = reinterpret_cast<uint16_t*>(local_ptr);
    auto local_K  = local_Q + block_size_r * padded_head;
    auto local_P  = local_K + block_size_c * padded_head;
    auto local_Vh = local_P + p_fp16_elems;
    auto local_S  = reinterpret_cast<float*>(local_Vh + v_fp16_elems);

    // Determine global/local row index
    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto l_row = threadIdx.x;

    auto g_row_offset = g_row * head_dim;
    auto l_row_offset = l_row * head_dim;

    // Load Q_i into shared memory, pad to padded_head
    for (uint32_t col = 0; col < padded_head; ++col) {
        if (col < head_dim)
            local_Q[l_row * padded_head + col] = float_to_fp16(Q_ptr[g_row_offset + col]);
        else
            local_Q[l_row * padded_head + col] = 0;
    }

    // Ensure all threads have loaded Q before proceeding
    __syncthreads();

    // Initialize O_i in registers (accumulates l_i * O_i)
    float O_buf[HEAD_DIM_MAX];
    for (uint32_t col = 0; col < head_dim; ++col)
      O_buf[col] = 0.0f;

    // Initialize m_i, l_i
    float m = -INFINITY;
    float l = 0.0f;

    tcu_ctx::fragment_a   fragA;
    tcu_ctx::fragment_b   fragB;
    tcu_ctx::fragment_acc fragC;
    tcu_ctx::fragment_a   fragP;
    tcu_ctx::fragment_b   fragV;
    tcu_ctx::fragment_acc fragO;

    // Loop over blocks of K and V
    for (uint32_t j = 0; j < seq_len; j += block_size_c) {
        uint32_t block_offset = j * head_dim;

        // Load K_j and V_j^T into shared memory (col-major fp16, padded to padded_head)
        for (uint32_t row = l_row; row < block_size_c; row += block_size_r) {
          auto row_offset = row * head_dim;
          for (uint32_t col = 0; col < padded_head; ++col) {
            auto offset = col * block_size_c + row;
            if (row < block_size_c && col < head_dim) {
              local_K[offset]  = float_to_fp16(K_ptr[block_offset + row_offset + col]);
              local_Vh[offset] = float_to_fp16(V_ptr[block_offset + row_offset + col]);
            } else {
              local_K[offset]  = 0;
              local_Vh[offset] = 0;
            }
          }
        }

        __syncthreads();

        // Compute S_ij = Q_i · K_j^T using tensor cores (tile across k-tiles)
        tcu_ctx::fill_fragment(fragC, 0);
        for (uint32_t kt = 0; kt < padded_head; kt += TCU_K) {
          auto a_ptr = local_Q + kt;
          auto b_ptr = local_K + kt * block_size_c;
          tcu_ctx::load_matrix_sync(fragA, a_ptr, padded_head);
          tcu_ctx::load_matrix_sync<vt::col_major>(fragB, b_ptr, block_size_c);
          tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
        }

        // Store the scores tile back to shared memory
        tcu_ctx::store_matrix_sync(local_S, fragC, block_size_c);

        __syncthreads();

        // Thread's row of S_ij
        float sp_buf[BLOCK_SIZE_C_MAX];
        for (uint32_t k = 0; k < block_size_c; ++k) {
          auto key_idx = j + k;
          float val = local_S[l_row * block_size_c + k] * scale;
          sp_buf[k] = (causal && key_idx > g_row) ? -INFINITY : val;
        }

        // Row max
        float rowmax = sp_buf[0];
        for (uint32_t k = 1; k < block_size_c; ++k)
          if (sp_buf[k] > rowmax) rowmax = sp_buf[k];

        // Compute exp even if rowmax is -INFINITY to avoid deadlock
        // (All threads must follow the same control flow)
        // Thread's row of P_ij
        for (uint32_t k = 0; k < block_size_c; ++k)
          sp_buf[k] = expf(sp_buf[k] - rowmax);

        // Row sum
        float rowsum = 0;
        for (uint32_t k = 0; k < block_size_c; ++k)
          rowsum += sp_buf[k];

        // Compute new m and l using streaming softmax update
        float new_m = (m > rowmax ? m : rowmax);
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);
        float new_l = old_weight * l + new_weight * rowsum;

        // Build P tile in fp16 (row-major)
        for (uint32_t col = 0; col < block_size_c; ++col) {
          uint16_t p_val = float_to_fp16(sp_buf[col]);
          local_P[l_row * block_size_c + col] = p_val;
        }

        __syncthreads();

        // Compute PV using tensor cores, tiled over head_dim
        for (uint32_t nt = 0; nt < padded_head; nt += TCU_K) {
          tcu_ctx::fill_fragment(fragO, 0);
          auto v_ptr = local_Vh + nt * block_size_c;
          tcu_ctx::load_matrix_sync(fragP, local_P, block_size_c);
          tcu_ctx::load_matrix_sync<vt::col_major>(fragV, v_ptr, block_size_c);
          tcu_ctx::mma_sync(fragO, fragP, fragV, fragO);
          tcu_ctx::store_matrix_sync(local_S + nt, fragO, padded_head);
        }

        __syncthreads();

        // Update O with streaming softmax combine (unnormalized, divide once at end)
        for (uint32_t k = 0; k < head_dim; ++k) {
          float dot = local_S[l_row * padded_head + k];
          float old_contrib = old_weight * O_buf[k];
          float new_contrib = new_weight * dot;
          O_buf[k] = old_contrib + new_contrib;
        }

        // Update m and l for next block
        m = new_m;
        l = new_l;

        __syncthreads();
    }

    // Normalize O and write back to HBM
    float inv_l = (l > 0) ? (1.0f / l) : 0.0f;
    for (uint32_t k = 0; k < head_dim; ++k)
    float inv_l = (l > 0) ? (1.0f / l) : 0.0f;
    for (uint32_t k = 0; k < head_dim; ++k)
      O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

static void kernel_body(kernel_arg_t *arg) {
    if (arg->kernel_type == 1) {
        flashattention_tcu(arg);
    } else {
        // Prefer fixed unrolled path for head_dim=8 and common block sizes.
        if (arg->head_dim == 8 && arg->block_size_r == 8 && arg->block_size_c == 8) {
            flashattention_simt_fixed<8, 8>(arg);
        } else if (arg->head_dim == 8 && arg->block_size_r == 4 && arg->block_size_c == 4) {
            flashattention_simt_fixed<4, 4>(arg);
        } else {
            flashattention_simt_generic(arg);
        }
    }
}

static void kernel_body(kernel_arg_t *arg) {
    if (arg->kernel_type == 1) {
        flashattention_tcu(arg);
    } else {
        // Prefer fixed unrolled path for head_dim=8 and common block sizes.
        if (arg->head_dim == 8 && arg->block_size_r == 8 && arg->block_size_c == 8) {
            flashattention_simt_fixed<8, 8>(arg);
        } else if (arg->head_dim == 8 && arg->block_size_r == 4 && arg->block_size_c == 4) {
            flashattention_simt_fixed<4, 4>(arg);
        } else {
            flashattention_simt_generic(arg);
        }
    }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
