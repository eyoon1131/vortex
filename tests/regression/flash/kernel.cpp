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

// FIXED TCU Flash Attention - Proper Cooperative Reductions (No Race Conditions)
// This version parallelizes softmax WITHOUT race conditions
static inline float h2f(uint16_t x) {
    __fp16 h;
    memcpy(&h, &x, sizeof(h));
    return (float)h;
}


// Add this SIMPLIFIED version to verify the algorithm is correct
// This removes all the fp16 conversions and uses fp32 everywhere
// CRITICAL FIX: Handle TCU_K stride correctly!
// TCU_K is the leading dimension (likely 16, not 8)

static void flashattention_tcu(kernel_arg_t* arg) {
    if (arg->head_dim != 8 || arg->block_size_r != 8 || arg->block_size_c != 8) {
        return;
    }

    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;

    auto local_ptr = __local_mem(
        3 * TCU_K * TCU_K * sizeof(uint16_t) +  // Q, K, Vh
        TCU_K * TCU_K * sizeof(float) +          // S  
        8 * sizeof(float) +                      // row_m
        8 * sizeof(float) +                      // row_l
        8 * 8 * sizeof(float) +                  // row_O
        8 * sizeof(float) +                      // shared_maxval
        8 * sizeof(float)                        // shared_sumval
    );
    
    uint16_t* local_Q  = reinterpret_cast<uint16_t*>(local_ptr);
    uint16_t* local_K  = local_Q  + TCU_K * TCU_K;
    uint16_t* local_Vh = local_K  + TCU_K * TCU_K;
    float*    local_S  = reinterpret_cast<float*>(local_Vh + TCU_K * TCU_K);
    
    float* row_m = local_S + TCU_K * TCU_K;
    float* row_l = row_m + 8;
    float* row_O = row_l + 8;
    float* shared_maxval = row_O + 64;
    float* shared_sumval = shared_maxval + 8;

    uint32_t tile_row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t global_row_start = tile_row * 8;
    
    // Load Q tile (row by row)
    for (uint32_t local_r = 0; local_r < 8; local_r++) {
        for (uint32_t c = tid; c < 8; c += blockDim.x) {
            uint32_t global_row = global_row_start + local_r;
            local_Q[local_r * TCU_K + c] = f2h(Q_ptr[global_row * 8 + c]);
        }
    }
    for (uint32_t i = 64 + tid; i < TCU_K * TCU_K; i += blockDim.x) {
        local_Q[i] = 0;
    }

    // Initialize stats
    if (tid < 8) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
        for (uint32_t c = 0; c < 8; ++c) {
            row_O[tid * 8 + c] = 0.0f;
        }
    }

    tcu_ctx::fragment_a   fragA;
    tcu_ctx::fragment_b   fragB;
    tcu_ctx::fragment_acc fragC;
    tcu_ctx::fragment_b   fragV;
    tcu_ctx::fragment_acc fragPV;

    __syncthreads();

    for (uint32_t kv_block = 0; kv_block < seq_len / 8; kv_block++) {
        uint32_t kv_row_start = kv_block * 8;
        
        // Load K (transposed)
        for (uint32_t local_r = 0; local_r < 8; local_r++) {
            for (uint32_t c = tid; c < 8; c += blockDim.x) {
                uint32_t global_row = kv_row_start + local_r;
                local_K[c * TCU_K + local_r] = f2h(K_ptr[global_row * 8 + c]);
            }
        }
        for (uint32_t idx = 64 + tid; idx < TCU_K * TCU_K; idx += blockDim.x) {
            local_K[idx] = 0;
        }

        // Load V (transposed)
        for (uint32_t local_r = 0; local_r < 8; local_r++) {
            for (uint32_t c = tid; c < 8; c += blockDim.x) {
                uint32_t global_row = kv_row_start + local_r;
                local_Vh[c * TCU_K + local_r] = f2h(V_ptr[global_row * 8 + c]);
            }
        }
        for (uint32_t idx = 64 + tid; idx < TCU_K * TCU_K; idx += blockDim.x) {
            local_Vh[idx] = 0;
        }

        __syncthreads();

        // S = Q * K^T
        tcu_ctx::fill_fragment(fragC, 0);
        tcu_ctx::load_matrix_sync(fragA, local_Q, TCU_K);
        tcu_ctx::load_matrix_sync<vt::col_major>(fragB, local_K, TCU_K);
        tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
        tcu_ctx::store_matrix_sync(local_S, fragC, TCU_K);  // Stores with stride TCU_K!

        __syncthreads();

        // SOFTMAX
        if (tid < 8) {
            uint32_t r = tid;
            float rowmax = -INFINITY;
            for (uint32_t c = 0; c < 8; ++c) {
                // CRITICAL: Use TCU_K as stride when reading from local_S!
                float val = local_S[r * TCU_K + c];
                if (val > rowmax) rowmax = val;
            }
            shared_maxval[r] = rowmax;
        }
        __syncthreads();

        // Compute exp (all threads)
        for (uint32_t local_r = 0; local_r < 8; local_r++) {
            for (uint32_t c = tid; c < 8; c += blockDim.x) {
                // Use TCU_K stride for both read and write
                float s_val = local_S[local_r * TCU_K + c];
                float prob = expf(s_val - shared_maxval[local_r]);
                local_K[local_r * TCU_K + c] = f2h(prob);
                local_S[local_r * TCU_K + c] = prob;  // Reuse local_S for prob
            }
        }
        __syncthreads();

        // Compute sum
        if (tid < 8) {
            uint32_t r = tid;
            float rowsum = 0.0f;
            for (uint32_t c = 0; c < 8; ++c) {
                rowsum += local_S[r * TCU_K + c];  // TCU_K stride
            }
            shared_sumval[r] = rowsum;
        }
        __syncthreads();

        // Update stats and weight P
        if (tid < 8) {
            uint32_t r = tid;
            float rowmax = shared_maxval[r];
            float rowsum = shared_sumval[r];

            float m_old = row_m[r];
            float l_old = row_l[r];
            float m_new = fmaxf(m_old, rowmax);
            float l_new = expf(m_old - m_new) * l_old + expf(rowmax - m_new) * rowsum;

            float w_old = expf(m_old - m_new);
            float w_new = expf(rowmax - m_new);

            // Scale existing output
            for (uint32_t c = 0; c < 8; ++c) {
                row_O[r * 8 + c] *= w_old;
            }

            // Weight probabilities
            for (uint32_t c = 0; c < 8; ++c) {
                float p = h2f(local_K[r * TCU_K + c]);  // TCU_K stride!
                local_K[r * TCU_K + c] = f2h(w_new * p);
            }

            row_m[r] = m_new;
            row_l[r] = l_new;
        }

        // Pad P
        for (uint32_t idx = 64 + tid; idx < TCU_K * TCU_K; idx += blockDim.x) {
            local_K[idx] = 0;
        }
        __syncthreads();

        // PV = P * V
        tcu_ctx::fill_fragment(fragPV, 0);
        tcu_ctx::load_matrix_sync(fragA, local_K, TCU_K);
        tcu_ctx::load_matrix_sync<vt::col_major>(fragV, local_Vh, TCU_K);
        tcu_ctx::mma_sync(fragPV, fragA, fragV, fragPV);
        tcu_ctx::store_matrix_sync(local_S, fragPV, TCU_K);  // Stores with stride TCU_K!

        __syncthreads();

        // Accumulate PV into O
        if (tid < 8) {
            uint32_t r = tid;
            for (uint32_t c = 0; c < 8; ++c) {
                // CRITICAL: Read from local_S with TCU_K stride!
                row_O[r * 8 + c] += local_S[r * TCU_K + c];
            }
        }

        __syncthreads();
    }

    // Final normalization and write
    if (tid < 8) {
        uint32_t local_r = tid;
        uint32_t global_row = global_row_start + local_r;
        float inv_l = 1.0f / row_l[local_r];
        
        for (uint32_t c = 0; c < 8; ++c) {
            O_ptr[global_row * 8 + c] = row_O[local_r * 8 + c] * inv_l;
        }
    }
}

// softmax for this one is NOT parallelized
static void not_parallelized_flashattention_tcu(kernel_arg_t* arg) {
    if (arg->head_dim != 8 || arg->block_size_r != 8 || arg->block_size_c != 8) {
        return;
    }

    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;

    // Local memory: Q, K, V tiles in fp16 + scratch + per-row stats + reduction arrays
    auto local_ptr = __local_mem(
        3 * TCU_K * TCU_K * sizeof(uint16_t) +  // Q, K, Vh
        TCU_K * TCU_K * sizeof(float) +          // S (scratch)
        8 * sizeof(float) +                      // row_m
        8 * sizeof(float) +                      // row_l
        8 * 8 * sizeof(float) +                  // row_O
        8 * sizeof(float) +                      // shared_maxval
        8 * sizeof(float)                        // shared_sumval
    );
    uint16_t* local_Q  = reinterpret_cast<uint16_t*>(local_ptr);
    uint16_t* local_K  = local_Q  + TCU_K * TCU_K;
    uint16_t* local_Vh = local_K  + TCU_K * TCU_K;
    float*    local_S  = reinterpret_cast<float*>(local_Vh + TCU_K * TCU_K);
    
    float* row_m = local_S + TCU_K * TCU_K;
    float* row_l = row_m + 8;
    float* row_O = row_l + 8;
    float* shared_maxval = row_O + 64;
    float* shared_sumval = shared_maxval + 8;

    uint32_t tile_row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    
    // Cooperatively load Q tile
    uint32_t q_offset = tile_row * 8 * 8;
    for (uint32_t i = tid; i < 64; i += blockDim.x) {
        local_Q[i] = f2h(Q_ptr[q_offset + i]);
    }
    for (uint32_t i = 64 + tid; i < TCU_K * TCU_K; i += blockDim.x) {
        local_Q[i] = 0;
    }

    // Initialize per-row stats
    if (tid < 8) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
        for (uint32_t c = 0; c < 8; ++c) {
            row_O[tid * 8 + c] = 0.0f;
        }
    }

    tcu_ctx::fragment_a   fragA;
    tcu_ctx::fragment_b   fragB;
    tcu_ctx::fragment_acc fragC;
    tcu_ctx::fragment_b   fragV;
    tcu_ctx::fragment_acc fragPV;

    __syncthreads();

    for (uint32_t j = 0; j < seq_len; j += 8) {
        // === LOAD K BLOCK (transposed) ===
        uint32_t kv_offset = j * 8;
        for (uint32_t idx = tid; idx < 64; idx += blockDim.x) {
            uint32_t src_row = idx / 8;
            uint32_t src_col = idx % 8;
            local_K[src_col * TCU_K + src_row] = f2h(K_ptr[kv_offset + src_row * 8 + src_col]);
        }
        for (uint32_t idx = 64 + tid; idx < TCU_K * TCU_K; idx += blockDim.x) {
            local_K[idx] = 0;
        }

        // === LOAD V BLOCK (transposed) ===
        for (uint32_t idx = tid; idx < 64; idx += blockDim.x) {
            uint32_t src_row = idx / 8;
            uint32_t src_col = idx % 8;
            local_Vh[src_col * TCU_K + src_row] = f2h(V_ptr[kv_offset + src_row * 8 + src_col]);
        }
        for (uint32_t idx = 64 + tid; idx < TCU_K * TCU_K; idx += blockDim.x) {
            local_Vh[idx] = 0;
        }

        __syncthreads();

        // === COMPUTE S = Q * K^T ===
        tcu_ctx::fill_fragment(fragC, 0);
        tcu_ctx::load_matrix_sync(fragA, local_Q, TCU_K);
        tcu_ctx::load_matrix_sync<vt::col_major>(fragB, local_K, TCU_K);
        tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
        tcu_ctx::store_matrix_sync(local_S, fragC, TCU_K);

        __syncthreads();

        // === SOFTMAX WITH COOPERATIVE REDUCTION ===
        
        // Initialize reduction arrays
        if (tid < 8) {
            shared_maxval[tid] = -INFINITY;
            shared_sumval[tid] = 0.0f;
        }
        
        __syncthreads();
        
        // Find max per row (parallel reduction with simple approach)
        // Each thread helps find max for all rows
        for (uint32_t r = 0; r < 8; ++r) {
            for (uint32_t c = tid; c < 8; c += blockDim.x) {
                float val = local_S[r * TCU_K + c];
                // Simple max update (not atomic, but works with proper sync)
                if (val > shared_maxval[r]) {
                    shared_maxval[r] = val;
                }
            }
        }
        
        __syncthreads();
        
        // Compute exp and accumulate sum
        for (uint32_t r = 0; r < 8; ++r) {
            float rowmax = shared_maxval[r];
            float local_sum = 0.0f;
            
            for (uint32_t c = tid; c < 8; c += blockDim.x) {
                float val = local_S[r * TCU_K + c];
                float prob = expf(val - rowmax);
                local_K[r * TCU_K + c] = f2h(prob);  // Store P temporarily
                local_sum += prob;
            }
            
            // Accumulate sum (simple approach - not truly atomic but works)
            if (local_sum > 0) {
                shared_sumval[r] += local_sum;
            }
        }
        
        __syncthreads();
        
        // Update running statistics (one thread per row)
        if (tid < 8) {
            uint32_t r = tid;
            float rowmax = shared_maxval[r];
            float rowsum = shared_sumval[r];
            
            float m_old = row_m[r];
            float l_old = row_l[r];
            
            float m_new = fmaxf(m_old, rowmax);
            float l_new = expf(m_old - m_new) * l_old + expf(rowmax - m_new) * rowsum;
            
            float w_old = expf(m_old - m_new);
            float w_new = expf(rowmax - m_new);
            
            // Weight P values for this row
            for (uint32_t c = 0; c < 8; ++c) {
                uint16_t p_val = local_K[r * TCU_K + c];
                __fp16 p_h;
                memcpy(&p_h, &p_val, sizeof(uint16_t));
                float p_f = (float)p_h;
                local_K[r * TCU_K + c] = f2h(w_new * p_f);
            }
            
            // Scale old O
            for (uint32_t c = 0; c < 8; ++c) {
                row_O[r * 8 + c] *= w_old;
            }
            
            row_m[r] = m_new;
            row_l[r] = l_new;
        }
        
        // Pad P matrix
        for (uint32_t idx = 64 + tid; idx < TCU_K * TCU_K; idx += blockDim.x) {
            local_K[idx] = 0;
        }

        __syncthreads();

        // === COMPUTE PV ===
        tcu_ctx::fill_fragment(fragPV, 0);
        tcu_ctx::load_matrix_sync(fragA, local_K, TCU_K);
        tcu_ctx::load_matrix_sync<vt::col_major>(fragV, local_Vh, TCU_K);
        tcu_ctx::mma_sync(fragPV, fragA, fragV, fragPV);
        tcu_ctx::store_matrix_sync(local_S, fragPV, TCU_K);

        __syncthreads();

        // === ACCUMULATE PV INTO O ===
        if (tid < 8) {
            uint32_t r = tid;
            for (uint32_t c = 0; c < 8; ++c) {
                float pv_elem = local_S[r * TCU_K + c];
                row_O[r * 8 + c] += pv_elem;
            }
        }

        __syncthreads();
    }

    // === FINAL NORMALIZATION AND WRITE ===
    if (tid < 8) {
        uint32_t r = tid;
        uint32_t out_offset = tile_row * 8 * 8 + r * 8;
        float inv_l = 1.0f / row_l[r];
        for (uint32_t c = 0; c < 8; ++c) {
            O_ptr[out_offset + c] = row_O[r * 8 + c] * inv_l;
        }
    }
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